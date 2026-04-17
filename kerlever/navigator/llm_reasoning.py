"""Strategy Navigator LLM reasoning — Phase 3 LLM-based decision making.

Assembles context for the LLM, parses structured responses, validates
decisions against tabu/exhausted constraints, and implements retry logic.

Spec: docs/navigator/spec.md §6.3
"""

from __future__ import annotations

import json
import re

from kerlever.llm_client import LLMClientProtocol
from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DerivedSignals, LLMDecision
from kerlever.types import (
    CrossCandidateAnalysis,
    Mode,
    OptimizationState,
    SubMode,
    TabuEntry,
)

_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

_VALID_EXPLOIT_SUB_MODES: frozenset[str] = frozenset(
    {"param_search", "local_rewrite", "pattern_apply"}
)
_VALID_EXPLORE_SUB_MODES: frozenset[str] = frozenset({"de_novo", "recombination"})

_SYSTEM_PROMPT = """\
You are an optimization strategy advisor for CUDA kernel optimization. \
You will receive a summary of the current optimization state and must decide \
the next strategy.

Decide between:
- "exploit": Refine the current best kernel with targeted modifications.
- "explore": Try a fundamentally different approach.

If explore:
- "de_novo": Generate a completely new kernel from scratch.
- "recombination": Combine winning traits from multiple past candidates.

Provide your confidence level in the decision.

Return ONLY a single JSON object with exactly five fields:
{
    "mode": "exploit" | "explore",
    "direction": "<optimization_target_tag>",
    "sub_mode": "de_novo" | "recombination" | ... | null,
    "reasoning": "<1-2 sentence justification>",
    "confidence": "high" | "medium" | "low"
}

Do not include any other text outside the JSON object.\
"""


def assemble_llm_context(
    state: OptimizationState,
    signals: DerivedSignals,
    cross_analysis: CrossCandidateAnalysis | None,
    config: NavigatorConfig,
) -> str:
    """Assemble the user prompt for LLM reasoning, capped at budget.

    Items are included in priority order per spec §6.3. If the budget
    is exceeded, lower-priority items are truncated or omitted.

    Priority order:
    1. Current bottleneck tags
    2. Performance trend
    3. Top-3 candidates by latency
    4. Last round's mode, direction, improvement
    5. Exhausted directions
    6. Cross-candidate analysis
    7. Direction history with visit counts

    Args:
        state: Full optimization state.
        signals: Derived signals from Phase 1.
        cross_analysis: Cross-candidate analysis output (may be None).
        config: Navigator configuration.

    Returns:
        Assembled context string within token budget.
    """
    sections: list[str] = []

    # 1. Current bottleneck assessment: primary_tag and evidence
    if state.bottleneck_history:
        latest = state.bottleneck_history[-1]
        if latest.primary_tag is not None:
            evidence_str = ", ".join(f"{k}={v:.4f}" for k, v in latest.evidence.items())
            sections.append(
                f"Current bottleneck: {latest.primary_tag}"
                + (f" (evidence: {evidence_str})" if evidence_str else "")
            )

    # 2. Performance trend
    trend_parts = [f"avg_delta={signals.avg_delta:.4f}"]
    if signals.is_plateau:
        trend_parts.append("PLATEAU detected")
    if signals.is_regress:
        trend_parts.append("REGRESSION detected")
    sections.append(f"Performance trend: {', '.join(trend_parts)}")

    # 3. Top-3 candidates by objective score (from round summaries)
    rounds_with_score = [r for r in state.rounds if r.best_objective_score is not None]
    rounds_with_score.sort(
        key=lambda r: (
            r.best_objective_score
            if r.best_objective_score is not None
            else float("inf")
        )
    )
    top3 = rounds_with_score[:3]
    if top3:
        lines = []
        for r in top3:
            lines.append(
                f"  Round {r.round_number}: score={r.best_objective_score:.4f} "
                f"({r.mode.value} / {r.direction})"
            )
        sections.append("Top-3 rounds by objective score:\n" + "\n".join(lines))

    # 4. Last round info
    if state.rounds:
        last = state.rounds[-1]
        gain = last.rel_gain_vs_prev_best
        gain_str = f"{gain:.4f}" if gain is not None else "none"
        sections.append(
            f"Last round: mode={last.mode.value}, direction={last.direction}, "
            f"rel_gain={gain_str}"
        )

    # 5. Exhausted directions
    if signals.exhausted_directions:
        sections.append(
            f"Exhausted directions: {', '.join(sorted(signals.exhausted_directions))}"
        )

    # 6. Cross-candidate analysis
    if cross_analysis is not None:
        if cross_analysis.recombination_hints:
            hint_lines = []
            for hint in cross_analysis.recombination_hints[:2]:
                hint_lines.append(
                    "  "
                    f"{hint.hint_id}: parents={hint.parent_candidates}, "
                    f"gene_map={hint.gene_map}, "
                    f"evidence={hint.evidence_candidate_hashes}, "
                    f"confidence={hint.confidence}, "
                    f"risks={hint.risk_flags}"
                )
            sections.append("Structured recombination hints:\n" + "\n".join(hint_lines))
        if cross_analysis.avoid_patterns:
            avoid_lines = []
            for pattern in cross_analysis.avoid_patterns[:3]:
                evidence_str = ", ".join(
                    f"{key}={value:.4f}" for key, value in pattern.evidence.items()
                )
                avoid_lines.append(
                    "  "
                    f"{pattern.pattern} from {pattern.source_candidate_hash}: "
                    f"evidence={{{evidence_str}}}, "
                    f"shapes={pattern.affected_shape_ids}, "
                    f"confidence={pattern.confidence}, scope={pattern.scope}"
                )
            sections.append(
                "Structured avoid patterns (evidence context, not hard "
                "constraints):\n" + "\n".join(avoid_lines)
            )
        if cross_analysis.insights:
            sections.append(
                f"Cross-candidate insights: {'; '.join(cross_analysis.insights[:3])}"
            )
        if cross_analysis.recombination_suggestions:
            suggestions = cross_analysis.recombination_suggestions[:2]
            sections.append(f"Recombination suggestions: {'; '.join(suggestions)}")

    # 7. Direction history (from attempt records)
    if signals.direction_attempt_counts:
        dir_lines = [
            f"  {d}: {c} attempts"
            for d, c in sorted(signals.direction_attempt_counts.items())
        ]
        sections.append("Direction history:\n" + "\n".join(dir_lines))

    # Tabu entries context (important for the LLM to avoid tabu directions)
    active_tabu = [
        e for e in state.tabu_entries if e.expires_after_round >= state.current_round
    ]
    if active_tabu:
        tabu_strs = [
            f"{e.direction}(base={e.base_kernel_hash or 'any'})" for e in active_tabu
        ]
        sections.append(f"Active tabu entries: {', '.join(tabu_strs)}")

    # Join and truncate to budget (rough: 1 token ~= 4 chars)
    full_context = "\n\n".join(sections)
    max_chars = config.llm_context_budget * 4
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars]

    return full_context


def parse_llm_decision(raw: str) -> LLMDecision:
    """Parse the LLM's raw text response into an LLMDecision.

    Strips markdown code fences if present, then parses JSON.
    Validates the five expected fields.

    Args:
        raw: Raw text response from the LLM.

    Returns:
        Parsed LLMDecision.

    Raises:
        ValueError: If the response cannot be parsed or is missing fields.
    """
    cleaned = _strip_markdown_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Validate required fields
    required_fields = {"mode", "direction", "reasoning", "confidence"}
    missing = required_fields - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Parse mode
    mode_str = str(data["mode"]).upper()
    try:
        mode = Mode(mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid mode: {data['mode']!r}. Must be 'exploit' or 'explore'."
        ) from None

    # Parse direction
    direction = str(data["direction"])
    if not direction:
        raise ValueError("Direction must be a non-empty string.")

    # Parse sub_mode
    sub_mode: SubMode | None = None
    raw_sub_mode = data.get("sub_mode")
    if raw_sub_mode is not None:
        sub_mode_str = str(raw_sub_mode).upper()
        try:
            sub_mode = SubMode(sub_mode_str)
        except ValueError:
            raise ValueError(
                f"Invalid sub_mode: {raw_sub_mode!r}. Must be one of "
                f"'de_novo', 'recombination', 'param_search', 'local_rewrite', "
                f"'pattern_apply', or null."
            ) from None

        # Validate sub_mode matches mode
        if (
            mode == Mode.EXPLOIT
            and raw_sub_mode.lower() not in _VALID_EXPLOIT_SUB_MODES
        ):
            raise ValueError(
                f"sub_mode {raw_sub_mode!r} is not valid for EXPLOIT mode."
            )
        if (
            mode == Mode.EXPLORE
            and raw_sub_mode.lower() not in _VALID_EXPLORE_SUB_MODES
        ):
            raise ValueError(
                f"sub_mode {raw_sub_mode!r} is not valid for EXPLORE mode."
            )

    # Parse confidence
    confidence = str(data["confidence"]).lower()
    if confidence not in _CONFIDENCE_ORDER:
        raise ValueError(
            f"Invalid confidence: {data['confidence']!r}. "
            f"Must be 'high', 'medium', or 'low'."
        )

    # Parse reasoning
    reasoning = str(data["reasoning"])

    return LLMDecision(
        mode=mode,
        direction=direction,
        sub_mode=sub_mode,
        reasoning=reasoning,
        confidence=confidence,
    )


def validate_llm_decision(
    decision: LLMDecision,
    tabu_entries: list[TabuEntry],
    exhausted: set[str],
    config: NavigatorConfig,
    *,
    current_round: int = 0,
    base_kernel_hash: str | None = None,
) -> str | None:
    """Validate an LLM decision against tabu and exhaustion constraints.

    Tabu checking matches on (base_kernel_hash, direction) with expiry.
    An entry is active when expires_after_round >= current_round.

    Returns None if the decision is valid, or a string describing the
    validation failure.

    Args:
        decision: Parsed LLM decision.
        tabu_entries: Current TabuEntry records from optimization state.
        exhausted: Set of exhausted direction names.
        config: Navigator configuration.
        current_round: Current optimization round for tabu expiry check.
        base_kernel_hash: Current incumbent kernel hash for tabu matching.

    Returns:
        None if valid, error message string otherwise.

    Invariant: INV-NAV-005 (tabu matches on base_hash + direction pair)
    """
    # Check direction not blocked by active tabu entry for current base kernel
    for entry in tabu_entries:
        if (
            entry.direction == decision.direction
            and entry.base_kernel_hash == base_kernel_hash
            and entry.expires_after_round >= current_round
        ):
            return (
                f"Direction {decision.direction!r} is in the tabu list. "
                f"Choose a different direction."
            )

    # Check direction not exhausted
    if decision.direction in exhausted:
        return (
            f"Direction {decision.direction!r} is exhausted. "
            f"Choose a different direction."
        )

    # Check confidence meets minimum
    min_level = _CONFIDENCE_ORDER.get(config.llm_confidence_min, 1)
    actual_level = _CONFIDENCE_ORDER.get(decision.confidence, 0)
    if actual_level < min_level:
        return (
            f"Confidence {decision.confidence!r} is below minimum "
            f"{config.llm_confidence_min!r}. Provide a higher confidence "
            f"or a different decision."
        )

    return None


async def run_llm_reasoning(
    state: OptimizationState,
    signals: DerivedSignals,
    cross_analysis: CrossCandidateAnalysis | None,
    client: LLMClientProtocol,
    config: NavigatorConfig,
) -> LLMDecision:
    """Run LLM reasoning with retry logic.

    Assembles context, calls the LLM, parses and validates the response.
    On first failure, retries with a narrowed prompt. On second failure,
    raises an exception so the caller can fall back to UCB1.

    All LLM calls are wrapped in try/except per INV-NAV-003.

    Args:
        state: Full optimization state.
        signals: Derived signals from Phase 1.
        cross_analysis: Cross-candidate analysis output.
        client: LLM client for making completion calls.
        config: Navigator configuration.

    Returns:
        Validated LLMDecision.

    Raises:
        RuntimeError: If both attempts fail (caller should fall back to UCB1).

    Implements: REQ-NAV-006, SCN-NAV-006-01 through SCN-NAV-006-05
    Invariant: INV-NAV-003 (LLM failures never stall the system)
    """
    context = assemble_llm_context(state, signals, cross_analysis, config)
    # Active tabu entries for validation
    tabu_entries = [
        e for e in state.tabu_entries if e.expires_after_round >= state.current_round
    ]
    exhausted = signals.exhausted_directions
    base_kernel_hash = state.incumbent.kernel_hash

    last_error = ""

    for attempt in range(2):
        try:
            # Construct user prompt
            user_prompt = context
            if attempt == 1 and last_error:
                # Narrowed prompt: append the specific validation error
                user_prompt = (
                    f"{context}\n\n"
                    f"IMPORTANT: Your previous response was "
                    f"invalid: {last_error}\n"
                    f"Please return ONLY a valid JSON object "
                    f"with the five required fields."
                )

            raw_response = await client.complete(_SYSTEM_PROMPT, user_prompt)
            decision = parse_llm_decision(raw_response)

            # Validate the parsed decision
            error = validate_llm_decision(
                decision,
                tabu_entries,
                exhausted,
                config,
                current_round=state.current_round,
                base_kernel_hash=base_kernel_hash,
            )
            if error is not None:
                last_error = error
                continue

            return decision

        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    raise RuntimeError(f"LLM reasoning failed after 2 attempts: {last_error}")


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from a string."""
    stripped = text.strip()
    pattern = re.compile(r"^```(?:json)?\s*\n(.*)\n\s*```$", re.DOTALL)
    match = pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped
