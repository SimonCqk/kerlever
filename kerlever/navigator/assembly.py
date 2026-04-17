"""Strategy Navigator assembly — Phase 4 directive construction.

Takes a gate result or LLM decision and assembles a complete
StrategyDirective with all required fields populated.

Spec: docs/navigator/spec.md §6.5
"""

from __future__ import annotations

import re

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import GateResult, LLMDecision
from kerlever.types import (
    CrossCandidateAnalysis,
    Mode,
    OptimizationState,
    RecombinationHint,
    StrategyDirective,
    SubMode,
    TabuEntry,
)


def assemble_directive(
    decision: GateResult | LLMDecision,
    state: OptimizationState,
    cross_analysis: CrossCandidateAnalysis | None,
    config: NavigatorConfig,
) -> StrategyDirective:
    """Assemble a complete StrategyDirective from a gate or LLM decision.

    Follows the 7-step process defined in spec §6.5:
    1. Extract core fields
    2. Apply tabu filter
    3. Determine candidate count
    4. Set base kernel hash
    5. Populate mode-specific fields
    6. Assemble tabu list for output
    7. Construct and return StrategyDirective

    Args:
        decision: GateResult from Phase 2 or LLMDecision from Phase 3.
        state: Full optimization state.
        cross_analysis: Cross-candidate analysis output (may be None).
        config: Navigator configuration.

    Returns:
        A fully populated StrategyDirective.

    Implements: REQ-NAV-008
    Invariant: INV-NAV-005 (tabu matches on base_hash + direction pair)
    Invariant: INV-NAV-006 (every directive has valid mode and non-empty reason)
    """
    # Step 1: Extract core fields
    mode = decision.mode
    direction = decision.direction
    sub_mode = decision.sub_mode

    reason = decision.reason if isinstance(decision, GateResult) else decision.reasoning

    # Step 2: Apply tabu filter
    # Check whether (base_kernel_hash, direction) pair is in active tabu entries
    base_hash = state.incumbent.kernel_hash if mode == Mode.EXPLOIT else None
    if base_hash is not None and _is_tabu(base_hash, direction, state):
        # The direction is tabu for this base kernel — note in reason
        reason = f"{reason} (note: direction was tabu for current base kernel)"

    # Step 3: Determine candidate count
    if mode == Mode.EXPLOIT:
        num_candidates = config.exploit_candidates
    else:
        num_candidates = config.explore_candidates

    # Step 4: Set base kernel hash
    if mode == Mode.EXPLOIT:
        base_kernel_hash: str | None = state.incumbent.kernel_hash
    elif sub_mode == SubMode.DE_NOVO or sub_mode == SubMode.RECOMBINATION:
        base_kernel_hash = None
    else:
        # EXPLORE without specific sub_mode
        base_kernel_hash = None

    # Step 5: Populate mode-specific fields
    parent_candidates: list[str] | None = None
    gene_map: dict[str, str] | None = None
    search_range: dict[str, list[float]] | None = None
    hard_constraints: list[str] | None = None

    if mode == Mode.EXPLOIT:
        # Infer sub_mode if not already set
        if sub_mode is None:
            sub_mode = _infer_exploit_sub_mode(direction)

        # Populate search_range for PARAM_SEARCH
        if sub_mode == SubMode.PARAM_SEARCH:
            search_range = _derive_search_range(direction)

        # Hardware constraints apply to all modes
        hard_constraints = _derive_hard_constraints()

    elif mode == Mode.EXPLORE:
        if sub_mode == SubMode.DE_NOVO:
            # De novo: no parents, no gene map, no search range
            parent_candidates = None
            gene_map = None
            search_range = None
        elif sub_mode == SubMode.RECOMBINATION:
            # Recombination needs at least two real parents. If analysis does
            # not provide them, degrade to a normal de novo explore directive
            # instead of fabricating a one-parent recombination.
            parent_candidates = _derive_parent_candidates(cross_analysis)
            if len(parent_candidates) < 2:
                sub_mode = SubMode.DE_NOVO
                parent_candidates = None
                gene_map = None
                search_range = None
            else:
                gene_map = _derive_gene_map(cross_analysis, parent_candidates)

        # Hardware constraints apply to explore mode too
        hard_constraints = _derive_hard_constraints()

    # Step 6: Assemble active tabu entries for output
    tabu_output: list[TabuEntry] = [
        e for e in state.tabu_entries if e.expires_after_round >= state.current_round
    ]

    # Step 7: Construct and return StrategyDirective
    return StrategyDirective(
        mode=mode,
        direction=direction,
        reason=reason,
        base_kernel_hash=base_kernel_hash,
        num_candidates=num_candidates,
        tabu=tabu_output,
        sub_mode=sub_mode,
        parent_candidates=parent_candidates,
        gene_map=gene_map,
        search_range=search_range,
        hard_constraints=hard_constraints,
    )


def _is_tabu(
    base_hash: str,
    direction: str,
    state: OptimizationState,
) -> bool:
    """Check if a (base_hash, direction) pair is blocked by an active TabuEntry.

    An entry is active when expires_after_round >= current_round.
    Matching is on both base_kernel_hash and direction.

    Invariant: INV-NAV-005 (tabu matches on base_hash + direction pair)
    """
    for entry in state.tabu_entries:
        if (
            entry.base_kernel_hash == base_hash
            and entry.direction == direction
            and entry.expires_after_round >= state.current_round
        ):
            return True

    return False


def _infer_exploit_sub_mode(direction: str) -> SubMode:
    """Infer the exploit sub-mode from the direction string.

    Directions targeting parameter tuning use PARAM_SEARCH.
    Directions targeting code structure use LOCAL_REWRITE.
    Directions applying known patterns use PATTERN_APPLY.
    Default to LOCAL_REWRITE if ambiguous.
    """
    param_keywords = {
        "launch_bounds",
        "tile_size",
        "block_size",
        "grid_size",
        "param",
        "tune",
        "fine_tune",
        "search",
    }
    pattern_keywords = {
        "pattern",
        "apply",
        "template",
        "vectorize",
        "unroll",
        "prefetch",
    }

    direction_lower = direction.lower()

    for keyword in param_keywords:
        if keyword in direction_lower:
            return SubMode.PARAM_SEARCH

    for keyword in pattern_keywords:
        if keyword in direction_lower:
            return SubMode.PATTERN_APPLY

    return SubMode.LOCAL_REWRITE


def _derive_search_range(direction: str) -> dict[str, list[float]]:
    """Derive parameter search ranges from the direction.

    Returns default parameter bounds relevant to CUDA kernel optimization.
    """
    direction_lower = direction.lower()

    ranges: dict[str, list[float]] = {}

    if "launch_bounds" in direction_lower or "block_size" in direction_lower:
        ranges["block_size"] = [32, 64, 128, 256, 512, 1024]

    if "tile_size" in direction_lower:
        ranges["tile_size"] = [16, 32, 64]

    # Default search range if nothing specific matched
    if not ranges:
        ranges["launch_bounds"] = [128, 256]
        ranges["tile_size"] = [16, 32, 64]

    return ranges


def _derive_hard_constraints() -> list[str]:
    """Derive hardware constraints for the Coding Agent.

    Returns standard CUDA hardware limits.
    """
    return [
        "smem <= 48KB",
        "registers <= 255",
    ]


def _derive_parent_candidates(
    cross_analysis: CrossCandidateAnalysis | None,
) -> list[str]:
    """Derive parent kernel hashes for recombination.

    Prefer structured cross-candidate recombination hints. Legacy
    winning_genes remain a compatibility fallback.

    Implements: REQ-NAV-010
    """
    parents: list[str] = []

    hint = _first_valid_recombination_hint(cross_analysis)
    if hint is not None:
        parents.extend(_sanitize_parent_candidates(hint.parent_candidates))
    elif cross_analysis is not None and cross_analysis.winning_genes:
        # Legacy fallback: historical tests used winning_genes as hashes.
        parents.extend(_sanitize_parent_candidates(cross_analysis.winning_genes))

    return parents[:2] if len(parents) >= 2 else []


_HASH_LIKE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$")


def _looks_like_candidate_hash(value: str) -> bool:
    """Return whether a legacy value is plausibly a candidate hash.

    Legacy cross-analysis used simple hash strings in ``winning_genes``.
    Rich analyzer compatibility messages and descriptive legacy strings are
    not safe recombination parents and must be ignored.
    """
    return _HASH_LIKE_PATTERN.fullmatch(value.strip()) is not None


def _sanitize_parent_candidates(values: list[str]) -> list[str]:
    """Keep unique hash-like parent candidates in source order."""
    parents: list[str] = []
    for value in values:
        parent_hash = value.strip()
        if not _looks_like_candidate_hash(parent_hash):
            continue
        if parent_hash not in parents:
            parents.append(parent_hash)
    return parents


def _derive_gene_map(
    cross_analysis: CrossCandidateAnalysis | None,
    parent_candidates: list[str],
) -> dict[str, str]:
    """Derive gene map for recombination from cross-candidate analysis.

    Maps code section names to parent hashes.

    Implements: REQ-NAV-010
    """
    gene_map: dict[str, str] = {}

    parent_set = set(parent_candidates)
    hint = _first_valid_recombination_hint(cross_analysis)
    if hint is not None:
        return {
            section: parent_hash
            for section, parent_hash in hint.gene_map.items()
            if section and parent_hash in parent_set
        }

    return gene_map if gene_map else {}


def _first_valid_recombination_hint(
    cross_analysis: CrossCandidateAnalysis | None,
) -> RecombinationHint | None:
    """Return the first structured hint usable for recombination assembly."""
    if cross_analysis is None:
        return None

    for hint in cross_analysis.recombination_hints:
        parents = _sanitize_parent_candidates(hint.parent_candidates)
        parent_set = set(parents)
        if len(parent_set) < 2:
            continue
        if not hint.gene_map:
            continue
        if all(parent_hash in parent_set for parent_hash in hint.gene_map.values()):
            return hint

    return None
