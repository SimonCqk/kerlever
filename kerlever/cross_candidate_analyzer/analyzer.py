"""Cross-Candidate Analyzer — deterministic and optional LLM synthesis.

Spec: docs/cross-candidate-analyzer/spec.md
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from kerlever.llm_client import LLMClientProtocol
from kerlever.types import (
    AvoidPattern,
    CandidateGene,
    CandidateOutcome,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    ProblemSpec,
    RecombinationHint,
    SemanticDelta,
)

logger = logging.getLogger(__name__)

_ELIGIBLE_OUTCOMES = frozenset(
    {
        CandidateOutcome.IMPROVED,
        CandidateOutcome.BASELINE_MATCH,
        CandidateOutcome.REGRESSION,
    }
)

_NON_REGRESSING_OUTCOMES = frozenset(
    {CandidateOutcome.IMPROVED, CandidateOutcome.BASELINE_MATCH}
)

_MEASURED_EVIDENCE_KEYS = frozenset(
    {
        "benchmark.objective_score.value",
        "benchmark.objective_score.relative_to_baseline",
        "benchmark.objective_score.relative_to_incumbent",
        "benchmark.shape_results.latency_p50_us",
        "benchmark.shape_results.latency_p95_us",
        "static_analysis.registers_per_thread",
        "static_analysis.smem_bytes_per_block",
        "static_analysis.spill_stores",
        "static_analysis.spill_loads",
        "static_analysis.occupancy_estimate_pct",
        "profile.metrics.achieved_occupancy_pct",
        "profile.metrics.dram_throughput_pct_of_peak",
        "profile.metrics.sm_throughput_pct_of_peak",
        "profile.metrics.l2_hit_rate_pct",
        "profile.metrics.warp_stall_memory_dependency_pct",
        "profile.metrics.warp_stall_exec_dependency_pct",
        "profile.metrics.tensor_core_utilization_pct",
        "profile.metrics.arithmetic_intensity_flop_per_byte",
        "profile.assessment.evidence",
        "profile.assessment.primary_tag",
    }
)

_MEASURED_NUMERIC_EVIDENCE_KEYS = frozenset(
    {
        "objective_score",
        "relative_to_baseline",
        "relative_to_incumbent",
        "benchmark.objective_score.value",
        "benchmark.objective_score.relative_to_baseline",
        "benchmark.objective_score.relative_to_incumbent",
    }
)


@dataclass(frozen=True)
class _CandidateEvidence:
    """Normalized evidence extracted from one eligible candidate."""

    candidate_hash: str
    parent_hashes: list[str]
    outcome: CandidateOutcome
    direction: str
    mode: str
    sub_mode: str | None
    features: list[str]
    evidence_refs: list[str]
    numeric_evidence: dict[str, float]
    shape_ids: list[str]
    profile_tags: list[str]
    source_snippet: str


class CrossCandidateAnalyzer:
    """Analyze benchmarked candidates into evidence-backed rich outputs.

    The analyzer first extracts deterministic facts from correctness-passing,
    benchmarked candidates. If an LLM client is configured, it may synthesize
    richer semantic claims, but claims are accepted only after schema,
    evidence, and regression-safety validation. LLM failure falls back to the
    deterministic analysis.

    Implements: REQ-CCA-001 through REQ-CCA-009
    Invariant: INV-CCA-004 (LLM failure cannot stall the loop)
    """

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        """Initialize the analyzer.

        Args:
            llm_client: Optional LLM client used for JSON-only semantic
                synthesis. When omitted, analysis is deterministic-only.
        """
        self._llm_client = llm_client

    async def analyze(
        self,
        top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
        problem_spec: ProblemSpec,
    ) -> CrossCandidateAnalysis:
        """Analyze eligible benchmarked candidates.

        Args:
            top_k_results: Round-local candidate/evaluation pairs. Only
                benchmarked IMPROVED, BASELINE_MATCH, and REGRESSION results
                with compile success and correctness pass are analyzed.
            problem_spec: The optimization target used for shape weights and
                prompt context.

        Returns:
            CrossCandidateAnalysis with legacy fields and rich structured
            fields populated from validated evidence.

        Implements: REQ-CCA-002, REQ-CCA-003, REQ-CCA-004
        Invariant: INV-CCA-002 (failed candidates never influence analysis)
        """
        evidence = self._extract_evidence(top_k_results, problem_spec)
        deterministic = self._build_deterministic_analysis(evidence)

        if self._llm_client is None or not evidence:
            return self._ensure_serializable(deterministic)

        llm_analysis = await self._try_llm_synthesis(
            problem_spec,
            evidence,
            deterministic,
        )
        if llm_analysis is None:
            return self._ensure_serializable(deterministic)

        return self._ensure_serializable(llm_analysis)

    def _extract_evidence(
        self,
        top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
        problem_spec: ProblemSpec,
    ) -> list[_CandidateEvidence]:
        """Extract deterministic evidence from eligible inputs.

        Implements: REQ-CCA-002, REQ-CCA-003
        """
        seen: set[str] = set()
        shape_weights = {
            case.shape_id: case.weight for case in problem_spec.shape_cases
        }
        evidence: list[_CandidateEvidence] = []

        for candidate, result in top_k_results:
            if candidate.code_hash in seen:
                continue
            if not self._is_eligible_pair(candidate, result):
                continue

            seen.add(candidate.code_hash)
            benchmark = result.benchmark
            if benchmark is None:
                continue

            evidence_refs = [
                "benchmark.objective_score.value",
                "benchmark.objective_score.relative_to_baseline",
                "benchmark.objective_score.relative_to_incumbent",
            ]
            numeric_evidence: dict[str, float] = {
                "objective_score": benchmark.objective_score.value,
                "relative_to_baseline": (
                    benchmark.objective_score.relative_to_baseline
                ),
                "relative_to_incumbent": (
                    benchmark.objective_score.relative_to_incumbent
                ),
            }
            shape_ids: list[str] = []
            for shape_result in benchmark.shape_results:
                shape_ids.append(shape_result.shape_id)
                evidence_refs.extend(
                    [
                        "benchmark.shape_results.latency_p50_us",
                        "benchmark.shape_results.latency_p95_us",
                    ]
                )
                prefix = f"shape.{shape_result.shape_id}"
                numeric_evidence[f"{prefix}.latency_p50_us"] = (
                    shape_result.latency_p50_us
                )
                numeric_evidence[f"{prefix}.latency_p95_us"] = (
                    shape_result.latency_p95_us
                )
                numeric_evidence[f"{prefix}.weight"] = shape_weights.get(
                    shape_result.shape_id,
                    1.0,
                )
                if shape_result.stdev_us is not None:
                    numeric_evidence[f"{prefix}.stdev_us"] = shape_result.stdev_us
                numeric_evidence[f"{prefix}.run_count"] = float(shape_result.run_count)

            if result.static_analysis is not None:
                static_values = result.static_analysis.model_dump(exclude_none=True)
                for key, value in static_values.items():
                    evidence_refs.append(f"static_analysis.{key}")
                    numeric_evidence[f"static_analysis.{key}"] = float(value)

            profile_tags: list[str] = []
            if result.profile is not None:
                metric_values = result.profile.metrics.model_dump(exclude_none=True)
                for key, value in metric_values.items():
                    evidence_refs.append(f"profile.metrics.{key}")
                    numeric_evidence[f"profile.metrics.{key}"] = float(value)

                profile_tags = list(result.profile.assessment.tags)
                if result.profile.assessment.primary_tag is not None:
                    evidence_refs.append("profile.assessment.primary_tag")
                if result.profile.assessment.evidence:
                    evidence_refs.append("profile.assessment.evidence")
                    for key, value in result.profile.assessment.evidence.items():
                        numeric_evidence[f"profile.assessment.{key}"] = float(value)

            evidence.append(
                _CandidateEvidence(
                    candidate_hash=candidate.code_hash,
                    parent_hashes=list(candidate.parent_hashes),
                    outcome=result.outcome,
                    direction=candidate.intent.direction,
                    mode=candidate.intent.mode.value,
                    sub_mode=(
                        candidate.intent.sub_mode.value
                        if candidate.intent.sub_mode is not None
                        else None
                    ),
                    features=_detect_source_features(candidate.source_code),
                    evidence_refs=sorted(set(evidence_refs)),
                    numeric_evidence=numeric_evidence,
                    shape_ids=shape_ids,
                    profile_tags=profile_tags,
                    source_snippet=candidate.source_code[:1200],
                )
            )

        return evidence

    @staticmethod
    def _is_eligible_pair(
        candidate: KernelCandidate,
        result: EvaluationResult,
    ) -> bool:
        """Return whether a candidate/result pair can influence analysis."""
        return (
            candidate.code_hash == result.candidate_hash
            and result.outcome in _ELIGIBLE_OUTCOMES
            and result.benchmark is not None
            and result.compile_status.value == "SUCCESS"
            and result.correctness is not None
            and result.correctness.passed
        )

    def _build_deterministic_analysis(
        self,
        evidence: list[_CandidateEvidence],
    ) -> CrossCandidateAnalysis:
        """Build conservative rich output without LLM assistance.

        Implements: REQ-CCA-005, REQ-CCA-006, REQ-CCA-007, REQ-CCA-008
        """
        semantic_deltas = [self._semantic_delta(record) for record in evidence]
        candidate_genes = self._candidate_genes(evidence)
        avoid_patterns = self._avoid_patterns(evidence)
        recombination_hints = self._recombination_hints(candidate_genes, evidence)

        return _bridge_legacy_fields(
            semantic_deltas=semantic_deltas,
            candidate_genes=candidate_genes,
            recombination_hints=recombination_hints,
            avoid_patterns=avoid_patterns,
            extra_insights=[],
        )

    @staticmethod
    def _semantic_delta(record: _CandidateEvidence) -> SemanticDelta:
        """Create a measured semantic delta for one candidate."""
        feature_summary = ", ".join(record.features) if record.features else "no tags"
        summary = (
            f"{record.candidate_hash} ended {record.outcome.value} with "
            f"features: {feature_summary}; objective="
            f"{record.numeric_evidence['objective_score']:.4f}"
        )
        confidence: LiteralConfidence = "medium" if record.features else "low"
        return SemanticDelta(
            candidate_hash=record.candidate_hash,
            parent_hashes=record.parent_hashes,
            outcome=record.outcome,
            summary=summary,
            changed_features=record.features,
            evidence_refs=record.evidence_refs,
            confidence=confidence,
        )

    def _candidate_genes(
        self,
        evidence: list[_CandidateEvidence],
    ) -> list[CandidateGene]:
        """Extract conservative positive genes from non-regressing candidates."""
        genes: list[CandidateGene] = []
        for record in evidence:
            if record.outcome not in _NON_REGRESSING_OUTCOMES:
                continue
            if not record.features:
                continue

            relative = record.numeric_evidence["relative_to_incumbent"]
            if record.outcome == CandidateOutcome.BASELINE_MATCH and relative > 1.0:
                continue

            feature = _choose_positive_feature(record.features)
            gene_type = _feature_to_gene_type(feature)
            risk_flags = _risk_flags(record)
            confidence: LiteralConfidence = "medium" if relative < 1.0 else "low"
            gene_id = f"gene_{len(genes) + 1}_{gene_type}_{record.candidate_hash}"
            genes.append(
                CandidateGene(
                    gene_id=gene_id,
                    source_candidate_hash=record.candidate_hash,
                    gene_type=gene_type,
                    description=(
                        f"{feature} observed in {record.candidate_hash} with "
                        f"{record.outcome.value.lower()} objective evidence"
                    ),
                    evidence={
                        key: value
                        for key, value in record.numeric_evidence.items()
                        if key
                        in {
                            "objective_score",
                            "relative_to_baseline",
                            "relative_to_incumbent",
                        }
                    },
                    affected_shape_ids=record.shape_ids,
                    risk_flags=risk_flags,
                    confidence=confidence,
                )
            )
        return genes

    @staticmethod
    def _avoid_patterns(evidence: list[_CandidateEvidence]) -> list[AvoidPattern]:
        """Extract local avoid patterns from benchmarked regressions."""
        patterns: list[AvoidPattern] = []
        for record in evidence:
            if record.outcome != CandidateOutcome.REGRESSION:
                continue

            pattern = _choose_risky_feature(record.features)
            relative = record.numeric_evidence["relative_to_incumbent"]
            pattern_id = f"avoid_{len(patterns) + 1}_{pattern}_{record.candidate_hash}"
            patterns.append(
                AvoidPattern(
                    pattern_id=pattern_id,
                    source_candidate_hash=record.candidate_hash,
                    pattern=pattern,
                    reason=(
                        f"{pattern} co-occurred with measured regression "
                        f"relative_to_incumbent={relative:.4f}"
                    ),
                    evidence={
                        key: value
                        for key, value in record.numeric_evidence.items()
                        if key
                        in {
                            "objective_score",
                            "relative_to_baseline",
                            "relative_to_incumbent",
                        }
                    },
                    affected_shape_ids=record.shape_ids,
                    confidence="medium" if relative > 1.05 else "low",
                )
            )
        return patterns

    @staticmethod
    def _recombination_hints(
        genes: list[CandidateGene],
        evidence: list[_CandidateEvidence],
    ) -> list[RecombinationHint]:
        """Create hints for complementary genes from different parents."""
        outcome_by_hash = {record.candidate_hash: record.outcome for record in evidence}
        gene_map: dict[str, str] = {}
        parents: list[str] = []
        risk_flags: list[str] = []

        for gene in genes:
            if (
                outcome_by_hash.get(gene.source_candidate_hash)
                not in _NON_REGRESSING_OUTCOMES
            ):
                continue
            if gene.gene_type in gene_map:
                continue
            if gene.source_candidate_hash in parents and len(parents) < 2:
                continue
            gene_map[gene.gene_type] = gene.source_candidate_hash
            if gene.source_candidate_hash not in parents:
                parents.append(gene.source_candidate_hash)
            risk_flags.extend(gene.risk_flags)
            if len(parents) >= 2 and len(gene_map) >= 2:
                break

        if len(parents) < 2:
            return []

        return [
            RecombinationHint(
                hint_id=f"hint_1_{'_'.join(parents)}",
                parent_candidates=parents,
                gene_map=gene_map,
                expected_benefit=(
                    "Combine complementary evidence-backed genes from "
                    f"{', '.join(parents)}; benefit is hypothesis-only until "
                    "benchmarked."
                ),
                evidence_candidate_hashes=parents,
                required_constraints=[],
                risk_flags=sorted(set(risk_flags)),
                confidence="low"
                if any(g.confidence == "low" for g in genes)
                else "medium",
            )
        ]

    async def _try_llm_synthesis(
        self,
        problem_spec: ProblemSpec,
        evidence: list[_CandidateEvidence],
        deterministic: CrossCandidateAnalysis,
    ) -> CrossCandidateAnalysis | None:
        """Run JSON-only LLM synthesis with one retry and fallback."""
        if self._llm_client is None:
            return None

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(problem_spec, evidence, deterministic)
        last_error = ""

        for attempt in range(2):
            try:
                prompt = user_prompt
                if attempt == 1:
                    prompt = (
                        f"{user_prompt}\n\n"
                        f"Your previous JSON was invalid: {last_error}\n"
                        "Return ONLY a single valid JSON object matching the schema."
                    )
                raw = await self._llm_client.complete(system_prompt, prompt)
                candidate = _parse_llm_analysis(raw)
                return self._validate_llm_analysis(candidate, evidence, deterministic)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("Cross-candidate LLM synthesis attempt failed: %s", exc)

        return None

    def _validate_llm_analysis(
        self,
        data: dict[str, object],
        evidence: list[_CandidateEvidence],
        deterministic: CrossCandidateAnalysis,
    ) -> CrossCandidateAnalysis:
        """Validate LLM rich entries against evidence and regression safety."""
        allowed_keys = {
            "semantic_deltas",
            "candidate_genes",
            "recombination_hints",
            "avoid_patterns",
            "insights",
        }
        extra_keys = set(data) - allowed_keys
        if extra_keys:
            raise ValueError(f"Unexpected LLM fields: {sorted(extra_keys)}")

        outcome_by_hash = {record.candidate_hash: record.outcome for record in evidence}
        known_hashes = set(outcome_by_hash)

        try:
            semantic_deltas = [
                SemanticDelta.model_validate(item)
                for item in _list_field(data, "semantic_deltas")
            ]
            candidate_genes = [
                CandidateGene.model_validate(item)
                for item in _list_field(data, "candidate_genes")
            ]
            recombination_hints = [
                RecombinationHint.model_validate(item)
                for item in _list_field(data, "recombination_hints")
            ]
            avoid_patterns = [
                AvoidPattern.model_validate(item)
                for item in _list_field(data, "avoid_patterns")
            ]
        except ValidationError as exc:
            raise ValueError(f"LLM schema validation failed: {exc}") from exc

        for delta in semantic_deltas:
            _validate_delta(delta, known_hashes)
        for gene in candidate_genes:
            _validate_gene(gene, outcome_by_hash)
        for hint in recombination_hints:
            _validate_hint(hint, outcome_by_hash)
        for pattern in avoid_patterns:
            _validate_avoid_pattern(pattern, outcome_by_hash)

        insights = [str(item) for item in _list_field(data, "insights")]
        merged_deltas = semantic_deltas or deterministic.semantic_deltas
        merged_genes = candidate_genes or deterministic.candidate_genes
        merged_hints = recombination_hints or deterministic.recombination_hints
        merged_avoid = avoid_patterns or deterministic.avoid_patterns

        return _bridge_legacy_fields(
            semantic_deltas=merged_deltas,
            candidate_genes=merged_genes,
            recombination_hints=merged_hints,
            avoid_patterns=merged_avoid,
            extra_insights=insights,
        )

    @staticmethod
    def _ensure_serializable(
        analysis: CrossCandidateAnalysis,
    ) -> CrossCandidateAnalysis:
        """Round-trip through JSON to enforce Pydantic serializability."""
        try:
            return CrossCandidateAnalysis.model_validate_json(
                analysis.model_dump_json()
            )
        except Exception:
            logger.exception("CrossCandidateAnalysis serialization failed")
            return CrossCandidateAnalysis(
                insights=["Cross-candidate analysis serialization failed."],
                winning_genes=[],
                recombination_suggestions=[],
            )


LiteralConfidence = Literal["low", "medium", "high"]


def _detect_source_features(source: str) -> list[str]:
    """Detect simple CUDA source feature tags without parsing CUDA."""
    lowered = source.lower()
    features: list[str] = []

    checks = [
        ("shared_memory", "__shared__" in lowered or "extern __shared__" in lowered),
        (
            "vectorized_type",
            any(token in lowered for token in ("float4", "float2", "half2", "int4")),
        ),
        ("warp_shuffle", "__shfl" in lowered or "cooperative_groups" in lowered),
        ("tensor_core", "wmma" in lowered or "mma.sync" in lowered),
        ("unroll", "#pragma unroll" in lowered),
        ("extra_sync", "__syncthreads" in lowered or "__syncwarp" in lowered),
        ("cp_async", "cp.async" in lowered or "__pipeline" in lowered),
        ("launch_bounds", "__launch_bounds__" in lowered),
        ("restrict_qualifier", "__restrict__" in lowered),
        (
            "indexing_change",
            any(token in lowered for token in ("threadidx", "blockidx", "stride")),
        ),
    ]
    for name, present in checks:
        if present:
            features.append(name)
    return features


def _choose_positive_feature(features: list[str]) -> str:
    """Choose a representative positive feature in stable priority order."""
    priority = [
        "shared_memory",
        "cp_async",
        "vectorized_type",
        "warp_shuffle",
        "tensor_core",
        "unroll",
        "launch_bounds",
        "restrict_qualifier",
        "indexing_change",
        "extra_sync",
    ]
    for feature in priority:
        if feature in features:
            return feature
    return features[0]


def _choose_risky_feature(features: list[str]) -> str:
    """Choose a representative risky feature for regression evidence."""
    priority = [
        "extra_sync",
        "shared_memory",
        "cp_async",
        "unroll",
        "launch_bounds",
        "vectorized_type",
        "warp_shuffle",
        "tensor_core",
        "indexing_change",
        "restrict_qualifier",
    ]
    for feature in priority:
        if feature in features:
            return feature
    return "objective_regression"


def _feature_to_gene_type(feature: str) -> str:
    """Map a detected feature tag to a semantic gene type."""
    mapping = {
        "shared_memory": "tiling",
        "cp_async": "memory_access",
        "vectorized_type": "memory_access",
        "warp_shuffle": "instruction_mix",
        "tensor_core": "instruction_mix",
        "unroll": "compute_loop",
        "launch_bounds": "launch_config",
        "restrict_qualifier": "memory_access",
        "indexing_change": "memory_access",
        "extra_sync": "synchronization",
    }
    return mapping.get(feature, "algorithmic_structure")


def _risk_flags(record: _CandidateEvidence) -> list[str]:
    """Derive conservative risk flags from measured evidence."""
    flags: list[str] = []
    if len(record.shape_ids) == 1:
        flags.append("shape_specific")
    registers = record.numeric_evidence.get("static_analysis.registers_per_thread")
    if registers is not None and registers >= 96:
        flags.append("register_pressure")
    smem = record.numeric_evidence.get("static_analysis.smem_bytes_per_block")
    if smem is not None and smem >= 48 * 1024:
        flags.append("smem_pressure")
    if record.numeric_evidence.get("relative_to_incumbent", 1.0) >= 0.99:
        flags.append("noise_sensitive")
    return flags


def _bridge_legacy_fields(
    *,
    semantic_deltas: list[SemanticDelta],
    candidate_genes: list[CandidateGene],
    recombination_hints: list[RecombinationHint],
    avoid_patterns: list[AvoidPattern],
    extra_insights: list[str],
) -> CrossCandidateAnalysis:
    """Build legacy strings from validated rich entries only."""
    insights = list(extra_insights)
    insights.extend(delta.summary for delta in semantic_deltas[:4])
    insights.extend(pattern.reason for pattern in avoid_patterns[:3])
    if not insights:
        insights.append("No evidence-backed cross-candidate differences identified.")

    winning_genes = [
        (f"{gene.gene_type} from {gene.source_candidate_hash}: {gene.description}")
        for gene in candidate_genes
    ]
    if not winning_genes:
        winning_genes = ["No evidence-backed reusable genes identified."]

    recombination_suggestions = [
        (
            f"{hint.hint_id}: parents={hint.parent_candidates}, "
            f"gene_map={hint.gene_map}, rationale={hint.expected_benefit}"
        )
        for hint in recombination_hints
    ]

    return CrossCandidateAnalysis(
        insights=insights,
        winning_genes=winning_genes,
        recombination_suggestions=recombination_suggestions,
        semantic_deltas=semantic_deltas,
        candidate_genes=candidate_genes,
        recombination_hints=recombination_hints,
        avoid_patterns=avoid_patterns,
    )


def _build_system_prompt() -> str:
    """Build the JSON-only LLM system prompt."""
    return (
        "You are a CUDA semantic comparison assistant, not a strategy policy "
        "owner. Use only the supplied structured evidence. Do not claim "
        "speedups without measured evidence. Treat regressions as negative "
        "evidence only. Return exactly one JSON object with keys "
        "semantic_deltas, candidate_genes, recombination_hints, "
        "avoid_patterns, and insights."
    )


def _build_user_prompt(
    problem_spec: ProblemSpec,
    evidence: list[_CandidateEvidence],
    deterministic: CrossCandidateAnalysis,
) -> str:
    """Build bounded structured context for LLM synthesis."""
    payload = {
        "problem": {
            "op_name": problem_spec.op_name,
            "op_semantics": problem_spec.op_semantics,
            "dtype": problem_spec.dtype,
            "target_gpu": problem_spec.target_gpu,
            "objective": problem_spec.objective.model_dump(),
            "shape_cases": [shape.model_dump() for shape in problem_spec.shape_cases],
        },
        "candidate_evidence": [
            {
                "candidate_hash": item.candidate_hash,
                "parent_hashes": item.parent_hashes,
                "outcome": item.outcome.value,
                "direction": item.direction,
                "mode": item.mode,
                "sub_mode": item.sub_mode,
                "features": item.features,
                "evidence_refs": item.evidence_refs,
                "numeric_evidence": item.numeric_evidence,
                "shape_ids": item.shape_ids,
                "profile_tags": item.profile_tags,
                "source_snippet": item.source_snippet,
            }
            for item in evidence
        ],
        "deterministic_candidate_genes": [
            gene.model_dump() for gene in deterministic.candidate_genes
        ],
        "deterministic_avoid_patterns": [
            pattern.model_dump() for pattern in deterministic.avoid_patterns
        ],
        "instruction": (
            "Compile/correctness/error outcomes are absent by design. "
            "Return JSON only and do not discuss them."
        ),
    }
    return json.dumps(payload, sort_keys=True)


def _parse_llm_analysis(raw: str) -> dict[str, object]:
    """Parse a strict JSON object response from the LLM."""
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _list_field(data: dict[str, object], key: str) -> list[Any]:
    """Return a JSON list field or an empty list when omitted."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Field {key!r} must be a list")
    return value


def _has_measured_ref(refs: list[str]) -> bool:
    """Return whether at least one evidence ref names measured evidence."""
    return any(ref in _MEASURED_EVIDENCE_KEYS for ref in refs)


def _has_measured_numeric_evidence(evidence: dict[str, float]) -> bool:
    """Return whether numeric evidence contains a measured metric key."""
    return any(_is_measured_numeric_key(key) for key in evidence)


def _is_measured_numeric_key(key: str) -> bool:
    """Return whether a numeric evidence key comes from measured data."""
    if key in _MEASURED_NUMERIC_EVIDENCE_KEYS:
        return True
    if key.startswith("shape."):
        parts = key.split(".")
        return len(parts) == 3 and parts[2] in {
            "latency_p50_us",
            "latency_p95_us",
            "stdev_us",
        }
    return key.startswith(
        ("static_analysis.", "profile.metrics.", "profile.assessment.")
    )


def _validate_delta(delta: SemanticDelta, known_hashes: set[str]) -> None:
    """Validate a semantic delta against known measured candidates."""
    if delta.candidate_hash not in known_hashes:
        raise ValueError(f"Unknown semantic delta candidate: {delta.candidate_hash}")
    if not _has_measured_ref(delta.evidence_refs):
        raise ValueError("SemanticDelta lacks measured evidence_refs")


def _validate_gene(
    gene: CandidateGene,
    outcome_by_hash: dict[str, CandidateOutcome],
) -> None:
    """Validate a positive gene against measured, non-regressing evidence."""
    outcome = outcome_by_hash.get(gene.source_candidate_hash)
    if outcome is None:
        raise ValueError(f"Unknown gene source: {gene.source_candidate_hash}")
    if outcome not in _NON_REGRESSING_OUTCOMES:
        raise ValueError("CandidateGene source must be non-regressing")
    if not _has_measured_numeric_evidence(gene.evidence):
        raise ValueError("CandidateGene lacks measured numeric evidence")


def _validate_hint(
    hint: RecombinationHint,
    outcome_by_hash: dict[str, CandidateOutcome],
) -> None:
    """Validate a recombination hint against non-regressing parents."""
    if len(hint.parent_candidates) < 2:
        raise ValueError("RecombinationHint needs at least two parents")
    for parent_hash in hint.parent_candidates:
        if outcome_by_hash.get(parent_hash) not in _NON_REGRESSING_OUTCOMES:
            raise ValueError("RecombinationHint parent must be non-regressing")
    if not hint.evidence_candidate_hashes:
        raise ValueError("RecombinationHint lacks evidence candidate hashes")
    for evidence_hash in hint.evidence_candidate_hashes:
        if evidence_hash not in outcome_by_hash:
            raise ValueError(f"Unknown hint evidence hash: {evidence_hash}")
    for section, parent_hash in hint.gene_map.items():
        if not section:
            raise ValueError("RecombinationHint gene_map section cannot be empty")
        if parent_hash not in hint.parent_candidates:
            raise ValueError("RecombinationHint gene_map points outside parents")


def _validate_avoid_pattern(
    pattern: AvoidPattern,
    outcome_by_hash: dict[str, CandidateOutcome],
) -> None:
    """Validate an avoid pattern against measured regression evidence."""
    outcome = outcome_by_hash.get(pattern.source_candidate_hash)
    if outcome is None:
        raise ValueError(
            f"Unknown avoid pattern source: {pattern.source_candidate_hash}"
        )
    if outcome != CandidateOutcome.REGRESSION:
        raise ValueError("AvoidPattern source must be a benchmarked regression")
    if not _has_measured_numeric_evidence(pattern.evidence):
        raise ValueError("AvoidPattern lacks measured numeric evidence")
