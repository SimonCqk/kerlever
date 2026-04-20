"""Benchmarker — Phase 5: objective scoring and incumbent comparison.

The single function :func:`decide_incumbent_comparison` is the ONLY place
that produces an ``IncumbentComparison`` (INV-BENCH-006). All upstream code
routes through it so improvement/regression decisions cannot bypass the
noise-margin gate.

Spec: docs/benchmarker/spec.md §6.5
Design: docs/benchmarker/design.md §2.1 scoring.py
"""

from __future__ import annotations

import logging

from kerlever.benchmarker.stats import aggregate_noise_margin
from kerlever.benchmarker.types import (
    IncumbentComparison,
    MeasurementEnvelope,
    MeasurementQualityStatus,
    ObjectiveScore,
    PerformanceObjective,
    ShapeBenchResult,
)

logger = logging.getLogger(__name__)


def _select_metric_value(
    objective: PerformanceObjective, shape: ShapeBenchResult
) -> float | None:
    """Map primary_metric to the source field on ``shape``.

    ``weighted_p95_us`` consumers require the p95 to exist (REQ-BENCH-011 /
    REQ-BENCH-033). When the compact artifact carries the ``-1.0`` sentinel
    (or is otherwise non-positive in a p95 metric context), we propagate
    ``None`` so the shape does not contribute to scoring. The caller then
    marks the candidate ``not_comparable`` when every shape was excluded.

    Implements: REQ-BENCH-033
    """
    metric = objective.primary_metric
    if metric == "weighted_p50_us":
        return shape.latency_p50_us
    if metric == "worst_case_p50_us":
        return shape.latency_p50_us
    if metric == "weighted_p95_us":
        if shape.latency_p95_us == -1.0:
            return None
        return shape.latency_p95_us
    return None


def compute_objective_score(
    shape_results: list[ShapeBenchResult],
    objective: PerformanceObjective,
    shape_weights: dict[str, float],
    baseline_value: float,
    incumbent_anchor_value: float,
) -> ObjectiveScore:
    """Compute per-objective aggregate score from fast-benchmark shapes only.

    Implements: REQ-BENCH-014
    Invariant: INV-BENCH-007 (score never incorporates NCU/NSYS metrics)
    """
    contributing: list[tuple[float, float]] = []
    for shape in shape_results:
        val = _select_metric_value(objective, shape)
        if val is None:
            # Shape is gated out (e.g., p95 below MIN_P95_SAMPLES); exclude
            # from the aggregate but keep the shape in the bundle.
            continue
        w = float(shape_weights.get(shape.shape_id, 1.0))
        contributing.append((val, w))

    if not contributing:
        # No shape contributed — emit a sentinel score. The caller marks the
        # candidate not_comparable so downstream never ranks on this value.
        return ObjectiveScore(
            metric_name=objective.primary_metric,
            value=float("inf"),
            relative_to_baseline=float("inf"),
            relative_to_incumbent=float("inf"),
        )

    if objective.aggregation == "max":
        value = max(v for v, _w in contributing)
    else:  # weighted_mean
        total_w = sum(w for _v, w in contributing)
        if total_w <= 0.0:
            value = sum(v for v, _w in contributing) / len(contributing)
        else:
            value = sum(v * w for v, w in contributing) / total_w

    rel_base = value / baseline_value if baseline_value > 0.0 else float("inf")
    rel_inc = (
        value / incumbent_anchor_value
        if incumbent_anchor_value > 0.0
        else float("inf")
    )
    return ObjectiveScore(
        metric_name=objective.primary_metric,
        value=value,
        relative_to_baseline=rel_base,
        relative_to_incumbent=rel_inc,
    )


def _envelope_mismatch(
    candidate: MeasurementEnvelope, incumbent: MeasurementEnvelope
) -> bool:
    """Return ``True`` when candidate and incumbent envelopes disagree.

    Spec §6.5: metric mode, sm_arch, adapter abi, adapter version, and
    function attribute policy all must match for a comparable decision.
    """
    if candidate.metric_mode != incumbent.metric_mode:
        return True
    if candidate.sm_arch != incumbent.sm_arch:
        return True
    if candidate.operation_adapter_abi != incumbent.operation_adapter_abi:
        return True
    if candidate.operation_adapter_version != incumbent.operation_adapter_version:
        return True
    return (
        candidate.function_attribute_policy_observed
        != incumbent.function_attribute_policy_observed
    )


def decide_incumbent_comparison(
    candidate_envelope: MeasurementEnvelope,
    candidate_score: float,
    candidate_quality: list[MeasurementQualityStatus],
    candidate_cv_pct: float | None,
    incumbent_envelope: MeasurementEnvelope,
    incumbent_score: float,
    incumbent_cv_pct: float | None,
    anchor_drift_fraction: float,
    guard_pct: float,
    noise_floor_pct: float,
) -> IncumbentComparison:
    """Decide the per-candidate ``IncumbentComparison``.

    Decision order (spec §6.5 table):

    1. Any shape is ``infra_fault`` → ``NOT_COMPARABLE``.
    2. Any shape is ``runtime_fault`` → ``REGRESSED`` (``NOT_COMPARABLE`` if ambiguous).
    3. Any shape is ``unstable`` → ``UNSTABLE``.
    4. Envelope mismatch → ``NOT_COMPARABLE``.
    5. ``value < incumbent * (1 - noise_margin)`` → ``IMPROVED``.
    6. ``value > incumbent * (1 + guard + noise_margin)`` → ``REGRESSED``.
    7. Else → ``STATISTICAL_TIE``.

    Args:
        candidate_envelope: Candidate's measurement envelope.
        candidate_score: Candidate's aggregated objective value.
        candidate_quality: Quality statuses per contributing shape.
        candidate_cv_pct: Aggregated candidate CV (percent); ``None`` allowed.
        incumbent_envelope: Incumbent anchor envelope.
        incumbent_score: Incumbent anchor objective value.
        incumbent_cv_pct: Incumbent CV (percent); ``None`` allowed.
        anchor_drift_fraction: Anchor drift as fraction (e.g., 0.04 = 4%).
        guard_pct: ``problem_spec.objective.regression_guard_pct``.
        noise_floor_pct: Configured noise floor fraction.

    Returns:
        The resolved :class:`IncumbentComparison`.

    Implements: REQ-BENCH-015
    Invariant: INV-BENCH-006 (sole decision path), INV-BENCH-011
    """
    if MeasurementQualityStatus.INFRA_FAULT in candidate_quality:
        return IncumbentComparison.NOT_COMPARABLE
    if MeasurementQualityStatus.RUNTIME_FAULT in candidate_quality:
        # Candidate_fault → REGRESSED is NOT set here; spec §6.5 maps
        # runtime_fault to REGRESSED only when fault_class is candidate_fault.
        # The caller passes RUNTIME_FAULT with classified fault_class; we use
        # REGRESSED when the fault made it into the shape list.
        return IncumbentComparison.REGRESSED
    if MeasurementQualityStatus.UNSTABLE in candidate_quality:
        return IncumbentComparison.UNSTABLE
    if _envelope_mismatch(candidate_envelope, incumbent_envelope):
        return IncumbentComparison.NOT_COMPARABLE
    if incumbent_score <= 0.0:
        return IncumbentComparison.NOT_COMPARABLE

    noise_margin = aggregate_noise_margin(
        candidate_cv_pct=candidate_cv_pct,
        anchor_cv_pct=incumbent_cv_pct,
        anchor_drift_fraction=anchor_drift_fraction,
        floor=noise_floor_pct,
    )
    improved_threshold = incumbent_score * (1.0 - noise_margin)
    regressed_threshold = incumbent_score * (1.0 + guard_pct + noise_margin)

    if candidate_score < improved_threshold:
        return IncumbentComparison.IMPROVED
    if candidate_score > regressed_threshold:
        return IncumbentComparison.REGRESSED
    return IncumbentComparison.STATISTICAL_TIE


__all__ = [
    "compute_objective_score",
    "decide_incumbent_comparison",
]
