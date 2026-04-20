"""Benchmarker — Phase 3: fast-benchmark plan calibration.

Translates spec §6.3 warmup/iteration calibration into a deterministic
per-(candidate, shape) plan. The plan records cache policy, metric mode,
and repetition structure; it does NOT perform scoring or anchors.

Spec: docs/benchmarker/spec.md §6.3
Design: docs/benchmarker/design.md §2.1 plan.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from kerlever.benchmarker.config import CalibrationConfig
from kerlever.benchmarker.cuda_driver import CudaFunction
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    CachePolicy,
    FunctionAttributePolicy,
    MeasurementQualityStatus,
    MetricMode,
    ShapeCase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedCandidate:
    """Candidate after cubin load + function resolution + attribute apply.

    The worker constructs these after ``cuda_driver.load_module`` and
    ``cuda_driver.get_function`` succeed; the plan phase consumes them
    read-only.
    """

    candidate_hash: str
    function: CudaFunction
    launch_args_factory: Any
    adapter_iteration_semantics: AdapterIterationSemantics
    function_attribute_policy_observed: FunctionAttributePolicy
    block_dim: tuple[int, int, int]
    grid_dim: tuple[int, int, int] | None
    dynamic_smem_bytes: int


@dataclass(frozen=True)
class SamplePlan:
    """Calibration output for one (candidate, shape) pair."""

    iterations_per_sample: int
    repetitions: int
    warmup_count: int
    warning_reason: str | None = None
    quality_bias: MeasurementQualityStatus = MeasurementQualityStatus.VALID


@dataclass(frozen=True)
class CalibratedPlan:
    """Aggregate Phase 3 output across every (candidate, shape)."""

    sample_plans: dict[tuple[str, str], SamplePlan]
    requested_cache_policy: CachePolicy
    effective_cache_policy: CachePolicy
    cache_policy_reason: str | None
    metric_mode: MetricMode
    adapter_iteration_semantics_per_candidate: dict[str, AdapterIterationSemantics]
    function_attribute_policy_observed_per_candidate: dict[str, FunctionAttributePolicy]
    calibration_warnings: dict[tuple[str, str], str] = field(default_factory=dict)


def _calibrate_one(
    cfg: CalibrationConfig,
    candidate: LoadedCandidate,
    shape: ShapeCase,
    launcher: Any,
) -> SamplePlan:
    """Perform iteration calibration for one (candidate, shape) pair.

    Spec §6.3:
    * Start at ``iterations_per_sample = 1``.
    * Double until elapsed >= min_timed_batch_ms or cap.
    * Halve (to 1) when elapsed > max_timed_batch_ms.
    * Warn on ``calibration_upper_bound_unmet`` / ``calibration_lower_bound_unmet``.

    ``launcher`` is a callable accepting ``(candidate, shape, iterations)`` and
    returning elapsed ms. The worker passes a real launcher; calibration can
    be dry-run by passing a closure that returns a synthetic fixed time.

    Implements: REQ-BENCH-004
    """
    semantics = candidate.adapter_iteration_semantics
    if semantics == AdapterIterationSemantics.NOT_REPEATABLE:
        return SamplePlan(
            iterations_per_sample=1,
            repetitions=cfg.repetitions,
            warmup_count=cfg.warmup_min_runs,
            warning_reason="not_repeatable_one_launch_per_sample",
        )

    iterations = 1
    elapsed_ms = 0.0
    warning_reason: str | None = None

    # Prime: single launch calibration sample.
    elapsed_ms = float(launcher(candidate, shape, iterations))

    # Doubling phase: seek the lower bound.
    while (
        elapsed_ms < cfg.min_timed_batch_ms
        and iterations * 2 <= cfg.max_iterations_per_sample
    ):
        iterations *= 2
        elapsed_ms = float(launcher(candidate, shape, iterations))

    if elapsed_ms < cfg.min_timed_batch_ms:
        warning_reason = "calibration_lower_bound_unmet"

    # Upper-bound contraction.
    while elapsed_ms > cfg.max_timed_batch_ms and iterations > 1:
        iterations //= 2
        elapsed_ms = float(launcher(candidate, shape, iterations))

    if elapsed_ms > cfg.max_timed_batch_ms and iterations == 1:
        warning_reason = "calibration_upper_bound_unmet"

    return SamplePlan(
        iterations_per_sample=iterations,
        repetitions=cfg.repetitions,
        warmup_count=cfg.warmup_min_runs,
        warning_reason=warning_reason,
        quality_bias=(
            MeasurementQualityStatus.VALID_WITH_WARNING
            if warning_reason
            else MeasurementQualityStatus.VALID
        ),
    )


def calibrate(
    candidates: list[LoadedCandidate],
    shapes: list[ShapeCase],
    cfg: CalibrationConfig,
    metric_mode: MetricMode,
    requested_cache_policy: CachePolicy,
    effective_cache_policy: CachePolicy,
    cache_policy_reason: str | None,
    launcher: Any,
) -> CalibratedPlan:
    """Run Phase 3 calibration across the full batch.

    Args:
        candidates: Loaded candidates (post cubin load).
        shapes: Objective shapes to calibrate against.
        cfg: Calibration tunables (spec §6.8).
        metric_mode: Envelope's metric mode (dispatched by the harness).
        requested_cache_policy: Original request cache policy.
        effective_cache_policy: After interleave-auto-promotion (Phase 1).
        cache_policy_reason: ``"interleaved_batch_requires_rotation"`` or ``None``.
        launcher: Callable ``(candidate, shape, iterations) -> elapsed_ms``
            used for calibration probes.

    Implements: REQ-BENCH-004, REQ-BENCH-009
    """
    sample_plans: dict[tuple[str, str], SamplePlan] = {}
    warnings: dict[tuple[str, str], str] = {}
    for cand in candidates:
        for shape in shapes:
            plan = _calibrate_one(cfg, cand, shape, launcher)
            key = (cand.candidate_hash, shape.shape_id)
            sample_plans[key] = plan
            if plan.warning_reason is not None:
                warnings[key] = plan.warning_reason

    return CalibratedPlan(
        sample_plans=sample_plans,
        requested_cache_policy=requested_cache_policy,
        effective_cache_policy=effective_cache_policy,
        cache_policy_reason=cache_policy_reason,
        metric_mode=metric_mode,
        adapter_iteration_semantics_per_candidate={
            c.candidate_hash: c.adapter_iteration_semantics for c in candidates
        },
        function_attribute_policy_observed_per_candidate={
            c.candidate_hash: c.function_attribute_policy_observed
            for c in candidates
        },
        calibration_warnings=warnings,
    )


__all__ = [
    "CalibratedPlan",
    "LoadedCandidate",
    "SamplePlan",
    "calibrate",
]
