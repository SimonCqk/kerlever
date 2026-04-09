"""Spec Builder deterministic checks — Stage 1 validation.

Six categories of pure, reproducible checks that require no LLM.

Spec: docs/spec-builder/spec.md §6.2
"""

from __future__ import annotations

from kerlever.spec_builder.types import ValidationIssue
from kerlever.types import ProblemSpec

_KNOWN_DTYPES: frozenset[str] = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)

_KNOWN_GPUS: frozenset[str] = frozenset(
    {
        "a100",
        "h100",
        "a10",
        "l4",
        "l40",
        "t4",
        "v100",
        "a6000",
        "rtx3090",
        "rtx4090",
    }
)

_MAX_DIM: int = 2**31 - 1

_ALLOWED_PRIMARY_METRICS: frozenset[str] = frozenset(
    {"weighted_p50_us", "weighted_p95_us", "worst_case_p50_us"}
)

_ALLOWED_AGGREGATIONS: frozenset[str] = frozenset({"weighted_mean", "max"})


def run_deterministic_checks(spec: ProblemSpec) -> list[ValidationIssue]:
    """Run all six deterministic check categories on a resolved ProblemSpec.

    All checks run unconditionally — no short-circuit between categories.
    The spec's ``reference_kernel`` field must already be resolved to
    inline CUDA source before calling this function.

    Args:
        spec: A ProblemSpec with resolved reference_kernel.

    Returns:
        A list of validation issues (may be empty if everything passes).

    Implements: REQ-SB-001, SCN-SB-001-01, SCN-SB-001-02, SCN-SB-001-03,
        SCN-SB-001-04, SCN-SB-001-05
    Invariant: INV-SB-002 (pure function, no side effects)
    """
    issues: list[ValidationIssue] = []
    issues.extend(_check_reference_kernel(spec))
    issues.extend(_check_shape_cases(spec))
    issues.extend(_check_dtype(spec))
    issues.extend(_check_objective(spec))
    issues.extend(_check_target_gpu(spec))
    return issues


def _check_reference_kernel(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 2: Reference kernel quality after resolution."""
    issues: list[ValidationIssue] = []
    non_ws = len(
        spec.reference_kernel.replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("\r", "")
    )
    if non_ws < 20:
        issues.append(
            ValidationIssue(
                dimension="reference_kernel",
                severity="warn",
                message=(
                    f"Reference kernel has only {non_ws} non-whitespace characters; "
                    "this seems suspiciously short"
                ),
            )
        )
    return issues


def _check_shape_cases(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 3: Shape cases validation.

    Validates that shape_cases is non-empty, each has a non-empty unique
    shape_id, positive dims within 32-bit range, positive weight,
    correctness_tolerance in (0,1) if provided, and at least one has
    profile=True (warn if not).

    Implements: SCN-SB-001-03, SCN-SB-001-04
    """
    issues: list[ValidationIssue] = []

    if not spec.shape_cases:
        issues.append(
            ValidationIssue(
                dimension="shape_cases",
                severity="fail",
                message="shape_cases must be a non-empty list",
            )
        )
        return issues

    # Check for unique shape_ids
    seen_ids: set[str] = set()
    any_profile = False

    for i, sc in enumerate(spec.shape_cases):
        # shape_id must be non-empty
        if not sc.shape_id:
            issues.append(
                ValidationIssue(
                    dimension="shape_cases",
                    severity="fail",
                    message=f"shape_cases[{i}].shape_id must be a non-empty string",
                )
            )
        elif sc.shape_id in seen_ids:
            issues.append(
                ValidationIssue(
                    dimension="shape_cases",
                    severity="fail",
                    message=(
                        f"shape_cases[{i}].shape_id {sc.shape_id!r} is a duplicate"
                    ),
                )
            )
        else:
            seen_ids.add(sc.shape_id)

        # dims must be non-empty list of positive ints
        if not sc.dims:
            issues.append(
                ValidationIssue(
                    dimension="shape_cases",
                    severity="fail",
                    message=f"shape_cases[{i}].dims must be a non-empty list",
                )
            )
        else:
            for j, dim in enumerate(sc.dims):
                if not isinstance(dim, int) or isinstance(dim, bool):
                    issues.append(
                        ValidationIssue(
                            dimension="shape_cases",
                            severity="fail",
                            message=(
                                f"shape_cases[{i}].dims[{j}] must be an integer, "
                                f"got {type(dim).__name__}"
                            ),
                        )
                    )
                elif dim <= 0:
                    issues.append(
                        ValidationIssue(
                            dimension="shape_cases",
                            severity="fail",
                            message=(
                                f"shape_cases[{i}].dims[{j}] must be a positive "
                                f"integer, got {dim}"
                            ),
                        )
                    )
                elif dim > _MAX_DIM:
                    issues.append(
                        ValidationIssue(
                            dimension="shape_cases",
                            severity="fail",
                            message=(
                                f"shape_cases[{i}].dims[{j}] exceeds maximum 32-bit "
                                f"int ({_MAX_DIM}), got {dim}"
                            ),
                        )
                    )

        # weight must be > 0
        if sc.weight <= 0:
            issues.append(
                ValidationIssue(
                    dimension="shape_cases",
                    severity="fail",
                    message=(f"shape_cases[{i}].weight must be > 0, got {sc.weight}"),
                )
            )

        # correctness_tolerance if provided must be in (0, 1)
        if sc.correctness_tolerance is not None and not (
            0 < sc.correctness_tolerance < 1
        ):
            issues.append(
                ValidationIssue(
                    dimension="shape_cases",
                    severity="fail",
                    message=(
                        f"shape_cases[{i}].correctness_tolerance must be in "
                        f"(0, 1) exclusive, got {sc.correctness_tolerance}"
                    ),
                )
            )

        if sc.profile:
            any_profile = True

    # Warn if no shape has profile: true
    if not any_profile:
        issues.append(
            ValidationIssue(
                dimension="shape_cases",
                severity="warn",
                message=(
                    "No ShapeCase has profile=true; deep profiling will have "
                    "no designated shapes"
                ),
            )
        )

    return issues


def _check_dtype(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 4: Dtype recognition."""
    issues: list[ValidationIssue] = []
    if spec.dtype not in _KNOWN_DTYPES:
        issues.append(
            ValidationIssue(
                dimension="dtype",
                severity="fail",
                message=f"Unrecognized dtype: {spec.dtype!r}",
            )
        )
    return issues


def _check_objective(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 5: Objective and metric validation.

    Validates target_metric_value > 0, objective.primary_metric is
    recognized, objective.aggregation is recognized,
    objective.regression_guard_pct >= 0, and max_rounds >= 1.

    Implements: SCN-SB-001-05
    """
    issues: list[ValidationIssue] = []

    if spec.target_metric_value <= 0:
        issues.append(
            ValidationIssue(
                dimension="objective",
                severity="fail",
                message=(
                    f"target_metric_value must be positive, "
                    f"got {spec.target_metric_value}"
                ),
            )
        )

    if spec.objective.primary_metric not in _ALLOWED_PRIMARY_METRICS:
        issues.append(
            ValidationIssue(
                dimension="objective",
                severity="fail",
                message=(
                    f"Unrecognized objective.primary_metric: "
                    f"{spec.objective.primary_metric!r}"
                ),
            )
        )

    if spec.objective.aggregation not in _ALLOWED_AGGREGATIONS:
        issues.append(
            ValidationIssue(
                dimension="objective",
                severity="fail",
                message=(
                    f"Unrecognized objective.aggregation: "
                    f"{spec.objective.aggregation!r}"
                ),
            )
        )

    if spec.objective.regression_guard_pct < 0:
        issues.append(
            ValidationIssue(
                dimension="objective",
                severity="fail",
                message=(
                    f"objective.regression_guard_pct must be >= 0, "
                    f"got {spec.objective.regression_guard_pct}"
                ),
            )
        )

    if spec.max_rounds < 1:
        issues.append(
            ValidationIssue(
                dimension="objective",
                severity="fail",
                message=f"max_rounds must be >= 1, got {spec.max_rounds}",
            )
        )

    return issues


def _check_target_gpu(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 6: Target GPU recognition (warn, not fail)."""
    issues: list[ValidationIssue] = []
    if spec.target_gpu.lower() not in _KNOWN_GPUS:
        issues.append(
            ValidationIssue(
                dimension="target_gpu",
                severity="warn",
                message=f"Unrecognized target GPU: {spec.target_gpu!r}",
            )
        )
    return issues
