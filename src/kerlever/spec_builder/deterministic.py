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


def run_deterministic_checks(spec: ProblemSpec) -> list[ValidationIssue]:
    """Run all six deterministic check categories on a resolved ProblemSpec.

    All checks run unconditionally — no short-circuit between categories.
    The spec's ``reference_kernel`` field must already be resolved to
    inline CUDA source before calling this function.

    Args:
        spec: A ProblemSpec with resolved reference_kernel.

    Returns:
        A list of validation issues (may be empty if everything passes).

    Implements: REQ-SB-001, SCN-SB-001-01, SCN-SB-001-02
    Invariant: INV-SB-002 (pure function, no side effects)
    """
    issues: list[ValidationIssue] = []
    issues.extend(_check_reference_kernel(spec))
    issues.extend(_check_shapes(spec))
    issues.extend(_check_dtype(spec))
    issues.extend(_check_numeric(spec))
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


def _check_shapes(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 3: Shapes and dimensions validation."""
    issues: list[ValidationIssue] = []
    if not spec.shapes:
        issues.append(
            ValidationIssue(
                dimension="shapes",
                severity="fail",
                message="shapes must be a non-empty list",
            )
        )
        return issues

    for i, shape in enumerate(spec.shapes):
        if not shape:
            issues.append(
                ValidationIssue(
                    dimension="shapes",
                    severity="fail",
                    message=f"shapes[{i}] must be a non-empty list",
                )
            )
            continue
        for j, dim in enumerate(shape):
            if not isinstance(dim, int) or isinstance(dim, bool):
                issues.append(
                    ValidationIssue(
                        dimension="shapes",
                        severity="fail",
                        message=(
                            f"shapes[{i}][{j}] must be an integer, "
                            f"got {type(dim).__name__}"
                        ),
                    )
                )
            elif dim <= 0:
                issues.append(
                    ValidationIssue(
                        dimension="shapes",
                        severity="fail",
                        message=(
                            f"shapes[{i}][{j}] must be a positive integer, got {dim}"
                        ),
                    )
                )
            elif dim > _MAX_DIM:
                issues.append(
                    ValidationIssue(
                        dimension="shapes",
                        severity="fail",
                        message=(
                            f"shapes[{i}][{j}] exceeds maximum 32-bit int "
                            f"({_MAX_DIM}), got {dim}"
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


def _check_numeric(spec: ProblemSpec) -> list[ValidationIssue]:
    """Check 5: Numeric sanity checks."""
    issues: list[ValidationIssue] = []

    if spec.baseline_perf_us <= 0:
        issues.append(
            ValidationIssue(
                dimension="numeric",
                severity="fail",
                message=(
                    f"baseline_perf_us must be positive, got {spec.baseline_perf_us}"
                ),
            )
        )

    if spec.target_perf_us <= 0:
        issues.append(
            ValidationIssue(
                dimension="numeric",
                severity="fail",
                message=f"target_perf_us must be positive, got {spec.target_perf_us}",
            )
        )

    if spec.target_perf_us > spec.baseline_perf_us:
        issues.append(
            ValidationIssue(
                dimension="numeric",
                severity="fail",
                message=(
                    f"target_perf_us ({spec.target_perf_us}) must be <= "
                    f"baseline_perf_us ({spec.baseline_perf_us})"
                ),
            )
        )

    if not (0 < spec.tolerance < 1):
        issues.append(
            ValidationIssue(
                dimension="numeric",
                severity="fail",
                message=f"tolerance must be in (0, 1) exclusive, got {spec.tolerance}",
            )
        )

    if spec.max_rounds < 1:
        issues.append(
            ValidationIssue(
                dimension="numeric",
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
