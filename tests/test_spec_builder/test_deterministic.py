"""Tests for deterministic validation checks.

Spec: docs/spec-builder/spec.md §6.2
"""

from __future__ import annotations

import pytest

from kerlever.spec_builder.deterministic import run_deterministic_checks
from kerlever.types import ProblemSpec


def _make_valid_spec(**overrides: object) -> ProblemSpec:
    """Create a valid ProblemSpec with optional field overrides."""
    defaults: dict[str, object] = {
        "op_name": "matmul",
        "op_semantics": "C[M,N] = A[M,K] @ B[K,N]",
        "shapes": [[4096, 4096, 4096]],
        "dtype": "float16",
        "target_gpu": "A100",
        "baseline_perf_us": 5.0,
        "target_perf_us": 1.0,
        "tolerance": 0.05,
        "max_rounds": 20,
        "reference_kernel": (
            "__global__ void matmul(const half* A, const half* B, half* C, "
            "int M, int N, int K) { /* naive impl */ }"
        ),
    }
    defaults.update(overrides)
    return ProblemSpec.model_validate(defaults)


class TestValidSpecPasses:
    """SCN-SB-001-01: Valid spec passes all deterministic checks."""

    def test_valid_spec_returns_no_fail_issues(self) -> None:
        spec = _make_valid_spec()
        issues = run_deterministic_checks(spec)
        fail_issues = [i for i in issues if i.severity == "fail"]
        assert fail_issues == []

    def test_valid_spec_may_have_no_issues(self) -> None:
        spec = _make_valid_spec()
        issues = run_deterministic_checks(spec)
        # A valid spec should have no fail issues
        assert all(i.severity != "fail" for i in issues)


class TestDtypeCheck:
    """Check 4: dtype recognition."""

    def test_unrecognized_dtype_fails(self) -> None:
        spec = _make_valid_spec(dtype="imaginary_type")
        issues = run_deterministic_checks(spec)
        dtype_fails = [
            i for i in issues if i.dimension == "dtype" and i.severity == "fail"
        ]
        assert len(dtype_fails) == 1
        assert "imaginary_type" in dtype_fails[0].message

    @pytest.mark.parametrize(
        "dtype",
        [
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
        ],
    )
    def test_all_known_dtypes_pass(self, dtype: str) -> None:
        spec = _make_valid_spec(dtype=dtype)
        issues = run_deterministic_checks(spec)
        dtype_fails = [
            i for i in issues if i.dimension == "dtype" and i.severity == "fail"
        ]
        assert dtype_fails == []


class TestNumericChecks:
    """Check 5: numeric sanity."""

    def test_target_greater_than_baseline_fails(self) -> None:
        spec = _make_valid_spec(baseline_perf_us=1.0, target_perf_us=5.0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("target_perf_us" in i.message for i in numeric_fails)

    def test_negative_target_fails(self) -> None:
        spec = _make_valid_spec(target_perf_us=-1.0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("target_perf_us" in i.message for i in numeric_fails)

    def test_negative_baseline_fails(self) -> None:
        spec = _make_valid_spec(baseline_perf_us=-1.0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("baseline_perf_us" in i.message for i in numeric_fails)

    def test_tolerance_out_of_range_fails(self) -> None:
        spec = _make_valid_spec(tolerance=0.0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("tolerance" in i.message for i in numeric_fails)

    def test_tolerance_of_one_fails(self) -> None:
        spec = _make_valid_spec(tolerance=1.0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("tolerance" in i.message for i in numeric_fails)

    def test_max_rounds_zero_fails(self) -> None:
        spec = _make_valid_spec(max_rounds=0)
        issues = run_deterministic_checks(spec)
        numeric_fails = [
            i for i in issues if i.dimension == "numeric" and i.severity == "fail"
        ]
        assert any("max_rounds" in i.message for i in numeric_fails)


class TestTargetGpuCheck:
    """Check 6: target GPU recognition."""

    def test_unknown_gpu_warns(self) -> None:
        spec = _make_valid_spec(target_gpu="QUANTUM_GPU_3000")
        issues = run_deterministic_checks(spec)
        gpu_warns = [
            i for i in issues if i.dimension == "target_gpu" and i.severity == "warn"
        ]
        assert len(gpu_warns) == 1
        assert "QUANTUM_GPU_3000" in gpu_warns[0].message

    def test_known_gpu_does_not_warn(self) -> None:
        spec = _make_valid_spec(target_gpu="A100")
        issues = run_deterministic_checks(spec)
        gpu_warns = [i for i in issues if i.dimension == "target_gpu"]
        assert gpu_warns == []

    def test_known_gpu_case_insensitive(self) -> None:
        spec = _make_valid_spec(target_gpu="a100")
        issues = run_deterministic_checks(spec)
        gpu_warns = [i for i in issues if i.dimension == "target_gpu"]
        assert gpu_warns == []


class TestShapesCheck:
    """Check 3: shapes and dimensions."""

    def test_empty_shapes_fails(self) -> None:
        spec = _make_valid_spec(shapes=[])
        issues = run_deterministic_checks(spec)
        shape_fails = [
            i for i in issues if i.dimension == "shapes" and i.severity == "fail"
        ]
        assert len(shape_fails) >= 1
        assert any("non-empty" in i.message for i in shape_fails)

    def test_negative_dimension_fails(self) -> None:
        spec = _make_valid_spec(shapes=[[-1, 4096]])
        issues = run_deterministic_checks(spec)
        shape_fails = [
            i for i in issues if i.dimension == "shapes" and i.severity == "fail"
        ]
        assert len(shape_fails) >= 1
        assert any("positive" in i.message for i in shape_fails)

    def test_zero_dimension_fails(self) -> None:
        spec = _make_valid_spec(shapes=[[0, 4096]])
        issues = run_deterministic_checks(spec)
        shape_fails = [
            i for i in issues if i.dimension == "shapes" and i.severity == "fail"
        ]
        assert len(shape_fails) >= 1

    def test_dimension_exceeds_max_fails(self) -> None:
        spec = _make_valid_spec(shapes=[[2**31, 4096]])
        issues = run_deterministic_checks(spec)
        shape_fails = [
            i for i in issues if i.dimension == "shapes" and i.severity == "fail"
        ]
        assert len(shape_fails) >= 1
        assert any("exceeds" in i.message for i in shape_fails)

    def test_empty_inner_shape_fails(self) -> None:
        spec = _make_valid_spec(shapes=[[]])
        issues = run_deterministic_checks(spec)
        shape_fails = [
            i for i in issues if i.dimension == "shapes" and i.severity == "fail"
        ]
        assert len(shape_fails) >= 1


class TestReferenceKernelCheck:
    """Check 2: reference kernel quality after resolution."""

    def test_short_kernel_warns(self) -> None:
        spec = _make_valid_spec(reference_kernel="__global__ void f(){}")
        issues = run_deterministic_checks(spec)
        ref_warns = [
            i
            for i in issues
            if i.dimension == "reference_kernel" and i.severity == "warn"
        ]
        assert len(ref_warns) == 1
        assert "short" in ref_warns[0].message
