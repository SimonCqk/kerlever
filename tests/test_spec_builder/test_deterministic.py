"""Tests for deterministic validation checks.

Spec: docs/spec-builder/spec.md §6.2
"""

from __future__ import annotations

from typing import Any

import pytest

from kerlever.spec_builder.deterministic import run_deterministic_checks
from kerlever.types import ProblemSpec


def _make_valid_spec(**overrides: Any) -> ProblemSpec:
    """Create a valid ProblemSpec with optional field overrides."""
    defaults: dict[str, Any] = {
        "op_name": "matmul",
        "op_semantics": "C[M,N] = A[M,K] @ B[K,N]",
        "shape_cases": [
            {
                "shape_id": "4k_square",
                "dims": [4096, 4096, 4096],
                "weight": 1.0,
                "profile": True,
            },
        ],
        "dtype": "float16",
        "target_gpu": "A100",
        "objective": {
            "primary_metric": "weighted_p50_us",
            "aggregation": "weighted_mean",
            "regression_guard_pct": 0.0,
        },
        "target_metric_value": 1.0,
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


class TestObjectiveChecks:
    """Check 5: objective and metric validation."""

    def test_negative_target_metric_value_fails(self) -> None:
        """SCN-SB-001-05: target_metric_value must be > 0."""
        spec = _make_valid_spec(target_metric_value=-1.0)
        issues = run_deterministic_checks(spec)
        obj_fails = [
            i for i in issues if i.dimension == "objective" and i.severity == "fail"
        ]
        assert any("target_metric_value" in i.message for i in obj_fails)

    def test_zero_target_metric_value_fails(self) -> None:
        spec = _make_valid_spec(target_metric_value=0.0)
        issues = run_deterministic_checks(spec)
        obj_fails = [
            i for i in issues if i.dimension == "objective" and i.severity == "fail"
        ]
        assert any("target_metric_value" in i.message for i in obj_fails)

    def test_negative_regression_guard_pct_fails(self) -> None:
        """SCN-SB-001-05: regression_guard_pct must be >= 0."""
        spec = _make_valid_spec(
            objective={
                "primary_metric": "weighted_p50_us",
                "aggregation": "weighted_mean",
                "regression_guard_pct": -5.0,
            }
        )
        issues = run_deterministic_checks(spec)
        obj_fails = [
            i for i in issues if i.dimension == "objective" and i.severity == "fail"
        ]
        assert any("regression_guard_pct" in i.message for i in obj_fails)

    def test_max_rounds_zero_fails(self) -> None:
        spec = _make_valid_spec(max_rounds=0)
        issues = run_deterministic_checks(spec)
        obj_fails = [
            i for i in issues if i.dimension == "objective" and i.severity == "fail"
        ]
        assert any("max_rounds" in i.message for i in obj_fails)

    def test_valid_objective_passes(self) -> None:
        spec = _make_valid_spec()
        issues = run_deterministic_checks(spec)
        obj_fails = [
            i for i in issues if i.dimension == "objective" and i.severity == "fail"
        ]
        assert obj_fails == []


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


class TestShapeCasesCheck:
    """Check 3: shape cases validation."""

    def test_empty_shape_cases_fails(self) -> None:
        spec = _make_valid_spec(shape_cases=[])
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("non-empty" in i.message for i in sc_fails)

    def test_negative_dimension_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "neg",
                    "dims": [-1, 4096],
                    "weight": 1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("positive" in i.message for i in sc_fails)

    def test_zero_dimension_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "zero",
                    "dims": [0, 4096],
                    "weight": 1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1

    def test_dimension_exceeds_max_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "big",
                    "dims": [2**31, 4096],
                    "weight": 1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("exceeds" in i.message for i in sc_fails)

    def test_empty_dims_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "empty_dims",
                    "dims": [],
                    "weight": 1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1

    def test_duplicate_shape_id_fails(self) -> None:
        """SCN-SB-001-03: Duplicate shape_id fails validation."""
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "square_4k",
                    "dims": [4096, 4096, 4096],
                    "weight": 1.0,
                    "profile": True,
                },
                {
                    "shape_id": "square_4k",
                    "dims": [8192, 128, 4096],
                    "weight": 0.5,
                },
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("duplicate" in i.message.lower() for i in sc_fails)

    def test_empty_shape_id_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {"shape_id": "", "dims": [4096], "weight": 1.0, "profile": True}
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("non-empty" in i.message for i in sc_fails)

    def test_zero_weight_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "zero_weight",
                    "dims": [4096],
                    "weight": 0.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert len(sc_fails) >= 1
        assert any("weight" in i.message for i in sc_fails)

    def test_negative_weight_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "neg_weight",
                    "dims": [4096],
                    "weight": -1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert any("weight" in i.message for i in sc_fails)

    def test_correctness_tolerance_out_of_range_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "bad_tol",
                    "dims": [4096],
                    "weight": 1.0,
                    "correctness_tolerance": 0.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert any("correctness_tolerance" in i.message for i in sc_fails)

    def test_correctness_tolerance_of_one_fails(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "tol_one",
                    "dims": [4096],
                    "weight": 1.0,
                    "correctness_tolerance": 1.0,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert any("correctness_tolerance" in i.message for i in sc_fails)

    def test_valid_correctness_tolerance_passes(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "good_tol",
                    "dims": [4096, 4096, 4096],
                    "weight": 1.0,
                    "correctness_tolerance": 0.01,
                    "profile": True,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert sc_fails == []

    def test_no_profile_shape_warns(self) -> None:
        """SCN-SB-001-04: No profile shape emits warning."""
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "no_profile",
                    "dims": [4096, 4096, 4096],
                    "weight": 1.0,
                    "profile": False,
                }
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_warns = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "warn"
        ]
        assert len(sc_warns) >= 1
        assert any("profile" in i.message.lower() for i in sc_warns)
        # Must not cause failure
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert sc_fails == []

    def test_no_profile_default_warns(self) -> None:
        """Omitting profile field (defaults to false) should also warn."""
        spec = _make_valid_spec(
            shape_cases=[
                {"shape_id": "default_no_profile", "dims": [4096, 4096], "weight": 1.0}
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_warns = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "warn"
        ]
        assert len(sc_warns) >= 1

    def test_multiple_shapes_with_profile_passes(self) -> None:
        spec = _make_valid_spec(
            shape_cases=[
                {
                    "shape_id": "a",
                    "dims": [4096, 4096, 4096],
                    "weight": 1.0,
                    "profile": True,
                },
                {
                    "shape_id": "b",
                    "dims": [8192, 128, 4096],
                    "weight": 0.5,
                    "profile": False,
                },
            ]
        )
        issues = run_deterministic_checks(spec)
        sc_fails = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "fail"
        ]
        assert sc_fails == []
        sc_warns = [
            i for i in issues if i.dimension == "shape_cases" and i.severity == "warn"
        ]
        assert sc_warns == []


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
