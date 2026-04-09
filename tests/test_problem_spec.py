"""Tests for problem_spec YAML loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from kerlever.problem_spec import load_problem_spec
from kerlever.types import ProblemSpec


def test_load_valid_yaml(tmp_path: Path) -> None:
    """Load a valid YAML file and verify all fields are populated."""
    yaml_content = """\
op_name: matmul
op_semantics: "C[M,N] = A[M,K] @ B[K,N]"
dtype: float16
target_gpu: A100
shape_cases:
  - shape_id: "4k_square"
    dims: [4096, 4096, 4096]
    weight: 1.0
    profile: true
objective:
  primary_metric: weighted_p50_us
  aggregation: weighted_mean
  regression_guard_pct: 0.02
target_metric_value: 1.0
max_rounds: 20
reference_kernel: |
  __global__ void matmul() {}
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml_content)

    spec = load_problem_spec(spec_file)

    assert isinstance(spec, ProblemSpec)
    assert spec.op_name == "matmul"
    assert spec.op_semantics == "C[M,N] = A[M,K] @ B[K,N]"
    assert len(spec.shape_cases) == 1
    assert spec.shape_cases[0].shape_id == "4k_square"
    assert spec.shape_cases[0].dims == [4096, 4096, 4096]
    assert spec.shape_cases[0].weight == 1.0
    assert spec.shape_cases[0].profile is True
    assert spec.dtype == "float16"
    assert spec.target_gpu == "A100"
    assert spec.objective.primary_metric == "weighted_p50_us"
    assert spec.objective.aggregation == "weighted_mean"
    assert spec.objective.regression_guard_pct == 0.02
    assert spec.target_metric_value == 1.0
    assert spec.max_rounds == 20
    assert "matmul" in spec.reference_kernel


def test_load_missing_field(tmp_path: Path) -> None:
    """Loading YAML with a missing required field raises ValidationError."""
    yaml_content = """\
op_name: matmul
dtype: float16
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml_content)

    with pytest.raises(ValueError):  # pydantic.ValidationError subclass
        load_problem_spec(spec_file)


def test_load_nonexistent_file(tmp_path: Path) -> None:
    """Loading from a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_problem_spec(tmp_path / "nonexistent.yaml")


def test_load_example_spec() -> None:
    """Load the example matmul_spec.yaml from the examples directory."""
    spec_file = Path(__file__).parent.parent / "examples" / "matmul_spec.yaml"
    if not spec_file.exists():
        pytest.skip("Example spec file not found")

    spec = load_problem_spec(spec_file)
    assert spec.op_name == "matmul"
    assert spec.max_rounds == 20
    assert spec.target_metric_value == 1.0
    assert len(spec.shape_cases) > 0
    assert spec.objective.primary_metric == "weighted_p50_us"


def test_load_multiple_shape_cases(tmp_path: Path) -> None:
    """Load YAML with multiple shape cases and verify all are parsed."""
    yaml_content = """\
op_name: matmul
op_semantics: "C = A @ B"
dtype: float16
target_gpu: A100
shape_cases:
  - shape_id: "small"
    dims: [1024, 1024, 1024]
    weight: 0.3
    profile: false
  - shape_id: "medium"
    dims: [4096, 4096, 4096]
    weight: 1.0
    profile: true
  - shape_id: "large"
    dims: [8192, 8192, 8192]
    weight: 0.5
    correctness_tolerance: 0.01
    profile: true
objective:
  primary_metric: weighted_p50_us
  aggregation: weighted_mean
  regression_guard_pct: 0.02
target_metric_value: 1.0
max_rounds: 10
reference_kernel: |
  __global__ void matmul() {}
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml_content)

    spec = load_problem_spec(spec_file)

    assert len(spec.shape_cases) == 3
    assert spec.shape_cases[0].shape_id == "small"
    assert spec.shape_cases[0].weight == 0.3
    assert spec.shape_cases[1].shape_id == "medium"
    assert spec.shape_cases[1].profile is True
    assert spec.shape_cases[2].correctness_tolerance == 0.01
