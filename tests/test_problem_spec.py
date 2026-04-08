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
shapes:
  - [4096, 4096, 4096]
dtype: float16
target_gpu: A100
baseline_perf_us: 5.0
target_perf_us: 1.0
tolerance: 0.05
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
    assert spec.shapes == [[4096, 4096, 4096]]
    assert spec.dtype == "float16"
    assert spec.target_gpu == "A100"
    assert spec.baseline_perf_us == 5.0
    assert spec.target_perf_us == 1.0
    assert spec.tolerance == 0.05
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
    assert spec.target_perf_us == 1.0
