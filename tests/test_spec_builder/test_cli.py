"""Tests for the Spec Builder CLI entry point.

Spec: docs/spec-builder/spec.md §6.6
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EXAMPLE_SPEC = _PROJECT_ROOT / "examples" / "matmul_spec.yaml"


class TestCLIValidateNoLLM:
    """--validate with --no-llm should work without API key."""

    def test_valid_spec_exits_zero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kerlever.spec_builder",
                "--validate",
                str(_EXAMPLE_SPEC),
                "--no-llm",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env={"PATH": "", "PYTHONPATH": str(_PROJECT_ROOT / "src")},
            timeout=30,
        )
        assert result.returncode == 0
        assert '"is_valid": true' in result.stdout


class TestCLIInvalidSpec:
    """Invalid spec should exit 1."""

    def test_invalid_spec_exits_one(self, tmp_path: Path) -> None:
        bad_spec = tmp_path / "bad.yaml"
        bad_spec.write_text(
            "op_name: test\n"
            "op_semantics: test\n"
            "shape_cases:\n"
            "  - shape_id: empty_dims\n"
            "    dims: []\n"
            "    weight: 1.0\n"
            "dtype: imaginary_type\n"
            "target_gpu: A100\n"
            "objective:\n"
            "  primary_metric: weighted_p50_us\n"
            "  aggregation: weighted_mean\n"
            "  regression_guard_pct: 0.0\n"
            "target_metric_value: 1.0\n"
            "max_rounds: 20\n"
            "reference_kernel: |\n"
            "  __global__ void f() {\n"
            "    int x = 1; int y = 2; int z = x + y;\n"
            "  }\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "kerlever.spec_builder",
                "--validate",
                str(bad_spec),
                "--no-llm",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env={"PATH": "", "PYTHONPATH": str(_PROJECT_ROOT / "src")},
            timeout=30,
        )
        assert result.returncode == 1
        assert '"is_valid": false' in result.stdout


class TestCLINoArgs:
    """No args should exit 2."""

    def test_no_args_exits_two(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "kerlever.spec_builder"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env={"PATH": "", "PYTHONPATH": str(_PROJECT_ROOT / "src")},
            timeout=30,
        )
        assert result.returncode == 2
