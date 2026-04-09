"""Kerlever problem_spec — YAML loading and validation for ProblemSpec.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kerlever.types import ProblemSpec


def load_problem_spec(path: Path) -> ProblemSpec:
    """Load a ProblemSpec from a YAML file.

    Reads the YAML file at the given path and validates it against
    the ProblemSpec Pydantic model. The YAML must contain shape_cases
    (list of ShapeCase objects) and an objective (PerformanceObjective).

    Args:
        path: Path to the YAML specification file.

    Returns:
        A validated ProblemSpec instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If the YAML data does not match ProblemSpec.

    Implements: REQ-ORCH-008, SCN-ORCH-008-01
    """
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return ProblemSpec.model_validate(data)
