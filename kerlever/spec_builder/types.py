"""Spec Builder types — validation result types.

Spec: docs/spec-builder/spec.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, computed_field

ValidationSeverity = Literal["pass", "warn", "fail"]

ValidationDimension = Literal[
    "schema",
    "reference_kernel",
    "shape_cases",
    "dtype",
    "objective",
    "target_gpu",
    "consistency",
    "specificity",
    "feasibility",
    "completeness",
    "kernel_quality",
    "parse_error",
]


class ValidationIssue(BaseModel):
    """A single validation finding with dimension, severity, and message."""

    dimension: ValidationDimension
    severity: ValidationSeverity
    message: str


class ValidationResult(BaseModel):
    """Aggregated validation result.

    ``is_valid`` is True when no issue has severity="fail".
    """

    issues: list[ValidationIssue] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """Spec is valid when there are no fail-severity issues."""
        return not any(issue.severity == "fail" for issue in self.issues)
