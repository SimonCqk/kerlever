"""``OperationAdapter`` protocol + common adapter types.

Spec: docs/compiler-service/spec.md §6.2
Design: docs/compiler-service/design.md §4.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from kerlever.compiler_service.types import (
    CandidateRole,
    ComparisonMode,
    KernelExecutionSpec,
)
from kerlever.types import ProblemSpec, ShapeCase


@dataclass(frozen=True)
class InputBundle:
    """Inputs allocated by an adapter for one shape case."""

    shape_id: str
    buffers: dict[str, bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class ShapeComparisonResult:
    """Per-shape correctness comparison result."""

    shape_id: str
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None


@runtime_checkable
class OperationAdapter(Protocol):
    """Stable behavioral boundary between the core and op-specific logic.

    The core phases NEVER branch on ``op_name`` (INV-CS-013); everything
    operation-specific flows through this Protocol.
    """

    op_name: ClassVar[str]

    def adapter_version(self) -> str:
        """Return the adapter-version string baked into ``artifact_key``."""

    def abi_contract(self) -> tuple[str, str]:
        """Return ``(abi_name, abi_version)``."""

    def default_block_dim(self, problem_spec: ProblemSpec) -> tuple[int, int, int]:
        """Conservative default block dim for legacy inference (spec §6.2)."""

    def default_tolerance(self, dtype: str) -> float:
        """Per-dtype tolerance default."""

    def comparison_mode(self, dtype: str) -> ComparisonMode:
        """``tolerance`` for floats unless explicitly ``exact`` (spec §6.2)."""

    def high_risk_shape_ids(self, problem_spec: ProblemSpec) -> set[str]:
        """Return shape ids the adapter considers high-risk for sanitizer."""

    def validate_problem_spec(self, problem_spec: ProblemSpec) -> str | None:
        """Return an interface-contract error reason, or None when supported."""

    def allocate_inputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        seed: int,
    ) -> InputBundle:
        """Return deterministic inputs for one shape case."""

    def build_harness_source(
        self,
        execution_spec: KernelExecutionSpec,
        problem_spec: ProblemSpec,
        role: CandidateRole,
        kernel_source: str,
    ) -> str:
        """Render a CUDA harness for the given role + kernel source."""

    def compare_outputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        reference_output: Path,
        candidate_output: Path,
        tolerance: float,
        comparison_mode: ComparisonMode,
    ) -> ShapeComparisonResult:
        """Compare two output artifacts and return a per-shape result."""
