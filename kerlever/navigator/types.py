"""Strategy Navigator types — internal data types for the Navigator module.

Spec: docs/navigator/spec.md §5 (Internal Types)
"""

from __future__ import annotations

from pydantic import BaseModel

from kerlever.types import Mode, SubMode


class DerivedSignals(BaseModel, frozen=True):
    """Output of Phase 1 signal computation.

    All fields are derived deterministically from the OptimizationState.
    This is a value object — immutable once constructed.
    """

    avg_delta: float
    is_plateau: bool
    is_regress: bool
    stable_bottleneck: str | None
    new_bottleneck: str | None
    consecutive_exploit_rounds: int
    direction_attempt_counts: dict[str, int]
    exhausted_directions: set[str]


class GateResult(BaseModel, frozen=True):
    """Output of a matched deterministic gate in Phase 2."""

    mode: Mode
    direction: str
    reason: str
    sub_mode: SubMode | None = None


class LLMDecision(BaseModel, frozen=True):
    """Parsed output of LLM reasoning in Phase 3."""

    mode: Mode
    direction: str
    sub_mode: SubMode | None = None
    reasoning: str
    confidence: str


class DirectionStats(BaseModel, frozen=True):
    """Per-direction performance tracking for UCB1 scoring."""

    direction: str
    visits: int
    total_perf_gain: float
    avg_perf_gain: float
