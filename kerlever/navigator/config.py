"""Strategy Navigator configuration — 11 tunable parameters.

Spec: docs/navigator/spec.md §5 (Navigator Configuration)
"""

from __future__ import annotations

from pydantic import BaseModel


class NavigatorConfig(BaseModel, frozen=True):
    """Configuration parameters for the Strategy Navigator.

    All defaults match the spec §5 and docs/strategy-navigator.md
    Configuration Parameters table.
    """

    plateau_threshold: float = 0.02
    """Minimum avg improvement to not be considered plateau (2%)."""

    plateau_rounds: int = 3
    """(N) Consecutive exploit rounds below threshold before forcing explore."""

    stable_rounds: int = 3
    """(K) Rounds with same bottleneck tag to trigger stability check."""

    max_direction_attempts: int = 3
    """(M) Attempts before marking a direction exhausted."""

    tabu_window: int = 5
    """(W) Rounds to keep a (hash, tag) pair in tabu list."""

    target_threshold: float = 0.95
    """Performance ratio to target that triggers exploit-only mode."""

    llm_context_budget: int = 2048
    """Max tokens for LLM reasoning context."""

    llm_confidence_min: str = "medium"
    """Minimum LLM confidence to accept (high > medium > low)."""

    ucb1_c: float = 1.414
    """UCB1 exploration coefficient (sqrt(2))."""

    exploit_candidates: int = 5
    """Number of candidates to generate in exploit mode."""

    explore_candidates: int = 3
    """Number of candidates to generate in explore mode."""
