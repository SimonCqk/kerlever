"""Strategy Navigator UCB1 — Phase 3 deterministic fallback direction selection.

UCB1 (Upper Confidence Bound 1) selects a direction by balancing
exploitation of historically good directions against exploration
of under-tried ones.

Spec: docs/navigator/spec.md §6.4
"""

from __future__ import annotations

import math

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DirectionStats
from kerlever.types import OptimizationState


def compute_direction_stats(state: OptimizationState) -> list[DirectionStats]:
    """Compute per-direction performance statistics from round history.

    For each unique direction in the round summaries, computes visits,
    total performance gain, and average performance gain. None values
    in improvement_over_prev_best are treated as 0.0.

    Args:
        state: Full optimization state with rounds history.

    Returns:
        List of DirectionStats, one per unique direction.
    """
    visits: dict[str, int] = {}
    total_gain: dict[str, float] = {}

    for r in state.rounds:
        direction = r.direction
        visits[direction] = visits.get(direction, 0) + 1
        gain = (
            r.improvement_over_prev_best
            if r.improvement_over_prev_best is not None
            else 0.0
        )
        total_gain[direction] = total_gain.get(direction, 0.0) + gain

    stats: list[DirectionStats] = []
    for direction in visits:
        v = visits[direction]
        tg = total_gain[direction]
        avg = tg / v if v > 0 else 0.0
        stats.append(
            DirectionStats(
                direction=direction,
                visits=v,
                total_perf_gain=tg,
                avg_perf_gain=avg,
            )
        )

    return stats


def ucb1_select(
    stats: list[DirectionStats],
    total_rounds: int,
    exhausted: set[str],
    config: NavigatorConfig,
) -> str:
    """Select the direction with the highest UCB1 score.

    UCB1(d) = avg_perf_gain(d) + C * sqrt(ln(total_rounds) / visits(d))

    Edge cases:
    - visits=0 for a direction: UCB1 score is +infinity (selected first).
    - total_rounds=0: select by avg_perf_gain only (ln(0) treated as 0).
    - All directions exhausted: pick best from exhausted set.

    Among directions with equal scores (e.g., multiple unvisited),
    the first one encountered is selected (deterministic ordering).

    Args:
        stats: Per-direction performance statistics.
        total_rounds: Number of completed rounds.
        exhausted: Set of exhausted direction names.
        config: Navigator configuration (ucb1_c coefficient).

    Returns:
        The selected direction string.

    Implements: REQ-NAV-007
    """
    if not stats:
        return "initial_exploration"

    # Separate non-exhausted and exhausted candidates
    non_exhausted = [s for s in stats if s.direction not in exhausted]
    candidates = non_exhausted if non_exhausted else stats

    # Compute ln(total_rounds) once; treat ln(0) as 0
    ln_total = math.log(total_rounds) if total_rounds > 0 else 0.0

    best_direction = candidates[0].direction
    best_score = float("-inf")

    for s in candidates:
        if s.visits == 0:
            score = float("inf")
        elif ln_total == 0.0:
            # total_rounds=0 edge case: select by avg_gain only
            score = s.avg_perf_gain
        else:
            exploration_bonus = config.ucb1_c * math.sqrt(ln_total / s.visits)
            score = s.avg_perf_gain + exploration_bonus

        if score > best_score:
            best_score = score
            best_direction = s.direction

    return best_direction
