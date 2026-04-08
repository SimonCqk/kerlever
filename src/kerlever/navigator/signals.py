"""Strategy Navigator signals — Phase 1 derived signal computation.

Pure function that derives decision signals from the accumulated
optimization state. Deterministic, side-effect-free.

Spec: docs/navigator/spec.md §6.1
"""

from __future__ import annotations

from collections import Counter

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DerivedSignals
from kerlever.types import Mode, OptimizationState


def compute_derived_signals(
    state: OptimizationState,
    config: NavigatorConfig,
) -> DerivedSignals:
    """Compute derived signals from the optimization state.

    This is the entry point for Phase 1. All signal values are derived
    deterministically from the state and config — no side effects.

    Args:
        state: Full accumulated optimization state across rounds.
        config: Navigator configuration parameters.

    Returns:
        DerivedSignals with all fields populated.

    Implements: REQ-NAV-002, REQ-NAV-005
    Invariant: INV-NAV-002 (deterministic and side-effect-free)
    """
    avg_delta = _compute_avg_delta(state, config)
    consecutive_exploit = _compute_consecutive_exploit_rounds(state)
    is_plateau = (
        avg_delta < config.plateau_threshold
        and consecutive_exploit >= config.plateau_rounds
    )
    is_regress = avg_delta < 0.0
    stable_bn = _compute_stable_bottleneck(state, config)
    new_bn = _compute_new_bottleneck(state)
    direction_counts = _compute_direction_attempt_counts(state)
    exhausted = _compute_exhausted_directions(stable_bn, direction_counts, config)

    return DerivedSignals(
        avg_delta=avg_delta,
        is_plateau=is_plateau,
        is_regress=is_regress,
        stable_bottleneck=stable_bn,
        new_bottleneck=new_bn,
        consecutive_exploit_rounds=consecutive_exploit,
        direction_attempt_counts=direction_counts,
        exhausted_directions=exhausted,
    )


def _compute_avg_delta(
    state: OptimizationState,
    config: NavigatorConfig,
) -> float:
    """Compute average improvement over the last N rounds.

    Collects improvement_over_prev_best from the most recent
    plateau_rounds entries. None values are treated as 0.0.
    Returns 0.0 when the rounds list is empty.
    """
    if not state.rounds:
        return 0.0

    recent = state.rounds[-config.plateau_rounds :]
    deltas = [
        r.improvement_over_prev_best
        if r.improvement_over_prev_best is not None
        else 0.0
        for r in recent
    ]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _compute_consecutive_exploit_rounds(state: OptimizationState) -> int:
    """Count consecutive EXPLOIT rounds backward from the most recent.

    Stops at the first EXPLORE round or beginning of history.
    Returns 0 on round 0 (no history).
    """
    count = 0
    for r in reversed(state.rounds):
        if r.mode == Mode.EXPLOIT:
            count += 1
        else:
            break
    return count


def _compute_stable_bottleneck(
    state: OptimizationState,
    config: NavigatorConfig,
) -> str | None:
    """Find a stable bottleneck tag across the last K rounds.

    A bottleneck is stable if all K rounds share at least one common tag
    (intersection of the K tag lists is non-empty). If multiple tags are
    common, the one with the highest frequency across all K lists wins.

    Returns None if fewer than K rounds of history exist or any of the
    K lists is empty.
    """
    k = config.stable_rounds
    history = state.bottleneck_history

    if len(history) < k:
        return None

    last_k = history[-k:]

    # Any empty list breaks the streak
    for tags in last_k:
        if not tags:
            return None

    # Find common tags across all K rounds
    common = set(last_k[0])
    for tags in last_k[1:]:
        common &= set(tags)

    if not common:
        return None

    # Among common tags, pick the most frequent across all K lists
    counter: Counter[str] = Counter()
    for tags in last_k:
        for tag in tags:
            if tag in common:
                counter[tag] += 1

    # most_common returns in descending order; first is the most frequent
    return counter.most_common(1)[0][0]


def _compute_new_bottleneck(state: OptimizationState) -> str | None:
    """Find a bottleneck tag from the latest round not seen in any prior round.

    Returns None on round 0 (no history), when the latest round has no tags,
    or when all tags have been seen before.
    """
    history = state.bottleneck_history

    if not history:
        return None

    latest_tags = history[-1]
    if not latest_tags:
        return None

    # Collect all tags from prior rounds (everything except the latest)
    prior_tags: set[str] = set()
    for tags in history[:-1]:
        prior_tags.update(tags)

    # Spec: "On round 0 (no history at all), new_bottleneck = None"
    # Empty history is handled above. If history has 1 entry,
    # prior_tags is empty so all latest tags are "new". This is
    # correct because the orchestrator appends history after each
    # round, so 1 entry means current_round >= 1.

    for tag in latest_tags:
        if tag not in prior_tags:
            return tag

    return None


def _compute_direction_attempt_counts(
    state: OptimizationState,
) -> dict[str, int]:
    """Count how many rounds used each direction.

    Iterates over all round summaries and tallies each direction value.
    """
    counts: dict[str, int] = {}
    for r in state.rounds:
        counts[r.direction] = counts.get(r.direction, 0) + 1
    return counts


def _compute_exhausted_directions(
    stable_bottleneck: str | None,
    direction_counts: dict[str, int],
    config: NavigatorConfig,
) -> set[str]:
    """Determine which directions are exhausted.

    A direction is exhausted if the stable bottleneck it targets has been
    attempted M or more times. In practice: if stable_bottleneck is not None
    and its attempt count >= max_direction_attempts, it is exhausted.

    The exhausted set is cumulative within a run but since we recompute
    from state each time, we only detect the current exhaustion status.
    """
    exhausted: set[str] = set()

    if stable_bottleneck is None:
        return exhausted

    if direction_counts.get(stable_bottleneck, 0) >= config.max_direction_attempts:
        exhausted.add(stable_bottleneck)

    return exhausted
