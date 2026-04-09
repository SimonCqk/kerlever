"""Strategy Navigator signals — Phase 1 derived signal computation.

Pure function that derives decision signals from the accumulated
optimization state. Deterministic, side-effect-free.

Spec: docs/navigator/spec.md §6.1
"""

from __future__ import annotations

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
    """Compute average relative gain over the last N rounds.

    Collects rel_gain_vs_prev_best from the most recent
    plateau_rounds entries. None values are treated as 0.0.
    Returns 0.0 when the rounds list is empty.
    """
    if not state.rounds:
        return 0.0

    recent = state.rounds[-config.plateau_rounds :]
    deltas = [
        r.rel_gain_vs_prev_best if r.rel_gain_vs_prev_best is not None else 0.0
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
    """Find a stable bottleneck primary_tag across the last K assessments.

    A bottleneck is stable if all K most recent BottleneckAssessment entries
    share the same primary_tag (and it is not None).

    Returns None if fewer than K assessments exist or any has
    primary_tag = None.
    """
    k = config.stable_rounds
    history = state.bottleneck_history

    if len(history) < k:
        return None

    last_k = history[-k:]

    # All K assessments must have the same non-None primary_tag
    first_tag = last_k[0].primary_tag
    if first_tag is None:
        return None

    for assessment in last_k[1:]:
        if assessment.primary_tag != first_tag:
            return None

    return first_tag


def _compute_new_bottleneck(state: OptimizationState) -> str | None:
    """Find a new primary_tag from the latest BottleneckAssessment.

    Compares the latest assessment's primary_tag against all prior
    assessments' primary_tag values. Returns the tag if it was never
    seen as a primary_tag before, else None.

    Returns None on round 0 (no history), when the latest assessment
    has primary_tag = None, or when the tag has been seen before.
    """
    history = state.bottleneck_history

    if not history:
        return None

    latest_tag = history[-1].primary_tag
    if latest_tag is None:
        return None

    # Collect all primary_tags from prior assessments
    prior_tags: set[str | None] = set()
    for assessment in history[:-1]:
        prior_tags.add(assessment.primary_tag)

    # Spec: "On round 0 (no history at all), new_bottleneck = None"
    # Empty history is handled above. If history has 1 entry,
    # prior_tags is empty so the latest primary_tag is "new". This is
    # correct because the orchestrator appends history after each
    # round, so 1 entry means current_round >= 1.

    if latest_tag not in prior_tags:
        return latest_tag

    return None


def _compute_direction_attempt_counts(
    state: OptimizationState,
) -> dict[str, int]:
    """Count how many attempts used each direction.

    Iterates over all AttemptRecord entries in state.attempts and
    tallies each direction value. This uses typed attempt records
    rather than round summaries for more accurate counting.
    """
    counts: dict[str, int] = {}
    for attempt in state.attempts:
        counts[attempt.direction] = counts.get(attempt.direction, 0) + 1
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
