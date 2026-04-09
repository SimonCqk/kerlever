"""Strategy Navigator gates — Phase 2 deterministic gate checks.

Five gates evaluated in strict priority order. First match wins.
All gates are pure conditional checks on computed signals.

Spec: docs/navigator/spec.md §6.2
"""

from __future__ import annotations

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DerivedSignals, GateResult
from kerlever.types import Mode, OptimizationState, ProblemSpec, SubMode


def check_gates(
    signals: DerivedSignals,
    state: OptimizationState,
    problem_spec: ProblemSpec,
    config: NavigatorConfig,
) -> GateResult | None:
    """Check five deterministic gates in priority order.

    Returns the first GateResult whose condition is met, or None if
    no gate matches (decision proceeds to Phase 3 LLM reasoning).

    The gate evaluation order is fixed and cannot be reordered:
    1. Cold start (round 0)
    2. Plateau (exploit stagnation)
    3. Near target (close to goal)
    4. New bottleneck (novel signal → defer to LLM)
    5. Exhausted direction (direction depleted)

    Args:
        signals: Derived signals from Phase 1.
        state: Full optimization state.
        problem_spec: Problem specification with target performance.
        config: Navigator configuration.

    Returns:
        GateResult if a gate matched, None otherwise.

    Implements: REQ-NAV-001, REQ-NAV-002, REQ-NAV-003, REQ-NAV-004, REQ-NAV-005
    Invariant: INV-NAV-001 (gate evaluation order is fixed, first-match-wins)
    """
    # Gate 1: Cold Start
    result = _gate_cold_start(state)
    if result is not None:
        return result

    # Gate 2: Plateau
    result = _gate_plateau(signals, state, config)
    if result is not None:
        return result

    # Gate 3: Near Target
    result = _gate_near_target(state, problem_spec, config)
    if result is not None:
        return result

    # Gate 4: New Bottleneck → return None to trigger LLM
    if signals.new_bottleneck is not None:
        return None

    # Gate 5: Exhausted Direction
    result = _gate_exhausted_direction(signals)
    if result is not None:
        return result

    # No gate matched → proceed to Phase 3
    return None


def _gate_cold_start(state: OptimizationState) -> GateResult | None:
    """Gate 1: Cold start — round 0 with no optimization history.

    Implements: REQ-NAV-001, SCN-NAV-001-01
    """
    if state.current_round == 0:
        return GateResult(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason=(
                "Cold start: no optimization history, performing broad initial search"
            ),
            sub_mode=SubMode.DE_NOVO,
        )
    return None


def _gate_plateau(
    signals: DerivedSignals,
    state: OptimizationState,
    config: NavigatorConfig,
) -> GateResult | None:
    """Gate 2: Plateau — avg improvement below threshold for N exploit rounds.

    Implements: REQ-NAV-002, SCN-NAV-002-01
    """
    if signals.is_plateau:
        # Use the latest assessment's primary_tag or generic direction
        direction = "structural_change"
        if state.bottleneck_history:
            tag = state.bottleneck_history[-1].primary_tag
            if tag is not None:
                direction = tag

        return GateResult(
            mode=Mode.EXPLORE,
            direction=direction,
            reason=(
                f"Plateau detected: avg improvement {signals.avg_delta:.1%} "
                f"over {config.plateau_rounds} consecutive exploit rounds "
                f"below {config.plateau_threshold:.1%} threshold"
            ),
        )
    return None


def _gate_near_target(
    state: OptimizationState,
    problem_spec: ProblemSpec,
    config: NavigatorConfig,
) -> GateResult | None:
    """Gate 3: Near target — incumbent objective score within threshold of target.

    Condition: incumbent.objective_score.value <= target_metric_value / target_threshold
    Since lower objective score is better, this means the kernel is close to the goal.

    Implements: REQ-NAV-003, SCN-NAV-003-01
    """
    incumbent_score = state.incumbent.objective_score.value

    threshold_value = problem_spec.target_metric_value / config.target_threshold
    if incumbent_score <= threshold_value:
        # Use current bottleneck primary_tag or generic direction
        direction = "fine_tune"
        if state.bottleneck_history:
            tag = state.bottleneck_history[-1].primary_tag
            if tag is not None:
                direction = tag

        pct = problem_spec.target_metric_value / incumbent_score
        return GateResult(
            mode=Mode.EXPLOIT,
            direction=direction,
            reason=(
                f"Near target: incumbent is within {pct:.1%} of target, "
                f"fine-tuning only"
            ),
            sub_mode=SubMode.PARAM_SEARCH,
        )
    return None


def _gate_exhausted_direction(signals: DerivedSignals) -> GateResult | None:
    """Gate 5: Exhausted direction — stable bottleneck with M+ attempts.

    Implements: REQ-NAV-005, SCN-NAV-005-01
    """
    if (
        signals.stable_bottleneck is not None
        and signals.stable_bottleneck in signals.exhausted_directions
    ):
        count = signals.direction_attempt_counts.get(signals.stable_bottleneck, 0)
        return GateResult(
            mode=Mode.EXPLORE,
            direction="structural_change",
            reason=(
                f"Direction exhausted: {signals.stable_bottleneck} attempted "
                f"{count} times against stable bottleneck "
                f"{signals.stable_bottleneck}, trying new approach"
            ),
        )
    return None
