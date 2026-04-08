"""Tests for Navigator UCB1 fallback direction selection.

Spec: docs/navigator/spec.md §6.4
"""

from __future__ import annotations

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DirectionStats
from kerlever.navigator.ucb1 import compute_direction_stats, ucb1_select
from kerlever.types import Mode, OptimizationState, ProblemSpec, RoundSummary


def _make_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        shapes=[[1024, 1024], [1024, 1024]],
        dtype="float32",
        target_gpu="A100",
        baseline_perf_us=100.0,
        target_perf_us=10.0,
        tolerance=0.05,
        max_rounds=20,
        reference_kernel="__global__ void k() {}",
    )


def _make_state(**kwargs: object) -> OptimizationState:
    defaults: dict[str, object] = {
        "problem_spec": _make_problem_spec(),
        "current_round": 0,
    }
    defaults.update(kwargs)
    return OptimizationState(**defaults)  # type: ignore[arg-type]


class TestComputeDirectionStats:
    """Direction statistics computation from round history."""

    def test_counts_visits_and_gains(self) -> None:
        state = _make_state(
            current_round=3,
            rounds=[
                RoundSummary(
                    round_number=0,
                    mode=Mode.EXPLOIT,
                    direction="reduce_memory",
                    num_candidates=3,
                    num_improved=1,
                    improvement_over_prev_best=0.05,
                ),
                RoundSummary(
                    round_number=1,
                    mode=Mode.EXPLOIT,
                    direction="reduce_memory",
                    num_candidates=3,
                    num_improved=1,
                    improvement_over_prev_best=0.03,
                ),
                RoundSummary(
                    round_number=2,
                    mode=Mode.EXPLORE,
                    direction="structural_change",
                    num_candidates=3,
                    num_improved=0,
                    improvement_over_prev_best=None,
                ),
            ],
        )
        stats = compute_direction_stats(state)
        stats_by_dir = {s.direction: s for s in stats}

        assert stats_by_dir["reduce_memory"].visits == 2
        assert stats_by_dir["reduce_memory"].total_perf_gain == 0.08
        assert stats_by_dir["reduce_memory"].avg_perf_gain == 0.04

        assert stats_by_dir["structural_change"].visits == 1
        assert stats_by_dir["structural_change"].total_perf_gain == 0.0
        assert stats_by_dir["structural_change"].avg_perf_gain == 0.0


class TestUCB1NumericScore:
    """Verify UCB1 score computation with known values."""

    def test_known_score_values(self) -> None:
        # SCN-NAV-007-01: A=5 visits/3% gain, B=1 visit/5% gain,
        # C=5 visits/4% gain
        stats = [
            DirectionStats(
                direction="A",
                visits=5,
                total_perf_gain=0.15,
                avg_perf_gain=0.03,
            ),
            DirectionStats(
                direction="B",
                visits=1,
                total_perf_gain=0.05,
                avg_perf_gain=0.05,
            ),
            DirectionStats(
                direction="C",
                visits=5,
                total_perf_gain=0.20,
                avg_perf_gain=0.04,
            ),
        ]
        config = NavigatorConfig(ucb1_c=1.414)

        # With total_rounds=11:
        # UCB1(A) = 0.03 + 1.414*sqrt(ln(11)/5)
        #         = 0.03 + 1.414*sqrt(2.3979/5)
        #         = 0.03 + 1.414*0.6925 = 0.03 + 0.979 = 1.009
        # UCB1(B) = 0.05 + 1.414*sqrt(ln(11)/1)
        #         = 0.05 + 1.414*1.5485 = 0.05 + 2.189 = 2.239
        # UCB1(C) = 0.04 + 1.414*sqrt(ln(11)/5)
        #         = 0.04 + 0.979 = 1.019
        # B wins due to exploration bonus
        result = ucb1_select(stats, 11, set(), config)
        assert result == "B"


class TestUCB1UnvisitedDirection:
    """Unvisited directions get infinite score and are selected first."""

    def test_unvisited_selected_first(self) -> None:
        stats = [
            DirectionStats(
                direction="A",
                visits=3,
                total_perf_gain=0.15,
                avg_perf_gain=0.05,
            ),
            DirectionStats(
                direction="B",
                visits=0,
                total_perf_gain=0.0,
                avg_perf_gain=0.0,
            ),
        ]
        config = NavigatorConfig(ucb1_c=1.414)

        result = ucb1_select(stats, 5, set(), config)
        assert result == "B"


class TestUCB1AllUnvisited:
    """All directions unvisited: deterministic selection (first one)."""

    def test_all_unvisited_selects_first(self) -> None:
        stats = [
            DirectionStats(
                direction="A",
                visits=0,
                total_perf_gain=0.0,
                avg_perf_gain=0.0,
            ),
            DirectionStats(
                direction="B",
                visits=0,
                total_perf_gain=0.0,
                avg_perf_gain=0.0,
            ),
        ]
        config = NavigatorConfig(ucb1_c=1.414)

        result = ucb1_select(stats, 0, set(), config)
        # First unvisited direction is selected deterministically
        assert result == "A"


class TestUCB1AllExhausted:
    """When all directions are exhausted, pick best from exhausted set."""

    def test_all_exhausted_picks_best(self) -> None:
        stats = [
            DirectionStats(
                direction="A",
                visits=3,
                total_perf_gain=0.06,
                avg_perf_gain=0.02,
            ),
            DirectionStats(
                direction="B",
                visits=3,
                total_perf_gain=0.15,
                avg_perf_gain=0.05,
            ),
        ]
        config = NavigatorConfig(ucb1_c=1.414)
        exhausted = {"A", "B"}

        result = ucb1_select(stats, 6, exhausted, config)
        # B has higher avg_perf_gain, so should be selected
        assert result == "B"


class TestUCB1EmptyStats:
    """Edge case: no stats available."""

    def test_empty_stats_returns_default(self) -> None:
        config = NavigatorConfig(ucb1_c=1.414)
        result = ucb1_select([], 0, set(), config)
        assert result == "initial_exploration"
