"""Tests for Navigator UCB1 fallback direction selection.

Spec: docs/navigator/spec.md §6.4
"""

from __future__ import annotations

import pytest

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.types import DirectionStats
from kerlever.navigator.ucb1 import compute_direction_stats, ucb1_select
from kerlever.types import (
    AttemptRecord,
    BaselineArtifact,
    CandidateOutcome,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    RoundSummary,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
)


def _make_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        dtype="float32",
        target_gpu="A100",
        shape_cases=[ShapeCase(shape_id="s1", dims=[1024, 1024])],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
        ),
        target_metric_value=10.0,
        max_rounds=20,
        reference_kernel="__global__ void k() {}",
    )


def _make_baseline(
    kernel_hash: str = "baseline_hash",
    score_value: float = 100.0,
) -> BaselineArtifact:
    return BaselineArtifact(
        kernel_hash=kernel_hash,
        source_code="__global__ void k() {}",
        compile_artifact=StaticAnalysis(),
        benchmark_results=[
            ShapeBenchResult(
                shape_id="s1",
                latency_p50_us=score_value,
                latency_p95_us=score_value * 1.1,
                run_count=10,
            ),
        ],
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=score_value,
            relative_to_baseline=1.0,
            relative_to_incumbent=1.0,
        ),
    )


def _make_state(**kwargs: object) -> OptimizationState:
    ps = _make_problem_spec()
    bl = _make_baseline()
    defaults: dict[str, object] = {
        "problem_spec": ps,
        "baseline": bl,
        "incumbent": bl,
        "current_round": 0,
        "rounds": [],
        "attempts": [],
        "tabu_entries": [],
        "bottleneck_history": [],
    }
    defaults.update(kwargs)
    return OptimizationState(**defaults)  # type: ignore[arg-type]


class TestComputeDirectionStats:
    """Direction statistics computation from attempt records."""

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
                    rel_gain_vs_prev_best=0.05,
                ),
                RoundSummary(
                    round_number=1,
                    mode=Mode.EXPLOIT,
                    direction="reduce_memory",
                    num_candidates=3,
                    num_improved=1,
                    rel_gain_vs_prev_best=0.03,
                ),
                RoundSummary(
                    round_number=2,
                    mode=Mode.EXPLORE,
                    direction="structural_change",
                    num_candidates=3,
                    num_improved=0,
                    rel_gain_vs_prev_best=None,
                ),
            ],
            attempts=[
                AttemptRecord(
                    round_number=0,
                    candidate_hash="h0",
                    base_kernel_hash="baseline_hash",
                    direction="reduce_memory",
                    sub_mode=None,
                    outcome=CandidateOutcome.IMPROVED,
                ),
                AttemptRecord(
                    round_number=1,
                    candidate_hash="h1",
                    base_kernel_hash="baseline_hash",
                    direction="reduce_memory",
                    sub_mode=None,
                    outcome=CandidateOutcome.IMPROVED,
                ),
                AttemptRecord(
                    round_number=2,
                    candidate_hash="h2",
                    base_kernel_hash=None,
                    direction="structural_change",
                    sub_mode=None,
                    outcome=CandidateOutcome.REGRESSION,
                ),
            ],
        )
        stats = compute_direction_stats(state)
        stats_by_dir = {s.direction: s for s in stats}

        assert stats_by_dir["reduce_memory"].visits == 2
        assert stats_by_dir["reduce_memory"].total_perf_gain == pytest.approx(0.08)
        assert stats_by_dir["reduce_memory"].avg_perf_gain == pytest.approx(0.04)

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
