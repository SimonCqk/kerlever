"""Tests for Navigator Phase 1 signal computation.

Spec: docs/navigator/spec.md §6.1
"""

from __future__ import annotations

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.signals import compute_derived_signals
from kerlever.types import (
    AttemptRecord,
    BaselineArtifact,
    BottleneckAssessment,
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
    """Create a minimal ProblemSpec for testing."""
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        dtype="float32",
        target_gpu="A100",
        shape_cases=[
            ShapeCase(shape_id="s1", dims=[1024, 1024]),
        ],
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
    """Create a minimal BaselineArtifact for testing."""
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
    """Create an OptimizationState with overrides."""
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


def _make_round(
    round_number: int,
    mode: Mode,
    direction: str,
    rel_gain: float | None = None,
) -> RoundSummary:
    """Create a RoundSummary."""
    return RoundSummary(
        round_number=round_number,
        mode=mode,
        direction=direction,
        num_candidates=3,
        num_improved=1,
        best_objective_score=50.0,
        rel_gain_vs_prev_best=rel_gain,
    )


def _make_attempt(
    round_number: int,
    direction: str,
    outcome: CandidateOutcome = CandidateOutcome.IMPROVED,
) -> AttemptRecord:
    """Create an AttemptRecord."""
    return AttemptRecord(
        round_number=round_number,
        candidate_hash=f"hash_{round_number}",
        base_kernel_hash="baseline_hash",
        direction=direction,
        sub_mode=None,
        outcome=outcome,
    )


def _make_assessment(
    primary_tag: str | None,
    tags: list[str] | None = None,
) -> BottleneckAssessment:
    """Create a BottleneckAssessment."""
    return BottleneckAssessment(
        tags=tags if tags is not None else ([primary_tag] if primary_tag else []),
        primary_tag=primary_tag,
        evidence={},
        rule_trace=[],
    )


class TestRound0Defaults:
    """Round 0: all signals should have safe default values."""

    def test_avg_delta_zero_on_empty_rounds(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.avg_delta == 0.0

    def test_is_plateau_false_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.is_plateau is False

    def test_is_regress_false_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.is_regress is False

    def test_stable_bottleneck_none_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.stable_bottleneck is None

    def test_new_bottleneck_none_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.new_bottleneck is None

    def test_consecutive_exploit_zero_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.consecutive_exploit_rounds == 0

    def test_direction_counts_empty_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.direction_attempt_counts == {}

    def test_exhausted_empty_on_round_0(self) -> None:
        state = _make_state(current_round=0)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.exhausted_directions == set()


class TestPlateau:
    """Plateau detection: 3 consecutive exploit rounds with small improvement."""

    def test_three_exploit_rounds_below_threshold(self) -> None:
        rounds = [
            _make_round(0, Mode.EXPLORE, "initial_exploration", 5.0),
            _make_round(1, Mode.EXPLOIT, "reduce_memory", 0.01),
            _make_round(2, Mode.EXPLOIT, "reduce_memory", 0.005),
            _make_round(3, Mode.EXPLOIT, "reduce_memory", 0.003),
        ]
        state = _make_state(current_round=4, rounds=rounds)
        config = NavigatorConfig(plateau_threshold=0.02, plateau_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.is_plateau is True

    def test_no_plateau_with_explore_interspersed(self) -> None:
        rounds = [
            _make_round(0, Mode.EXPLOIT, "reduce_memory", 0.01),
            _make_round(1, Mode.EXPLORE, "structural_change", 0.005),
            _make_round(2, Mode.EXPLOIT, "reduce_memory", 0.003),
        ]
        state = _make_state(current_round=3, rounds=rounds)
        config = NavigatorConfig(plateau_threshold=0.02, plateau_rounds=3)
        signals = compute_derived_signals(state, config)
        # consecutive_exploit_rounds is only 1 (round 2),
        # so plateau condition not met
        assert signals.is_plateau is False

    def test_no_plateau_above_threshold(self) -> None:
        rounds = [
            _make_round(0, Mode.EXPLOIT, "reduce_memory", 0.05),
            _make_round(1, Mode.EXPLOIT, "reduce_memory", 0.04),
            _make_round(2, Mode.EXPLOIT, "reduce_memory", 0.03),
        ]
        state = _make_state(current_round=3, rounds=rounds)
        config = NavigatorConfig(plateau_threshold=0.02, plateau_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.is_plateau is False


class TestStableBottleneck:
    """Stable bottleneck: same primary_tag for K consecutive assessments."""

    def test_three_rounds_same_bottleneck(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment("memory_bandwidth"),
            _make_assessment("memory_bandwidth"),
        ]
        state = _make_state(
            current_round=3,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "memory_bandwidth") for i in range(3)],
        )
        config = NavigatorConfig(stable_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.stable_bottleneck == "memory_bandwidth"

    def test_different_bottlenecks_no_stable(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment("occupancy"),
            _make_assessment("memory_bandwidth"),
        ]
        state = _make_state(
            current_round=3,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "reduce_memory") for i in range(3)],
        )
        config = NavigatorConfig(stable_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.stable_bottleneck is None

    def test_fewer_than_k_rounds_no_stable(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment("memory_bandwidth"),
        ]
        state = _make_state(
            current_round=2,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "reduce_memory") for i in range(2)],
        )
        config = NavigatorConfig(stable_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.stable_bottleneck is None

    def test_none_primary_tag_breaks_streak(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment(None),
            _make_assessment("memory_bandwidth"),
        ]
        state = _make_state(
            current_round=3,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "reduce_memory") for i in range(3)],
        )
        config = NavigatorConfig(stable_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.stable_bottleneck is None


class TestNewBottleneck:
    """New bottleneck: a primary_tag in the latest assessment never seen before."""

    def test_new_tag_detected(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment("tensor_core_not_triggered"),
        ]
        state = _make_state(
            current_round=2,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "reduce_memory") for i in range(2)],
        )
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.new_bottleneck == "tensor_core_not_triggered"

    def test_no_new_tag(self) -> None:
        history = [
            _make_assessment("memory_bandwidth"),
            _make_assessment("memory_bandwidth"),
        ]
        state = _make_state(
            current_round=2,
            bottleneck_history=history,
            rounds=[_make_round(i, Mode.EXPLOIT, "reduce_memory") for i in range(2)],
        )
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.new_bottleneck is None


class TestDirectionAttemptCounts:
    """Direction attempt counting from AttemptRecord entries."""

    def test_direction_counts(self) -> None:
        attempts = [
            _make_attempt(0, "reduce_memory"),
            _make_attempt(1, "reduce_memory"),
            _make_attempt(2, "structural_change"),
            _make_attempt(3, "reduce_memory"),
        ]
        state = _make_state(current_round=4, attempts=attempts)
        signals = compute_derived_signals(state, NavigatorConfig())
        assert signals.direction_attempt_counts == {
            "reduce_memory": 3,
            "structural_change": 1,
        }


class TestExhaustedDirections:
    """Exhausted direction: stable bottleneck + M attempts on same direction."""

    def test_exhausted_when_stable_and_max_attempts(self) -> None:
        history = [
            _make_assessment("reduce_memory"),
            _make_assessment("reduce_memory"),
            _make_assessment("reduce_memory"),
        ]
        attempts = [
            _make_attempt(0, "reduce_memory"),
            _make_attempt(1, "reduce_memory"),
            _make_attempt(2, "reduce_memory"),
        ]
        state = _make_state(
            current_round=3,
            attempts=attempts,
            bottleneck_history=history,
        )
        config = NavigatorConfig(
            stable_rounds=3,
            max_direction_attempts=3,
        )
        signals = compute_derived_signals(state, config)
        assert "reduce_memory" in signals.exhausted_directions

    def test_not_exhausted_with_fewer_attempts(self) -> None:
        history = [
            _make_assessment("reduce_memory"),
            _make_assessment("reduce_memory"),
            _make_assessment("reduce_memory"),
        ]
        attempts = [
            _make_attempt(0, "reduce_memory"),
            _make_attempt(1, "reduce_memory"),
        ]
        state = _make_state(
            current_round=2,
            attempts=attempts,
            bottleneck_history=history,
        )
        config = NavigatorConfig(
            stable_rounds=3,
            max_direction_attempts=3,
        )
        signals = compute_derived_signals(state, config)
        assert signals.exhausted_directions == set()


class TestNoneImprovementTreatedAsZero:
    """None rel_gain_vs_prev_best values should be treated as 0.0."""

    def test_none_improvement_treated_as_zero(self) -> None:
        rounds = [
            _make_round(0, Mode.EXPLOIT, "reduce_memory", None),
            _make_round(1, Mode.EXPLOIT, "reduce_memory", None),
            _make_round(2, Mode.EXPLOIT, "reduce_memory", None),
        ]
        state = _make_state(current_round=3, rounds=rounds)
        config = NavigatorConfig(plateau_threshold=0.02, plateau_rounds=3)
        signals = compute_derived_signals(state, config)
        assert signals.avg_delta == 0.0
        assert signals.is_plateau is True
