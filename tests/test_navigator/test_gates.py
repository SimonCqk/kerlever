"""Tests for Navigator Phase 2 deterministic gate checks.

Spec: docs/navigator/spec.md §6.2
"""

from __future__ import annotations

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.gates import check_gates
from kerlever.navigator.types import DerivedSignals
from kerlever.types import (
    BaselineArtifact,
    BottleneckAssessment,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    SubMode,
)


def _make_problem_spec(**kwargs: object) -> ProblemSpec:
    """Create a ProblemSpec with overrides."""
    defaults: dict[str, object] = {
        "op_name": "matmul",
        "op_semantics": "C = A @ B",
        "dtype": "float32",
        "target_gpu": "A100",
        "shape_cases": [ShapeCase(shape_id="s1", dims=[1024, 1024])],
        "objective": PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
        ),
        "target_metric_value": 10.0,
        "max_rounds": 20,
        "reference_kernel": "__global__ void k() {}",
    }
    defaults.update(kwargs)
    return ProblemSpec(**defaults)  # type: ignore[arg-type]


def _make_baseline(
    kernel_hash: str = "baseline_hash",
    score_value: float = 100.0,
) -> BaselineArtifact:
    """Create a minimal BaselineArtifact."""
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


def _make_signals(**kwargs: object) -> DerivedSignals:
    """Create DerivedSignals with overrides."""
    defaults: dict[str, object] = {
        "avg_delta": 0.0,
        "is_plateau": False,
        "is_regress": False,
        "stable_bottleneck": None,
        "new_bottleneck": None,
        "consecutive_exploit_rounds": 0,
        "direction_attempt_counts": {},
        "exhausted_directions": set(),
    }
    defaults.update(kwargs)
    return DerivedSignals(**defaults)  # type: ignore[arg-type]


class TestGate1ColdStart:
    """Gate 1: round 0 always produces EXPLORE DE_NOVO."""

    def test_round_0_returns_explore_de_novo(self) -> None:
        state = _make_state(current_round=0)
        signals = _make_signals()
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        assert result is not None
        assert result.mode == Mode.EXPLORE
        assert result.sub_mode == SubMode.DE_NOVO
        assert "cold start" in result.reason.lower()


class TestGate2Plateau:
    """Gate 2: plateau forces EXPLORE."""

    def test_plateau_returns_explore(self) -> None:
        state = _make_state(
            current_round=4,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )
        signals = _make_signals(
            is_plateau=True,
            consecutive_exploit_rounds=3,
        )
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        assert result is not None
        assert result.mode == Mode.EXPLORE
        assert "plateau" in result.reason.lower()

    def test_no_plateau_when_flag_false(self) -> None:
        state = _make_state(current_round=4)
        signals = _make_signals(is_plateau=False)
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        # No gate should match (not round 0, not plateau, not near target,
        # no new bottleneck, no exhausted direction)
        assert result is None


class TestGate3NearTarget:
    """Gate 3: near target forces EXPLOIT PARAM_SEARCH."""

    def test_near_target_returns_exploit(self) -> None:
        # target = 10.0, threshold = 0.95
        # Gate fires when incumbent.objective_score.value <= 10.0 / 0.95 = 10.526
        incumbent = _make_baseline(kernel_hash="best_hash", score_value=10.5)
        state = _make_state(
            current_round=5,
            incumbent=incumbent,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["occupancy"],
                    primary_tag="occupancy",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )
        signals = _make_signals()
        spec = _make_problem_spec(target_metric_value=10.0)
        config = NavigatorConfig(target_threshold=0.95)

        result = check_gates(signals, state, spec, config)

        assert result is not None
        assert result.mode == Mode.EXPLOIT
        assert result.sub_mode == SubMode.PARAM_SEARCH
        assert "near target" in result.reason.lower()

    def test_far_from_target_does_not_trigger(self) -> None:
        incumbent = _make_baseline(kernel_hash="best_hash", score_value=50.0)
        state = _make_state(
            current_round=5,
            incumbent=incumbent,
        )
        signals = _make_signals()
        spec = _make_problem_spec(target_metric_value=10.0)
        config = NavigatorConfig(target_threshold=0.95)

        result = check_gates(signals, state, spec, config)

        assert result is None


class TestGate4NewBottleneck:
    """Gate 4: new bottleneck returns None (defer to LLM)."""

    def test_new_bottleneck_returns_none(self) -> None:
        state = _make_state(current_round=3)
        signals = _make_signals(new_bottleneck="tensor_core_not_triggered")
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        assert result is None


class TestGate5ExhaustedDirection:
    """Gate 5: exhausted direction forces EXPLORE."""

    def test_exhausted_returns_explore(self) -> None:
        state = _make_state(current_round=7)
        signals = _make_signals(
            stable_bottleneck="reduce_register_pressure",
            exhausted_directions={"reduce_register_pressure"},
            direction_attempt_counts={"reduce_register_pressure": 3},
        )
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        assert result is not None
        assert result.mode == Mode.EXPLORE
        assert "exhausted" in result.reason.lower()

    def test_stable_but_not_exhausted(self) -> None:
        state = _make_state(current_round=5)
        signals = _make_signals(
            stable_bottleneck="reduce_register_pressure",
            exhausted_directions=set(),
            direction_attempt_counts={"reduce_register_pressure": 2},
        )
        spec = _make_problem_spec()
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        # Not exhausted, no other gate matches
        assert result is None


class TestNoMatch:
    """When no gate matches, result is None (proceed to LLM)."""

    def test_no_match_returns_none(self) -> None:
        incumbent = _make_baseline(kernel_hash="best_hash", score_value=50.0)
        state = _make_state(
            current_round=5,
            incumbent=incumbent,
        )
        signals = _make_signals()
        spec = _make_problem_spec(target_metric_value=10.0)
        config = NavigatorConfig()

        result = check_gates(signals, state, spec, config)

        assert result is None


class TestGatePriority:
    """Gates are evaluated in priority order — near target beats exhausted."""

    def test_near_target_beats_exhausted(self) -> None:
        # Both near target and exhausted conditions are true
        incumbent = _make_baseline(kernel_hash="best_hash", score_value=10.5)
        state = _make_state(
            current_round=7,
            incumbent=incumbent,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["occupancy"],
                    primary_tag="occupancy",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )
        signals = _make_signals(
            stable_bottleneck="reduce_register_pressure",
            exhausted_directions={"reduce_register_pressure"},
            direction_attempt_counts={"reduce_register_pressure": 3},
        )
        spec = _make_problem_spec(target_metric_value=10.0)
        config = NavigatorConfig(target_threshold=0.95)

        result = check_gates(signals, state, spec, config)

        # Near target (Gate 3) should win over exhausted (Gate 5)
        assert result is not None
        assert result.mode == Mode.EXPLOIT
        assert "near target" in result.reason.lower()

    def test_plateau_beats_near_target(self) -> None:
        # Both plateau and near target conditions are true
        incumbent = _make_baseline(kernel_hash="best_hash", score_value=10.5)
        state = _make_state(
            current_round=5,
            incumbent=incumbent,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )
        signals = _make_signals(
            is_plateau=True,
            consecutive_exploit_rounds=3,
        )
        spec = _make_problem_spec(target_metric_value=10.0)
        config = NavigatorConfig(target_threshold=0.95)

        result = check_gates(signals, state, spec, config)

        # Plateau (Gate 2) has higher priority than near target (Gate 3)
        assert result is not None
        assert result.mode == Mode.EXPLORE
        assert "plateau" in result.reason.lower()
