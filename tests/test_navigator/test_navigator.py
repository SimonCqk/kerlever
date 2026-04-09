"""Integration tests for the StrategyNavigator class.

Tests the full Phase 1-5 flow with stub LLM client.

Spec: docs/navigator/spec.md §6.6
"""

from __future__ import annotations

import json

import pytest

from kerlever.navigator import StrategyNavigator
from kerlever.navigator.config import NavigatorConfig
from kerlever.types import (
    BaselineArtifact,
    BottleneckAssessment,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    RoundSummary,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    SubMode,
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


class StubLLMClient:
    """Stub LLM client for integration tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class TestColdStart:
    """Round 0 always produces EXPLORE DE_NOVO."""

    @pytest.mark.asyncio
    async def test_cold_start_explore_de_novo(self) -> None:
        navigator = StrategyNavigator()
        state = _make_state(current_round=0)
        spec = _make_problem_spec()

        directive = await navigator.decide(spec, state, None, None)

        assert directive.mode == Mode.EXPLORE
        assert directive.sub_mode == SubMode.DE_NOVO
        assert directive.base_kernel_hash is None
        assert "cold start" in directive.reason.lower()


class TestMultiRoundSequence:
    """Multi-round sequence: cold start -> exploit -> plateau -> explore."""

    @pytest.mark.asyncio
    async def test_cold_start_then_exploit_then_plateau(self) -> None:
        config = NavigatorConfig(
            plateau_threshold=0.02,
            plateau_rounds=3,
        )
        llm_response = json.dumps(
            {
                "mode": "exploit",
                "direction": "reduce_memory_bandwidth",
                "sub_mode": None,
                "reasoning": "Focus on memory optimization",
                "confidence": "high",
            }
        )
        client = StubLLMClient([llm_response] * 10)
        navigator = StrategyNavigator(llm_client=client, config=config)
        spec = _make_problem_spec()

        # Round 0: cold start
        state = _make_state(current_round=0)
        d0 = await navigator.decide(spec, state, None, None)
        assert d0.mode == Mode.EXPLORE
        assert d0.sub_mode == SubMode.DE_NOVO

        # Round 1-3: exploit with small improvements
        incumbent = _make_baseline(kernel_hash="hash_best", score_value=78.5)
        rounds = [
            RoundSummary(
                round_number=0,
                mode=Mode.EXPLORE,
                direction="initial_exploration",
                num_candidates=3,
                num_improved=1,
                best_objective_score=80.0,
                rel_gain_vs_prev_best=0.20,
            ),
            RoundSummary(
                round_number=1,
                mode=Mode.EXPLOIT,
                direction="reduce_memory_bandwidth",
                num_candidates=5,
                num_improved=1,
                best_objective_score=79.0,
                rel_gain_vs_prev_best=0.01,
            ),
            RoundSummary(
                round_number=2,
                mode=Mode.EXPLOIT,
                direction="reduce_memory_bandwidth",
                num_candidates=5,
                num_improved=1,
                best_objective_score=78.5,
                rel_gain_vs_prev_best=0.005,
            ),
            RoundSummary(
                round_number=3,
                mode=Mode.EXPLOIT,
                direction="reduce_memory_bandwidth",
                num_candidates=5,
                num_improved=0,
                best_objective_score=78.5,
                rel_gain_vs_prev_best=0.0,
            ),
        ]
        state_4 = _make_state(
            current_round=4,
            incumbent=incumbent,
            rounds=rounds,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )

        d4 = await navigator.decide(spec, state_4, rounds[-1], None)

        # After 3 consecutive exploit rounds with avg improvement
        # below 2%, plateau should be detected
        assert d4.mode == Mode.EXPLORE
        assert "plateau" in d4.reason.lower()


class TestLLMFailureFallback:
    """LLM failure degrades to UCB1 fallback."""

    @pytest.mark.asyncio
    async def test_llm_failure_uses_ucb1(self) -> None:
        # LLM always returns garbage
        client = StubLLMClient(["not json"] * 5)
        navigator = StrategyNavigator(llm_client=client)
        spec = _make_problem_spec()

        # State with some history (not round 0, no gate matches)
        incumbent = _make_baseline(kernel_hash="hash_0", score_value=80.0)
        rounds = [
            RoundSummary(
                round_number=0,
                mode=Mode.EXPLORE,
                direction="initial_exploration",
                num_candidates=3,
                num_improved=1,
                best_objective_score=80.0,
                rel_gain_vs_prev_best=0.20,
            ),
        ]
        state = _make_state(
            current_round=1,
            incumbent=incumbent,
            rounds=rounds,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )

        directive = await navigator.decide(spec, state, rounds[0], None)

        # Should still return a valid directive (UCB1 fallback)
        assert directive.mode == Mode.EXPLOIT
        assert "ucb1" in directive.reason.lower()


class TestNoLLMClient:
    """No LLM client -> pure deterministic + UCB1."""

    @pytest.mark.asyncio
    async def test_no_llm_uses_ucb1_directly(self) -> None:
        navigator = StrategyNavigator(llm_client=None)
        spec = _make_problem_spec()

        # State with history, no gate matches
        incumbent = _make_baseline(kernel_hash="hash_0", score_value=80.0)
        rounds = [
            RoundSummary(
                round_number=0,
                mode=Mode.EXPLORE,
                direction="initial_exploration",
                num_candidates=3,
                num_improved=1,
                best_objective_score=80.0,
                rel_gain_vs_prev_best=0.20,
            ),
        ]
        state = _make_state(
            current_round=1,
            incumbent=incumbent,
            rounds=rounds,
            bottleneck_history=[
                BottleneckAssessment(
                    tags=["memory_bandwidth"],
                    primary_tag="memory_bandwidth",
                    evidence={},
                    rule_trace=[],
                ),
            ],
        )

        directive = await navigator.decide(spec, state, rounds[0], None)

        # Should use UCB1 (no LLM available)
        assert directive.mode is not None
        assert directive.direction is not None
        assert len(directive.reason) > 0


class TestProtocolConformance:
    """StrategyNavigator satisfies StrategyNavigatorProtocol."""

    def test_has_decide_method(self) -> None:
        navigator = StrategyNavigator()
        assert hasattr(navigator, "decide")
        assert callable(navigator.decide)

    @pytest.mark.asyncio
    async def test_decide_returns_strategy_directive(self) -> None:
        from kerlever.types import StrategyDirective

        navigator = StrategyNavigator()
        spec = _make_problem_spec()
        state = _make_state(current_round=0)

        result = await navigator.decide(spec, state, None, None)

        assert isinstance(result, StrategyDirective)
