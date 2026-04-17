"""Tests for Navigator Phase 3 LLM reasoning.

Uses stub LLM client for testing. Covers parse, validate, retry,
and double-failure scenarios.

Spec: docs/navigator/spec.md §6.3
"""

from __future__ import annotations

import json

import pytest

from kerlever.navigator.config import NavigatorConfig
from kerlever.navigator.llm_reasoning import (
    assemble_llm_context,
    parse_llm_decision,
    run_llm_reasoning,
    validate_llm_decision,
)
from kerlever.navigator.types import DerivedSignals, LLMDecision
from kerlever.types import (
    AvoidPattern,
    BaselineArtifact,
    CrossCandidateAnalysis,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    RecombinationHint,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    SubMode,
    TabuEntry,
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
        "current_round": 3,
        "rounds": [],
        "attempts": [],
        "tabu_entries": [],
        "bottleneck_history": [],
    }
    defaults.update(kwargs)
    return OptimizationState(**defaults)  # type: ignore[arg-type]


def _make_signals(**kwargs: object) -> DerivedSignals:
    defaults: dict[str, object] = {
        "avg_delta": 0.01,
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


def _valid_llm_response(
    mode: str = "exploit",
    direction: str = "reduce_memory_bandwidth",
    sub_mode: str | None = None,
    reasoning: str = "Test reasoning",
    confidence: str = "high",
) -> str:
    """Build a valid LLM JSON response string."""
    data = {
        "mode": mode,
        "direction": direction,
        "sub_mode": sub_mode,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    return json.dumps(data)


class StubLLMClient:
    """Stub LLM client that returns preconfigured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class TestParseLLMDecision:
    """Parsing valid and invalid LLM responses."""

    def test_valid_response_produces_decision(self) -> None:
        raw = _valid_llm_response()
        decision = parse_llm_decision(raw)

        assert decision.mode == Mode.EXPLOIT
        assert decision.direction == "reduce_memory_bandwidth"
        assert decision.confidence == "high"
        assert decision.reasoning == "Test reasoning"

    def test_valid_with_sub_mode(self) -> None:
        raw = _valid_llm_response(mode="explore", sub_mode="de_novo")
        decision = parse_llm_decision(raw)

        assert decision.mode == Mode.EXPLORE
        assert decision.sub_mode == SubMode.DE_NOVO

    def test_markdown_fenced_response(self) -> None:
        inner = _valid_llm_response()
        raw = f"```json\n{inner}\n```"
        decision = parse_llm_decision(raw)

        assert decision.mode == Mode.EXPLOIT

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_llm_decision("not json at all")

    def test_missing_fields_raises(self) -> None:
        raw = json.dumps({"mode": "exploit"})
        with pytest.raises(ValueError, match="Missing required"):
            parse_llm_decision(raw)

    def test_invalid_mode_raises(self) -> None:
        raw = json.dumps(
            {
                "mode": "invalid",
                "direction": "x",
                "reasoning": "r",
                "confidence": "high",
            }
        )
        with pytest.raises(ValueError, match="Invalid mode"):
            parse_llm_decision(raw)

    def test_invalid_confidence_raises(self) -> None:
        raw = json.dumps(
            {
                "mode": "exploit",
                "direction": "x",
                "reasoning": "r",
                "confidence": "very_high",
            }
        )
        with pytest.raises(ValueError, match="Invalid confidence"):
            parse_llm_decision(raw)


class TestValidateLLMDecision:
    """Validation of parsed decisions against constraints."""

    def test_valid_decision_returns_none(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory",
            reasoning="test",
            confidence="high",
        )
        config = NavigatorConfig(llm_confidence_min="medium")
        error = validate_llm_decision(decision, [], set(), config)
        assert error is None

    def test_tabu_direction_fails(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory",
            reasoning="test",
            confidence="high",
        )
        config = NavigatorConfig()
        tabu_entries = [
            TabuEntry(
                base_kernel_hash="baseline_hash",
                direction="reduce_memory",
                sub_mode=None,
                round_number=1,
                expires_after_round=10,
            ),
        ]
        error = validate_llm_decision(
            decision,
            tabu_entries,
            set(),
            config,
            current_round=3,
            base_kernel_hash="baseline_hash",
        )
        assert error is not None
        assert "tabu" in error.lower()

    def test_tabu_direction_different_base_hash_allowed(self) -> None:
        """Same direction on different base kernel should not be blocked."""
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory",
            reasoning="test",
            confidence="high",
        )
        config = NavigatorConfig()
        tabu_entries = [
            TabuEntry(
                base_kernel_hash="hash_A",
                direction="reduce_memory",
                sub_mode=None,
                round_number=1,
                expires_after_round=10,
            ),
        ]
        error = validate_llm_decision(
            decision,
            tabu_entries,
            set(),
            config,
            current_round=3,
            base_kernel_hash="hash_B",
        )
        assert error is None

    def test_exhausted_direction_fails(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory",
            reasoning="test",
            confidence="high",
        )
        config = NavigatorConfig()
        error = validate_llm_decision(decision, [], {"reduce_memory"}, config)
        assert error is not None
        assert "exhausted" in error.lower()

    def test_low_confidence_fails(self) -> None:
        decision = LLMDecision(
            mode=Mode.EXPLOIT,
            direction="reduce_memory",
            reasoning="test",
            confidence="low",
        )
        config = NavigatorConfig(llm_confidence_min="medium")
        error = validate_llm_decision(decision, [], set(), config)
        assert error is not None
        assert "confidence" in error.lower()


class TestRunLLMReasoningSuccess:
    """LLM returns valid decision on first attempt."""

    @pytest.mark.asyncio
    async def test_valid_response_succeeds(self) -> None:
        client = StubLLMClient([_valid_llm_response()])
        state = _make_state()
        signals = _make_signals()
        config = NavigatorConfig()

        decision = await run_llm_reasoning(state, signals, None, client, config)

        assert decision.mode == Mode.EXPLOIT
        assert decision.direction == "reduce_memory_bandwidth"


class TestRunLLMReasoningRetry:
    """LLM fails first attempt, succeeds on retry."""

    @pytest.mark.asyncio
    async def test_malformed_then_valid(self) -> None:
        client = StubLLMClient(
            [
                "not json",
                _valid_llm_response(),
            ]
        )
        state = _make_state()
        signals = _make_signals()
        config = NavigatorConfig()

        decision = await run_llm_reasoning(state, signals, None, client, config)

        assert decision.mode == Mode.EXPLOIT

    @pytest.mark.asyncio
    async def test_tabu_direction_then_valid(self) -> None:
        tabu_response = _valid_llm_response(direction="reduce_memory")
        valid_response = _valid_llm_response(direction="optimize_shared_memory")
        client = StubLLMClient([tabu_response, valid_response])
        state = _make_state(
            tabu_entries=[
                TabuEntry(
                    base_kernel_hash="baseline_hash",
                    direction="reduce_memory",
                    sub_mode=None,
                    round_number=1,
                    expires_after_round=10,
                ),
            ],
        )
        signals = _make_signals()
        config = NavigatorConfig()

        decision = await run_llm_reasoning(state, signals, None, client, config)

        assert decision.direction == "optimize_shared_memory"

    @pytest.mark.asyncio
    async def test_low_confidence_then_valid(self) -> None:
        low_conf = _valid_llm_response(confidence="low")
        high_conf = _valid_llm_response(confidence="high")
        client = StubLLMClient([low_conf, high_conf])
        state = _make_state()
        signals = _make_signals()
        config = NavigatorConfig(llm_confidence_min="medium")

        decision = await run_llm_reasoning(state, signals, None, client, config)

        assert decision.confidence == "high"


class TestRunLLMReasoningDoubleFailure:
    """LLM fails both attempts -> raises RuntimeError."""

    @pytest.mark.asyncio
    async def test_double_malformed_raises(self) -> None:
        client = StubLLMClient(["not json", "still not json"])
        state = _make_state()
        signals = _make_signals()
        config = NavigatorConfig()

        with pytest.raises(RuntimeError, match="failed after 2"):
            await run_llm_reasoning(state, signals, None, client, config)

    @pytest.mark.asyncio
    async def test_double_tabu_raises(self) -> None:
        tabu_resp = _valid_llm_response(direction="blocked_dir")
        client = StubLLMClient([tabu_resp, tabu_resp])
        state = _make_state(
            tabu_entries=[
                TabuEntry(
                    base_kernel_hash="baseline_hash",
                    direction="blocked_dir",
                    sub_mode=None,
                    round_number=1,
                    expires_after_round=10,
                ),
            ],
        )
        signals = _make_signals()
        config = NavigatorConfig()

        with pytest.raises(RuntimeError, match="failed after 2"):
            await run_llm_reasoning(state, signals, None, client, config)


class TestAssembleLLMContext:
    """Cross-analysis context assembly."""

    def test_includes_structured_hints_and_avoid_patterns(self) -> None:
        """Structured cross-analysis fields appear in LLM context."""
        cross_analysis = CrossCandidateAnalysis(
            insights=["legacy insight"],
            winning_genes=[],
            recombination_suggestions=[],
            recombination_hints=[
                RecombinationHint(
                    hint_id="hint_1",
                    parent_candidates=["hash_A", "hash_B"],
                    gene_map={
                        "memory_access": "hash_A",
                        "compute_loop": "hash_B",
                    },
                    expected_benefit="Combine measured strengths.",
                    evidence_candidate_hashes=["hash_A", "hash_B"],
                    risk_flags=["shape_specific"],
                    confidence="medium",
                )
            ],
            avoid_patterns=[
                AvoidPattern(
                    pattern_id="avoid_1",
                    source_candidate_hash="hash_bad",
                    pattern="extra_sync",
                    reason="extra sync regressed",
                    evidence={"relative_to_incumbent": 1.2},
                    affected_shape_ids=["s1"],
                    confidence="medium",
                )
            ],
        )

        context = assemble_llm_context(
            _make_state(),
            _make_signals(),
            cross_analysis,
            NavigatorConfig(),
        )

        assert "Structured recombination hints" in context
        assert "hash_A" in context
        assert "memory_access" in context
        assert "Structured avoid patterns" in context
        assert "extra_sync" in context
        assert "relative_to_incumbent=1.2000" in context
