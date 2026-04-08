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
    parse_llm_decision,
    run_llm_reasoning,
    validate_llm_decision,
)
from kerlever.navigator.types import DerivedSignals, LLMDecision
from kerlever.types import Mode, OptimizationState, ProblemSpec, SubMode


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
        "current_round": 3,
        "rounds": [],
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
        error = validate_llm_decision(decision, ["reduce_memory"], set(), config)
        assert error is not None
        assert "tabu" in error.lower()

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
            tabu_list=["reduce_memory"],
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
        state = _make_state(tabu_list=["blocked_dir"])
        signals = _make_signals()
        config = NavigatorConfig()

        with pytest.raises(RuntimeError, match="failed after 2"):
            await run_llm_reasoning(state, signals, None, client, config)
