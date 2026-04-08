"""Tests for LLM judge.

Spec: docs/spec-builder/spec.md §6.3
"""

from __future__ import annotations

import json

import pytest

from kerlever.spec_builder.llm_judge import run_llm_judge
from kerlever.types import ProblemSpec


def _make_valid_spec() -> ProblemSpec:
    """Create a valid ProblemSpec for testing."""
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C[M,N] = A[M,K] @ B[K,N]",
        shapes=[[4096, 4096, 4096]],
        dtype="float16",
        target_gpu="A100",
        baseline_perf_us=5.0,
        target_perf_us=1.0,
        tolerance=0.05,
        max_rounds=20,
        reference_kernel=(
            "__global__ void matmul(const half* A, const half* B, half* C, "
            "int M, int N, int K) { /* naive impl */ }"
        ),
    )


def _valid_judge_response() -> str:
    """Return a well-formed LLM judge JSON response."""
    data = [
        {"dimension": "consistency", "severity": "pass", "reason": "ok"},
        {"dimension": "specificity", "severity": "pass", "reason": "ok"},
        {"dimension": "feasibility", "severity": "warn", "reason": "aggressive"},
        {"dimension": "completeness", "severity": "pass", "reason": "ok"},
        {"dimension": "kernel_quality", "severity": "pass", "reason": "ok"},
    ]
    return json.dumps(data)


class StubLLMClient:
    """Stub LLM client that returns preconfigured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return next response from the preconfigured list."""
        idx = self._call_count
        self._call_count += 1
        return self._responses[idx]


class FailingLLMClient:
    """Stub LLM client that raises an exception."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Always raise an exception."""
        raise ConnectionError("network error")


class TestLLMJudgeSuccess:
    """SCN-SB-004-01: LLM judge returns well-formed JSON."""

    @pytest.mark.asyncio
    async def test_valid_json_produces_five_issues(self) -> None:
        client = StubLLMClient([_valid_judge_response()])
        spec = _make_valid_spec()
        issues = await run_llm_judge(spec, client)
        assert len(issues) == 5
        dimensions = {i.dimension for i in issues}
        assert dimensions == {
            "consistency",
            "specificity",
            "feasibility",
            "completeness",
            "kernel_quality",
        }

    @pytest.mark.asyncio
    async def test_valid_json_with_markdown_fences(self) -> None:
        response = f"```json\n{_valid_judge_response()}\n```"
        client = StubLLMClient([response])
        spec = _make_valid_spec()
        issues = await run_llm_judge(spec, client)
        assert len(issues) == 5


class TestLLMJudgeRetry:
    """SCN-SB-004-02: LLM judge retries on parse failure then succeeds."""

    @pytest.mark.asyncio
    async def test_retry_on_first_failure(self) -> None:
        client = StubLLMClient(
            [
                "not valid json at all",
                _valid_judge_response(),
            ]
        )
        spec = _make_valid_spec()
        issues = await run_llm_judge(spec, client)
        assert len(issues) == 5
        assert client._call_count == 2


class TestLLMJudgeDegrade:
    """SCN-SB-004-03: LLM judge degrades after two parse failures."""

    @pytest.mark.asyncio
    async def test_two_failures_produce_parse_error(self) -> None:
        client = StubLLMClient(
            [
                "garbage response 1",
                "garbage response 2",
            ]
        )
        spec = _make_valid_spec()
        issues = await run_llm_judge(spec, client)
        assert len(issues) == 1
        assert issues[0].dimension == "parse_error"
        assert issues[0].severity == "fail"

    @pytest.mark.asyncio
    async def test_exception_degrades_to_parse_error(self) -> None:
        client = FailingLLMClient()
        spec = _make_valid_spec()
        issues = await run_llm_judge(spec, client)
        assert len(issues) == 1
        assert issues[0].dimension == "parse_error"
        assert issues[0].severity == "fail"
        assert "network error" in issues[0].message
