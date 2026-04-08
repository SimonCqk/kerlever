"""Tests for interactive collection mode.

Spec: docs/spec-builder/spec.md §6.5
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from kerlever.spec_builder.interactive import interactive_collect


def _make_extraction_response(
    extracted: dict[str, object],
    follow_up: str = "",
) -> str:
    """Build a JSON extraction response string."""
    return json.dumps({"extracted": extracted, "follow_up": follow_up})


_VALID_JUDGE_RESPONSE = json.dumps(
    [
        {"dimension": "consistency", "severity": "pass", "reason": "ok"},
        {"dimension": "specificity", "severity": "pass", "reason": "ok"},
        {"dimension": "feasibility", "severity": "pass", "reason": "ok"},
        {"dimension": "completeness", "severity": "pass", "reason": "ok"},
        {"dimension": "kernel_quality", "severity": "pass", "reason": "ok"},
    ]
)


class StubLLMClientForInteractive:
    """Stub LLM client that returns different responses based on call count."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return next response from the preconfigured list."""
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


class TestInteractiveFullFlow:
    """SCN-SB-005-01: Interactive collection completes full flow."""

    @pytest.mark.asyncio
    async def test_complete_flow_outputs_yaml(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # LLM responses: first extraction provides all fields, then validation passes
        all_fields = {
            "op_name": "matmul",
            "op_semantics": "C[M,N] = A[M,K] @ B[K,N]",
            "shapes": [[4096, 4096, 4096]],
            "dtype": "float16",
            "target_gpu": "A100",
            "baseline_perf_us": 5.0,
            "target_perf_us": 1.0,
            "tolerance": 0.05,
            "max_rounds": 20,
            "reference_kernel": (
                "__global__ void matmul(const half* A, const half* B, "
                "half* C, int M, int N, int K) { /* naive */ }"
            ),
        }

        extraction_response = _make_extraction_response(
            all_fields,
            follow_up="",
        )

        client = StubLLMClientForInteractive([extraction_response])

        # Mock input to provide one message then EOF
        input_mock = AsyncMock(side_effect=["I want to optimize matmul"])

        with (
            patch("kerlever.spec_builder.interactive.asyncio.to_thread", input_mock),
            pytest.raises(SystemExit) as exc_info,
        ):
            await interactive_collect(client, no_llm=True)

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "op_name" in captured.out
        assert "matmul" in captured.out


class TestInteractivePartialInput:
    """SCN-SB-005-02: Interactive handles partial input with follow-ups."""

    @pytest.mark.asyncio
    async def test_partial_input_asks_followup(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # First response: only partial fields
        partial_response = _make_extraction_response(
            {"op_name": "matmul", "op_semantics": "C[M,N] = A[M,K] @ B[K,N]"},
            follow_up="What are the matrix dimensions and data type?",
        )

        # Second response: remaining fields
        remaining_fields = {
            "shapes": [[4096, 4096, 4096]],
            "dtype": "float16",
            "target_gpu": "A100",
            "baseline_perf_us": 5.0,
            "target_perf_us": 1.0,
            "tolerance": 0.05,
            "max_rounds": 20,
            "reference_kernel": (
                "__global__ void matmul(const half* A, const half* B, "
                "half* C, int M, int N, int K) { /* naive */ }"
            ),
        }
        remaining_response = _make_extraction_response(remaining_fields, follow_up="")

        client = StubLLMClientForInteractive([partial_response, remaining_response])

        call_count = 0

        async def mock_to_thread(fn: object, *args: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I want to optimize matmul C = A @ B"
            elif call_count == 2:
                return (
                    "4096x4096 float16 on A100, "
                    "baseline 5us target 1us, "
                    "tolerance 0.05, 20 rounds"
                )
            raise EOFError

        with (
            patch(
                "kerlever.spec_builder.interactive.asyncio.to_thread",
                mock_to_thread,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await interactive_collect(client, no_llm=True)

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Should have asked the follow-up question
        assert "matrix dimensions" in captured.out
