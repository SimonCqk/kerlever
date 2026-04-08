"""Tests for the LLM code generation, parsing, and retry logic.

Uses a stub LLM client to test code extraction, validation, retry,
and candidate assembly.

Spec: docs/coding-agent/spec.md §6.5
"""

from __future__ import annotations

import pytest

from kerlever.coding_agent.config import CodingAgentConfig
from kerlever.coding_agent.generator import (
    build_intent_tag,
    compute_code_hash,
    generate_one_candidate,
    parse_cuda_from_response,
)
from kerlever.coding_agent.hardware import get_gpu_spec
from kerlever.coding_agent.playbook import get_relevant_playbook
from kerlever.types import Mode, ProblemSpec, StrategyDirective, SubMode

# Valid CUDA kernel that passes all validation checks
VALID_CUDA = """\
__launch_bounds__(256, 2)
__global__ void matmul_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        half sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""


def _make_problem_spec() -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="Matrix multiplication C = A @ B",
        shapes=[[1024, 1024], [1024, 1024]],
        dtype="float16",
        target_gpu="A100",
        baseline_perf_us=100.0,
        target_perf_us=50.0,
        tolerance=0.05,
        max_rounds=10,
        reference_kernel="// ref",
    )


def _make_directive(
    mode: Mode = Mode.EXPLOIT,
    sub_mode: SubMode = SubMode.LOCAL_REWRITE,
    direction: str = "reduce_register_pressure",
) -> StrategyDirective:
    return StrategyDirective(
        mode=mode,
        direction=direction,
        reason="test",
        base_kernel_hash="parent123",
        num_candidates=3,
        tabu=[],
        sub_mode=sub_mode,
    )


class StubLLMClient:
    """Test stub that returns configurable responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


class FailingLLMClient:
    """Test stub that always raises an exception."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ConnectionError("LLM service unavailable")


class TestParseCudaFromResponse:
    """Tests for CUDA code extraction from LLM responses."""

    def test_parse_cuda_block(self) -> None:
        """Extract code from ```cuda block."""
        response = f"Here is the kernel:\n```cuda\n{VALID_CUDA}\n```"
        code = parse_cuda_from_response(response)
        assert code is not None
        assert "__global__" in code

    def test_parse_generic_code_block(self) -> None:
        """Extract code from generic ``` block when no ```cuda found."""
        response = f"Here is the kernel:\n```\n{VALID_CUDA}\n```"
        code = parse_cuda_from_response(response)
        assert code is not None
        assert "__global__" in code

    def test_parse_c_code_block(self) -> None:
        """Extract code from ```c block."""
        response = f"Here is the kernel:\n```c\n{VALID_CUDA}\n```"
        code = parse_cuda_from_response(response)
        assert code is not None
        assert "__global__" in code

    def test_parse_raw_global_function(self) -> None:
        """SCN-CA-006-01: Extract __global__ from raw text."""
        response = f"The kernel is:\n{VALID_CUDA}\nThat's it."
        code = parse_cuda_from_response(response)
        assert code is not None
        assert "__global__" in code

    def test_parse_no_code_returns_none(self) -> None:
        """No extractable code returns None."""
        response = "I cannot generate a kernel for this task."
        code = parse_cuda_from_response(response)
        assert code is None

    def test_parse_empty_response(self) -> None:
        """Empty response returns None."""
        code = parse_cuda_from_response("")
        assert code is None


class TestComputeCodeHash:
    """Tests for code hash computation."""

    def test_deterministic(self) -> None:
        """SCN-CA-007-01: Same source produces same hash."""
        hash1 = compute_code_hash(VALID_CUDA)
        hash2 = compute_code_hash(VALID_CUDA)
        assert hash1 == hash2

    def test_length_16(self) -> None:
        """Hash is 16 hex characters."""
        h = compute_code_hash(VALID_CUDA)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_code_different_hash(self) -> None:
        """Different code produces different hash."""
        h1 = compute_code_hash(VALID_CUDA)
        h2 = compute_code_hash(VALID_CUDA + "\n// modified")
        assert h1 != h2


class TestBuildIntentTag:
    """Tests for intent tag construction."""

    def test_format(self) -> None:
        """SCN-CA-007-02: Intent tag has expected format."""
        directive = _make_directive(
            sub_mode=SubMode.LOCAL_REWRITE,
            direction="reduce_register_pressure",
        )
        tag = build_intent_tag(directive, 0, SubMode.LOCAL_REWRITE)
        assert tag == "local_rewrite_reduce_register_pressure_0"

    def test_de_novo_format(self) -> None:
        """DE_NOVO intent tag format."""
        directive = _make_directive(
            mode=Mode.EXPLORE,
            sub_mode=SubMode.DE_NOVO,
            direction="initial_exploration",
        )
        tag = build_intent_tag(directive, 2, SubMode.DE_NOVO)
        assert tag == "de_novo_initial_exploration_2"


class TestGenerateOneCandidate:
    """Tests for per-candidate generation flow."""

    @pytest.fixture()
    def config(self) -> CodingAgentConfig:
        return CodingAgentConfig()

    @pytest.fixture()
    def problem_spec(self) -> ProblemSpec:
        return _make_problem_spec()

    @pytest.fixture()
    def directive(self) -> StrategyDirective:
        return _make_directive()

    async def test_successful_generation(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
    ) -> None:
        """Valid LLM response produces a KernelCandidate."""
        llm = StubLLMClient([f"```cuda\n{VALID_CUDA}\n```"])
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_register_pressure", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source="// existing",
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.LOCAL_REWRITE,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        assert result is not None
        assert "__global__" in result.source_code
        assert result.mode == Mode.EXPLOIT
        assert result.sub_mode == SubMode.LOCAL_REWRITE
        assert result.parent_hash == "parent123"
        assert len(result.code_hash) == 16

    async def test_retry_on_invalid_code(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
    ) -> None:
        """SCN-CA-006-02: First failure triggers retry."""
        # First response has no __global__, second has valid code
        llm = StubLLMClient(
            [
                "```cuda\nvoid bad_kernel() { }\n```",
                f"```cuda\n{VALID_CUDA}\n```",
            ]
        )
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_register_pressure", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source="// existing",
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.LOCAL_REWRITE,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        assert result is not None
        assert llm.call_count == 2  # Retried once

    async def test_skip_after_two_failures(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
    ) -> None:
        """SCN-CA-006-03: Second failure skips candidate."""
        llm = StubLLMClient(
            [
                "```cuda\nvoid bad() { }\n```",
                "```cuda\nvoid also_bad() { }\n```",
            ]
        )
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_register_pressure", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source="// existing",
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.LOCAL_REWRITE,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        assert result is None
        assert llm.call_count == 2

    async def test_llm_exception_triggers_retry(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
    ) -> None:
        """SCN-CA-006-04: LLM exception triggers retry."""
        llm = FailingLLMClient()
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_register_pressure", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source="// existing",
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.LOCAL_REWRITE,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        # Both attempts raise exception -> skip
        assert result is None

    async def test_no_code_block_triggers_retry(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
    ) -> None:
        """SCN-CA-006-01: No code block triggers retry."""
        llm = StubLLMClient(
            [
                "I cannot generate this kernel.",
                f"```cuda\n{VALID_CUDA}\n```",
            ]
        )
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_register_pressure", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source="// existing",
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.LOCAL_REWRITE,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        assert result is not None
        assert llm.call_count == 2

    async def test_de_novo_parent_hash_is_none(
        self,
        config: CodingAgentConfig,
        problem_spec: ProblemSpec,
    ) -> None:
        """DE_NOVO candidates have parent_hash = None."""
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.DE_NOVO)
        llm = StubLLMClient([f"```cuda\n{VALID_CUDA}\n```"])
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("initial_exploration", gpu, "matmul")

        result = await generate_one_candidate(
            llm_client=llm,
            system_prompt="test system",
            problem_spec=problem_spec,
            directive=directive,
            current_best_source=None,
            candidate_index=0,
            config=config,
            effective_sub_mode=SubMode.DE_NOVO,
            gpu_spec=gpu,
            playbook_layers=layers,
        )

        assert result is not None
        assert result.parent_hash is None
