"""End-to-end integration tests for the CodingAgent class.

Tests the full generate() flow with stub LLM clients.

Spec: docs/coding-agent/spec.md §6.6
"""

from __future__ import annotations

from kerlever.coding_agent import CodingAgent
from kerlever.types import Mode, ProblemSpec, StrategyDirective, SubMode

# Valid CUDA kernels for test stubs — each is slightly different to avoid dedup
VALID_KERNELS = [
    """\
__launch_bounds__(256, 2)
__global__ void matmul_v0(
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
}""",
    """\
__launch_bounds__(128, 4)
__global__ void matmul_v1(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int tid = threadIdx.x;
    int row = blockIdx.y * 128 + tid;
    int col = blockIdx.x;
    if (row < M && col < N) {
        half acc = 0;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}""",
    """\
__launch_bounds__(512, 1)
__global__ void matmul_v2(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if (row < M && col < N) {
        half result = 0;
        for (int k = 0; k < K; k++) {
            result += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = result;
    }
}""",
]


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
        reference_kernel="__global__ void ref() { int x = 1; }",
    )


class IndexedStubLLMClient:
    """Stub LLM that returns different valid kernels per call."""

    def __init__(self, kernels: list[str] | None = None) -> None:
        self._kernels = kernels or VALID_KERNELS
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        idx = self._call_count % len(self._kernels)
        self._call_count += 1
        return f"```cuda\n{self._kernels[idx]}\n```"

    @property
    def call_count(self) -> int:
        return self._call_count


class PartialFailureLLMClient:
    """Stub LLM that fails for specific call indices."""

    def __init__(
        self,
        fail_indices: set[int],
        valid_kernel: str,
    ) -> None:
        self._fail_indices = fail_indices
        self._valid_kernel = valid_kernel
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        idx = self._call_count
        self._call_count += 1
        if idx in self._fail_indices:
            return "I cannot generate this kernel."
        return f"```cuda\n{self._valid_kernel}\n```"


class AllFailureLLMClient:
    """Stub LLM that always fails."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise ConnectionError("LLM unavailable")


class TestCodingAgentDeNovo:
    """SCN-CA-001-01: DE_NOVO generation."""

    async def test_generates_n_candidates(self) -> None:
        """DE_NOVO generates N kernel candidates."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        assert len(candidates) == 3
        for c in candidates:
            assert "__global__" in c.source_code
            assert c.mode == Mode.EXPLORE
            assert c.sub_mode == SubMode.DE_NOVO
            assert c.parent_hash is None

    async def test_all_hashes_distinct(self) -> None:
        """All code_hash values are distinct."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        hashes = [c.code_hash for c in candidates]
        assert len(set(hashes)) == len(hashes)

    async def test_partial_failure_returns_fewer(self) -> None:
        """SCN-CA-001-02: LLM partial failure returns fewer candidates."""
        # Calls 0,1 produce no code (fail both initial + retry);
        # calls 2,3 are retries for 0,1;
        # call 4 produces valid code (candidate 2, first attempt)
        llm = PartialFailureLLMClient(
            fail_indices={0, 1, 2, 3},
            valid_kernel=VALID_KERNELS[0],
        )
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        # At least the third candidate should succeed
        assert len(candidates) >= 1
        assert len(candidates) < 3


class TestCodingAgentExploit:
    """SCN-CA-002-01: EXPLOIT/LOCAL_REWRITE generation."""

    async def test_local_rewrite_with_parent(self) -> None:
        """LOCAL_REWRITE sets parent_hash from directive."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLOIT,
            direction="reduce_register_pressure",
            reason="test",
            base_kernel_hash="parent_hash_abc",
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.LOCAL_REWRITE,
        )

        candidates = await agent.generate(spec, directive, VALID_KERNELS[0])

        assert len(candidates) > 0
        for c in candidates:
            assert c.parent_hash == "parent_hash_abc"
            assert c.mode == Mode.EXPLOIT
            assert c.sub_mode == SubMode.LOCAL_REWRITE

    async def test_exploit_without_best_source_falls_back(self) -> None:
        """SCN-CA-002-02: EXPLOIT without current_best_source falls back to DE_NOVO."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLOIT,
            direction="reduce_register_pressure",
            reason="test",
            base_kernel_hash="parent_hash_abc",
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.LOCAL_REWRITE,
        )

        # current_best_source is None -> falls back to DE_NOVO-style
        candidates = await agent.generate(spec, directive, None)

        assert len(candidates) > 0
        # Mode is still EXPLOIT (reflecting original directive per spec)
        for c in candidates:
            assert c.mode == Mode.EXPLOIT


class TestCodingAgentErrorHandling:
    """Tests for error handling and edge cases."""

    async def test_all_failures_returns_empty_list(self) -> None:
        """SCN-CA-006-05: All candidates fail returns empty list."""
        llm = AllFailureLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        assert candidates == []

    async def test_deduplication(self) -> None:
        """SCN-CA-007-03: Duplicate hash is deduplicated."""
        # All candidates get the same kernel -> should be deduped
        same_kernel = [VALID_KERNELS[0]]
        llm = IndexedStubLLMClient(kernels=same_kernel)
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=3,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        # Should be deduplicated to 1
        assert len(candidates) == 1

    async def test_intent_tag_contains_direction(self) -> None:
        """SCN-CA-007-02: intent_tag references direction."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = _make_problem_spec()
        directive = StrategyDirective(
            mode=Mode.EXPLOIT,
            direction="reduce_register_pressure",
            reason="test",
            base_kernel_hash="parent",
            num_candidates=1,
            tabu=[],
            sub_mode=SubMode.LOCAL_REWRITE,
        )

        candidates = await agent.generate(spec, directive, VALID_KERNELS[0])

        assert len(candidates) == 1
        assert "reduce_register_pressure" in candidates[0].intent_tag

    async def test_unknown_gpu_does_not_crash(self) -> None:
        """REQ-CA-011: Unknown GPU returns valid candidates."""
        llm = IndexedStubLLMClient()
        agent = CodingAgent(llm)
        spec = ProblemSpec(
            op_name="matmul",
            op_semantics="Matrix multiplication",
            shapes=[[1024, 1024]],
            dtype="float16",
            target_gpu="IMAGINARY_GPU_9999",
            baseline_perf_us=100.0,
            target_perf_us=50.0,
            tolerance=0.05,
            max_rounds=10,
            reference_kernel="// ref",
        )
        directive = StrategyDirective(
            mode=Mode.EXPLORE,
            direction="initial_exploration",
            reason="test",
            base_kernel_hash=None,
            num_candidates=1,
            tabu=[],
            sub_mode=SubMode.DE_NOVO,
        )

        candidates = await agent.generate(spec, directive, None)

        assert len(candidates) >= 1


class TestCodingAgentProtocolConformance:
    """Verify CodingAgent satisfies CodingAgentProtocol."""

    def test_protocol_conformance(self) -> None:
        """CodingAgent is a structural subtype of CodingAgentProtocol."""
        from kerlever.protocols import CodingAgentProtocol

        llm = IndexedStubLLMClient()
        agent: CodingAgentProtocol = CodingAgent(llm)

        # Verify the method signature matches
        assert hasattr(agent, "generate")
        assert callable(agent.generate)
