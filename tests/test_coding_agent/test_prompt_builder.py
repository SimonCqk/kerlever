"""Tests for prompt construction — system and user prompts.

Spec: docs/coding-agent/spec.md §6.3
"""

from __future__ import annotations

from kerlever.coding_agent.hardware import get_gpu_spec
from kerlever.coding_agent.playbook import LAYER_1, get_relevant_playbook
from kerlever.coding_agent.prompt_builder import (
    build_retry_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from kerlever.types import (
    Mode,
    PerformanceObjective,
    ProblemSpec,
    ShapeCase,
    StrategyDirective,
    SubMode,
    TabuEntry,
)


def _make_problem_spec() -> ProblemSpec:
    """Create a test ProblemSpec."""
    return ProblemSpec(
        op_name="matmul",
        op_semantics="Matrix multiplication C = A @ B",
        shape_cases=[
            ShapeCase(shape_id="s0", dims=[1024, 1024]),
            ShapeCase(shape_id="s1", dims=[1024, 1024]),
        ],
        dtype="float16",
        target_gpu="A100",
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
        ),
        target_metric_value=50.0,
        max_rounds=10,
        reference_kernel="__global__ void ref_matmul() { /* ref */ }",
    )


def _make_directive(
    mode: Mode = Mode.EXPLOIT,
    direction: str = "reduce_register_pressure",
    sub_mode: SubMode | None = SubMode.LOCAL_REWRITE,
    num_candidates: int = 3,
) -> StrategyDirective:
    """Create a test StrategyDirective."""
    return StrategyDirective(
        mode=mode,
        direction=direction,
        reason="test",
        base_kernel_hash="abc123",
        num_candidates=num_candidates,
        tabu=[
            TabuEntry(
                base_kernel_hash=None,
                direction="old_approach_1",
                sub_mode=None,
                round_number=0,
                expires_after_round=5,
            ),
        ],
        sub_mode=sub_mode,
        search_range={"block_size": [128.0, 256.0, 512.0]},
        hard_constraints=["max_smem_48KB"],
        parent_candidates=["hash_A", "hash_B"],
        gene_map={"memory_access": "hash_A", "compute_loop": "hash_B"},
    )


class TestBuildSystemPrompt:
    """Tests for system prompt construction."""

    def test_includes_role_declaration(self) -> None:
        """System prompt includes CUDA expert role."""
        gpu = get_gpu_spec("A100")
        layers = [LAYER_1]
        prompt = build_system_prompt(gpu, layers)
        assert "CUDA kernel optimization expert" in prompt

    def test_includes_code_standards(self) -> None:
        """System prompt includes mandatory code standards."""
        gpu = get_gpu_spec("A100")
        layers = [LAYER_1]
        prompt = build_system_prompt(gpu, layers)
        assert "__launch_bounds__" in prompt
        assert "__restrict__" in prompt
        assert "multiple of 32" in prompt
        assert "bounds-check" in prompt
        assert "ceiling division" in prompt

    def test_includes_gpu_constraints(self) -> None:
        """System prompt includes GPU constraint info."""
        gpu = get_gpu_spec("A100")
        layers = [LAYER_1]
        prompt = build_system_prompt(gpu, layers)
        assert "sm_80" in prompt
        assert "164" in prompt

    def test_includes_playbook(self) -> None:
        """System prompt includes playbook techniques."""
        gpu = get_gpu_spec("A100")
        layers = get_relevant_playbook("reduce_memory_bandwidth", gpu, "matmul")
        prompt = build_system_prompt(gpu, layers)
        assert "block_size_tuning" in prompt
        assert "coalesced_access" in prompt

    def test_includes_output_format(self) -> None:
        """System prompt includes output format instruction."""
        gpu = get_gpu_spec("A100")
        layers = [LAYER_1]
        prompt = build_system_prompt(gpu, layers)
        assert "```cuda" in prompt
        assert "host code" in prompt


class TestBuildUserPromptLocalRewrite:
    """Tests for LOCAL_REWRITE user prompt."""

    def test_includes_direction(self) -> None:
        """LOCAL_REWRITE prompt includes optimization direction."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.LOCAL_REWRITE)
        prompt = build_user_prompt(
            spec, directive, "// existing kernel", 0, SubMode.LOCAL_REWRITE
        )
        assert "reduce_register_pressure" in prompt

    def test_includes_current_best(self) -> None:
        """LOCAL_REWRITE prompt includes current best kernel."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.LOCAL_REWRITE)
        prompt = build_user_prompt(
            spec, directive, "// existing kernel code", 0, SubMode.LOCAL_REWRITE
        )
        assert "existing kernel code" in prompt

    def test_includes_task_instruction(self) -> None:
        """LOCAL_REWRITE prompt includes rewrite task."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.LOCAL_REWRITE)
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.LOCAL_REWRITE
        )
        assert "local rewrite" in prompt.lower() or "rewrite" in prompt.lower()

    def test_includes_constraints(self) -> None:
        """LOCAL_REWRITE prompt includes hard constraints."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.LOCAL_REWRITE)
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.LOCAL_REWRITE
        )
        assert "max_smem_48KB" in prompt

    def test_includes_tabu(self) -> None:
        """LOCAL_REWRITE prompt includes tabu list."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.LOCAL_REWRITE)
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.LOCAL_REWRITE
        )
        assert "old_approach_1" in prompt


class TestBuildUserPromptParamSearch:
    """Tests for PARAM_SEARCH user prompt."""

    def test_includes_search_range(self) -> None:
        """PARAM_SEARCH prompt includes search range."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.PARAM_SEARCH)
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.PARAM_SEARCH
        )
        assert "block_size" in prompt

    def test_includes_specific_params(self) -> None:
        """PARAM_SEARCH prompt includes specific parameter values."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.PARAM_SEARCH)
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.PARAM_SEARCH
        )
        assert "128" in prompt  # First value for candidate 0


class TestBuildUserPromptDeNovo:
    """Tests for DE_NOVO user prompt."""

    def test_includes_op_semantics(self) -> None:
        """DE_NOVO prompt includes operation semantics."""
        spec = _make_problem_spec()
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.DE_NOVO)
        prompt = build_user_prompt(spec, directive, None, 0, SubMode.DE_NOVO)
        assert "Matrix multiplication" in prompt

    def test_includes_shapes_and_dtype(self) -> None:
        """DE_NOVO prompt includes shapes and dtype."""
        spec = _make_problem_spec()
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.DE_NOVO)
        prompt = build_user_prompt(spec, directive, None, 0, SubMode.DE_NOVO)
        assert "1024" in prompt
        assert "float16" in prompt

    def test_includes_reference_kernel(self) -> None:
        """DE_NOVO prompt includes reference kernel."""
        spec = _make_problem_spec()
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.DE_NOVO)
        prompt = build_user_prompt(spec, directive, None, 0, SubMode.DE_NOVO)
        assert "ref_matmul" in prompt


class TestBuildUserPromptRecombination:
    """Tests for RECOMBINATION user prompt."""

    def test_includes_gene_map(self) -> None:
        """RECOMBINATION prompt includes gene map."""
        spec = _make_problem_spec()
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.RECOMBINATION)
        prompt = build_user_prompt(
            spec, directive, "// parent code", 0, SubMode.RECOMBINATION
        )
        assert "memory_access" in prompt
        assert "hash_A" in prompt

    def test_includes_parent_hashes(self) -> None:
        """RECOMBINATION prompt includes parent candidate hashes."""
        spec = _make_problem_spec()
        directive = _make_directive(mode=Mode.EXPLORE, sub_mode=SubMode.RECOMBINATION)
        prompt = build_user_prompt(
            spec, directive, "// parent code", 0, SubMode.RECOMBINATION
        )
        assert "hash_A" in prompt
        assert "hash_B" in prompt


class TestBuildUserPromptPatternApply:
    """Tests for PATTERN_APPLY user prompt."""

    def test_includes_pattern_name(self) -> None:
        """PATTERN_APPLY prompt includes the pattern direction."""
        spec = _make_problem_spec()
        directive = _make_directive(
            sub_mode=SubMode.PATTERN_APPLY,
            direction="shared_memory_tiling",
        )
        prompt = build_user_prompt(
            spec, directive, "// kernel", 0, SubMode.PATTERN_APPLY
        )
        assert "shared_memory_tiling" in prompt


class TestVariationHints:
    """Tests for candidate variation mechanism."""

    def test_different_candidates_get_different_hints(self) -> None:
        """Each candidate index gets a different variation hint."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.DE_NOVO)
        prompts = [
            build_user_prompt(spec, directive, None, i, SubMode.DE_NOVO)
            for i in range(3)
        ]
        # Each prompt should have a "Variation hint" section
        for prompt in prompts:
            assert "Variation hint" in prompt

        # At least two should be different
        hints = [p.split("Variation hint")[1] for p in prompts]
        assert len(set(hints)) > 1

    def test_variation_hint_wraps_around(self) -> None:
        """Candidate index beyond hint count wraps around."""
        spec = _make_problem_spec()
        directive = _make_directive(sub_mode=SubMode.DE_NOVO)
        # Should not crash even with high index
        prompt = build_user_prompt(spec, directive, None, 100, SubMode.DE_NOVO)
        assert "Variation hint" in prompt


class TestRetryPrompt:
    """Tests for retry user prompt construction."""

    def test_includes_original_prompt(self) -> None:
        """Retry prompt includes original prompt content."""
        retry = build_retry_user_prompt("Original task", "Missing __global__")
        assert "Original task" in retry

    def test_includes_error_details(self) -> None:
        """Retry prompt includes error details."""
        retry = build_retry_user_prompt("Original task", "Missing __global__")
        assert "Missing __global__" in retry

    def test_includes_fix_instruction(self) -> None:
        """Retry prompt asks LLM to fix issues."""
        retry = build_retry_user_prompt("Original task", "error details")
        assert "fix" in retry.lower() or "try again" in retry.lower()
