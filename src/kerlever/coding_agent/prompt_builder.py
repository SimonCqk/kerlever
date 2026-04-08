"""Coding Agent prompt builder — system and user prompt construction.

Builds LLM prompts tailored to the requested sub-mode. The system prompt
is static per generation call; the user prompt varies per candidate.

Spec: docs/coding-agent/spec.md §6.3
"""

from __future__ import annotations

from kerlever.coding_agent.hardware import format_gpu_spec
from kerlever.coding_agent.playbook import format_playbook_layers
from kerlever.coding_agent.types import GPUSpec, PlaybookLayer
from kerlever.types import ProblemSpec, StrategyDirective, SubMode

# Variation hints to encourage diversity among candidates
_VARIATION_HINTS = [
    "Focus on reducing loop-carried dependencies.",
    "Try increasing thread coarsening to reduce per-thread register count.",
    "Explore a different memory access pattern.",
    "Prioritize minimizing shared memory usage in this variant.",
    "Use a different tiling strategy than your other outputs.",
    "Explore a warp-level optimization approach.",
    "Focus on maximizing instruction-level parallelism.",
    "Try a different loop ordering or unrolling strategy.",
    "Prioritize occupancy over per-thread performance.",
    "Explore vectorized memory operations in this variant.",
]


def build_system_prompt(
    gpu_spec: GPUSpec,
    playbook_layers: list[PlaybookLayer],
) -> str:
    """Build the system prompt for LLM code generation.

    Assembles the role declaration, code standards, GPU constraints,
    playbook layers, and output format instruction.

    Args:
        gpu_spec: Target GPU specification.
        playbook_layers: Relevant playbook layers for this generation.

    Returns:
        Complete system prompt string.

    Implements: REQ-CA-003, REQ-CA-004
    """
    gpu_summary = format_gpu_spec(gpu_spec)
    playbook_text = format_playbook_layers(playbook_layers)

    role = (
        "You are a CUDA kernel optimization expert. "
        "Your task is to write a single, complete __global__ kernel "
        "function. Return your code in a ```cuda code block. "
        "Do not include host code."
    )
    standards = (
        "## Code Standards (mandatory)\n"
        "- __launch_bounds__(maxThreadsPerBlock, "
        "minBlocksPerMultiprocessor) must be declared on every "
        "kernel.\n"
        "- All pointer parameters must use __restrict__ and const "
        "where the parameter is read-only.\n"
        "- Block size must be a multiple of 32 (warp size).\n"
        "- All global memory accesses must be bounds-checked.\n"
        "- Grid size computation must use ceiling division: "
        "(N + BLOCK_SIZE - 1) / BLOCK_SIZE."
    )
    output_fmt = (
        "## Output Format\n"
        "Return exactly one ```cuda code block containing a "
        "complete kernel function. Do not include host code, "
        "main functions, or kernel launch syntax."
    )

    return (
        f"{role}\n\n"
        f"{standards}\n\n"
        f"## Target GPU Constraints\n{gpu_summary}\n\n"
        f"## Optimization Playbook (relevant layers)\n"
        f"{playbook_text}\n\n"
        f"{output_fmt}"
    )


def build_user_prompt(
    problem_spec: ProblemSpec,
    directive: StrategyDirective,
    current_best_source: str | None,
    candidate_index: int,
    effective_sub_mode: SubMode,
) -> str:
    """Build the user prompt for a specific candidate.

    The prompt structure varies by sub-mode. A variation hint is appended
    to encourage diversity among candidates.

    Args:
        problem_spec: Target problem specification.
        directive: Strategy directive from the Navigator.
        current_best_source: Current best kernel source (may be None).
        candidate_index: Index of this candidate (0-based).
        effective_sub_mode: The effective sub-mode after fallback logic.

    Returns:
        Complete user prompt string.

    Implements: REQ-CA-001, REQ-CA-002, REQ-CA-008, REQ-CA-009, REQ-CA-010
    """
    if effective_sub_mode == SubMode.LOCAL_REWRITE:
        prompt = _build_local_rewrite_prompt(directive, current_best_source or "")
    elif effective_sub_mode == SubMode.PARAM_SEARCH:
        prompt = _build_param_search_prompt(
            directive, current_best_source or "", candidate_index
        )
    elif effective_sub_mode == SubMode.PATTERN_APPLY:
        prompt = _build_pattern_apply_prompt(directive, current_best_source or "")
    elif effective_sub_mode == SubMode.RECOMBINATION:
        prompt = _build_recombination_prompt(directive, current_best_source)
    else:
        # DE_NOVO (default)
        prompt = _build_de_novo_prompt(problem_spec, directive)

    # Append variation hint for diversity
    hint = _VARIATION_HINTS[candidate_index % len(_VARIATION_HINTS)]
    prompt += f"\n\nVariation hint (candidate {candidate_index}): {hint}"

    return prompt


def _build_local_rewrite_prompt(
    directive: StrategyDirective,
    current_best_source: str,
) -> str:
    """Build user prompt for EXPLOIT/LOCAL_REWRITE sub-mode."""
    parts = [
        f"Optimization direction: {directive.direction}",
        f"Current best kernel:\n```cuda\n{current_best_source}\n```",
        (
            f"Task: Apply a targeted local rewrite to the above kernel to "
            f"improve {directive.direction}."
        ),
    ]

    if directive.hard_constraints:
        parts.append(f"Constraints: {', '.join(directive.hard_constraints)}")

    if directive.tabu:
        parts.append(
            "Do not replicate these previously attempted approaches: "
            f"{', '.join(directive.tabu)}"
        )

    return "\n\n".join(parts)


def _build_param_search_prompt(
    directive: StrategyDirective,
    current_best_source: str,
    candidate_index: int,
) -> str:
    """Build user prompt for EXPLOIT/PARAM_SEARCH sub-mode."""
    parts = [
        f"Optimization direction: {directive.direction}",
        f"Current best kernel:\n```cuda\n{current_best_source}\n```",
    ]

    if directive.search_range:
        parts.append(f"Parameter search range: {directive.search_range}")

        # Select a specific parameter point for this candidate
        specific_params = _select_search_params(directive.search_range, candidate_index)
        parts.append(
            f"Task: Generate a variant of the above kernel with the "
            f"following parameter values: {specific_params}."
        )
    else:
        parts.append(
            "Task: Generate a variant of the above kernel with different "
            "parameter values."
        )

    if directive.hard_constraints:
        parts.append(f"Constraints: {', '.join(directive.hard_constraints)}")

    return "\n\n".join(parts)


def _build_pattern_apply_prompt(
    directive: StrategyDirective,
    current_best_source: str,
) -> str:
    """Build user prompt for EXPLOIT/PATTERN_APPLY sub-mode."""
    parts = [
        f"Optimization direction: {directive.direction}",
        f"Current best kernel:\n```cuda\n{current_best_source}\n```",
        (
            f"Task: Apply the {directive.direction} optimization pattern "
            f"to the above kernel."
        ),
    ]

    if directive.hard_constraints:
        parts.append(f"Constraints: {', '.join(directive.hard_constraints)}")

    return "\n\n".join(parts)


def _build_de_novo_prompt(
    problem_spec: ProblemSpec,
    directive: StrategyDirective,
) -> str:
    """Build user prompt for EXPLORE/DE_NOVO sub-mode."""
    parts = [
        f"Target operation: {problem_spec.op_semantics}",
        f"Input shapes: {problem_spec.shapes}, dtype: {problem_spec.dtype}",
        f"Optimization direction: {directive.direction}",
        (
            f"Task: Implement a high-performance {problem_spec.op_name} "
            f"kernel from scratch."
        ),
    ]

    if problem_spec.reference_kernel:
        parts.append(
            "Reference kernel (behavioral reference, do not copy):\n"
            f"```cuda\n{problem_spec.reference_kernel}\n```"
        )

    return "\n\n".join(parts)


def _build_recombination_prompt(
    directive: StrategyDirective,
    current_best_source: str | None,
) -> str:
    """Build user prompt for EXPLORE/RECOMBINATION sub-mode."""
    parts: list[str] = []

    # Include parent sources if available
    if directive.parent_candidates and len(directive.parent_candidates) >= 2:
        # Use current_best_source as context for parent retrieval
        parts.append(f"Parent A (hash: {directive.parent_candidates[0]}):")
        if current_best_source:
            parts.append(f"```cuda\n{current_best_source}\n```")
        parts.append(f"Parent B (hash: {directive.parent_candidates[1]}):")
        parts.append("(Source code should be provided by the orchestrator)")
    elif current_best_source:
        parts.append(f"Parent source:\n```cuda\n{current_best_source}\n```")

    if directive.gene_map:
        parts.append(f"Gene map: {directive.gene_map}")

    parts.append(
        "Task: Combine the specified code sections from the parent "
        "kernels into a single kernel."
    )

    if directive.hard_constraints:
        parts.append(f"Constraints: {', '.join(directive.hard_constraints)}")

    return "\n\n".join(parts)


def _select_search_params(
    search_range: dict[str, list[float]],
    candidate_index: int,
) -> dict[str, float]:
    """Select a specific parameter point from the search range.

    For each parameter, selects a value by cycling through the available
    values based on the candidate index.

    Args:
        search_range: Parameter name to list of candidate values.
        candidate_index: Index of the current candidate.

    Returns:
        Dict mapping parameter names to specific values.
    """
    result: dict[str, float] = {}
    for param_name, values in search_range.items():
        if values:
            result[param_name] = values[candidate_index % len(values)]
    return result


def build_retry_user_prompt(
    original_prompt: str,
    error_details: str,
) -> str:
    """Build a retry user prompt with error feedback appended.

    Args:
        original_prompt: The original user prompt.
        error_details: Description of why the previous attempt failed.

    Returns:
        Retry user prompt with error context.

    Implements: REQ-CA-006
    """
    return (
        f"{original_prompt}\n\n"
        f"Your previous attempt failed: {error_details}. "
        f"Please fix these issues and try again."
    )
