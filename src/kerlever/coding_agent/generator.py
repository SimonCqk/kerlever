"""Coding Agent generator — LLM code generation, parsing, and retry logic.

Handles the per-candidate generation flow: LLM call, response parsing,
code validation, retry on failure, and KernelCandidate assembly.

Spec: docs/coding-agent/spec.md §6.5
"""

from __future__ import annotations

import hashlib
import logging
import re

from kerlever.coding_agent.code_validator import has_errors, validate_code
from kerlever.coding_agent.config import CodingAgentConfig
from kerlever.coding_agent.prompt_builder import (
    build_retry_user_prompt,
    build_user_prompt,
)
from kerlever.coding_agent.types import GPUSpec, PlaybookLayer
from kerlever.llm_client import LLMClientProtocol
from kerlever.types import (
    KernelCandidate,
    ProblemSpec,
    StrategyDirective,
    SubMode,
)

logger = logging.getLogger(__name__)

# Patterns for extracting code blocks from LLM response
_CUDA_BLOCK_PATTERN = re.compile(r"```cuda\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK_PATTERN = re.compile(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)
_GLOBAL_EXTRACT_PATTERN = re.compile(
    r"(__global__\s+(?:void|[\w:]+)\s+\w+\s*\()", re.DOTALL
)


def parse_cuda_from_response(response: str) -> str | None:
    """Parse CUDA code from an LLM response.

    Tries three extraction strategies in order:
    1. ```cuda code block
    2. ```c or ``` (generic) code block
    3. Raw __global__ function extraction

    Args:
        response: Raw LLM response text.

    Returns:
        Extracted CUDA code string, or None if extraction fails.

    Implements: REQ-CA-006, SCN-CA-006-01
    """
    # Strategy 1: ```cuda block
    match = _CUDA_BLOCK_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    # Strategy 2: ```c or ``` generic block
    match = _GENERIC_BLOCK_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    # Strategy 3: Extract __global__ function from raw text
    match = _GLOBAL_EXTRACT_PATTERN.search(response)
    if match:
        start = match.start()
        # Find the matching closing brace
        depth = 0
        pos = start
        found_open = False
        while pos < len(response):
            if response[pos] == "{":
                depth += 1
                found_open = True
            elif response[pos] == "}":
                depth -= 1
                if found_open and depth == 0:
                    return response[start : pos + 1].strip()
            pos += 1

    return None


def compute_code_hash(source_code: str) -> str:
    """Compute the SHA-256 hash of source code, truncated to 16 hex chars.

    Args:
        source_code: The CUDA source code.

    Returns:
        First 16 hex characters of the SHA-256 hash.

    Implements: REQ-CA-007
    Invariant: INV-CA-003 (deterministic and collision-resistant)
    """
    return hashlib.sha256(source_code.encode()).hexdigest()[:16]


def build_intent_tag(
    directive: StrategyDirective,
    candidate_index: int,
    effective_sub_mode: SubMode,
) -> str:
    """Build the intent tag for a candidate.

    Format: "{sub_mode}_{direction}_{index}"

    Args:
        directive: The strategy directive.
        candidate_index: Index of the candidate.
        effective_sub_mode: The effective sub-mode after fallback.

    Returns:
        Intent tag string.

    Implements: REQ-CA-007, SCN-CA-007-02
    """
    mode_tag = effective_sub_mode.value.lower()
    direction = directive.direction
    return f"{mode_tag}_{direction}_{candidate_index}"


async def generate_one_candidate(
    llm_client: LLMClientProtocol,
    system_prompt: str,
    problem_spec: ProblemSpec,
    directive: StrategyDirective,
    current_best_source: str | None,
    candidate_index: int,
    config: CodingAgentConfig,
    effective_sub_mode: SubMode,
    gpu_spec: GPUSpec,
    playbook_layers: list[PlaybookLayer],
) -> KernelCandidate | None:
    """Generate a single kernel candidate via LLM call with retry.

    This is the per-candidate generation flow from §6.5. On validation
    failure, retries once with error feedback. On second failure, returns
    None (candidate skipped).

    Args:
        llm_client: The LLM client for code generation.
        system_prompt: Shared system prompt for all candidates.
        problem_spec: Target problem specification.
        directive: Strategy directive.
        current_best_source: Current best kernel source (may be None).
        candidate_index: Index of this candidate.
        config: Coding agent configuration.
        effective_sub_mode: The effective sub-mode after fallback.
        gpu_spec: Target GPU specification (unused directly but kept for
            potential future prompt refinement).
        playbook_layers: Relevant playbook layers (unused directly but
            kept for potential future prompt refinement).

    Returns:
        A KernelCandidate if generation succeeds, None if skipped.

    Implements: REQ-CA-006, INV-CA-002
    """
    user_prompt = build_user_prompt(
        problem_spec,
        directive,
        current_best_source,
        candidate_index,
        effective_sub_mode,
    )

    # Attempt 1
    code = await _attempt_generation(
        llm_client, system_prompt, user_prompt, problem_spec.dtype, config
    )
    if code is not None:
        return _assemble_candidate(code, directive, candidate_index, effective_sub_mode)

    # Retry (attempt 2) with error feedback
    error_details = "Failed to produce valid CUDA code on first attempt."
    retry_prompt = build_retry_user_prompt(user_prompt, error_details)

    code = await _attempt_generation(
        llm_client, system_prompt, retry_prompt, problem_spec.dtype, config
    )
    if code is not None:
        return _assemble_candidate(code, directive, candidate_index, effective_sub_mode)

    logger.warning("Candidate %d skipped after 2 failed attempts", candidate_index)
    return None


async def _attempt_generation(
    llm_client: LLMClientProtocol,
    system_prompt: str,
    user_prompt: str,
    dtype: str,
    config: CodingAgentConfig,
) -> str | None:
    """Attempt a single LLM generation cycle: call, parse, validate.

    Args:
        llm_client: The LLM client.
        system_prompt: System prompt.
        user_prompt: User prompt.
        dtype: Expected dtype for validation.
        config: Configuration.

    Returns:
        Validated CUDA code string, or None on failure.
    """
    try:
        response = await llm_client.complete(system_prompt, user_prompt)
    except Exception:
        logger.warning("LLM call failed", exc_info=True)
        return None

    code = parse_cuda_from_response(response)
    if code is None:
        logger.warning("Failed to parse CUDA code from LLM response")
        return None

    # Truncate if exceeds max length
    if len(code) > config.max_code_length:
        logger.warning(
            "Generated code exceeds max_code_length (%d > %d), truncating",
            len(code),
            config.max_code_length,
        )
        code = code[: config.max_code_length]

    # Validate
    issues = validate_code(code, dtype)
    if has_errors(issues):
        error_msgs = [issue.message for issue in issues if issue.severity == "error"]
        logger.warning("Code validation failed: %s", "; ".join(error_msgs))
        return None

    return code


def _assemble_candidate(
    source_code: str,
    directive: StrategyDirective,
    candidate_index: int,
    effective_sub_mode: SubMode,
) -> KernelCandidate:
    """Assemble a KernelCandidate from validated source code.

    Args:
        source_code: Validated CUDA source code.
        directive: Strategy directive.
        candidate_index: Index of the candidate.
        effective_sub_mode: The effective sub-mode.

    Returns:
        Assembled KernelCandidate.

    Implements: REQ-CA-007, INV-CA-006
    """
    code_hash = compute_code_hash(source_code)
    intent_tag = build_intent_tag(directive, candidate_index, effective_sub_mode)

    # For DE_NOVO, parent_hash is None per spec
    parent_hash: str | None
    if effective_sub_mode == SubMode.DE_NOVO:
        parent_hash = None
    else:
        parent_hash = directive.base_kernel_hash

    return KernelCandidate(
        code_hash=code_hash,
        source_code=source_code,
        intent_tag=intent_tag,
        parent_hash=parent_hash,
        mode=directive.mode,
        sub_mode=directive.sub_mode,
    )
