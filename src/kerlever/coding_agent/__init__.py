"""Coding Agent — CUDA kernel code generator for the optimization loop.

Public API: ``CodingAgent`` class satisfying ``CodingAgentProtocol``.

Spec: docs/coding-agent/spec.md
"""

from __future__ import annotations

import asyncio
import logging

from kerlever.coding_agent.config import CodingAgentConfig
from kerlever.coding_agent.generator import generate_one_candidate
from kerlever.coding_agent.hardware import get_gpu_spec
from kerlever.coding_agent.playbook import get_relevant_playbook
from kerlever.coding_agent.prompt_builder import build_system_prompt
from kerlever.coding_agent.types import GPUSpec, PlaybookLayer
from kerlever.llm_client import LLMClientProtocol
from kerlever.types import (
    BaselineArtifact,
    KernelCandidate,
    Mode,
    ProblemSpec,
    StrategyDirective,
    SubMode,
)

__all__ = ["CodingAgent"]

logger = logging.getLogger(__name__)


class CodingAgent:
    """CUDA kernel code generator driven by LLM and structured playbook.

    Combines an LLM client with a 6-layer CUDA optimization playbook and
    GPU hardware constraint table to generate kernel candidates. Satisfies
    ``CodingAgentProtocol`` from ``kerlever.protocols``.

    Implements: REQ-CA-001 through REQ-CA-011
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: CodingAgentConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._config = config or CodingAgentConfig()

    async def generate(
        self,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
        incumbent: BaselineArtifact,
    ) -> list[KernelCandidate]:
        """Generate kernel candidates based on the directive.

        Follows the 4-step generate() flow from spec §6.6:
        1. Resolve context (GPU spec, playbook, fallback logic)
        2. Build system prompt
        3. Generate candidates concurrently
        4. Collect, deduplicate, and return

        Args:
            problem_spec: Target problem specification.
            directive: Strategy directive from the Navigator.
            incumbent: The current incumbent BaselineArtifact. The Coding
                Agent uses incumbent.source_code as the base for
                exploit-mode mutations.

        Returns:
            List of KernelCandidate objects. May be empty if all
            candidates fail.

        Implements: REQ-CA-001 through REQ-CA-011
        Invariant: INV-CA-001 (every candidate has __global__)
        Invariant: INV-CA-002 (LLM failures never propagate)
        Invariant: INV-CA-006 (mode/sub_mode match directive)
        """
        # Extract current best source from incumbent for prompt building
        current_best_source = incumbent.source_code if incumbent else None

        # Step 1: Resolve context
        gpu_spec = get_gpu_spec(problem_spec.target_gpu)
        playbook_layers = get_relevant_playbook(
            directive.direction, gpu_spec, problem_spec.op_name
        )

        # Determine effective sub_mode with fallback logic
        effective_sub_mode = _resolve_effective_sub_mode(directive, current_best_source)

        # Step 2: Build system prompt (shared across all candidates)
        system_prompt = build_system_prompt(gpu_spec, playbook_layers)

        # Step 3: Generate candidates concurrently
        results: list[KernelCandidate | None] = []

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        _safe_generate_one(
                            self._llm_client,
                            system_prompt,
                            problem_spec,
                            directive,
                            current_best_source,
                            i,
                            self._config,
                            effective_sub_mode,
                            gpu_spec,
                            playbook_layers,
                        )
                    )
                    for i in range(directive.num_candidates)
                ]
        except ExceptionGroup:
            # TaskGroup may raise ExceptionGroup if tasks raise unexpected
            # errors not caught by _safe_generate_one. Treat as all failed.
            logger.warning(
                "TaskGroup raised ExceptionGroup during generation",
                exc_info=True,
            )
            return []

        for task in tasks:
            results.append(task.result())

        # Step 4: Collect and deduplicate
        candidates: list[KernelCandidate] = []
        seen_hashes: set[str] = set()

        for result in results:
            if result is None:
                continue
            if result.code_hash in seen_hashes:
                logger.info(
                    "Deduplicating candidate with hash %s",
                    result.code_hash,
                )
                continue
            seen_hashes.add(result.code_hash)
            candidates.append(result)

        return candidates


def _resolve_effective_sub_mode(
    directive: StrategyDirective,
    current_best_source: str | None,
) -> SubMode:
    """Resolve the effective sub-mode, with fallback to DE_NOVO.

    When the directive specifies an EXPLOIT sub-mode but no current best
    source is available, falls back to DE_NOVO.

    Args:
        directive: Strategy directive.
        current_best_source: Current best kernel source.

    Returns:
        The effective SubMode to use for generation.
    """
    sub_mode = directive.sub_mode or SubMode.DE_NOVO

    # Fallback: EXPLOIT without current_best_source -> DE_NOVO-style
    if (
        directive.mode == Mode.EXPLOIT
        and not current_best_source
        and sub_mode
        in (SubMode.LOCAL_REWRITE, SubMode.PARAM_SEARCH, SubMode.PATTERN_APPLY)
    ):
        logger.warning(
            "EXPLOIT sub_mode %s requested but current_best_source is empty; "
            "falling back to DE_NOVO-style generation",
            sub_mode,
        )
        return SubMode.DE_NOVO

    return sub_mode


async def _safe_generate_one(
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
    """Wrap generate_one_candidate with catch-all exception handling.

    Ensures that any unexpected exception from a single candidate task
    does not propagate to TaskGroup, which would cancel other candidates.

    Implements: INV-CA-002
    """
    try:
        return await generate_one_candidate(
            llm_client=llm_client,
            system_prompt=system_prompt,
            problem_spec=problem_spec,
            directive=directive,
            current_best_source=current_best_source,
            candidate_index=candidate_index,
            config=config,
            effective_sub_mode=effective_sub_mode,
            gpu_spec=gpu_spec,
            playbook_layers=playbook_layers,
        )
    except Exception:
        logger.warning(
            "Unexpected error generating candidate %d",
            candidate_index,
            exc_info=True,
        )
        return None
