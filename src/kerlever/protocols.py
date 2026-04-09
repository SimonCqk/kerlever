"""Kerlever protocols — async Protocol interfaces for downstream services.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

from typing import Protocol

from kerlever.types import (
    BaselineArtifact,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    OptimizationState,
    ProblemSpec,
    RoundSummary,
    StrategyDirective,
)


class StrategyNavigatorProtocol(Protocol):
    """Protocol for the Strategy Navigator service."""

    async def decide(
        self,
        problem_spec: ProblemSpec,
        optimization_state: OptimizationState,
        round_summary: RoundSummary | None,
        cross_analysis: CrossCandidateAnalysis | None,
    ) -> StrategyDirective:
        """Decide the strategy directive for the next round."""
        ...


class CodingAgentProtocol(Protocol):
    """Protocol for the Coding Agent service."""

    async def generate(
        self,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
        incumbent: BaselineArtifact,
    ) -> list[KernelCandidate]:
        """Generate kernel candidates based on the directive.

        Args:
            problem_spec: The optimization target specification.
            directive: The strategy directive for this round.
            incumbent: The current incumbent BaselineArtifact. The Coding
                Agent uses incumbent.source_code as the base for
                exploit-mode mutations.
        """
        ...


class GPUPipelineProtocol(Protocol):
    """Protocol for the GPU evaluation pipeline service."""

    async def evaluate(
        self,
        candidate: KernelCandidate,
        problem_spec: ProblemSpec,
        baseline: BaselineArtifact,
        incumbent: BaselineArtifact,
    ) -> EvaluationResult:
        """Evaluate a kernel candidate on GPU hardware.

        Args:
            candidate: The kernel candidate to evaluate.
            problem_spec: The optimization target specification.
            baseline: The original measured (or V1 synthetic) baseline
                artifact. Used to compute ObjectiveScore.relative_to_baseline.
            incumbent: The current best kernel artifact. Used for
                regression detection and ObjectiveScore.relative_to_incumbent.
        """
        ...


class CrossCandidateAnalyzerProtocol(Protocol):
    """Protocol for the Cross-Candidate Analyzer service."""

    async def analyze(
        self,
        top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
        problem_spec: ProblemSpec,
    ) -> CrossCandidateAnalysis:
        """Analyze passing candidates to extract insights."""
        ...
