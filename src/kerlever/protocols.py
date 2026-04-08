"""Kerlever protocols — async Protocol interfaces for downstream services.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

from typing import Protocol

from kerlever.types import (
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
        current_best_source: str | None,
    ) -> list[KernelCandidate]:
        """Generate kernel candidates based on the directive."""
        ...


class GPUPipelineProtocol(Protocol):
    """Protocol for the GPU evaluation pipeline service."""

    async def evaluate(
        self,
        candidate: KernelCandidate,
        problem_spec: ProblemSpec,
        current_best_latency_us: float | None,
    ) -> EvaluationResult:
        """Evaluate a kernel candidate on GPU hardware."""
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
