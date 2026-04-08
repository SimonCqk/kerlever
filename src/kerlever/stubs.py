"""Kerlever stubs — stub implementations of all four Protocols for V1 testing.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

import hashlib
import random

from kerlever.types import (
    BenchResult,
    CandidateOutcome,
    CompileResult,
    CompileStatus,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    Mode,
    OptimizationState,
    ProblemSpec,
    ProfileResult,
    RoundSummary,
    StrategyDirective,
    SubMode,
)


class StubStrategyNavigator:
    """Stub Strategy Navigator that always returns a fixed exploit directive."""

    async def decide(
        self,
        problem_spec: ProblemSpec,
        optimization_state: OptimizationState,
        round_summary: RoundSummary | None,
        cross_analysis: CrossCandidateAnalysis | None,
    ) -> StrategyDirective:
        """Return a fixed exploit directive."""
        return StrategyDirective(
            mode=Mode.EXPLOIT,
            direction="optimize_memory_access",
            reason="Stub: always exploit memory access optimization",
            base_kernel_hash=optimization_state.global_best_hash,
            num_candidates=3,
            tabu=list(optimization_state.tabu_list),
        )


class StubCodingAgent:
    """Stub Coding Agent that generates N dummy kernels with unique hashes."""

    async def generate(
        self,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
        current_best_source: str | None,
    ) -> list[KernelCandidate]:
        """Generate dummy kernel candidates with unique content hashes."""
        candidates: list[KernelCandidate] = []
        for i in range(directive.num_candidates):
            source_code = (
                f"// Stub kernel candidate {i}\n"
                f"// Direction: {directive.direction}\n"
                f"// Mode: {directive.mode.value}\n"
                f"__global__ void kernel_{i}() {{\n"
                f"    // dummy kernel body round={id(directive)} idx={i}\n"
                f"}}\n"
            )
            code_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]
            candidates.append(
                KernelCandidate(
                    code_hash=code_hash,
                    source_code=source_code,
                    intent_tag=f"stub_opt_{directive.direction}_{i}",
                    parent_hash=directive.base_kernel_hash,
                    mode=directive.mode,
                    sub_mode=SubMode.LOCAL_REWRITE,
                )
            )
        return candidates


class StubGPUPipeline:
    """Stub GPU Pipeline that returns random perf variations.

    Uses a seeded random generator for reproducibility. Simulates
    progressive improvement: each round has a chance to produce
    candidates with better latency than the baseline.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._eval_count = 0

    async def evaluate(
        self,
        candidate: KernelCandidate,
        problem_spec: ProblemSpec,
        current_best_latency_us: float | None,
    ) -> EvaluationResult:
        """Evaluate a candidate with simulated random performance variation.

        Returns latency as 90%-105% of baseline, with progressive
        improvement tendency. Outcome is classified based on comparison
        to current best.
        """
        self._eval_count += 1

        # Simulate occasional compile failures (~10% chance)
        if self._rng.random() < 0.10:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_result=CompileResult(
                    status=CompileStatus.COMPILE_ERROR,
                    error_message="Stub: simulated compile error",
                ),
                outcome=CandidateOutcome.COMPILE_FAIL,
            )

        # Simulate performance: 90%-105% of baseline, with progressive
        # improvement (lower multiplier over time)
        base_latency = current_best_latency_us or problem_spec.baseline_perf_us
        # Progressive improvement: scale factor decreases with eval count
        progress_factor = max(0.70, 1.0 - self._eval_count * 0.02)
        multiplier = self._rng.uniform(0.90, 1.05) * progress_factor
        latency = base_latency * multiplier

        # Classify outcome
        if current_best_latency_us is None or latency < current_best_latency_us:
            outcome = CandidateOutcome.IMPROVED
        elif latency <= current_best_latency_us * (1 + problem_spec.tolerance):
            outcome = CandidateOutcome.BASELINE_MATCH
        else:
            outcome = CandidateOutcome.REGRESSION

        bench_result = BenchResult(
            latency_us=latency,
            p50_us=latency * 0.98,
            p95_us=latency * 1.05,
        )

        profile_result = ProfileResult(
            bottleneck_tags=["memory_bandwidth", "occupancy"],
            metrics={
                "achieved_bandwidth_pct": self._rng.uniform(30.0, 80.0),
                "occupancy_pct": self._rng.uniform(40.0, 90.0),
            },
        )

        return EvaluationResult(
            candidate_hash=candidate.code_hash,
            compile_result=CompileResult(
                status=CompileStatus.SUCCESS,
                register_count=self._rng.randint(16, 64),
                smem_bytes=self._rng.randint(0, 49152),
            ),
            bench_result=bench_result,
            profile_result=profile_result,
            outcome=outcome,
        )


class StubCrossCandidateAnalyzer:
    """Stub Cross-Candidate Analyzer that returns empty analysis."""

    async def analyze(
        self,
        top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
        problem_spec: ProblemSpec,
    ) -> CrossCandidateAnalysis:
        """Return an empty cross-candidate analysis."""
        return CrossCandidateAnalysis(
            insights=[],
            winning_genes=[],
            recombination_suggestions=[],
        )
