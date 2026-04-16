"""Kerlever stubs — stub implementations of all four Protocols for V1 testing.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

import hashlib
import random

from kerlever.types import (
    BaselineArtifact,
    BenchmarkBundle,
    BottleneckAssessment,
    CandidateIntent,
    CandidateOutcome,
    CompileStatus,
    CorrectnessResult,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    Mode,
    ObjectiveScore,
    OptimizationState,
    ProblemSpec,
    ProfileBundle,
    ProfileMetrics,
    RoundSummary,
    ShapeBenchResult,
    StaticAnalysis,
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
            base_kernel_hash=optimization_state.incumbent.kernel_hash,
            num_candidates=3,
            tabu=list(optimization_state.tabu_entries),
        )


class StubCodingAgent:
    """Stub Coding Agent that generates N dummy kernels with unique hashes."""

    async def generate(
        self,
        problem_spec: ProblemSpec,
        directive: StrategyDirective,
        incumbent: BaselineArtifact,
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

            # Determine parent_hashes based on mode
            if directive.mode == Mode.EXPLORE and directive.sub_mode == SubMode.DE_NOVO:
                parent_hashes: list[str] = []
            else:
                parent_hashes = (
                    [directive.base_kernel_hash]
                    if directive.base_kernel_hash is not None
                    else []
                )

            candidates.append(
                KernelCandidate(
                    code_hash=code_hash,
                    source_code=source_code,
                    parent_hashes=parent_hashes,
                    intent=CandidateIntent(
                        direction=directive.direction,
                        mode=directive.mode,
                        sub_mode=directive.sub_mode or SubMode.LOCAL_REWRITE,
                        rationale=f"Stub optimization #{i}",
                    ),
                )
            )
        return candidates


class StubGPUPipeline:
    """Stub GPU Pipeline that returns random perf variations.

    Uses a seeded random generator for reproducibility. Simulates
    progressive improvement: each round has a chance to produce
    candidates with better objective scores than the incumbent.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._eval_count = 0

    async def evaluate(
        self,
        candidate: KernelCandidate,
        problem_spec: ProblemSpec,
        baseline: BaselineArtifact,
        incumbent: BaselineArtifact,
    ) -> EvaluationResult:
        """Evaluate a candidate with simulated random performance variation.

        Returns per-shape benchmark results with progressive improvement
        tendency. Outcome is classified based on comparison to incumbent.
        """
        self._eval_count += 1

        # Simulate occasional compile failures (~10% chance)
        if self._rng.random() < 0.10:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_status=CompileStatus.COMPILE_ERROR,
                outcome=CandidateOutcome.COMPILE_FAIL,
            )

        # Compile succeeded — produce static analysis
        static_analysis = StaticAnalysis(
            registers_per_thread=self._rng.randint(16, 64),
            smem_bytes_per_block=self._rng.randint(0, 49152),
        )

        # Correctness always passes in stubs
        correctness = CorrectnessResult(passed=True)

        # Simulate performance: per-shape benchmark results
        # Progressive improvement: scale factor decreases with eval count
        progress_factor = max(0.70, 1.0 - self._eval_count * 0.02)
        incumbent_score = incumbent.objective_score.value
        baseline_score = baseline.objective_score.value

        shape_results: list[ShapeBenchResult] = []
        for sc in problem_spec.shape_cases:
            # Find incumbent's benchmark result for this shape, or use a default
            incumbent_latency = incumbent_score  # approximate
            for br in incumbent.benchmark_results:
                if br.shape_id == sc.shape_id:
                    incumbent_latency = br.latency_p50_us
                    break

            multiplier = self._rng.uniform(0.90, 1.05) * progress_factor
            latency_p50 = incumbent_latency * multiplier
            latency_p95 = latency_p50 * 1.05

            shape_results.append(
                ShapeBenchResult(
                    shape_id=sc.shape_id,
                    latency_p50_us=latency_p50,
                    latency_p95_us=latency_p95,
                    stdev_us=latency_p50 * 0.02,
                    run_count=10,
                )
            )

        # Compute objective score from shape results
        objective = problem_spec.objective
        if objective.aggregation == "weighted_mean":
            total_weight = sum(sc.weight for sc in problem_spec.shape_cases)
            if total_weight == 0:
                total_weight = 1.0
            weighted_sum = 0.0
            for sc, sr in zip(problem_spec.shape_cases, shape_results, strict=True):
                if objective.primary_metric in (
                    "weighted_p50_us",
                    "worst_case_p50_us",
                ):
                    weighted_sum += sc.weight * sr.latency_p50_us
                else:
                    weighted_sum += sc.weight * sr.latency_p95_us
            score_value = weighted_sum / total_weight
        else:
            # aggregation == "max"
            values: list[float] = []
            for sr in shape_results:
                if objective.primary_metric in (
                    "weighted_p50_us",
                    "worst_case_p50_us",
                ):
                    values.append(sr.latency_p50_us)
                else:
                    values.append(sr.latency_p95_us)
            score_value = max(values) if values else 0.0

        relative_to_baseline = score_value / baseline_score if baseline_score else 1.0
        relative_to_incumbent = (
            score_value / incumbent_score if incumbent_score else 1.0
        )

        objective_score = ObjectiveScore(
            metric_name=objective.primary_metric,
            value=score_value,
            relative_to_baseline=relative_to_baseline,
            relative_to_incumbent=relative_to_incumbent,
        )

        # Determine regression
        regression_guard = problem_spec.objective.regression_guard_pct
        regressed = relative_to_incumbent > (1.0 + regression_guard)

        # Classify outcome
        if score_value < incumbent_score:
            outcome = CandidateOutcome.IMPROVED
        elif regressed:
            outcome = CandidateOutcome.REGRESSION
        else:
            outcome = CandidateOutcome.BASELINE_MATCH

        benchmark = BenchmarkBundle(
            shape_results=shape_results,
            objective_score=objective_score,
            regressed_vs_incumbent=regressed,
        )

        # Profile bundle for shapes designated for profiling
        profile_bundle: ProfileBundle | None = None
        profile_shapes = [sc for sc in problem_spec.shape_cases if sc.profile]
        if profile_shapes:
            ps = profile_shapes[0]
            profile_bundle = ProfileBundle(
                shape_id=ps.shape_id,
                metrics=ProfileMetrics(
                    achieved_occupancy_pct=self._rng.uniform(40.0, 90.0),
                    dram_throughput_pct_of_peak=self._rng.uniform(30.0, 80.0),
                ),
                assessment=BottleneckAssessment(
                    tags=["memory_bandwidth", "occupancy"],
                    primary_tag="memory_bandwidth",
                    evidence={
                        "dram_throughput_pct_of_peak": self._rng.uniform(30.0, 80.0),
                        "achieved_occupancy_pct": self._rng.uniform(40.0, 90.0),
                    },
                    rule_trace=["check_memory_bound", "check_occupancy"],
                ),
            )

        return EvaluationResult(
            candidate_hash=candidate.code_hash,
            compile_status=CompileStatus.SUCCESS,
            static_analysis=static_analysis,
            correctness=correctness,
            benchmark=benchmark,
            profile=profile_bundle,
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
