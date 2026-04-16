"""Kerlever orchestrator — main optimization control loop.

Spec: docs/orchestrator/spec.md

The Orchestrator sequences calls to four downstream services via
async Protocols, manages global optimization state, enforces early-exit
rules, and persists every round's state to a workdir.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import Counter
from pathlib import Path

from kerlever.protocols import (
    CodingAgentProtocol,
    CrossCandidateAnalyzerProtocol,
    GPUPipelineProtocol,
    StrategyNavigatorProtocol,
)
from kerlever.state import StateManager
from kerlever.types import (
    AttemptRecord,
    BaselineArtifact,
    CandidateOutcome,
    CompileStatus,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    ObjectiveScore,
    OptimizationResult,
    OptimizationState,
    Phase,
    ProblemSpec,
    RoundState,
    RoundSummary,
    ShapeBenchResult,
    StaticAnalysis,
    TabuEntry,
)

logger = logging.getLogger(__name__)

# Outcomes that qualify for cross-candidate analysis input
_PASSING_OUTCOMES = frozenset(
    {CandidateOutcome.IMPROVED, CandidateOutcome.BASELINE_MATCH}
)

# Number of rounds a tabu entry remains active after creation.
# The spec says "a configurable constant" but does not specify the default.
_TABU_WINDOW = 5


def _bootstrap_baseline(problem_spec: ProblemSpec) -> BaselineArtifact:
    """Construct a synthetic BaselineArtifact from ProblemSpec declared values.

    V1 behavior: constructs synthetic benchmark results from shape_cases
    and computes an objective score from them. No real GPU evaluation.

    TODO(V2): Replace with real GPU pipeline bootstrap that compiles,
    validates, benchmarks, and profiles the reference kernel.

    Implements: REQ-ORCH-009, SCN-ORCH-009-03
    Invariant: INV-ORCH-006 (baseline and incumbent always present after bootstrap)
    """
    kernel_hash = hashlib.sha256(problem_spec.reference_kernel.encode()).hexdigest()[
        :16
    ]

    # Synthetic ShapeBenchResult for each shape_case with placeholder latency
    benchmark_results: list[ShapeBenchResult] = []
    for sc in problem_spec.shape_cases:
        benchmark_results.append(
            ShapeBenchResult(
                shape_id=sc.shape_id,
                latency_p50_us=problem_spec.target_metric_value * 5.0,
                latency_p95_us=problem_spec.target_metric_value * 6.0,
                run_count=1,
            )
        )

    # Compute objective score from synthetic benchmarks using the objective
    # definition. For V1, use a simple weighted aggregation of the primary
    # metric values.
    objective = problem_spec.objective
    if objective.aggregation == "weighted_mean":
        total_weight = sum(sc.weight for sc in problem_spec.shape_cases)
        if total_weight == 0:
            total_weight = 1.0
        weighted_sum = 0.0
        for sc, br in zip(problem_spec.shape_cases, benchmark_results, strict=True):
            if objective.primary_metric in (
                "weighted_p50_us",
                "worst_case_p50_us",
            ):
                weighted_sum += sc.weight * br.latency_p50_us
            else:
                weighted_sum += sc.weight * br.latency_p95_us
        score_value = weighted_sum / total_weight
    else:
        # aggregation == "max"
        values: list[float] = []
        for br in benchmark_results:
            if objective.primary_metric in (
                "weighted_p50_us",
                "worst_case_p50_us",
            ):
                values.append(br.latency_p50_us)
            else:
                values.append(br.latency_p95_us)
        score_value = max(values) if values else 0.0

    objective_score = ObjectiveScore(
        metric_name=objective.primary_metric,
        value=score_value,
        relative_to_baseline=1.0,
        relative_to_incumbent=1.0,
    )

    return BaselineArtifact(
        kernel_hash=kernel_hash,
        source_code=problem_spec.reference_kernel,
        compile_artifact=StaticAnalysis(),
        benchmark_results=benchmark_results,
        objective_score=objective_score,
        profile_bundle=None,
    )


class Orchestrator:
    """Main optimization loop controller.

    Drives the full kernel optimization cycle: bootstrap a measured baseline,
    request strategy, generate candidates, evaluate on GPU, analyze results,
    update state, repeat. The Orchestrator does not contain optimization
    intelligence — it is a state machine that sequences calls to four
    downstream services.

    Implements: REQ-ORCH-001, REQ-ORCH-002, REQ-ORCH-003, REQ-ORCH-006
    """

    def __init__(
        self,
        problem_spec: ProblemSpec,
        strategy_navigator: StrategyNavigatorProtocol,
        coding_agent: CodingAgentProtocol,
        gpu_pipeline: GPUPipelineProtocol,
        cross_analyzer: CrossCandidateAnalyzerProtocol,
        workdir: Path,
    ) -> None:
        self._problem_spec = problem_spec
        self._navigator = strategy_navigator
        self._coder = coding_agent
        self._pipeline = gpu_pipeline
        self._analyzer = cross_analyzer
        self._state_mgr = StateManager(workdir)

        # Bootstrap: construct baseline artifact from ProblemSpec
        baseline = _bootstrap_baseline(problem_spec)

        self._state = OptimizationState(
            problem_spec=problem_spec,
            baseline=baseline,
            incumbent=baseline,
        )
        self._total_candidates_evaluated = 0
        self._prev_cross_analysis: CrossCandidateAnalysis | None = None

    async def run(self) -> OptimizationResult:
        """Execute the optimization loop until target met or max rounds reached.

        Follows spec section 6.1 main loop exactly. Each round performs:
        strategy request, candidate generation, kernel persistence,
        concurrent evaluation, outcome classification, incumbent update,
        termination check, cross-analysis, state update, and persistence.

        Returns:
            The final optimization result with status, best kernel, and stats.

        Implements: REQ-ORCH-001 (loop termination)
        Invariant: INV-ORCH-001 (incumbent monotonically non-increasing)
        Invariant: INV-ORCH-002 (round counter strictly increasing)
        """
        status = "MAX_ROUNDS_REACHED"

        for round_num in range(self._problem_spec.max_rounds):
            self._state.current_round = round_num

            # Get the previous round summary (None for round 0)
            prev_summary = self._state.rounds[-1] if self._state.rounds else None

            # Snapshot the incumbent objective score before this round
            prev_incumbent_score = self._state.incumbent.objective_score.value

            # --- Step 1: Request strategy directive ---
            phase = Phase.AWAITING_STRATEGY
            directive = await self._navigator.decide(
                self._problem_spec,
                self._state,
                prev_summary,
                self._prev_cross_analysis,
            )

            # --- Step 2: Request kernel candidates ---
            phase = Phase.AWAITING_CODING
            candidates = await self._coder.generate(
                self._problem_spec,
                directive,
                self._state.incumbent,
            )

            # --- Step 3: Persist all candidate source files (INV-ORCH-003) ---
            for candidate in candidates:
                self._state_mgr.save_kernel(candidate.code_hash, candidate.source_code)

            # --- Step 4: Evaluate all candidates concurrently (REQ-ORCH-006) ---
            phase = Phase.AWAITING_EVALUATION
            eval_results = await self._evaluate_candidates_concurrently(candidates)
            self._total_candidates_evaluated += len(eval_results)

            # --- Step 5: Classify outcomes and filter results ---
            # Outcomes are already set by the pipeline; partition by outcome
            passing_results = [
                r for r in eval_results if r.outcome in _PASSING_OUTCOMES
            ]
            improved_results = [
                r for r in eval_results if r.outcome == CandidateOutcome.IMPROVED
            ]

            # --- Step 6: Update incumbent (REQ-ORCH-003) ---
            self._update_incumbent(improved_results, candidates)

            # --- Step 7: Check termination (REQ-ORCH-001) ---
            target_met = self._check_target_met()

            # --- Step 8: Cross-candidate analysis (REQ-ORCH-007) ---
            phase = Phase.ANALYSIS
            cross_analysis: CrossCandidateAnalysis | None = None
            if len(passing_results) >= 2:
                # Build (candidate, result) pairs for passing candidates
                candidate_by_hash = {c.code_hash: c for c in candidates}
                top_k = [
                    (candidate_by_hash[r.candidate_hash], r) for r in passing_results
                ]
                cross_analysis = await self._analyzer.analyze(top_k, self._problem_spec)
            self._prev_cross_analysis = cross_analysis

            # --- Step 9: Record attempt history, tabu, bottleneck ---
            self._record_attempts(round_num, candidates, eval_results)
            self._update_tabu_entries(round_num, candidates, eval_results)
            self._update_bottleneck_history(passing_results)

            # --- Step 10: Build round summary, persist state ---
            phase = Phase.ROUND_COMPLETE

            # Best objective score this round (from passing candidates)
            round_best_score: float | None = None
            passing_with_bench = [r for r in passing_results if r.benchmark is not None]
            if passing_with_bench:
                round_best_score = min(
                    r.benchmark.objective_score.value
                    for r in passing_with_bench
                    if r.benchmark is not None
                )

            # Compute absolute and relative gains vs previous incumbent
            abs_gain: float | None = None
            rel_gain: float | None = None
            current_incumbent_score = self._state.incumbent.objective_score.value
            if current_incumbent_score < prev_incumbent_score:
                abs_gain = prev_incumbent_score - current_incumbent_score
                rel_gain = abs_gain / prev_incumbent_score

            round_summary = RoundSummary(
                round_number=round_num,
                mode=directive.mode,
                direction=directive.direction,
                num_candidates=len(candidates),
                num_improved=len(improved_results),
                best_objective_score=round_best_score,
                abs_gain_vs_prev_best_us=abs_gain,
                rel_gain_vs_prev_best=rel_gain,
            )
            self._state.rounds.append(round_summary)

            # Build RoundState for persistence
            round_state = RoundState(
                round_number=round_num,
                phase=phase,
                directive=directive,
                candidates=candidates,
                evaluation_results=eval_results,
                cross_analysis=cross_analysis,
                best_candidate_hash=(
                    self._state.incumbent.kernel_hash if improved_results else None
                ),
                best_objective_score=round_best_score,
            )

            # Build decision log entry (spec 6.8)
            outcome_counts = Counter(r.outcome.value for r in eval_results)
            decision_entry: dict[str, object] = {
                "round_number": round_num,
                "directive": {
                    "mode": directive.mode.value,
                    "direction": directive.direction,
                    "reason": directive.reason,
                    "num_candidates": directive.num_candidates,
                },
                "outcomes": dict(outcome_counts),
                "best_objective_score_this_round": round_best_score,
                "incumbent_objective_score_after_round": (
                    self._state.incumbent.objective_score.value
                ),
                "improvement": abs_gain is not None,
            }
            self._state.decision_log.append(decision_entry)

            # Persist everything
            self._state_mgr.save_round(round_state)
            self._state_mgr.save_state(self._state)
            self._state_mgr.append_decision(decision_entry)

            logger.info(
                "Round %d complete: %d candidates, %d improved, "
                "best_score=%.3f, incumbent_score=%.3f",
                round_num,
                len(candidates),
                len(improved_results),
                round_best_score or 0.0,
                self._state.incumbent.objective_score.value,
            )

            # --- Step 11: Check if target met (break after persistence) ---
            if target_met:
                status = "TARGET_MET"
                break

        # Build and persist final result
        result = OptimizationResult(
            status=status,
            best_kernel_hash=self._state.incumbent.kernel_hash,
            best_objective_score=self._state.incumbent.objective_score.value,
            best_kernel_source=self._state.incumbent.source_code,
            total_rounds=len(self._state.rounds),
            total_candidates_evaluated=self._total_candidates_evaluated,
        )
        self._state_mgr.save_result(result)

        return result

    async def _evaluate_candidates_concurrently(
        self,
        candidates: list[KernelCandidate],
    ) -> list[EvaluationResult]:
        """Evaluate all candidates concurrently, isolating per-task failures.

        Uses asyncio.TaskGroup for concurrent evaluation. Each candidate
        evaluation is wrapped to catch exceptions and record ERROR outcome.

        Implements: REQ-ORCH-006, SCN-ORCH-006-01
        """
        results: list[EvaluationResult] = []

        async def _eval_one(candidate: KernelCandidate) -> EvaluationResult:
            """Evaluate a single candidate, catching exceptions.

            If the pipeline raises an unexpected exception, the candidate
            is recorded with outcome ERROR instead of propagating the error.
            """
            try:
                return await self._pipeline.evaluate(
                    candidate,
                    self._problem_spec,
                    self._state.baseline,
                    self._state.incumbent,
                )
            except Exception:
                logger.exception(
                    "Evaluation failed for candidate %s",
                    candidate.code_hash,
                )
                return EvaluationResult(
                    candidate_hash=candidate.code_hash,
                    compile_status=CompileStatus.COMPILE_ERROR,
                    outcome=CandidateOutcome.ERROR,
                )

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_eval_one(c)) for c in candidates]

        results = [task.result() for task in tasks]
        return results

    def _update_incumbent(
        self,
        improved_results: list[EvaluationResult],
        candidates: list[KernelCandidate],
    ) -> None:
        """Update the incumbent if any IMPROVED candidate is strictly better.

        Among IMPROVED candidates, selects the one with the lowest
        objective_score.value. Only updates if it is strictly less than
        the current incumbent's objective score.

        Implements: REQ-ORCH-003, SCN-ORCH-003-01
        Invariant: INV-ORCH-001 (monotonically non-increasing)
        """
        if not improved_results:
            return

        # Find the best among improved candidates
        best_result = min(
            improved_results,
            key=lambda r: (
                r.benchmark.objective_score.value
                if r.benchmark is not None
                else float("inf")
            ),
        )

        if best_result.benchmark is None:
            return

        new_score = best_result.benchmark.objective_score.value

        # Check if this is strictly better than current incumbent
        if new_score < self._state.incumbent.objective_score.value:
            # Find the candidate to get source code
            candidate_by_hash = {c.code_hash: c for c in candidates}
            best_candidate = candidate_by_hash[best_result.candidate_hash]

            # Construct new BaselineArtifact from the winning candidate
            self._state.incumbent = BaselineArtifact(
                kernel_hash=best_result.candidate_hash,
                source_code=best_candidate.source_code,
                compile_artifact=best_result.static_analysis or StaticAnalysis(),
                benchmark_results=best_result.benchmark.shape_results,
                objective_score=best_result.benchmark.objective_score,
                profile_bundle=best_result.profile,
            )

    def _check_target_met(self) -> bool:
        """Check if the incumbent's objective score meets the target.

        Implements: REQ-ORCH-001 (termination), SCN-ORCH-011-01

        Returns:
            True if incumbent.objective_score.value <= target_metric_value.
        """
        return (
            self._state.incumbent.objective_score.value
            <= self._problem_spec.target_metric_value
        )

    def _record_attempts(
        self,
        round_num: int,
        candidates: list[KernelCandidate],
        eval_results: list[EvaluationResult],
    ) -> None:
        """Record AttemptRecord for every candidate in this round.

        Implements: SCN-ORCH-010-01
        Invariant: INV-ORCH-007 (attempt records are append-only)
        """
        result_by_hash = {r.candidate_hash: r for r in eval_results}
        for candidate in candidates:
            evaluation = result_by_hash.get(candidate.code_hash)
            if evaluation is None:
                continue

            # Base kernel hash: first entry in parent_hashes, or None
            base_hash = candidate.parent_hashes[0] if candidate.parent_hashes else None

            # Objective score if benchmarking was reached
            obj_score: float | None = None
            if evaluation.benchmark is not None:
                obj_score = evaluation.benchmark.objective_score.value

            self._state.attempts.append(
                AttemptRecord(
                    round_number=round_num,
                    candidate_hash=candidate.code_hash,
                    base_kernel_hash=base_hash,
                    direction=candidate.intent.direction,
                    sub_mode=candidate.intent.sub_mode,
                    outcome=evaluation.outcome,
                    objective_score=obj_score,
                )
            )

    def _update_tabu_entries(
        self,
        round_num: int,
        candidates: list[KernelCandidate],
        eval_results: list[EvaluationResult],
    ) -> None:
        """Create TabuEntry entries for non-improving directions.

        A TabuEntry is created when a candidate's outcome is not IMPROVED,
        suppressing the same direction on the same base kernel for
        _TABU_WINDOW rounds.

        Implements: SCN-ORCH-010-01, SCN-ORCH-010-02
        """
        result_by_hash = {r.candidate_hash: r for r in eval_results}
        for candidate in candidates:
            evaluation = result_by_hash.get(candidate.code_hash)
            if evaluation is None:
                continue

            if evaluation.outcome != CandidateOutcome.IMPROVED:
                base_hash = (
                    candidate.parent_hashes[0] if candidate.parent_hashes else None
                )
                self._state.tabu_entries.append(
                    TabuEntry(
                        base_kernel_hash=base_hash,
                        direction=candidate.intent.direction,
                        sub_mode=candidate.intent.sub_mode,
                        round_number=round_num,
                        expires_after_round=round_num + _TABU_WINDOW,
                    )
                )

    def _update_bottleneck_history(
        self,
        passing_results: list[EvaluationResult],
    ) -> None:
        """Append BottleneckAssessment from the best profiled candidate.

        The BottleneckAssessment from the ProfileBundle of the best passing
        candidate (if profiling occurred) is appended to bottleneck_history.

        Spec: §6.6 — if no candidates were profiled, no entry is added.
        """
        profiled = [r for r in passing_results if r.profile is not None]
        if not profiled:
            return

        # Best profiled = lowest objective score among profiled passing
        best_profiled = min(
            profiled,
            key=lambda r: (
                r.benchmark.objective_score.value
                if r.benchmark is not None
                else float("inf")
            ),
        )
        if best_profiled.profile is not None:
            self._state.bottleneck_history.append(best_profiled.profile.assessment)
