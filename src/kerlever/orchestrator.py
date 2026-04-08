"""Kerlever orchestrator — main optimization control loop.

Spec: docs/orchestrator/spec.md

The Orchestrator sequences calls to four downstream services via
async Protocols, manages global optimization state, enforces early-exit
rules, and persists every round's state to a workdir.
"""

from __future__ import annotations

import asyncio
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
    CandidateOutcome,
    CompileResult,
    CompileStatus,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    OptimizationResult,
    OptimizationState,
    Phase,
    ProblemSpec,
    RoundState,
    RoundSummary,
)

logger = logging.getLogger(__name__)

# Outcomes that qualify for cross-candidate analysis input
_PASSING_OUTCOMES = frozenset(
    {CandidateOutcome.IMPROVED, CandidateOutcome.BASELINE_MATCH}
)


class Orchestrator:
    """Main optimization loop controller.

    Drives the full kernel optimization cycle: request strategy, generate
    candidates, evaluate on GPU, analyze results, update state, repeat.
    The Orchestrator does not contain optimization intelligence — it is
    a state machine that sequences calls to four downstream services.

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

        self._state = OptimizationState(problem_spec=problem_spec)
        self._total_candidates_evaluated = 0
        self._prev_cross_analysis: CrossCandidateAnalysis | None = None

    async def run(self) -> OptimizationResult:
        """Execute the optimization loop until target met or max rounds reached.

        Follows spec section 6.1 main loop exactly. Each round performs:
        strategy request, candidate generation, kernel persistence,
        concurrent evaluation, outcome classification, global best update,
        termination check, cross-analysis, state update, and persistence.

        Returns:
            The final optimization result with status, best kernel, and stats.

        Implements: REQ-ORCH-001 (loop termination)
        Invariant: INV-ORCH-001 (global best monotonically non-increasing)
        Invariant: INV-ORCH-002 (round counter strictly increasing)
        """
        status = "MAX_ROUNDS_REACHED"

        for round_num in range(self._problem_spec.max_rounds):
            self._state.current_round = round_num

            # Get the previous round summary (None for round 0)
            prev_summary = self._state.rounds[-1] if self._state.rounds else None

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
                self._state.global_best_source,
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

            # --- Step 6: Update global best (REQ-ORCH-003) ---
            prev_best_latency = self._state.global_best_latency_us
            self._update_global_best(improved_results, candidates)

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

            # --- Step 9: Update tabu and bottleneck history ---
            # Tabu: append all intent_tags from this round (spec 6.6)
            for candidate in candidates:
                self._state.tabu_list.append(candidate.intent_tag)

            # Bottleneck history: tags from passing candidates' profiles (spec 6.6)
            round_bottleneck_tags: list[str] = []
            for eval_r in passing_results:
                if eval_r.profile_result is not None:
                    round_bottleneck_tags.extend(eval_r.profile_result.bottleneck_tags)
            self._state.bottleneck_history.append(round_bottleneck_tags)

            # --- Step 10: Build round summary, persist state ---
            phase = Phase.ROUND_COMPLETE

            # Compute improvement delta
            improvement_delta: float | None = None
            if (
                self._state.global_best_latency_us is not None
                and prev_best_latency is not None
                and self._state.global_best_latency_us < prev_best_latency
            ):
                improvement_delta = (
                    prev_best_latency - self._state.global_best_latency_us
                )

            # Best latency this round (from best passing candidate)
            round_best_latency: float | None = None
            passing_with_bench = [
                r for r in passing_results if r.bench_result is not None
            ]
            if passing_with_bench:
                round_best_latency = min(
                    r.bench_result.latency_us
                    for r in passing_with_bench
                    if r.bench_result is not None
                )

            round_summary = RoundSummary(
                round_number=round_num,
                mode=directive.mode,
                direction=directive.direction,
                num_candidates=len(candidates),
                num_improved=len(improved_results),
                best_latency_us=round_best_latency,
                improvement_over_prev_best=improvement_delta,
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
                    self._state.global_best_hash if improved_results else None
                ),
                best_latency_us=round_best_latency,
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
                "best_latency_this_round": round_best_latency,
                "global_best_latency_after_round": (self._state.global_best_latency_us),
                "improvement": improvement_delta is not None,
            }
            self._state.decision_log.append(decision_entry)

            # Persist everything
            self._state_mgr.save_round(round_state)
            self._state_mgr.save_state(self._state)
            self._state_mgr.append_decision(decision_entry)

            logger.info(
                "Round %d complete: %d candidates, %d improved, "
                "best_latency=%.3f us, global_best=%.3f us",
                round_num,
                len(candidates),
                len(improved_results),
                round_best_latency or 0.0,
                self._state.global_best_latency_us or 0.0,
            )

            # --- Step 11: Check if target met (break after persistence) ---
            if target_met:
                status = "TARGET_MET"
                break

        # Build and persist final result
        result = OptimizationResult(
            status=status,
            best_kernel_hash=self._state.global_best_hash,
            best_latency_us=self._state.global_best_latency_us,
            best_kernel_source=self._state.global_best_source,
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
                    self._state.global_best_latency_us,
                )
            except Exception:
                logger.exception(
                    "Evaluation failed for candidate %s",
                    candidate.code_hash,
                )
                return EvaluationResult(
                    candidate_hash=candidate.code_hash,
                    compile_result=CompileResult(
                        status=CompileStatus.COMPILE_ERROR,
                        error_message="Evaluation infrastructure error",
                    ),
                    outcome=CandidateOutcome.ERROR,
                )

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_eval_one(c)) for c in candidates]

        results = [task.result() for task in tasks]
        return results

    def _update_global_best(
        self,
        improved_results: list[EvaluationResult],
        candidates: list[KernelCandidate],
    ) -> None:
        """Update the global best kernel if any IMPROVED candidate is strictly better.

        Among IMPROVED candidates, selects the one with the lowest latency.
        Only updates if it is strictly less than the current global best
        (or if no global best exists yet).

        Implements: REQ-ORCH-003, SCN-ORCH-003-01
        Invariant: INV-ORCH-001 (monotonically non-increasing)
        """
        if not improved_results:
            return

        # Find the best among improved candidates
        best_result = min(
            improved_results,
            key=lambda r: r.bench_result.latency_us if r.bench_result else float("inf"),
        )

        if best_result.bench_result is None:
            return

        new_latency = best_result.bench_result.latency_us

        # Check if this is strictly better than current best (or first best)
        if self._state.global_best_latency_us is None:
            should_update = True
        else:
            should_update = new_latency < self._state.global_best_latency_us

        if should_update:
            # Find the candidate to get source code
            candidate_by_hash = {c.code_hash: c for c in candidates}
            best_candidate = candidate_by_hash[best_result.candidate_hash]

            self._state.global_best_hash = best_result.candidate_hash
            self._state.global_best_latency_us = new_latency
            self._state.global_best_source = best_candidate.source_code

    def _check_target_met(self) -> bool:
        """Check if the global best latency meets the target within tolerance.

        Implements: REQ-ORCH-001 (termination)

        Returns:
            True if global_best_latency_us <= target_perf_us * (1 + tolerance).
        """
        if self._state.global_best_latency_us is None:
            return False
        threshold = self._problem_spec.target_perf_us * (
            1 + self._problem_spec.tolerance
        )
        return self._state.global_best_latency_us <= threshold
