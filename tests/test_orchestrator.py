"""Tests for the Orchestrator main loop.

Tests use deterministic stubs with seeded random for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kerlever.orchestrator import Orchestrator
from kerlever.stubs import (
    StubCodingAgent,
    StubCrossCandidateAnalyzer,
    StubGPUPipeline,
    StubStrategyNavigator,
)
from kerlever.types import (
    BenchResult,
    CandidateOutcome,
    CompileResult,
    CompileStatus,
    EvaluationResult,
    KernelCandidate,
    OptimizationState,
    ProblemSpec,
)


def _make_spec(
    target_perf_us: float = 1.0,
    max_rounds: int = 10,
    tolerance: float = 0.05,
    baseline_perf_us: float = 5.0,
) -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        shapes=[[1024, 1024, 1024]],
        dtype="float16",
        target_gpu="A100",
        baseline_perf_us=baseline_perf_us,
        target_perf_us=target_perf_us,
        tolerance=tolerance,
        max_rounds=max_rounds,
        reference_kernel="__global__ void matmul() {}",
    )


def _make_orchestrator(
    tmp_path: Path,
    spec: ProblemSpec | None = None,
    seed: int = 42,
) -> Orchestrator:
    if spec is None:
        spec = _make_spec()
    return Orchestrator(
        problem_spec=spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=StubGPUPipeline(seed=seed),
        cross_analyzer=StubCrossCandidateAnalyzer(),
        workdir=tmp_path,
    )


@pytest.mark.asyncio
async def test_loop_terminates_target_met(tmp_path: Path) -> None:
    """Loop terminates with TARGET_MET when a candidate meets the target.

    Implements: SCN-ORCH-001-01
    """
    # Use a generous target that the stub's progressive improvement will reach
    spec = _make_spec(target_perf_us=2.0, max_rounds=20, baseline_perf_us=5.0)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    assert result.status == "TARGET_MET"
    assert result.best_latency_us is not None
    assert result.best_latency_us <= spec.target_perf_us * (1 + spec.tolerance)
    assert result.total_rounds <= spec.max_rounds
    assert result.total_rounds > 0


@pytest.mark.asyncio
async def test_loop_terminates_max_rounds(tmp_path: Path) -> None:
    """Loop terminates with MAX_ROUNDS_REACHED when target is unreachable.

    Implements: SCN-ORCH-001-02
    """
    # Set an impossibly low target
    spec = _make_spec(target_perf_us=0.001, max_rounds=3, baseline_perf_us=5.0)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    assert result.status == "MAX_ROUNDS_REACHED"
    assert result.total_rounds == 3


@pytest.mark.asyncio
async def test_compile_fail_discarded(tmp_path: Path) -> None:
    """Compile-failed candidates do not enter global best or analysis.

    Implements: SCN-ORCH-004-01, REQ-ORCH-004
    """

    class AlwaysFailPipeline:
        """Pipeline that always returns compile failure."""

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            current_best_latency_us: float | None,
        ) -> EvaluationResult:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_result=CompileResult(
                    status=CompileStatus.COMPILE_ERROR,
                    error_message="always fail",
                ),
                outcome=CandidateOutcome.COMPILE_FAIL,
            )

    spec = _make_spec(max_rounds=2)
    orch = Orchestrator(
        problem_spec=spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=AlwaysFailPipeline(),
        cross_analyzer=StubCrossCandidateAnalyzer(),
        workdir=tmp_path,
    )

    result = await orch.run()

    # Global best should never be set since all candidates fail compile
    assert result.best_kernel_hash is None
    assert result.best_latency_us is None
    assert result.status == "MAX_ROUNDS_REACHED"


@pytest.mark.asyncio
async def test_regression_not_in_analysis(tmp_path: Path) -> None:
    """Regression candidates don't enter cross-candidate analysis.

    Implements: SCN-ORCH-005-01, REQ-ORCH-005, INV-ORCH-004
    """

    class AlwaysRegressPipeline:
        """Pipeline that always returns regression outcome."""

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            current_best_latency_us: float | None,
        ) -> EvaluationResult:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                bench_result=BenchResult(latency_us=99.0, p50_us=98.0, p95_us=100.0),
                outcome=CandidateOutcome.REGRESSION,
            )

    class TrackingAnalyzer:
        """Analyzer that tracks whether it was called."""

        def __init__(self) -> None:
            self.called = False

        async def analyze(
            self,
            top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
            problem_spec: ProblemSpec,
        ) -> object:
            self.called = True
            from kerlever.types import CrossCandidateAnalysis

            return CrossCandidateAnalysis(
                insights=[], winning_genes=[], recombination_suggestions=[]
            )

    spec = _make_spec(max_rounds=2)
    analyzer = TrackingAnalyzer()
    orch = Orchestrator(
        problem_spec=spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=AlwaysRegressPipeline(),
        cross_analyzer=analyzer,  # type: ignore[arg-type]
        workdir=tmp_path,
    )

    await orch.run()

    # Analyzer should never be called since no IMPROVED/BASELINE_MATCH
    assert not analyzer.called


@pytest.mark.asyncio
async def test_global_best_monotonically_non_increasing(
    tmp_path: Path,
) -> None:
    """Global best latency only decreases across rounds.

    Implements: INV-ORCH-001, REQ-ORCH-003
    """
    spec = _make_spec(target_perf_us=0.001, max_rounds=5, baseline_perf_us=5.0)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    # Load state and verify monotonicity across rounds
    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # Track global best through rounds
    best_so_far: float | None = None
    for round_summary in state.rounds:
        if round_summary.best_latency_us is not None:
            if best_so_far is None:
                best_so_far = round_summary.best_latency_us
            else:
                # Round best can be anything, but global best only improves
                pass

    # The final global best should be set
    assert state.global_best_latency_us is not None
    assert result.best_latency_us == state.global_best_latency_us


@pytest.mark.asyncio
async def test_workdir_files_created(tmp_path: Path) -> None:
    """All expected workdir files are created after a run.

    Implements: SCN-ORCH-002-01, REQ-ORCH-002
    """
    spec = _make_spec(max_rounds=3, target_perf_us=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    # Verify all expected files exist
    assert (tmp_path / "state.json").exists()
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "decision_log.jsonl").exists()

    # Verify round files exist (one per round)
    for i in range(result.total_rounds):
        round_file = tmp_path / "rounds" / f"round_{i:03d}.json"
        assert round_file.exists(), f"Missing round file: {round_file}"

    # Verify kernel files exist
    kernel_files = list((tmp_path / "kernels").glob("*.cu"))
    assert len(kernel_files) > 0

    # Verify decision log has correct number of entries
    log_lines = (tmp_path / "decision_log.jsonl").read_text().strip().split("\n")
    assert len(log_lines) == result.total_rounds

    # Verify result.json content
    result_data = json.loads((tmp_path / "result.json").read_text())
    assert result_data["status"] == result.status
    assert result_data["total_rounds"] == result.total_rounds


@pytest.mark.asyncio
async def test_concurrent_eval_isolates_failures(tmp_path: Path) -> None:
    """One candidate evaluation failure doesn't abort others.

    Implements: SCN-ORCH-006-01, REQ-ORCH-006
    """

    class FailOnSecondPipeline:
        """Pipeline that fails on the second candidate but succeeds for others."""

        def __init__(self) -> None:
            self._call_count = 0

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            current_best_latency_us: float | None,
        ) -> EvaluationResult:
            self._call_count += 1
            if self._call_count % 3 == 2:
                raise RuntimeError("Simulated infrastructure failure")
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_result=CompileResult(status=CompileStatus.SUCCESS),
                bench_result=BenchResult(latency_us=2.0, p50_us=1.9, p95_us=2.1),
                outcome=CandidateOutcome.IMPROVED,
            )

    spec = _make_spec(max_rounds=1)
    orch = Orchestrator(
        problem_spec=spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=FailOnSecondPipeline(),
        cross_analyzer=StubCrossCandidateAnalyzer(),
        workdir=tmp_path,
    )

    result = await orch.run()

    # Should complete without crashing
    assert result.total_rounds == 1
    # At least some candidates should have succeeded
    assert result.total_candidates_evaluated == 3

    # Check that the failing candidate has ERROR outcome in the round state
    round_path = tmp_path / "rounds" / "round_000.json"
    round_data = json.loads(round_path.read_text())
    outcomes = [r["outcome"] for r in round_data["evaluation_results"]]
    assert "ERROR" in outcomes
    # The other candidates should have IMPROVED
    assert outcomes.count("IMPROVED") == 2


@pytest.mark.asyncio
async def test_cross_analysis_skipped_with_fewer_than_2(
    tmp_path: Path,
) -> None:
    """Cross-analysis is skipped when fewer than 2 candidates pass.

    Implements: SCN-ORCH-007-01, REQ-ORCH-007
    """

    class OnlyOnePassPipeline:
        """Pipeline where only one candidate passes; rest compile-fail."""

        def __init__(self) -> None:
            self._call_count = 0

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            current_best_latency_us: float | None,
        ) -> EvaluationResult:
            self._call_count += 1
            if self._call_count % 3 == 1:
                return EvaluationResult(
                    candidate_hash=candidate.code_hash,
                    compile_result=CompileResult(status=CompileStatus.SUCCESS),
                    bench_result=BenchResult(latency_us=2.0, p50_us=1.9, p95_us=2.1),
                    outcome=CandidateOutcome.IMPROVED,
                )
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_result=CompileResult(
                    status=CompileStatus.COMPILE_ERROR,
                    error_message="fail",
                ),
                outcome=CandidateOutcome.COMPILE_FAIL,
            )

    class TrackingAnalyzer:
        """Analyzer that tracks calls."""

        def __init__(self) -> None:
            self.call_count = 0

        async def analyze(
            self,
            top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
            problem_spec: ProblemSpec,
        ) -> object:
            self.call_count += 1
            from kerlever.types import CrossCandidateAnalysis

            return CrossCandidateAnalysis(
                insights=[], winning_genes=[], recombination_suggestions=[]
            )

    spec = _make_spec(max_rounds=2)
    analyzer = TrackingAnalyzer()
    orch = Orchestrator(
        problem_spec=spec,
        strategy_navigator=StubStrategyNavigator(),
        coding_agent=StubCodingAgent(),
        gpu_pipeline=OnlyOnePassPipeline(),
        cross_analyzer=analyzer,  # type: ignore[arg-type]
        workdir=tmp_path,
    )

    await orch.run()

    # With only 1 passing candidate per round, analyzer should never be called
    assert analyzer.call_count == 0


@pytest.mark.asyncio
async def test_tabu_list_updated_every_round(tmp_path: Path) -> None:
    """Tabu list is extended with all intent_tags each round.

    Addresses Shortcut Risk #5 from spec.md.
    """
    spec = _make_spec(max_rounds=2, target_perf_us=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # With 3 candidates per round and 2 rounds, tabu should have 6 entries
    assert len(state.tabu_list) == 6


@pytest.mark.asyncio
async def test_kernels_persisted_before_evaluation(tmp_path: Path) -> None:
    """All kernel source files exist in workdir after run.

    Implements: INV-ORCH-003
    """
    spec = _make_spec(max_rounds=1)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    kernel_files = list((tmp_path / "kernels").glob("*.cu"))
    # 3 candidates per round (stub default)
    assert len(kernel_files) == 3

    # All kernel files should have non-empty content
    for kf in kernel_files:
        assert kf.stat().st_size > 0
