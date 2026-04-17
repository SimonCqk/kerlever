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
    BaselineArtifact,
    BenchmarkBundle,
    CandidateOutcome,
    CompileStatus,
    CorrectnessResult,
    CrossCandidateAnalysis,
    EvaluationResult,
    KernelCandidate,
    Mode,
    ObjectiveScore,
    OptimizationState,
    PerformanceObjective,
    ProblemSpec,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    StrategyDirective,
    SubMode,
)


def _make_spec(
    target_metric_value: float = 1.0,
    max_rounds: int = 10,
) -> ProblemSpec:
    return ProblemSpec(
        op_name="matmul",
        op_semantics="C = A @ B",
        dtype="float16",
        target_gpu="A100",
        shape_cases=[
            ShapeCase(
                shape_id="default",
                dims=[1024, 1024, 1024],
                weight=1.0,
                profile=True,
            ),
        ],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
            regression_guard_pct=0.0,
        ),
        target_metric_value=target_metric_value,
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
    # Baseline synthetic score is target_metric_value * 5.0 = 10.0
    # With progressive improvement, stubs should reach 2.0 within 20 rounds
    spec = _make_spec(target_metric_value=2.0, max_rounds=20)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    assert result.status == "TARGET_MET"
    assert result.best_objective_score is not None
    assert result.best_objective_score <= spec.target_metric_value
    assert result.total_rounds <= spec.max_rounds
    assert result.total_rounds > 0


@pytest.mark.asyncio
async def test_loop_terminates_max_rounds(tmp_path: Path) -> None:
    """Loop terminates with MAX_ROUNDS_REACHED when target is unreachable.

    Implements: SCN-ORCH-001-02
    """
    # Set an impossibly low target
    spec = _make_spec(target_metric_value=0.001, max_rounds=3)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    result = await orch.run()

    assert result.status == "MAX_ROUNDS_REACHED"
    assert result.total_rounds == 3


@pytest.mark.asyncio
async def test_compile_fail_discarded(tmp_path: Path) -> None:
    """Compile-failed candidates do not enter incumbent or analysis.

    Implements: SCN-ORCH-004-01, REQ-ORCH-004
    """

    class AlwaysFailPipeline:
        """Pipeline that always returns compile failure."""

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            baseline: BaselineArtifact,
            incumbent: BaselineArtifact,
        ) -> EvaluationResult:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_status=CompileStatus.COMPILE_ERROR,
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

    # Incumbent should still be the baseline (never updated)
    assert result.best_kernel_hash is not None  # baseline hash
    assert result.status == "MAX_ROUNDS_REACHED"

    # Verify incumbent was never updated beyond baseline
    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())
    assert state.incumbent.kernel_hash == state.baseline.kernel_hash


@pytest.mark.asyncio
async def test_regression_enters_analysis_as_negative_evidence(tmp_path: Path) -> None:
    """Benchmarked regressions enter analysis but never update incumbent.

    Implements: SCN-ORCH-007-02, REQ-ORCH-007
    """

    class AlwaysRegressPipeline:
        """Pipeline that always returns regression outcome."""

        async def evaluate(
            self,
            candidate: KernelCandidate,
            problem_spec: ProblemSpec,
            baseline: BaselineArtifact,
            incumbent: BaselineArtifact,
        ) -> EvaluationResult:
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_status=CompileStatus.SUCCESS,
                static_analysis=StaticAnalysis(),
                correctness=CorrectnessResult(passed=True),
                benchmark=BenchmarkBundle(
                    shape_results=[
                        ShapeBenchResult(
                            shape_id="default",
                            latency_p50_us=99.0,
                            latency_p95_us=100.0,
                            run_count=10,
                        ),
                    ],
                    objective_score=ObjectiveScore(
                        metric_name="weighted_p50_us",
                        value=99.0,
                        relative_to_baseline=2.0,
                        relative_to_incumbent=2.0,
                    ),
                    regressed_vs_incumbent=True,
                ),
                outcome=CandidateOutcome.REGRESSION,
            )

    class TrackingAnalyzer:
        """Analyzer that tracks whether it was called."""

        def __init__(self) -> None:
            self.called = False
            self.seen_outcomes: list[CandidateOutcome] = []

        async def analyze(
            self,
            top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
            problem_spec: ProblemSpec,
        ) -> CrossCandidateAnalysis:
            self.called = True
            self.seen_outcomes = [result.outcome for _, result in top_k_results]
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
        cross_analyzer=analyzer,
        workdir=tmp_path,
    )

    await orch.run()

    assert analyzer.called
    assert analyzer.seen_outcomes
    assert set(analyzer.seen_outcomes) == {CandidateOutcome.REGRESSION}

    state = OptimizationState.model_validate_json((tmp_path / "state.json").read_text())
    assert state.incumbent.kernel_hash == state.baseline.kernel_hash


@pytest.mark.asyncio
async def test_incumbent_monotonically_non_increasing(
    tmp_path: Path,
) -> None:
    """Incumbent objective score only decreases across rounds.

    Implements: INV-ORCH-001, REQ-ORCH-003
    """
    spec = _make_spec(target_metric_value=0.001, max_rounds=5)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    # Load state and verify incumbent score decreased or stayed same
    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # The incumbent should have a valid objective score
    assert state.incumbent.objective_score.value is not None

    # Verify that round summaries show non-negative gains (or None)
    for rs in state.rounds:
        if rs.abs_gain_vs_prev_best_us is not None:
            assert rs.abs_gain_vs_prev_best_us >= 0.0
        if rs.rel_gain_vs_prev_best is not None:
            assert rs.rel_gain_vs_prev_best >= 0.0


@pytest.mark.asyncio
async def test_workdir_files_created(tmp_path: Path) -> None:
    """All expected workdir files are created after a run.

    Implements: SCN-ORCH-002-01, REQ-ORCH-002
    """
    spec = _make_spec(max_rounds=3, target_metric_value=0.001)
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
            baseline: BaselineArtifact,
            incumbent: BaselineArtifact,
        ) -> EvaluationResult:
            self._call_count += 1
            if self._call_count % 3 == 2:
                raise RuntimeError("Simulated infrastructure failure")
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_status=CompileStatus.SUCCESS,
                static_analysis=StaticAnalysis(),
                benchmark=BenchmarkBundle(
                    shape_results=[
                        ShapeBenchResult(
                            shape_id="default",
                            latency_p50_us=2.0,
                            latency_p95_us=2.1,
                            run_count=10,
                        ),
                    ],
                    objective_score=ObjectiveScore(
                        metric_name="weighted_p50_us",
                        value=2.0,
                        relative_to_baseline=0.4,
                        relative_to_incumbent=0.4,
                    ),
                    regressed_vs_incumbent=False,
                ),
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
            baseline: BaselineArtifact,
            incumbent: BaselineArtifact,
        ) -> EvaluationResult:
            self._call_count += 1
            if self._call_count % 3 == 1:
                return EvaluationResult(
                    candidate_hash=candidate.code_hash,
                    compile_status=CompileStatus.SUCCESS,
                    static_analysis=StaticAnalysis(),
                    benchmark=BenchmarkBundle(
                        shape_results=[
                            ShapeBenchResult(
                                shape_id="default",
                                latency_p50_us=2.0,
                                latency_p95_us=2.1,
                                run_count=10,
                            ),
                        ],
                        objective_score=ObjectiveScore(
                            metric_name="weighted_p50_us",
                            value=2.0,
                            relative_to_baseline=0.4,
                            relative_to_incumbent=0.4,
                        ),
                        regressed_vs_incumbent=False,
                    ),
                    outcome=CandidateOutcome.IMPROVED,
                )
            return EvaluationResult(
                candidate_hash=candidate.code_hash,
                compile_status=CompileStatus.COMPILE_ERROR,
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
        ) -> CrossCandidateAnalysis:
            self.call_count += 1
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
        cross_analyzer=analyzer,
        workdir=tmp_path,
    )

    await orch.run()

    # With only 1 passing candidate per round, analyzer should never be called
    assert analyzer.call_count == 0


@pytest.mark.asyncio
async def test_attempt_records_created(tmp_path: Path) -> None:
    """AttemptRecords are created for every candidate in every round.

    Implements: SCN-ORCH-010-01, INV-ORCH-007
    """
    spec = _make_spec(max_rounds=2, target_metric_value=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # With 3 candidates per round and 2 rounds, should have 6 attempt records
    assert len(state.attempts) == 6

    # Each attempt record should have required fields
    for attempt in state.attempts:
        assert attempt.round_number >= 0
        assert attempt.candidate_hash != ""
        assert attempt.direction != ""
        assert attempt.outcome in CandidateOutcome


@pytest.mark.asyncio
async def test_tabu_entries_created(tmp_path: Path) -> None:
    """TabuEntry entries are created for non-improving directions.

    Implements: SCN-ORCH-010-01, SCN-ORCH-010-02
    """
    spec = _make_spec(max_rounds=2, target_metric_value=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # Tabu entries should be created for non-IMPROVED candidates
    assert len(state.tabu_entries) > 0

    # Each tabu entry should have an expiry round
    for entry in state.tabu_entries:
        assert entry.expires_after_round > entry.round_number
        assert entry.direction != ""


@pytest.mark.asyncio
async def test_round_summary_has_gain_fields(tmp_path: Path) -> None:
    """RoundSummary contains both absolute and relative gains.

    Implements: SCN-ORCH-012-01
    """
    spec = _make_spec(target_metric_value=0.001, max_rounds=5)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # At least one round should show improvement
    any_improvement = False
    for rs in state.rounds:
        if rs.abs_gain_vs_prev_best_us is not None:
            any_improvement = True
            assert rs.abs_gain_vs_prev_best_us > 0.0
            assert rs.rel_gain_vs_prev_best is not None
            assert rs.rel_gain_vs_prev_best > 0.0
            assert rs.rel_gain_vs_prev_best < 1.0  # relative gain < 100%

    assert any_improvement, "Expected at least one improving round"


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


@pytest.mark.asyncio
async def test_recombination_parent_sources_are_hydrated(tmp_path: Path) -> None:
    """Orchestrator hydrates available parent sources before Coding Agent.

    Implements: SCN-ORCH-013-01, SCN-ORCH-013-02
    """

    class RecombinationNavigator:
        """Navigator that requests recombination with available and missing parents."""

        async def decide(
            self,
            problem_spec: ProblemSpec,
            optimization_state: OptimizationState,
            round_summary: object,
            cross_analysis: CrossCandidateAnalysis | None,
        ) -> StrategyDirective:
            return StrategyDirective(
                mode=Mode.EXPLORE,
                direction="recombine",
                reason="test",
                num_candidates=0,
                tabu=[],
                sub_mode=SubMode.RECOMBINATION,
                parent_candidates=[
                    optimization_state.incumbent.kernel_hash,
                    "persisted_hash",
                    "missing_hash",
                ],
                gene_map={
                    "memory_access": optimization_state.incumbent.kernel_hash,
                    "compute_loop": "persisted_hash",
                },
            )

    class RecordingCodingAgent:
        """Coding Agent that records the hydrated directive."""

        def __init__(self) -> None:
            self.directive: StrategyDirective | None = None

        async def generate(
            self,
            problem_spec: ProblemSpec,
            directive: StrategyDirective,
            incumbent: BaselineArtifact,
        ) -> list[KernelCandidate]:
            self.directive = directive
            return []

    spec = _make_spec(max_rounds=1, target_metric_value=0.001)
    coder = RecordingCodingAgent()
    orch = Orchestrator(
        problem_spec=spec,
        strategy_navigator=RecombinationNavigator(),
        coding_agent=coder,
        gpu_pipeline=StubGPUPipeline(),
        cross_analyzer=StubCrossCandidateAnalyzer(),
        workdir=tmp_path,
    )
    persisted_source = "__global__ void persisted() {}"
    (tmp_path / "kernels" / "persisted_hash.cu").write_text(persisted_source)

    await orch.run()

    assert coder.directive is not None
    assert coder.directive.parent_sources is not None
    assert coder.directive.parent_sources["persisted_hash"] == persisted_source
    assert "missing_hash" not in coder.directive.parent_sources
    assert spec.reference_kernel in coder.directive.parent_sources.values()


@pytest.mark.asyncio
async def test_bottleneck_history_populated(tmp_path: Path) -> None:
    """Bottleneck history is populated from profiled candidates.

    Addresses spec 6.6: BottleneckAssessment from ProfileBundle of best
    profiled candidate is appended to bottleneck_history.
    """
    spec = _make_spec(max_rounds=3, target_metric_value=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # Bottleneck history should have entries (stubs produce profile data)
    assert len(state.bottleneck_history) > 0
    for assessment in state.bottleneck_history:
        assert len(assessment.tags) > 0
        assert len(assessment.evidence) > 0


@pytest.mark.asyncio
async def test_baseline_never_changes(tmp_path: Path) -> None:
    """Baseline artifact is never modified during the optimization loop.

    Implements: INV-ORCH-006
    """
    spec = _make_spec(max_rounds=3, target_metric_value=0.001)
    orch = _make_orchestrator(tmp_path, spec=spec, seed=42)

    await orch.run()

    state_path = tmp_path / "state.json"
    state = OptimizationState.model_validate_json(state_path.read_text())

    # Baseline should still match the reference kernel
    assert state.baseline.source_code == spec.reference_kernel
    assert state.baseline.kernel_hash is not None
