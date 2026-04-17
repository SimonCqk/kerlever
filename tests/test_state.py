"""Tests for StateManager workdir I/O and atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

from kerlever.state import StateManager
from kerlever.types import (
    BaselineArtifact,
    BenchmarkBundle,
    CandidateIntent,
    CandidateOutcome,
    CompileStatus,
    EvaluationResult,
    KernelCandidate,
    Mode,
    ObjectiveScore,
    OptimizationResult,
    OptimizationState,
    PerformanceObjective,
    Phase,
    ProblemSpec,
    RoundState,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
    StrategyDirective,
    SubMode,
)


def _make_problem_spec() -> ProblemSpec:
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
            ),
        ],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
        ),
        target_metric_value=1.0,
        max_rounds=10,
        reference_kernel="__global__ void matmul() {}",
    )


def _make_baseline() -> BaselineArtifact:
    return BaselineArtifact(
        kernel_hash="baseline_hash",
        source_code="__global__ void matmul() {}",
        compile_artifact=StaticAnalysis(),
        benchmark_results=[
            ShapeBenchResult(
                shape_id="default",
                latency_p50_us=5.0,
                latency_p95_us=6.0,
                run_count=10,
            ),
        ],
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=5.0,
            relative_to_baseline=1.0,
            relative_to_incumbent=1.0,
        ),
    )


def _make_directive() -> StrategyDirective:
    return StrategyDirective(
        mode=Mode.EXPLOIT,
        direction="optimize_memory",
        reason="test",
        num_candidates=2,
        tabu=[],
    )


def test_init_creates_workdir(tmp_path: Path) -> None:
    """StateManager creates workdir and subdirectories on init."""
    workdir = tmp_path / "testwork"
    StateManager(workdir)

    assert workdir.exists()
    assert (workdir / "rounds").is_dir()
    assert (workdir / "kernels").is_dir()


def test_save_and_load_state(tmp_path: Path) -> None:
    """State round-trip: save then load produces identical data."""
    mgr = StateManager(tmp_path)
    spec = _make_problem_spec()
    baseline = _make_baseline()
    incumbent = BaselineArtifact(
        kernel_hash="better_hash",
        source_code="__global__ void better() {}",
        compile_artifact=StaticAnalysis(),
        benchmark_results=[
            ShapeBenchResult(
                shape_id="default",
                latency_p50_us=2.5,
                latency_p95_us=3.0,
                run_count=10,
            ),
        ],
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=2.5,
            relative_to_baseline=0.5,
            relative_to_incumbent=1.0,
        ),
    )
    state = OptimizationState(
        problem_spec=spec,
        baseline=baseline,
        incumbent=incumbent,
        current_round=3,
    )

    mgr.save_state(state)
    loaded = mgr.load_state()

    assert loaded is not None
    assert loaded.current_round == 3
    assert loaded.incumbent.kernel_hash == "better_hash"
    assert loaded.incumbent.objective_score.value == 2.5
    assert loaded.baseline.kernel_hash == "baseline_hash"
    assert loaded.problem_spec.op_name == "matmul"


def test_load_state_returns_none_when_missing(tmp_path: Path) -> None:
    """load_state returns None when state.json does not exist."""
    mgr = StateManager(tmp_path)
    assert mgr.load_state() is None


def test_save_and_load_round(tmp_path: Path) -> None:
    """Round state round-trip: save then load produces identical data."""
    mgr = StateManager(tmp_path)
    directive = _make_directive()
    candidate = KernelCandidate(
        code_hash="hash1",
        source_code="__global__ void k() {}",
        parent_hashes=["parent_hash"],
        intent=CandidateIntent(
            direction="optimize_memory",
            mode=Mode.EXPLOIT,
            sub_mode=SubMode.LOCAL_REWRITE,
        ),
    )
    eval_result = EvaluationResult(
        candidate_hash="hash1",
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
                relative_to_incumbent=0.8,
            ),
            regressed_vs_incumbent=False,
        ),
        outcome=CandidateOutcome.IMPROVED,
    )
    round_state = RoundState(
        round_number=0,
        phase=Phase.ROUND_COMPLETE,
        directive=directive,
        candidates=[candidate],
        evaluation_results=[eval_result],
        best_candidate_hash="hash1",
        best_objective_score=2.0,
    )

    mgr.save_round(round_state)
    loaded = mgr.load_round(0)

    assert loaded is not None
    assert loaded.round_number == 0
    assert loaded.phase == Phase.ROUND_COMPLETE
    assert len(loaded.candidates) == 1
    assert loaded.candidates[0].code_hash == "hash1"
    assert loaded.best_objective_score == 2.0


def test_load_round_returns_none_when_missing(tmp_path: Path) -> None:
    """load_round returns None for nonexistent round."""
    mgr = StateManager(tmp_path)
    assert mgr.load_round(99) is None


def test_save_kernel(tmp_path: Path) -> None:
    """Kernel source is written to kernels/<hash>.cu."""
    mgr = StateManager(tmp_path)
    mgr.save_kernel("abc123", "__global__ void k() {}")

    kernel_file = tmp_path / "kernels" / "abc123.cu"
    assert kernel_file.exists()
    assert kernel_file.read_text() == "__global__ void k() {}"


def test_load_kernel_rejects_unsafe_hash_input(tmp_path: Path) -> None:
    """load_kernel refuses path traversal and non-hash-like input."""
    mgr = StateManager(tmp_path)
    (tmp_path / "secret.cu").write_text("secret", encoding="utf-8")

    assert mgr.load_kernel("../secret") is None
    assert mgr.load_kernel("not a hash") is None
    assert mgr.load_kernel(" hash_A") is None


def test_load_kernel_accepts_simple_legacy_hash_like_values(tmp_path: Path) -> None:
    """load_kernel keeps backward-compatible hash-like names."""
    mgr = StateManager(tmp_path)
    source = "__global__ void k() {}"
    mgr.save_kernel("hash_A", source)

    assert mgr.load_kernel("hash_A") == source


def test_append_decision(tmp_path: Path) -> None:
    """Decision log entries are appended as JSONL."""
    mgr = StateManager(tmp_path)
    entry1: dict[str, object] = {"round_number": 0, "status": "ok"}
    entry2: dict[str, object] = {"round_number": 1, "status": "improved"}

    mgr.append_decision(entry1)
    mgr.append_decision(entry2)

    log_path = tmp_path / "decision_log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["round_number"] == 0
    assert json.loads(lines[1])["round_number"] == 1


def test_save_result(tmp_path: Path) -> None:
    """Final result is written to result.json."""
    mgr = StateManager(tmp_path)
    result = OptimizationResult(
        status="TARGET_MET",
        best_kernel_hash="best_hash",
        best_objective_score=0.9,
        total_rounds=5,
        total_candidates_evaluated=15,
    )

    mgr.save_result(result)

    result_path = tmp_path / "result.json"
    assert result_path.exists()
    data = json.loads(result_path.read_text())
    assert data["status"] == "TARGET_MET"
    assert data["best_kernel_hash"] == "best_hash"
    assert data["best_objective_score"] == 0.9


def test_atomic_write_no_tmp_left(tmp_path: Path) -> None:
    """After atomic write, no .tmp files remain."""
    mgr = StateManager(tmp_path)
    spec = _make_problem_spec()
    baseline = _make_baseline()
    state = OptimizationState(
        problem_spec=spec,
        baseline=baseline,
        incumbent=baseline,
    )

    mgr.save_state(state)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
