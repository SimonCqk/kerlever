"""Kerlever types — all shared data types and enums.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Phase(StrEnum):
    """Orchestrator state machine phase within a round."""

    AWAITING_STRATEGY = "AWAITING_STRATEGY"
    AWAITING_CODING = "AWAITING_CODING"
    AWAITING_EVALUATION = "AWAITING_EVALUATION"
    ANALYSIS = "ANALYSIS"
    ROUND_COMPLETE = "ROUND_COMPLETE"


class Mode(StrEnum):
    """Strategy mode: exploit current best or explore new directions."""

    EXPLOIT = "EXPLOIT"
    EXPLORE = "EXPLORE"


class SubMode(StrEnum):
    """Sub-mode for finer-grained strategy classification."""

    PARAM_SEARCH = "PARAM_SEARCH"
    LOCAL_REWRITE = "LOCAL_REWRITE"
    PATTERN_APPLY = "PATTERN_APPLY"
    DE_NOVO = "DE_NOVO"
    RECOMBINATION = "RECOMBINATION"


class CompileStatus(StrEnum):
    """Compilation outcome status."""

    SUCCESS = "SUCCESS"
    COMPILE_ERROR = "COMPILE_ERROR"
    CORRECTNESS_FAIL = "CORRECTNESS_FAIL"


class CandidateOutcome(StrEnum):
    """Overall outcome for a candidate after evaluation.

    Implements: REQ-ORCH-003, REQ-ORCH-004, REQ-ORCH-005
    """

    IMPROVED = "IMPROVED"
    BASELINE_MATCH = "BASELINE_MATCH"
    REGRESSION = "REGRESSION"
    COMPILE_FAIL = "COMPILE_FAIL"
    CORRECTNESS_FAIL = "CORRECTNESS_FAIL"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Problem Specification Types
# ---------------------------------------------------------------------------


class ShapeCase(BaseModel):
    """A single workload shape with weight and profiling designation.

    Each shape case defines a specific set of dimensions for benchmarking,
    along with metadata controlling its importance in objective aggregation
    and whether it should be deep-profiled.
    """

    shape_id: str
    dims: list[int]
    weight: float = 1.0
    correctness_tolerance: float | None = None
    profile: bool = False


class PerformanceObjective(BaseModel):
    """Defines how candidates are scored and ranked.

    Controls which latency metric is optimized, how per-shape results
    are aggregated, and the regression detection threshold.
    """

    primary_metric: Literal["weighted_p50_us", "weighted_p95_us", "worst_case_p50_us"]
    aggregation: Literal["weighted_mean", "max"]
    regression_guard_pct: float = 0.0


class ProblemSpec(BaseModel):
    """Optimization target specification loaded from YAML.

    Defines the kernel operation, target hardware, workload shapes,
    performance objective, and loop budget.

    Implements: REQ-ORCH-008
    """

    op_name: str
    op_semantics: str
    dtype: str
    target_gpu: str
    shape_cases: list[ShapeCase]
    objective: PerformanceObjective
    target_metric_value: float
    max_rounds: int
    reference_kernel: str


# ---------------------------------------------------------------------------
# Evaluation Output Types (defined before Baseline types due to dependencies)
# ---------------------------------------------------------------------------


class StaticAnalysis(BaseModel):
    """Compile-time resource usage analysis.

    Contains register counts, shared memory usage, spill metrics, and
    estimated occupancy. All fields optional since not all compilers
    report all metrics.
    """

    registers_per_thread: int | None = None
    smem_bytes_per_block: int | None = None
    spill_stores: int | None = None
    spill_loads: int | None = None
    occupancy_estimate_pct: float | None = None


class ShapeBenchResult(BaseModel):
    """Benchmark result for a single shape case.

    Contains latency statistics from running the kernel on a specific
    set of dimensions.
    """

    shape_id: str
    latency_p50_us: float
    latency_p95_us: float
    stdev_us: float | None = None
    run_count: int


class ObjectiveScore(BaseModel):
    """Aggregate objective score for a kernel candidate.

    Lower values are better. The relative fields express the score
    as a ratio to the baseline and incumbent scores respectively
    (values < 1.0 indicate improvement).
    """

    metric_name: str
    value: float
    relative_to_baseline: float
    relative_to_incumbent: float


class ProfileMetrics(BaseModel):
    """Raw profiling metrics from GPU hardware counters.

    All fields optional since availability depends on the profiling
    tool and GPU architecture.
    """

    achieved_occupancy_pct: float | None = None
    dram_throughput_pct_of_peak: float | None = None
    sm_throughput_pct_of_peak: float | None = None
    l2_hit_rate_pct: float | None = None
    warp_stall_memory_dependency_pct: float | None = None
    warp_stall_exec_dependency_pct: float | None = None
    tensor_core_utilization_pct: float | None = None
    arithmetic_intensity_flop_per_byte: float | None = None


class BottleneckAssessment(BaseModel):
    """Bottleneck analysis derived from profiling metrics.

    Identifies performance bottlenecks with supporting evidence
    and a trace of which analysis rules fired.
    """

    tags: list[str]
    primary_tag: str | None = None
    evidence: dict[str, float]
    rule_trace: list[str]


class ProfileBundle(BaseModel):
    """Complete profiling output for a single shape case.

    Combines raw metrics with the derived bottleneck assessment.
    """

    shape_id: str
    metrics: ProfileMetrics
    assessment: BottleneckAssessment


class CorrectnessResult(BaseModel):
    """Result of correctness validation against reference output.

    Reports which shapes failed and the magnitude of errors observed.
    """

    passed: bool
    failing_shape_ids: list[str] = []
    max_abs_error: float | None = None
    max_rel_error: float | None = None


class BenchmarkBundle(BaseModel):
    """Complete benchmark output for a kernel candidate.

    Contains per-shape results, the aggregate objective score, and
    whether this candidate regressed against the incumbent.
    """

    shape_results: list[ShapeBenchResult]
    objective_score: ObjectiveScore
    regressed_vs_incumbent: bool


# ---------------------------------------------------------------------------
# Baseline and Incumbent Types
# ---------------------------------------------------------------------------


class BaselineArtifact(BaseModel):
    """Measured (or V1 synthetic) baseline artifact for a kernel.

    Contains the kernel source, compile analysis, benchmark results,
    objective score, and optional profiling data. Used as both the
    immutable baseline and the mutable incumbent in OptimizationState.

    Implements: REQ-ORCH-009
    """

    kernel_hash: str
    source_code: str
    compile_artifact: StaticAnalysis
    benchmark_results: list[ShapeBenchResult]
    objective_score: ObjectiveScore
    profile_bundle: ProfileBundle | None = None


# ---------------------------------------------------------------------------
# Candidate and Search Memory Types
# ---------------------------------------------------------------------------


class CandidateIntent(BaseModel):
    """Structured intent for a kernel candidate.

    Replaces the old flat intent_tag with a rich description of the
    optimization direction, mode, and rationale.
    """

    direction: str
    mode: Mode
    sub_mode: SubMode | None = None
    rationale: str | None = None


class KernelCandidate(BaseModel):
    """A kernel candidate produced by the Coding Agent."""

    code_hash: str
    source_code: str
    parent_hashes: list[str]
    intent: CandidateIntent


class AttemptRecord(BaseModel):
    """Record of a single candidate attempt within a round.

    Stored in OptimizationState.attempts for complete search history.
    """

    round_number: int
    candidate_hash: str
    base_kernel_hash: str | None
    direction: str
    sub_mode: SubMode | None
    outcome: CandidateOutcome
    objective_score: float | None = None


class TabuEntry(BaseModel):
    """Typed tabu entry with expiry information.

    Prevents re-exploring a direction from a given base kernel
    for a specified number of rounds.
    """

    base_kernel_hash: str | None
    direction: str
    sub_mode: SubMode | None
    round_number: int
    expires_after_round: int


# ---------------------------------------------------------------------------
# Evaluation Result
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Complete evaluation result for a kernel candidate.

    Structured with progressive fields: compile_status is always present,
    static_analysis is present if compile succeeded, correctness if compile
    succeeded, benchmark if correctness passed, profile if selected for
    deep profiling.

    Implements: REQ-ORCH-004, REQ-ORCH-005
    """

    candidate_hash: str
    compile_status: CompileStatus
    static_analysis: StaticAnalysis | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkBundle | None = None
    profile: ProfileBundle | None = None
    outcome: CandidateOutcome


# ---------------------------------------------------------------------------
# Strategy and Analysis Types
# ---------------------------------------------------------------------------


class StrategyDirective(BaseModel):
    """Directive from Strategy Navigator to Coding Agent.

    The five optional fields (sub_mode through hard_constraints) were added
    for the Strategy Navigator module and are backward-compatible -- existing
    code that constructs StrategyDirective without them continues to work.

    Implements: REQ-NAV-008, REQ-NAV-009
    """

    mode: Mode
    direction: str
    reason: str
    base_kernel_hash: str | None = None
    num_candidates: int
    tabu: list[TabuEntry]
    sub_mode: SubMode | None = None
    parent_candidates: list[str] | None = None
    gene_map: dict[str, str] | None = None
    search_range: dict[str, list[float]] | None = None
    hard_constraints: list[str] | None = None


class CrossCandidateAnalysis(BaseModel):
    """Output from cross-candidate analysis."""

    insights: list[str]
    winning_genes: list[str]
    recombination_suggestions: list[str]


# ---------------------------------------------------------------------------
# Round and State Types
# ---------------------------------------------------------------------------


class RoundState(BaseModel):
    """Complete state for a single optimization round."""

    round_number: int
    phase: Phase
    directive: StrategyDirective
    candidates: list[KernelCandidate]
    evaluation_results: list[EvaluationResult]
    cross_analysis: CrossCandidateAnalysis | None = None
    best_candidate_hash: str | None = None
    best_objective_score: float | None = None


class RoundSummary(BaseModel):
    """Compressed round summary for Strategy Navigator consumption."""

    round_number: int
    mode: Mode
    direction: str
    num_candidates: int
    num_improved: int
    best_objective_score: float | None = None
    abs_gain_vs_prev_best_us: float | None = None
    rel_gain_vs_prev_best: float | None = None


class OptimizationState(BaseModel):
    """Full accumulated optimization state across rounds.

    The baseline is set during bootstrap and never changed. The incumbent
    is initialized from the baseline and updated when a strictly better
    candidate is found.
    """

    problem_spec: ProblemSpec
    baseline: BaselineArtifact
    incumbent: BaselineArtifact
    current_round: int = 0
    rounds: list[RoundSummary] = []
    attempts: list[AttemptRecord] = []
    tabu_entries: list[TabuEntry] = []
    bottleneck_history: list[BottleneckAssessment] = []
    decision_log: list[dict[str, object]] = []


class OptimizationResult(BaseModel):
    """Final result of the optimization loop."""

    status: str
    best_kernel_hash: str | None = None
    best_objective_score: float | None = None
    best_kernel_source: str | None = None
    total_rounds: int = 0
    total_candidates_evaluated: int = 0
