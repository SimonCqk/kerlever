"""Kerlever types — all shared data types and enums.

Spec: docs/orchestrator/spec.md
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


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


class ProblemSpec(BaseModel):
    """Optimization target specification loaded from YAML.

    Defines the kernel operation, target hardware, performance baseline
    and target, and loop budget.
    """

    op_name: str
    op_semantics: str
    shapes: list[list[int]]
    dtype: str
    target_gpu: str
    baseline_perf_us: float
    target_perf_us: float
    tolerance: float
    max_rounds: int
    reference_kernel: str


class KernelCandidate(BaseModel):
    """A kernel candidate produced by the Coding Agent."""

    code_hash: str
    source_code: str
    intent_tag: str
    parent_hash: str | None = None
    mode: Mode
    sub_mode: SubMode | None = None


class CompileResult(BaseModel):
    """Result of compiling a kernel candidate."""

    status: CompileStatus
    error_message: str | None = None
    register_count: int | None = None
    smem_bytes: int | None = None


class BenchResult(BaseModel):
    """Benchmark result for a successfully compiled kernel."""

    latency_us: float
    p50_us: float
    p95_us: float


class ProfileResult(BaseModel):
    """Profiling result with bottleneck tags and metrics."""

    bottleneck_tags: list[str]
    metrics: dict[str, float]


class EvaluationResult(BaseModel):
    """Complete evaluation result for a kernel candidate."""

    candidate_hash: str
    compile_result: CompileResult
    bench_result: BenchResult | None = None
    profile_result: ProfileResult | None = None
    outcome: CandidateOutcome


class StrategyDirective(BaseModel):
    """Directive from Strategy Navigator to Coding Agent.

    The five optional fields (sub_mode through hard_constraints) were added
    for the Strategy Navigator module and are backward-compatible — existing
    code that constructs StrategyDirective without them continues to work.

    Implements: REQ-NAV-008, REQ-NAV-009
    """

    mode: Mode
    direction: str
    reason: str
    base_kernel_hash: str | None = None
    num_candidates: int
    tabu: list[str]
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


class RoundState(BaseModel):
    """Complete state for a single optimization round."""

    round_number: int
    phase: Phase
    directive: StrategyDirective
    candidates: list[KernelCandidate]
    evaluation_results: list[EvaluationResult]
    cross_analysis: CrossCandidateAnalysis | None = None
    best_candidate_hash: str | None = None
    best_latency_us: float | None = None


class RoundSummary(BaseModel):
    """Compressed round summary for Strategy Navigator consumption."""

    round_number: int
    mode: Mode
    direction: str
    num_candidates: int
    num_improved: int
    best_latency_us: float | None = None
    improvement_over_prev_best: float | None = None


class OptimizationState(BaseModel):
    """Full accumulated optimization state across rounds."""

    problem_spec: ProblemSpec
    current_round: int = 0
    global_best_hash: str | None = None
    global_best_latency_us: float | None = None
    global_best_source: str | None = None
    rounds: list[RoundSummary] = []
    tabu_list: list[str] = []
    bottleneck_history: list[list[str]] = []
    decision_log: list[dict[str, object]] = []


class OptimizationResult(BaseModel):
    """Final result of the optimization loop."""

    status: str
    best_kernel_hash: str | None = None
    best_latency_us: float | None = None
    best_kernel_source: str | None = None
    total_rounds: int = 0
    total_candidates_evaluated: int = 0
