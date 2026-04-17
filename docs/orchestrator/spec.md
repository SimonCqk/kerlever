# Orchestrator Module Specification

## §1 Overview

The Orchestrator is the global control loop of the Kerlever optimization system. It drives the full kernel optimization cycle: bootstrap a measured baseline, request a strategy, generate kernel candidates, evaluate them on GPU hardware, analyze results, update state, and repeat until the target performance objective is met or the round budget is exhausted.

The Orchestrator does not contain optimization intelligence. It is a state machine that sequences calls to four downstream services (all accessed via async Protocols), manages global optimization state, enforces early-exit rules for failed or regressing candidates, and persists every round's state to a workdir for auditability.

The Orchestrator is also the owner of all shared data types. Every type defined in §5 is part of the Orchestrator's public contract. Downstream modules import these types; they do not define their own.

**V1 scope:** The Orchestrator runs end-to-end with stub implementations of all four downstream Protocols. No real LLM calls, no real GPU evaluation. The goal is to prove the control flow, state management, early-exit paths, and persistence are correct.

**V2 (deferred):** Baseline bootstrap with real GPU pipeline. See REQ-ORCH-009.

---

## §2 Requirements

### Functional Requirements

**REQ-ORCH-001: Loop Termination** [traces SC-1]
The optimization loop must always terminate. It runs until either (a) the incumbent's objective score meets the target (`incumbent.objective_score.value <= target_metric_value`) or (b) the maximum round count is reached. The loop must not hang, deadlock, or crash under normal operation, including when downstream services return errors or empty results.

**REQ-ORCH-002: State Persistence** [traces SC-2]
After each round completes, the Orchestrator must persist the round state to the workdir. Upon loop termination, it must write a final result file. The workdir must contain:
- `state.json` — full optimization state snapshot (overwritten each round)
- `rounds/round_NNN.json` — per-round state (one file per round, zero-padded)
- `kernels/<hash>.cu` — source code for every generated kernel candidate
- `decision_log.jsonl` — append-only log of strategy decisions and round outcomes
- `result.json` — final optimization result (written once at termination)

**REQ-ORCH-003: Global Best Tracking** [traces SC-3]
The Orchestrator must maintain a global best kernel (the "incumbent") across all rounds. After evaluating candidates in a round, if any candidate achieves a better (lower) `objective_score.value` than the current incumbent (and passes compile + correctness), the incumbent must be updated. The incumbent's objective score must be monotonically non-increasing across rounds (a new incumbent is only set when strictly better).

**REQ-ORCH-004: Candidate Discard on Failure** [traces SC-4]
Candidates that fail compilation or correctness checks must be discarded immediately. They must not participate in incumbent comparison, cross-candidate analysis, or any downstream processing. Their outcome must be recorded as COMPILE_FAIL or CORRECTNESS_FAIL respectively.

**REQ-ORCH-005: Candidate Discard on Regression** [traces SC-4]
Candidates whose benchmark results regress against the incumbent (as determined by `BenchmarkBundle.regressed_vs_incumbent`) must be discarded from the "improving" pool. Their outcome must be recorded as REGRESSION, and they must never participate in incumbent comparison. Because they are correctness-passing and benchmarked, they may participate in cross-candidate analysis only as negative evidence for avoid patterns. The regression threshold is determined by the evaluation pipeline using the `PerformanceObjective.regression_guard_pct` — the Orchestrator acts on the `regressed_vs_incumbent` flag returned by the pipeline.

**REQ-ORCH-006: Concurrent Candidate Evaluation** [traces SC-1]
Multiple kernel candidates produced in a single round must be evaluated concurrently using asyncio.TaskGroup. A failure in one candidate's evaluation must not abort or corrupt the evaluation of other candidates in the same round.

**REQ-ORCH-007: Cross-Candidate Analysis Gate** [traces SC-1]
Cross-candidate analysis must only be invoked when a round produces two or more correctness-passing candidates that reached benchmarking (outcome is IMPROVED, BASELINE_MATCH, or REGRESSION). IMPROVED and BASELINE_MATCH candidates provide positive or neutral comparison evidence; REGRESSION candidates provide negative evidence only. If fewer than two benchmarked candidates are available, cross-candidate analysis is skipped for that round.

**REQ-ORCH-008: Problem Spec Loading** [traces SC-1]
The Orchestrator must accept a ProblemSpec (loaded from YAML) that defines the optimization target: operation name, shape cases with weights and tolerances, dtype, target GPU, performance objective definition, target metric value, maximum rounds, and reference kernel source.

**REQ-ORCH-009: Baseline Bootstrap** [traces SC-6]
Before round 0, the system must seed OptimizationState with a measured baseline artifact produced by compiling, validating, benchmarking, and profiling the reference kernel on the target GPU. The baseline artifact becomes both the initial baseline and the initial incumbent. If baseline bootstrap fails (compile error, correctness failure), the optimization loop must not start.

> **V2: implementation deferred until GPU Pipeline is ready.** Current behavior: construct BaselineArtifact from ProblemSpec declared values (shape_cases produce synthetic ShapeBenchResults, objective_score is computed from declared weights, profile_bundle is None). The bootstrap still populates `OptimizationState.baseline` and `OptimizationState.incumbent` so that all downstream logic referencing these fields works identically in V1 stubs and V2 real pipeline.

**REQ-ORCH-010: Recombination Parent Source Hydration** [traces SC-9]
When the Strategy Navigator returns an EXPLORE/RECOMBINATION directive with `parent_candidates`, the Orchestrator must resolve available parent source code before calling the Coding Agent and populate `StrategyDirective.parent_sources` with a mapping from parent hash to source body. Source lookup must check the current incumbent first and then persisted `kernels/<hash>.cu` files. Missing sources must not change the CodingAgentProtocol signature or block the round; unresolved hashes are simply absent from `parent_sources` and the Coding Agent degrades gracefully.

### Quality Gates

**QG-ORCH-001: Type Safety** [traces SC-5]
All source code must pass `mypy --strict` with no errors.

**QG-ORCH-002: Lint** [traces SC-5]
All source code must pass `ruff check` with no errors.

---

## §3 Scenarios

**SCN-ORCH-001-01: Normal loop to target met**
- GIVEN: a ProblemSpec with target_metric_value that is achievable
- AND: max_rounds = 10
- AND: the evaluation pipeline returns progressively improving objective scores across rounds
- WHEN: a candidate achieves an objective score value <= target_metric_value
- THEN: the loop terminates with status TARGET_MET
- AND: result.json contains the incumbent kernel, its objective score, and its source code
- AND: the number of completed rounds is <= max_rounds

**SCN-ORCH-001-02: Loop exhausts max_rounds**
- GIVEN: a ProblemSpec with max_rounds = 3
- AND: no candidate ever meets target_metric_value
- WHEN: 3 rounds complete
- THEN: the loop terminates with status MAX_ROUNDS_REACHED
- AND: result.json contains the best kernel found so far

**SCN-ORCH-002-01: Workdir completeness after termination**
- GIVEN: the Orchestrator runs for R rounds before terminating
- WHEN: the loop terminates (for any reason)
- THEN: workdir contains state.json with current optimization state
- AND: workdir contains rounds/round_000.json through rounds/round_{R-1:03d}.json
- AND: workdir contains kernels/<hash>.cu for every candidate generated across all rounds
- AND: workdir contains decision_log.jsonl with R entries (one per round)
- AND: workdir contains result.json

**SCN-ORCH-003-01: Incumbent updates when improvement found**
- GIVEN: current incumbent has objective_score.value = 2.0
- WHEN: a round produces a candidate with objective_score.value = 1.5 that passes correctness
- THEN: the incumbent is updated to that candidate's BaselineArtifact
- AND: incumbent.objective_score.value becomes 1.5

**SCN-ORCH-003-02: Incumbent unchanged when no improvement**
- GIVEN: current incumbent has objective_score.value = 2.0
- WHEN: all candidates in a round have objective_score.value >= 2.0
- THEN: incumbent remains unchanged

**SCN-ORCH-004-01: Compile-failed candidate is discarded**
- GIVEN: a round with 3 candidates
- AND: candidate B fails compilation
- WHEN: evaluation completes for all candidates
- THEN: candidate B's outcome is COMPILE_FAIL
- AND: candidate B is not considered for incumbent update
- AND: candidate B is not included in cross-candidate analysis input

**SCN-ORCH-004-02: Correctness-failed candidate is discarded**
- GIVEN: a round with 3 candidates
- AND: candidate C passes compilation but fails correctness
- WHEN: evaluation completes for all candidates
- THEN: candidate C's outcome is CORRECTNESS_FAIL
- AND: candidate C is not considered for incumbent update
- AND: candidate C is not included in cross-candidate analysis input

**SCN-ORCH-005-01: Regressing candidate is discarded**
- GIVEN: a current incumbent
- AND: candidate A's BenchmarkBundle has regressed_vs_incumbent = True
- WHEN: evaluation completes
- THEN: candidate A's outcome is REGRESSION
- AND: candidate A may be included in cross-candidate analysis input only as negative evidence

**SCN-ORCH-006-01: Concurrent evaluation isolates failures**
- GIVEN: a round with 4 candidates
- AND: the evaluation pipeline raises an unexpected exception for candidate 2
- WHEN: evaluation runs concurrently
- THEN: candidates 1, 3, and 4 still complete evaluation normally
- AND: candidate 2 is recorded with outcome ERROR

**SCN-ORCH-007-01: Cross-analysis skipped with fewer than 2 benchmarked candidates**
- GIVEN: a round produces 3 candidates
- AND: 2 fail compilation, 1 passes with IMPROVED outcome
- WHEN: the round reaches the analysis phase
- THEN: cross-candidate analysis is not invoked
- AND: the round proceeds directly to state update

**SCN-ORCH-007-02: Regression included as negative analysis evidence**
- GIVEN: a round produces one IMPROVED candidate and one REGRESSION candidate
- AND: both candidates passed correctness and have benchmark bundles
- WHEN: the round reaches the analysis phase
- THEN: cross-candidate analysis is invoked with both candidates
- AND: the REGRESSION candidate is available only as negative evidence
- AND: the REGRESSION candidate is not considered for incumbent update

**SCN-ORCH-008-01: ProblemSpec loaded from YAML**
- GIVEN: a YAML file with valid fields (op_name, shape_cases with weights, objective, target_metric_value, max_rounds, reference_kernel)
- WHEN: the Orchestrator is initialized with this file path
- THEN: a ProblemSpec is constructed with all fields populated
- AND: shape_cases contains ShapeCase entries with dims, weights, and optional tolerances
- AND: objective contains a PerformanceObjective with primary_metric and aggregation
- AND: the loop uses target_metric_value for termination checks

**SCN-ORCH-009-01: Baseline bootstrap seeds initial state (V2)**
- GIVEN: a ProblemSpec with a valid reference_kernel
- WHEN: the Orchestrator starts before round 0
- THEN: the reference kernel is compiled, validated for correctness, benchmarked on all objective shapes, and profiled on designated profile shapes
- AND: a BaselineArtifact is constructed from the measured results
- AND: OptimizationState.baseline is set to this artifact
- AND: OptimizationState.incumbent is set to this artifact

**SCN-ORCH-009-02: Baseline bootstrap failure aborts loop (V2)**
- GIVEN: a ProblemSpec whose reference_kernel fails compilation
- WHEN: the Orchestrator attempts baseline bootstrap
- THEN: the optimization loop does not start
- AND: an error is reported indicating bootstrap failure

**SCN-ORCH-009-03: Baseline bootstrap from declared values (V1)**
- GIVEN: a ProblemSpec with shape_cases and objective
- WHEN: the Orchestrator starts in V1 mode (no real GPU pipeline)
- THEN: a BaselineArtifact is constructed from declared values (synthetic benchmarks derived from shape_cases, computed objective_score, no profile)
- AND: OptimizationState.baseline and incumbent are set to this synthetic artifact
- AND: all downstream logic operates identically to the V2 path

**SCN-ORCH-010-01: Typed tabu entries recorded per candidate**
- GIVEN: a round produces 3 candidates with directions "tiling", "vectorize", "tiling"
- AND: the candidates have base_kernel_hashes from their intent
- WHEN: the round completes
- THEN: 3 AttemptRecord entries are appended to OptimizationState.attempts
- AND: TabuEntry entries are created for each (base_kernel_hash, direction) pair with an expiry round

**SCN-ORCH-010-02: Tabu entries expire**
- GIVEN: a TabuEntry with expires_after_round = 5
- WHEN: the current round is 6
- THEN: the tabu entry is no longer active
- AND: the direction it covers is available for reuse

**SCN-ORCH-011-01: Objective-based termination**
- GIVEN: target_metric_value = 1.0
- AND: incumbent.objective_score.value = 1.2
- WHEN: a candidate achieves objective_score.value = 0.95
- THEN: the incumbent is updated
- AND: since 0.95 <= 1.0, the loop terminates with status TARGET_MET

**SCN-ORCH-012-01: RoundSummary contains both absolute and relative gains**
- GIVEN: the incumbent before this round had objective_score.value = 2.0
- WHEN: a candidate in this round achieves objective_score.value = 1.5
- THEN: the RoundSummary for this round has abs_gain_vs_prev_best_us = 0.5
- AND: rel_gain_vs_prev_best = 0.25 (i.e. (2.0 - 1.5) / 2.0)

**SCN-ORCH-013-01: Recombination parent sources hydrated before generation**
- GIVEN: the Strategy Navigator returns a RECOMBINATION directive with parent_candidates = ["hash_A", "hash_B"]
- AND: `kernels/hash_A.cu` and `kernels/hash_B.cu` exist in the workdir
- WHEN: the Orchestrator prepares to call the Coding Agent
- THEN: `directive.parent_sources` maps "hash_A" and "hash_B" to their source code
- AND: the CodingAgentProtocol signature is unchanged

**SCN-ORCH-013-02: Missing parent source does not block the round**
- GIVEN: the Strategy Navigator returns a RECOMBINATION directive with parent_candidates = ["hash_A", "hash_missing"]
- AND: only `hash_A` can be resolved from incumbent or workdir storage
- WHEN: the Orchestrator hydrates parent sources
- THEN: `directive.parent_sources` contains only "hash_A"
- AND: candidate generation is still attempted

---

## §4 Invariants

**INV-ORCH-001: Incumbent objective score is monotonically non-increasing**
Once an incumbent is set, its objective score value can only decrease (improve) or stay the same. A candidate with a worse (higher) objective score must never replace the current incumbent.
*Enforcement:* The incumbent update logic compares `candidate_objective_score.value < incumbent.objective_score.value` before assignment. This check is the sole path to updating the incumbent.

**INV-ORCH-002: Round counter is strictly increasing**
The round counter increments by exactly 1 after each round. No round number is skipped or repeated.
*Enforcement:* The round counter is incremented at exactly one point in the loop — after all round processing (evaluation, analysis, persistence) completes. It is not modified elsewhere.

**INV-ORCH-003: Every generated kernel is persisted**
Every kernel candidate's source code is written to `kernels/<hash>.cu` before evaluation begins, regardless of whether it later fails compilation or is discarded.
*Enforcement:* Kernel source files are written immediately after receiving candidates from the Coding Agent, before dispatching evaluation.

**INV-ORCH-004: Failed candidates never reach analysis**
Candidates with outcome COMPILE_FAIL, CORRECTNESS_FAIL, or ERROR are filtered out before being passed to cross-candidate analysis. Candidates with outcome REGRESSION may reach analysis only because they are correctness-passing and benchmarked; they remain ineligible for incumbent update.
*Enforcement:* The cross-candidate analysis input is constructed by filtering evaluation results to outcomes IMPROVED, BASELINE_MATCH, or REGRESSION with a present BenchmarkBundle; the incumbent-update input remains filtered to IMPROVED only.

**INV-ORCH-005: State file writes are atomic**
State files (state.json, round files, result.json) must not be left in a partially written state after a crash.
*Enforcement:* All state writes use a write-to-temporary-then-rename pattern. Write to `<target>.tmp`, then `os.replace()` to the final path.

**INV-ORCH-006: Baseline and incumbent are always present after bootstrap**
After bootstrap completes (whether V1 synthetic or V2 measured), `OptimizationState.baseline` and `OptimizationState.incumbent` are never None. All downstream logic may assume they exist.
*Enforcement:* Bootstrap runs before the first round. The loop entry is gated on successful baseline construction. No code path can clear baseline or incumbent to None.

**INV-ORCH-007: Attempt records are append-only**
AttemptRecord entries are never modified or removed from `OptimizationState.attempts`. Each candidate evaluation in each round appends exactly one record.
*Enforcement:* The attempts list is only mutated via `.append()` at the single state-update point after evaluation completes.

---

## §5 Interfaces

### Protocol Signatures

The Orchestrator depends on exactly four downstream Protocols. All methods are async.

```
StrategyNavigatorProtocol:
    decide(problem_spec, optimization_state, round_summary, cross_analysis) -> StrategyDirective

CodingAgentProtocol:
    generate(problem_spec, directive, incumbent) -> list[KernelCandidate]

GPUPipelineProtocol:
    evaluate(candidate, problem_spec, baseline, incumbent) -> EvaluationResult

CrossCandidateAnalyzerProtocol:
    analyze(top_k_results, problem_spec) -> CrossCandidateAnalysis
```

Parameter semantics:
- `optimization_state`: Full accumulated state (rounds history, attempt records, tabu entries, bottleneck history, baseline, incumbent).
- `round_summary`: Compressed summary of the most recent round, consumed by Strategy Navigator.
- `cross_analysis`: Output from the previous round's cross-candidate analysis, or None on the first round.
- `incumbent`: The current incumbent BaselineArtifact. The Coding Agent uses `incumbent.source_code` as the base for exploit-mode mutations. The GPU Pipeline uses both baseline and incumbent for regression detection and objective score computation.
- `baseline`: The original measured (or V1 synthetic) BaselineArtifact. Used by the GPU Pipeline to compute `ObjectiveScore.relative_to_baseline`.
- `top_k_results`: List of (candidate, evaluation_result) pairs for correctness-passing candidates that reached benchmarking in the current round. This includes IMPROVED and BASELINE_MATCH candidates as positive/neutral evidence and REGRESSION candidates as negative evidence; compile, correctness, and infrastructure failures are excluded.

### Shared Types

All types are Pydantic BaseModel (for serialization) or Python enums. These types are defined by the Orchestrator module and imported by all downstream modules.

#### Problem Specification Types

**ShapeCase**:
- shape_id: str — unique identifier for this shape
- dims: list[int] — dimension values (e.g. [M, N, K] for matmul)
- weight: float = 1.0 — relative importance weight for objective aggregation
- correctness_tolerance: float | None = None — per-shape correctness tolerance override
- profile: bool = False — whether this shape is designated for deep profiling

**PerformanceObjective**:
- primary_metric: Literal["weighted_p50_us", "weighted_p95_us", "worst_case_p50_us"] — which latency metric to optimize
- aggregation: Literal["weighted_mean", "max"] — how per-shape results are aggregated into the objective score
- regression_guard_pct: float = 0.0 — percentage threshold for regression detection (0.0 means any regression is flagged)

**ProblemSpec** (from YAML):
- op_name: str
- op_semantics: str
- dtype: str
- target_gpu: str
- shape_cases: list[ShapeCase] — workload shapes with weights and tolerances
- objective: PerformanceObjective — how to score and rank candidates
- target_metric_value: float — objective score value at or below which the target is considered met
- max_rounds: int
- reference_kernel: str — source code of the reference/baseline kernel

#### Baseline and Incumbent Types

**StaticAnalysis**:
- registers_per_thread: int | None = None
- smem_bytes_per_block: int | None = None
- spill_stores: int | None = None
- spill_loads: int | None = None
- occupancy_estimate_pct: float | None = None

**ShapeBenchResult**:
- shape_id: str — which ShapeCase this result corresponds to
- latency_p50_us: float
- latency_p95_us: float
- stdev_us: float | None = None
- run_count: int

**ObjectiveScore**:
- metric_name: str — name of the metric (matches PerformanceObjective.primary_metric)
- value: float — the aggregate objective score (lower is better)
- relative_to_baseline: float — ratio relative to the baseline's objective score (< 1.0 means improvement)
- relative_to_incumbent: float — ratio relative to the incumbent's objective score (< 1.0 means improvement)

**BaselineArtifact**:
- kernel_hash: str
- source_code: str
- compile_artifact: StaticAnalysis
- benchmark_results: list[ShapeBenchResult]
- objective_score: ObjectiveScore
- profile_bundle: ProfileBundle | None = None

#### Candidate and Search Memory Types

**CandidateIntent**:
- direction: str — the optimization direction (e.g. "tiling", "vectorize", "memory_coalescing")
- mode: Mode
- sub_mode: SubMode | None = None
- rationale: str | None = None — optional human-readable explanation of why this direction was chosen

**KernelCandidate**:
- code_hash: str — content hash of source_code
- source_code: str
- parent_hashes: list[str] — hashes of kernels this candidate was derived from (empty for de novo)
- intent: CandidateIntent — structured intent replacing the old flat intent_tag

**AttemptRecord**:
- round_number: int
- candidate_hash: str
- base_kernel_hash: str | None — hash of the parent kernel (None for de novo)
- direction: str
- sub_mode: SubMode | None
- outcome: CandidateOutcome
- objective_score: float | None = None — the candidate's objective score if evaluation reached benchmarking

**TabuEntry**:
- base_kernel_hash: str | None — the parent kernel hash this tabu applies to (None means global)
- direction: str
- sub_mode: SubMode | None
- round_number: int — the round in which this tabu was created
- expires_after_round: int — the round after which this tabu expires (inclusive)

#### Evaluation Output Types

**CorrectnessResult**:
- passed: bool
- failing_shape_ids: list[str] = [] — which shapes failed correctness
- max_abs_error: float | None = None
- max_rel_error: float | None = None

**BenchmarkBundle**:
- shape_results: list[ShapeBenchResult] — per-shape benchmark results
- objective_score: ObjectiveScore — aggregate objective score for this candidate
- regressed_vs_incumbent: bool — whether this candidate regressed against the incumbent (determined by the pipeline using regression_guard_pct)

**ProfileMetrics**:
- achieved_occupancy_pct: float | None = None
- dram_throughput_pct_of_peak: float | None = None
- sm_throughput_pct_of_peak: float | None = None
- l2_hit_rate_pct: float | None = None
- warp_stall_memory_dependency_pct: float | None = None
- warp_stall_exec_dependency_pct: float | None = None
- tensor_core_utilization_pct: float | None = None
- arithmetic_intensity_flop_per_byte: float | None = None

**BottleneckAssessment**:
- tags: list[str] — all identified bottleneck tags
- primary_tag: str | None = None — the dominant bottleneck
- evidence: dict[str, float] — metric name to value pairs supporting the assessment
- rule_trace: list[str] — ordered list of rules that fired to produce this assessment

**ProfileBundle**:
- shape_id: str — which ShapeCase was profiled
- metrics: ProfileMetrics
- assessment: BottleneckAssessment

**EvaluationResult**:
- candidate_hash: str
- compile_status: CompileStatus — compile outcome (SUCCESS, COMPILE_ERROR, CORRECTNESS_FAIL)
- static_analysis: StaticAnalysis | None = None — resource usage analysis (present if compile succeeded)
- correctness: CorrectnessResult | None = None — correctness check result (present if compile succeeded)
- benchmark: BenchmarkBundle | None = None — benchmark results (present if correctness passed)
- profile: ProfileBundle | None = None — profile results (present if candidate was selected for deep profiling)
- outcome: CandidateOutcome

#### Strategy and Analysis Types

**StrategyDirective**:
- mode: Mode
- direction: str
- reason: str
- base_kernel_hash: str | None = None
- num_candidates: int
- tabu: list[TabuEntry]
- sub_mode: SubMode | None = None
- parent_candidates: list[str] | None = None
- gene_map: dict[str, str] | None = None
- search_range: dict[str, list[float]] | None = None
- hard_constraints: list[str] | None = None
- parent_sources: dict[str, str] | None = None — optional parent hash -> source mapping hydrated by Orchestrator for recombination before calling Coding Agent

**SemanticDelta**:
- candidate_hash: str
- parent_hashes: list[str]
- outcome: CandidateOutcome
- summary: str
- changed_features: list[str]
- evidence_refs: list[str]
- confidence: str

**CandidateGene**:
- gene_id: str
- source_candidate_hash: str
- gene_type: str
- description: str
- evidence: dict[str, float]
- affected_shape_ids: list[str]
- risk_flags: list[str] = []
- confidence: str

**RecombinationHint**:
- hint_id: str
- parent_candidates: list[str]
- gene_map: dict[str, str]
- expected_benefit: str
- evidence_candidate_hashes: list[str]
- required_constraints: list[str] = []
- risk_flags: list[str] = []
- confidence: str

**AvoidPattern**:
- pattern_id: str
- source_candidate_hash: str
- pattern: str
- reason: str
- evidence: dict[str, float]
- affected_shape_ids: list[str]
- scope: str = "candidate_local"
- confidence: str

**CrossCandidateAnalysis**:
- insights: list[str]
- winning_genes: list[str]
- recombination_suggestions: list[str]
- semantic_deltas: list[SemanticDelta] = []
- candidate_genes: list[CandidateGene] = []
- recombination_hints: list[RecombinationHint] = []
- avoid_patterns: list[AvoidPattern] = []

#### Round and State Types

**RoundState**:
- round_number: int
- phase: Phase
- directive: StrategyDirective
- candidates: list[KernelCandidate]
- evaluation_results: list[EvaluationResult]
- cross_analysis: CrossCandidateAnalysis | None = None
- best_candidate_hash: str | None = None
- best_objective_score: float | None = None

**RoundSummary**:
- round_number: int
- mode: Mode
- direction: str
- num_candidates: int
- num_improved: int
- best_objective_score: float | None = None — best objective score achieved in this round
- abs_gain_vs_prev_best_us: float | None = None — absolute improvement in objective score value (prev_best - current_best, in the objective's unit)
- rel_gain_vs_prev_best: float | None = None — relative improvement as a fraction ((prev_best - current_best) / prev_best)

**OptimizationState**:
- problem_spec: ProblemSpec
- baseline: BaselineArtifact — the measured (or V1 synthetic) baseline artifact; set during bootstrap, never changed
- incumbent: BaselineArtifact — the current best kernel artifact; initialized from baseline, updated when a strictly better candidate is found
- current_round: int = 0
- rounds: list[RoundSummary] = []
- attempts: list[AttemptRecord] = [] — complete history of every candidate attempt
- tabu_entries: list[TabuEntry] = [] — typed tabu entries with expiry
- bottleneck_history: list[BottleneckAssessment] = [] — per-round bottleneck assessments with evidence
- decision_log: list[dict] = [] — structured log entries (one per round)

**OptimizationResult**:
- status: str — TARGET_MET or MAX_ROUNDS_REACHED
- best_kernel_hash: str | None = None
- best_objective_score: float | None = None — the incumbent's final objective score value
- best_kernel_source: str | None = None
- total_rounds: int = 0
- total_candidates_evaluated: int = 0

#### Enums

- Phase: AWAITING_STRATEGY, AWAITING_CODING, AWAITING_EVALUATION, ANALYSIS, ROUND_COMPLETE
- Mode: EXPLOIT, EXPLORE
- SubMode: PARAM_SEARCH, LOCAL_REWRITE, PATTERN_APPLY, DE_NOVO, RECOMBINATION
- CompileStatus: SUCCESS, COMPILE_ERROR, CORRECTNESS_FAIL
- CandidateOutcome: IMPROVED, BASELINE_MATCH, REGRESSION, COMPILE_FAIL, CORRECTNESS_FAIL, ERROR

### YAML Input Format

```yaml
op_name: matmul
op_semantics: "C[M,N] = A[M,K] @ B[K,N]"
dtype: float16
target_gpu: A100
shape_cases:
  - shape_id: "small"
    dims: [1024, 1024, 1024]
    weight: 0.3
    profile: false
  - shape_id: "medium"
    dims: [4096, 4096, 4096]
    weight: 1.0
    profile: true
  - shape_id: "large"
    dims: [8192, 8192, 8192]
    weight: 0.5
    correctness_tolerance: 0.01
    profile: true
objective:
  primary_metric: weighted_p50_us
  aggregation: weighted_mean
  regression_guard_pct: 0.02
target_metric_value: 1.0
max_rounds: 20
reference_kernel: |
  __global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K) {
      // naive reference implementation
  }
```

---

## §6 Behavioral Specification

### 6.0 Baseline Bootstrap

Before the optimization loop begins, the Orchestrator must establish a measured baseline. This is a prerequisite — no round 0 search begins until the baseline is established.

**Bootstrap sequence:**

1. Resolve the reference kernel source from ProblemSpec.
2. Compile the reference kernel on the target GPU toolchain.
3. Validate correctness on all required shapes (every ShapeCase in shape_cases).
4. Benchmark on all objective shapes (all shape_cases, producing ShapeBenchResults).
5. Compute the ObjectiveScore from the benchmark results using the ProblemSpec's objective definition.
6. Deep-profile on designated profile shapes (shape_cases where profile = True).
7. Construct a BaselineArtifact from the results.
8. Set `OptimizationState.baseline` and `OptimizationState.incumbent` to this artifact.

**Failure handling:** If the reference kernel fails compilation or correctness, the optimization loop does not start. The system reports a bootstrap failure.

**V1 behavior:** In V1 (stub GPU pipeline), the bootstrap constructs a synthetic BaselineArtifact:
- `kernel_hash`: content hash of reference_kernel
- `source_code`: the reference_kernel text
- `compile_artifact`: StaticAnalysis with all None fields
- `benchmark_results`: one ShapeBenchResult per shape_case with placeholder latency values
- `objective_score`: computed from the synthetic benchmark results and the objective definition
- `profile_bundle`: None

The key guarantee is that `OptimizationState.baseline` and `OptimizationState.incumbent` are populated before round 0 regardless of V1 or V2 mode, so all downstream logic that references these fields works uniformly.

### 6.1 Main Loop

The Orchestrator operates as a single async function that runs a bounded loop. Each iteration of the loop is one "round" of optimization. The loop structure is:

```
Bootstrap: construct BaselineArtifact, seed OptimizationState with baseline and incumbent
For each round from 0 to max_rounds - 1:
    1. Request strategy directive from Strategy Navigator
    2. Hydrate parent_sources for recombination directives when parent source files are available
    3. Request kernel candidates from Coding Agent
    4. Persist all candidate source files to workdir
    5. Evaluate all candidates concurrently
    6. Classify outcomes and filter results
    7. Update incumbent if any candidate improved
    8. Check termination: if target met, break
    9. Run cross-candidate analysis (if >= 2 benchmarked candidates)
    10. Record attempt history, update tabu entries, update bottleneck history
    11. Build round summary, persist round state, append to decision log
    12. Increment round counter
Build and persist OptimizationResult
```

**First round special case:** On round 0, there is no prior round_summary or cross_analysis. The Orchestrator passes None for both when calling the Strategy Navigator. The incumbent is the baseline artifact constructed during bootstrap.

**Parent-source hydration:** If the returned directive has sub_mode RECOMBINATION and non-empty `parent_candidates`, the Orchestrator resolves source code before candidate generation. It first checks whether any requested hash equals the current incumbent hash, then checks `kernels/<hash>.cu` in the workdir. Resolved sources are stored in `directive.parent_sources`. Missing parent files are logged but do not alter the protocol call or abort the round.

### 6.2 State Machine

The Orchestrator tracks its current phase within each round for observability and debugging. The phase transitions are strictly sequential within a round:

```
AWAITING_STRATEGY
    -> Strategy Navigator returns directive
AWAITING_CODING
    -> Coding Agent returns candidates
AWAITING_EVALUATION
    -> All concurrent evaluations complete
ANALYSIS
    -> Cross-candidate analysis completes (or is skipped)
ROUND_COMPLETE
    -> State persisted, round counter incremented
    -> Back to AWAITING_STRATEGY (next round) or loop exit
```

The phase is informational — it records where the Orchestrator is in the current round. There are no conditional phase transitions or branches in the state machine. Every round passes through every phase in order (though the ANALYSIS phase may be a no-op when skipped per REQ-ORCH-007).

### 6.3 Early-Exit and Candidate Outcome Classification

After concurrent evaluation completes, each candidate has an EvaluationResult with an outcome field set by the GPU pipeline. The Orchestrator uses these outcomes to partition candidates:

| Outcome | Meaning | Enters incumbent-update? | Enters cross-analysis? |
|---------|---------|--------------------------|----------------------|
| IMPROVED | Objective score better than incumbent | Yes | Yes |
| BASELINE_MATCH | Objective score within tolerance of incumbent | No (not strictly better) | Yes |
| REGRESSION | Regressed against incumbent per regression guard | No | Yes, negative evidence only |
| COMPILE_FAIL | Compilation failed | No | No |
| CORRECTNESS_FAIL | Compiled but produced wrong results | No | No |
| ERROR | Unexpected error during evaluation | No | No |

**Incumbent-update rule:** Only candidates with outcome IMPROVED are eligible to become the new incumbent. Among multiple IMPROVED candidates in the same round, the one with the lowest `benchmark.objective_score.value` wins.

**Cross-analysis input:** Candidates with outcome IMPROVED, BASELINE_MATCH, or REGRESSION and a present BenchmarkBundle form the input to cross-candidate analysis. REGRESSION candidates are included only as negative evidence for avoid patterns and must not become incumbents or default recombination parents. If this set has fewer than 2 members, cross-analysis is skipped.

**Handling evaluation exceptions:** If a candidate's evaluation raises an unexpected exception (not a compile/correctness failure, but an infrastructure error), the Orchestrator catches it, records the candidate with outcome ERROR, and continues evaluating other candidates. The round proceeds normally with the remaining results.

### 6.4 Incumbent Update

After classifying outcomes:

1. Filter candidates to those with outcome IMPROVED.
2. If the filtered set is empty, skip update.
3. Among filtered candidates, select the one with the lowest `benchmark.objective_score.value`.
4. If no incumbent has been updated beyond the baseline yet, compare against `incumbent.objective_score.value`.
5. Set the new incumbent only if the candidate's objective score value is strictly less than the current incumbent's objective score value.
6. When updating, construct a new BaselineArtifact from the winning candidate's evaluation results (hash, source, static_analysis, benchmark_results, objective_score, profile_bundle) and assign it as the new `OptimizationState.incumbent`.

### 6.5 Termination Check

After updating the incumbent, check if the target has been met:

```
target_met = incumbent.objective_score.value <= problem_spec.target_metric_value
```

There is no separate tolerance margin. The target_metric_value in the ProblemSpec already represents the actual target the user wants to achieve. If the user wants a margin, they set a lower target_metric_value.

If target_met is true, the loop breaks immediately. The current round's state is still persisted before exiting.

If the round counter reaches max_rounds without meeting the target, the loop exits naturally. This is a safety limit, not a goal.

### 6.6 Tabu and Attempt History Management

**Attempt records:** After each round, for every candidate evaluated in that round, the Orchestrator creates an AttemptRecord containing:
- `round_number`: the current round
- `candidate_hash`: the candidate's code_hash
- `base_kernel_hash`: the first entry in the candidate's parent_hashes (or None for de novo candidates)
- `direction`: from the candidate's intent
- `sub_mode`: from the candidate's intent
- `outcome`: the candidate's evaluation outcome
- `objective_score`: the candidate's objective score value if evaluation reached benchmarking, else None

Each AttemptRecord is appended to `OptimizationState.attempts`.

**Tabu entries:** After recording attempt records, the Orchestrator creates TabuEntry entries for directions that should be temporarily suppressed. A TabuEntry is created when a candidate's outcome is not IMPROVED (i.e., the direction did not yield improvement on this base kernel). The entry records:
- `base_kernel_hash`: the parent kernel
- `direction`: the direction attempted
- `sub_mode`: the sub_mode attempted
- `round_number`: the current round
- `expires_after_round`: current round + tabu_window (a configurable constant)

The Orchestrator does not enforce tabu constraints itself — it tracks the entries and passes them to the Strategy Navigator via `OptimizationState.tabu_entries`. The Navigator is responsible for respecting active (non-expired) entries when choosing directions.

**Bottleneck history:** After each round, the BottleneckAssessment from the ProfileBundle of the best passing candidate (if profiling occurred) is appended to `OptimizationState.bottleneck_history`. If no candidates were profiled in this round, no entry is added for that round. The BottleneckAssessment preserves the full evidence and rule trace, not just the tag labels.

### 6.7 Round Summary Construction

After each round, the Orchestrator builds a RoundSummary containing:
- Round number
- Mode and direction from the directive
- Number of candidates generated
- Number with outcome IMPROVED
- `best_objective_score`: the best (lowest) objective score achieved by any passing candidate in this round, or None if no candidate passed
- `abs_gain_vs_prev_best_us`: the absolute improvement in the objective metric (previous incumbent objective score value minus this round's best objective score value), or None if no improvement occurred. Positive means improvement.
- `rel_gain_vs_prev_best`: the relative improvement as a fraction ((previous incumbent value - this round's best value) / previous incumbent value), or None if no improvement occurred. Positive means improvement. For example, if the previous incumbent scored 2.0 and the new best scores 1.5, rel_gain = 0.25.

This summary is stored in `OptimizationState.rounds` and passed to the Strategy Navigator in the next round.

### 6.8 Decision Log

After each round, the Orchestrator appends a structured entry to decision_log.jsonl. Each entry is a single JSON line containing:
- round_number
- directive (mode, direction, reason, num_candidates)
- outcomes summary (count per outcome type)
- best_objective_score_this_round
- incumbent_objective_score_after_round
- improvement (boolean: did incumbent improve this round?)

### 6.9 State Persistence

**StateManager** handles all filesystem operations. The Orchestrator calls it at defined points:

| When | What is written | File |
|------|----------------|------|
| After candidates received (loop step 4) | Each candidate's source code | `kernels/<hash>.cu` |
| After round completes (loop step 11) | Full optimization state snapshot | `state.json` |
| After round completes (loop step 11) | This round's complete state | `rounds/round_NNN.json` |
| After round completes (loop step 11) | Decision log entry (append) | `decision_log.jsonl` |
| After loop exits | Final result | `result.json` |

All writes to state.json, round files, and result.json use the atomic write pattern: write to `<path>.tmp`, then `os.replace(<path>.tmp, <path>)`. The decision_log.jsonl is append-only (open in append mode, write line, flush).

**State serialization notes:** The OptimizationState now contains BaselineArtifact (for baseline and incumbent), AttemptRecord lists, TabuEntry lists, and BottleneckAssessment lists. All of these are Pydantic BaseModels and serialize to JSON via `.model_dump()`. The state.json snapshot must faithfully represent the entire OptimizationState including these nested structures.

**Workdir initialization:** On startup, the Orchestrator creates the workdir directory and its subdirectories (`rounds/`, `kernels/`) if they do not exist. If state.json already exists in the workdir, V1 logs a warning and starts fresh (resume-from-crash is not a V1 requirement beyond basic detection).

### 6.10 Workdir Layout

```
<workdir>/
    state.json                  # OptimizationState snapshot (overwritten each round)
    result.json                 # OptimizationResult (written once at termination)
    decision_log.jsonl          # Append-only, one JSON line per round
    rounds/
        round_000.json          # RoundState for round 0
        round_001.json          # RoundState for round 1
        ...
    kernels/
        <hash1>.cu              # Source code of candidate with hash1
        <hash2>.cu              # Source code of candidate with hash2
        ...
```

### 6.11 CLI Entry Point

The module is runnable via `python -m kerlever <spec.yaml> <workdir>`. This:
1. Loads ProblemSpec from the YAML file
2. Runs baseline bootstrap (V1: synthetic, V2: real GPU pipeline)
3. Creates stub implementations of all four Protocols (V1)
4. Instantiates the Orchestrator with the ProblemSpec, stubs, and workdir path
5. Runs the async main loop via `asyncio.run()`
6. Exits with code 0 on success

---

## §7 Production Path Trace

This traces the data and decision flow for a complete optimization run, following the exact sequence an operator would observe.

**Trigger:** The operator invokes the CLI with a YAML spec file and a workdir path.

### Bootstrap Phase

1. **Spec loading.** The YAML file is parsed into a ProblemSpec. Shape cases, objective definition, target metric value, and reference kernel are extracted.

2. **Baseline construction.** The reference kernel is evaluated through the full pipeline: compile, correctness, benchmark across all shape_cases, profile on designated shapes. In V1, synthetic results are constructed from declared values. The result is a BaselineArtifact containing the kernel hash, source, static analysis, per-shape benchmark results, aggregate objective score, and profile bundle.

3. **State seeding.** An OptimizationState is created with baseline and incumbent both set to the baseline artifact. Attempts, tabu entries, bottleneck history, and rounds are all empty. The round counter is 0.

### Round N (repeated)

4. **Strategy request.** The Orchestrator sends the current optimization state, previous round summary (None if N=0), and previous cross-analysis (None if N=0) to the Strategy Navigator. The Navigator can access baseline, incumbent, attempt history, tabu entries, and bottleneck history from the state. It returns a StrategyDirective specifying mode, direction, and number of candidates.

5. **Candidate generation.** If the directive requests RECOMBINATION, the Orchestrator first hydrates `directive.parent_sources` from the incumbent or persisted kernel files for every selected parent hash it can resolve. The Orchestrator then sends the ProblemSpec, the directive, and the current incumbent to the Coding Agent. The Coding Agent uses `incumbent.source_code` as the base for exploit-mode mutations and `directive.parent_sources` for recombination prompts when present. It returns a list of KernelCandidate objects, each with a unique code_hash, source code, parent_hashes, and structured intent.

6. **Kernel persistence.** Each candidate's source code is written to `kernels/<hash>.cu` in the workdir. This happens before evaluation so that even if evaluation crashes, the source code is preserved.

7. **Concurrent evaluation.** All candidates are dispatched concurrently to the GPU Pipeline via asyncio.TaskGroup. Each evaluation receives the candidate, problem_spec, baseline, and incumbent. The pipeline compiles, checks correctness, benchmarks across shape_cases, computes the objective score relative to both baseline and incumbent, determines regression, and optionally profiles. If one evaluation raises an exception, the Orchestrator catches it and marks that candidate as ERROR. All other evaluations continue.

8. **Outcome classification.** Each EvaluationResult carries an outcome set by the pipeline. The Orchestrator partitions results by outcome.

9. **Incumbent update.** Among IMPROVED candidates, the one with the lowest objective score value is compared against the current incumbent. If strictly better, the incumbent is replaced with a new BaselineArtifact constructed from the winning candidate's results.

10. **Termination check.** If `incumbent.objective_score.value <= target_metric_value`, the loop sets status TARGET_MET and proceeds to persist the round state before exiting.

11. **Cross-candidate analysis.** If two or more candidates have outcome IMPROVED, BASELINE_MATCH, or REGRESSION with benchmark data, they are sent to the Cross-Candidate Analyzer. Regressions are included only as negative evidence. If fewer than two benchmarked candidates are available, this step is skipped.

12. **State update.** AttemptRecords are created for every candidate. TabuEntries are created for non-improving directions. BottleneckAssessment from the best profiled candidate is appended to bottleneck_history. A RoundSummary is constructed with both absolute and relative gains.

13. **Persistence.** The full RoundState is written to `rounds/round_NNN.json`. The OptimizationState snapshot is written to `state.json`. A decision log entry is appended to `decision_log.jsonl`.

14. **Next round.** The round counter increments. Control returns to step 4, unless the loop is exiting.

**On loop exit:** The Orchestrator builds an OptimizationResult (status, best kernel hash, best objective score, best source, total rounds, total candidates evaluated) and writes it to `result.json`.

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | Partial state on crash | Writing state.json directly without atomic rename | Corrupted state file if process killed mid-write | Use tmp-then-rename for all state writes (INV-ORCH-005) |
| 2 | Lost kernel source | Writing kernel files after evaluation instead of before | If evaluation crashes, the candidate source code is lost | Write kernels immediately after generation, before evaluation (INV-ORCH-003) |
| 3 | Cascading evaluation failure | Using a single try/except around TaskGroup instead of per-task handling | One failed candidate kills the entire round | Catch exceptions inside each evaluation task; record ERROR outcome per candidate |
| 4 | False incumbent | Not checking compile/correctness status before incumbent comparison | A candidate with wrong results could become the incumbent | Filter to IMPROVED-only before incumbent comparison (INV-ORCH-004) |
| 5 | Stale tabu data | Not recording attempt records for non-improving candidates | Strategy Navigator re-suggests the same direction on the same parent repeatedly | Always record AttemptRecords for all candidates regardless of outcome (INV-ORCH-007) |
| 6 | Optimizing against wrong baseline | Starting the loop without measuring the reference kernel | Strategy decisions are based on declared, not measured, performance | Run baseline bootstrap before round 0 (INV-ORCH-006); V1 uses synthetic but structurally identical baseline |
| 7 | Unit-inconsistent strategy signals | Storing only absolute gain and using it for percentage-based thresholds | Plateau detection and near-target logic produce wrong decisions | Store both absolute and relative gains in RoundSummary; downstream uses relative form for thresholds |
| 8 | Placeholder recombination parent | Passing only parent hashes to Coding Agent without resolving source bodies | The LLM invents or ignores a selected parent, producing fake recombination | Hydrate `StrategyDirective.parent_sources` before generation; missing sources are explicit and handled gracefully (REQ-ORCH-010) |

---

## §9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|-------------|-----------|
| SC-1: Loop runs to termination with stubs, no hang/crash | REQ-ORCH-001, REQ-ORCH-006, REQ-ORCH-007, REQ-ORCH-008 | SCN-ORCH-001-01, SCN-ORCH-001-02, SCN-ORCH-006-01, SCN-ORCH-007-01, SCN-ORCH-007-02, SCN-ORCH-008-01, SCN-ORCH-011-01 |
| SC-2: Workdir produces complete artifacts | REQ-ORCH-002 | SCN-ORCH-002-01 |
| SC-3: Incumbent updates correctly using objective score | REQ-ORCH-003 | SCN-ORCH-003-01, SCN-ORCH-003-02 |
| SC-4: Failed candidates discarded and regressions excluded from incumbent update | REQ-ORCH-004, REQ-ORCH-005 | SCN-ORCH-004-01, SCN-ORCH-004-02, SCN-ORCH-005-01, SCN-ORCH-007-02 |
| SC-5: mypy --strict and ruff check pass | QG-ORCH-001, QG-ORCH-002 | (quality gate, verified by CI) |
| SC-6: Baseline bootstrap seeds measured state before round 0 | REQ-ORCH-009 | SCN-ORCH-009-01, SCN-ORCH-009-02, SCN-ORCH-009-03 |
| SC-7: Typed search memory with attempt records and tabu entries | REQ-ORCH-003, REQ-ORCH-009 | SCN-ORCH-010-01, SCN-ORCH-010-02 |
| SC-8: Unit-consistent strategy signals (abs + rel gains) | REQ-ORCH-003 | SCN-ORCH-012-01 |
| SC-9: Recombination directives carry hydrated parent source when available | REQ-ORCH-010 | SCN-ORCH-013-01, SCN-ORCH-013-02 |
