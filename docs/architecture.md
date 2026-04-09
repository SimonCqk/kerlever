# Kerlever System Architecture

This document is the architecture source of truth for Kerlever.

It describes both:

- the current skeleton decomposition,
- the required architectural corrections so the system can become a rigorous CUDA operator optimization loop rather than just a well-structured control-flow prototype.

---

## Overview

Kerlever is an agent-driven CUDA kernel optimization loop:

```text
baseline bootstrap -> generate -> compile -> benchmark -> profile -> analyze -> navigate -> generate
```

The core design remains:

- few agents,
- strong deterministic services,
- structured data exchange,
- optimization decisions grounded in measured hardware behavior.

---

## Core Modules

```text
Agents (LLM)                           Services (Deterministic, Remote GPU Pod)
=========================              ============================================
  Orchestrator                           Compiler Service
  Strategy Navigator                       (compile + correctness + static analysis)
  Coding Agent                           Benchmarker
  Cross-Candidate Analyzer                 (fast bench + rank + deep profile)
                                         Profile Interpreter
```

### Responsibilities

| Module | Type | Role |
|---|---|---|
| Orchestrator | Agent | Global control loop, task state machine, persistence, termination |
| Strategy Navigator | Agent | Exploit/explore decision, direction selection, tabu and exhaustion logic |
| Coding Agent | Agent | Kernel generation and mutation |
| Cross-Candidate Analyzer | Agent | Semantic comparison between passing candidates, recombination hints |
| Compiler Service | Service | Compile, correctness validation, static resource analysis |
| Benchmarker | Service | Multi-shape benchmark, objective scoring, regression filtering, top-K selection |
| Profile Interpreter | Service | Rule-based interpretation of profiling metrics into bottleneck assessments |

---

## Corrected System DAG

The original skeleton was missing one critical stage: measured baseline seeding.

The corrected DAG is:

```text
Problem Spec / Oracle
        |
        v
Baseline Bootstrap
resolve reference -> compile -> correctness -> benchmark -> profile
        |
        v
OptimizationState seeded with measured baseline
        |
        v
Orchestrator
        |
        v
Strategy Navigator
        |
        v
Coding Agent
        |
        v
Compiler Service
        |
        v
Benchmarker
fast bench on all objective shapes -> regression filter -> top-K
        |
        v
Profile Interpreter
metrics -> bottleneck assessment with evidence
        |
        +--------------------------+
        v                          |
Cross-Candidate Analyzer           |
        |                          |
        +-------------> next round-+
```

### Why The Bootstrap Stage Exists

The loop must not begin from an empty incumbent state.

Before round 0, the system must know:

- whether the reference kernel compiles,
- whether it is correct on the required shapes,
- what the actual baseline performance is on the target GPU,
- what bottleneck the baseline exposes,
- what kernel exploit-mode mutations should start from.

Without that, the system is optimizing against a declared baseline, not a measured one.

---

## Architectural Corrections

The following corrections are mandatory for a rigorous optimization loop.

| Finding | Current Weakness | Architectural Correction |
|---|---|---|
| Baseline not seeded | loop starts with no measured parent kernel | add explicit baseline bootstrap stage |
| Unit mismatch in strategy signals | absolute and relative gains are conflated | store absolute and relative gains separately |
| Tabu state too weak | pairwise rule enforced with flat strings | use typed attempt records and typed tabu entries |
| Profile evidence lost | bottleneck tags survive, metrics disappear | preserve metrics, derived values, evidence, and rule trace |
| Objective too scalar | one latency number stands in for workload quality | rank candidates against an explicit multi-shape objective |

---

## Control Flow

### Stage 0: Baseline Bootstrap

Before the first optimization round:

1. Resolve `reference_kernel`.
2. Compile on the target GPU toolchain.
3. Validate correctness on required shapes.
4. Benchmark on all objective shapes.
5. Deep-profile on designated profile shapes.
6. Seed `OptimizationState` with the measured baseline artifact.

If any of these fail, the optimization loop does not start.

### Stage 1: Strategy Decision

The Strategy Navigator reasons over:

- the baseline artifact,
- the current incumbent artifact,
- typed attempt history,
- workload objective scores,
- bottleneck assessments with evidence.

### Stage 2: Candidate Generation

The Coding Agent generates:

- exploit candidates: local rewrites, parameter search, pattern apply,
- explore candidates: de novo, algorithmic changes, recombination.

Exploit mode should normally mutate a real parent kernel, not fall back silently to de novo unless that fallback is intentional.

### Stage 3: Candidate Evaluation

Candidate evaluation produces four distinct artifacts:

1. compile result,
2. correctness result,
3. benchmark result,
4. profile result.

Early exits remain:

- compile fail -> stop,
- correctness fail -> stop,
- benchmark regression -> stop before deep profile,
- only ranked candidates enter deep profile.

### Stage 4: Round Update

After evaluation, the system updates:

- incumbent if objective improves,
- typed attempt history,
- typed tabu entries,
- bottleneck history with evidence,
- round summaries with both absolute and relative gains.

---

## Early-Exit Paths

| Trigger Point | Condition | Action |
|---|---|---|
| Baseline Bootstrap | reference kernel fails compile or correctness | abort before round 0 |
| Compiler Service | candidate compile fail or correctness fail | discard candidate, feed failure context back |
| Benchmarker | candidate regresses against incumbent | discard before deep profile |
| Profile Interpreter / Navigator | stable bottleneck plus exhausted direction | force explore on next round |
| Orchestrator | target metric met | terminate and return incumbent |

---

## Revised Data Contracts

The skeleton contracts were too compressed. The system needs richer typed records.

These are architectural contracts, not byte-for-byte implementation requirements.

### Problem Specification

```python
class ShapeCase(BaseModel):
    shape_id: str
    dims: list[int]
    weight: float = 1.0
    correctness_tolerance: float | None = None
    profile: bool = False


class PerformanceObjective(BaseModel):
    primary_metric: Literal["weighted_p50_us", "weighted_p95_us", "worst_case_p50_us"]
    aggregation: Literal["weighted_mean", "max"]
    regression_guard_pct: float = 0.0


class ProblemSpec(BaseModel):
    op_name: str
    op_semantics: str
    dtype: str
    target_gpu: str
    shape_cases: list[ShapeCase]
    objective: PerformanceObjective
    target_metric_value: float
    max_rounds: int
    reference_kernel: str
```

Why this contract is better:

- `shape_cases` models the workload instead of a raw list,
- weights allow realistic traffic-aware optimization,
- `profile` flags which shapes deserve deep profiling,
- the objective becomes explicit instead of implied.

### Objective Computation

For a weighted latency objective:

```text
weighted_p50_us = sum(weight_i * p50_us_i) / sum(weight_i)
rel_gain_vs_prev_best = (prev_best - current) / prev_best
rel_gain_vs_baseline = (baseline - current) / baseline
```

Thresholded strategy gates should use relative gains or objective ratios, not raw microsecond deltas.

### Baseline and Incumbent Artifacts

```python
class StaticAnalysis(BaseModel):
    registers_per_thread: int | None = None
    smem_bytes_per_block: int | None = None
    spill_stores: int | None = None
    spill_loads: int | None = None
    occupancy_estimate_pct: float | None = None


class ShapeBenchResult(BaseModel):
    shape_id: str
    latency_p50_us: float
    latency_p95_us: float
    stdev_us: float | None = None
    run_count: int


class ObjectiveScore(BaseModel):
    metric_name: str
    value: float
    relative_to_baseline: float
    relative_to_incumbent: float


class BaselineArtifact(BaseModel):
    kernel_hash: str
    source_code: str
    compile_artifact: StaticAnalysis
    benchmark_results: list[ShapeBenchResult]
    objective_score: ObjectiveScore
    profile_bundle: "ProfileBundle | None"
```

Why this contract is better:

- baseline becomes measured system state, not user-declared metadata,
- incumbent and baseline become comparable artifacts,
- strategy logic can reason over like-for-like records.

### Candidate and Attempt History

```python
class CandidateIntent(BaseModel):
    direction: str
    mode: Mode
    sub_mode: SubMode | None = None
    rationale: str | None = None


class KernelCandidate(BaseModel):
    code_hash: str
    source_code: str
    parent_hashes: list[str]
    intent: CandidateIntent


class AttemptRecord(BaseModel):
    round_number: int
    candidate_hash: str
    base_kernel_hash: str | None
    direction: str
    sub_mode: SubMode | None
    outcome: CandidateOutcome
    objective_score: float | None = None
```

Why this contract is better:

- `parent_hashes` supports exploit and recombination,
- `AttemptRecord` becomes the source of truth for search memory,
- direction exhaustion and tabu logic become auditable.

### Typed Tabu Entries

```python
class TabuEntry(BaseModel):
    base_kernel_hash: str | None
    direction: str
    sub_mode: SubMode | None
    round_number: int
    expires_after_round: int
```

Why this contract is better:

- it matches the actual intended invariant,
- the same direction on a different base kernel remains legal,
- expiry becomes deterministic instead of approximate.

### Evaluation Output

```python
class CorrectnessResult(BaseModel):
    passed: bool
    failing_shape_ids: list[str] = []
    max_abs_error: float | None = None
    max_rel_error: float | None = None


class BenchmarkBundle(BaseModel):
    shape_results: list[ShapeBenchResult]
    objective_score: ObjectiveScore
    regressed_vs_incumbent: bool


class ProfileMetrics(BaseModel):
    achieved_occupancy_pct: float | None = None
    dram_throughput_pct_of_peak: float | None = None
    sm_throughput_pct_of_peak: float | None = None
    l2_hit_rate_pct: float | None = None
    warp_stall_memory_dependency_pct: float | None = None
    warp_stall_exec_dependency_pct: float | None = None
    tensor_core_utilization_pct: float | None = None
    arithmetic_intensity_flop_per_byte: float | None = None


class BottleneckAssessment(BaseModel):
    tags: list[str]
    primary_tag: str | None = None
    evidence: dict[str, float]
    rule_trace: list[str]


class ProfileBundle(BaseModel):
    shape_id: str
    metrics: ProfileMetrics
    assessment: BottleneckAssessment


class EvaluationResult(BaseModel):
    candidate_hash: str
    compile_status: CompileStatus
    static_analysis: StaticAnalysis | None = None
    correctness: CorrectnessResult | None = None
    benchmark: BenchmarkBundle | None = None
    profile: ProfileBundle | None = None
    outcome: CandidateOutcome
```

Why this contract is better:

- compile, correctness, benchmark, and profile are separated,
- bottleneck tags remain, but now carry evidence and provenance,
- the Profile Interpreter becomes explainable and testable.

### Optimization State

```python
class RoundSummary(BaseModel):
    round_number: int
    mode: Mode
    direction: str
    num_candidates: int
    num_improved: int
    best_objective_score: float | None = None
    abs_gain_vs_prev_best_us: float | None = None
    rel_gain_vs_prev_best: float | None = None


class OptimizationState(BaseModel):
    problem_spec: ProblemSpec
    baseline: BaselineArtifact
    incumbent: BaselineArtifact
    current_round: int = 0
    rounds: list[RoundSummary] = []
    attempts: list[AttemptRecord] = []
    tabu_entries: list[TabuEntry] = []
    bottleneck_history: list[BottleneckAssessment] = []
```

Why this contract is better:

- baseline and incumbent are explicit,
- relative and absolute gains are stored separately,
- bottleneck history keeps evidence instead of only labels.

---

## Revised Service Boundaries

The high-level module count can stay roughly the same. What changes is the strength of the interface between them.

### Compiler Service

Responsibilities:

- compile source into target-specific artifact,
- emit static analysis,
- run correctness validation across required shapes.

Suggested contract:

```python
compile_and_validate(
    candidate: KernelCandidate,
    problem_spec: ProblemSpec,
) -> tuple[CompileStatus, StaticAnalysis | None, CorrectnessResult | None]
```

### Benchmarker

Responsibilities:

- run fast benchmark across all objective shapes,
- compute aggregate objective score,
- classify regression against incumbent,
- collect raw profiling data for top-K candidates only.

Suggested contract:

```python
benchmark_candidate(
    candidate_hash: str,
    problem_spec: ProblemSpec,
    baseline: BaselineArtifact,
    incumbent: BaselineArtifact,
) -> BenchmarkBundle


collect_profile(
    candidate_hash: str,
    problem_spec: ProblemSpec,
    shape_ids: list[str],
) -> RawProfileBundle
```

### Profile Interpreter

Responsibilities:

- normalize raw profile counters,
- compute derived quantitative metrics,
- assign bottleneck tags,
- emit a primary bottleneck, supporting evidence, and rule trace.

Suggested contract:

```python
interpret_profile(raw_profile: RawProfileBundle) -> ProfileBundle
```

Why keep these boundaries:

- compile and correctness stay close to the toolchain,
- benchmarking owns objective scoring and top-K selection,
- profile interpretation stays deterministic and explainable.

---

## Navigator Contract Corrections

The Strategy Navigator should reason over typed measurements, not compressed strings.

It should consume:

- `baseline.objective_score`,
- `incumbent.objective_score`,
- `RoundSummary.rel_gain_vs_prev_best`,
- `AttemptRecord` history,
- `TabuEntry` history,
- `BottleneckAssessment.primary_tag` and `evidence`.

Corrected derived signals:

```text
avg_relative_gain = mean(last_n.rel_gain_vs_prev_best)
is_plateau = avg_relative_gain < plateau_threshold
is_regress = avg_relative_gain < 0
near_target = incumbent.objective_score.value <= target_metric_value / target_threshold
```

The important point is simple: thresholded gates must use unit-consistent ratios, not raw microsecond deltas.

---

## Minimal Migration Plan

The smallest implementation order that materially improves the architecture is:

### Phase A: Baseline Seeding

1. introduce `BaselineArtifact`,
2. add bootstrap before round 0,
3. initialize incumbent from the measured baseline.

### Phase B: Unit-Correct State

1. add `rel_gain_vs_prev_best`,
2. rank by explicit objective score,
3. rewrite plateau and near-target logic against relative metrics.

### Phase C: Search Memory Repair

1. add `AttemptRecord`,
2. add `TabuEntry`,
3. enforce pairwise tabu on `(base_kernel_hash, direction)`.

### Phase D: Profile Evidence Preservation

1. add `ProfileMetrics`,
2. add `BottleneckAssessment`,
3. preserve `rule_trace` and evidence.

### Phase E: Workload Objective

1. add `ShapeCase`,
2. add `PerformanceObjective`,
3. move ranking and incumbent selection to aggregate multi-shape objective scoring.

---

## Bottom Line

The original decomposition was directionally right:

- Orchestrator for sequencing,
- Strategy Navigator for search policy,
- Coding Agent for synthesis,
- deterministic services for evaluation.

The biggest issue was not the number of modules. It was contract strength.

The corrected architecture keeps the system simple, but makes it much harder to:

- optimize against the wrong baseline,
- use mathematically inconsistent strategy signals,
- over-prune the search space with weak tabu state,
- lose the evidence behind bottleneck judgments,
- overfit to a single benchmark point instead of the actual workload.
