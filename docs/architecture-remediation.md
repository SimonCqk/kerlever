# Kerlever Architecture Remediation

This document proposes the smallest architecture correction that upgrades the current skeleton from a control-flow prototype into a measurement-grounded kernel optimization system.

Scope:

- address findings 1-5 from the initial architecture review,
- preserve the current high-level module count where possible,
- optimize the data contracts so the system can support rigorous CUDA operator development.

Non-goals:

- resumability and checkpoint restore,
- distributed scheduling,
- a larger agent graph,
- productization concerns outside the optimization loop.

---

## 1. Summary of Required Corrections

| Finding | Current Weakness | Required Correction |
|---|---|---|
| 1. Baseline not seeded | loop starts with no measured parent kernel | add explicit baseline bootstrap stage |
| 2. Unit mismatch in strategy signals | absolute and relative gains are conflated | separate absolute deltas from relative gains in state |
| 3. Tabu state too weak | pairwise rule enforced with flat strings | store typed attempt records and typed tabu entries |
| 4. Profile evidence lost | bottleneck tags survive, metrics disappear | preserve raw metrics, derived metrics, and interpretation trace |
| 5. Objective too scalar | one latency number stands in for workload quality | store per-shape results and rank with an explicit objective |

---

## 2. Revised Architecture

The corrected architecture keeps the same core modules, but inserts one mandatory bootstrap stage and strengthens the contracts between services.

```
Problem Spec / Oracle
        |
        v
Baseline Bootstrap
compile -> correctness -> benchmark -> profile -> seed OptimizationState
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
fast bench on all objective shapes -> rank -> deep profile top-K
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

### Key Changes

1. The system now has a `Baseline Bootstrap` stage before round 0.
2. "Global best" is seeded from the measured baseline, not from the first generated candidate.
3. Candidate evaluation produces typed artifacts instead of a single compressed result.
4. The Navigator consumes unit-aware gains and typed attempt history.
5. The workload objective is explicit and can span multiple shapes.

---

## 3. Revised Control Flow

### Stage 0: Baseline Bootstrap

Before the first optimization round:

1. Resolve the `reference_kernel`.
2. Compile it on the target GPU toolchain.
3. Run correctness checks across all required shapes.
4. Run fast benchmark across all objective shapes.
5. Run deep profile on the designated profiling shape set.
6. Seed `OptimizationState` with the resulting baseline artifact.

If bootstrap fails, the optimization loop must not start.

This changes the semantics of round 0:

- round 0 is no longer "search with no history";
- round 0 is "first search step with a measured parent kernel and measured bottleneck evidence."

### Stage 1: Strategy Decision

The Navigator now reasons over:

- baseline artifact,
- current incumbent artifact,
- typed attempt history,
- per-shape objective scores,
- interpreted bottlenecks with supporting evidence.

### Stage 2: Candidate Generation

The Coding Agent still generates exploit and explore candidates, but exploit mode now always has a real parent unless the system explicitly chooses a de novo explore branch.

### Stage 3: Candidate Evaluation

Evaluation is split conceptually into four outputs:

1. compile artifact,
2. correctness artifact,
3. benchmark artifact,
4. profile artifact.

Early exits remain:

- compile fail: stop,
- correctness fail: stop,
- benchmark regression: stop before deep profile,
- only ranked candidates enter deep profile.

### Stage 4: State Update

State update now records:

- attempt lineage,
- objective score,
- per-shape benchmark results,
- profile evidence,
- bottleneck interpretation,
- tabu entries keyed by parent context.

---

## 4. Revised Data Contracts

The current contracts are too compressed. The system needs richer typed records.

The definitions below are architectural contracts, not exact implementation requirements.

### 4.1 Problem Specification

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

### Why This Is Better

- `shape_cases` replaces a bare list of shapes with workload semantics.
- `weight` lets the system optimize for real traffic mix.
- `profile` selects which shapes are worth deep profiling.
- `objective` makes ranking explicit instead of implicit.

### Objective Computation

For a weighted latency objective:

```text
weighted_p50_us = sum(weight_i * p50_us_i) / sum(weight_i)
rel_gain_vs_prev_best = (prev_best - current) / prev_best
rel_gain_vs_baseline = (baseline - current) / baseline
```

These relative gains, not raw microsecond deltas, should drive plateau and regression logic.

---

### 4.2 Baseline and Incumbent Artifacts

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

### Why This Is Better

- the baseline becomes measurable state, not a user-supplied scalar,
- the incumbent can use the same contract as the baseline,
- strategy logic can compare like-for-like artifacts.

---

### 4.3 Candidate and Attempt History

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

### Why This Is Better

- `parent_hashes` supports both exploit and recombination.
- `AttemptRecord` gives the Navigator exact search memory.
- attempt history becomes the source of truth for tabu and exhaustion logic.

---

### 4.4 Typed Tabu Entries

```python
class TabuEntry(BaseModel):
    base_kernel_hash: str | None
    direction: str
    sub_mode: SubMode | None
    round_number: int
    expires_after_round: int
```

### Why This Is Better

- it matches the intended invariant directly,
- the same direction on a new base kernel is allowed,
- expiry becomes deterministic and auditable.

---

### 4.5 Evaluation Output

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

### Why This Is Better

- compile, correctness, benchmark, and profile are no longer collapsed into one opaque record,
- bottleneck tags remain, but they carry evidence and rule provenance,
- the Profile Interpreter becomes explainable and testable.

---

### 4.6 Optimization State

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

### Why This Is Better

- the system carries both baseline and incumbent explicitly,
- relative and absolute gains are stored separately,
- bottleneck history retains evidence instead of a bare tag list.

---

## 5. Revised Service Boundaries

The module graph can stay almost unchanged, but the contracts need to be strengthened.

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
- emit evidence and rule trace.

Suggested contract:

```python
interpret_profile(raw_profile: RawProfileBundle) -> ProfileBundle
```

### Why Keep These Boundaries

- compile/correctness stays closest to the toolchain,
- benchmarking owns ranking and objective scoring,
- profile interpretation stays deterministic and explainable.

This preserves the original architecture while removing ambiguity from the data flow.

---

## 6. Navigator Contract Changes

The Navigator should no longer infer strategy from compressed strings alone.

It should consume:

- `baseline.objective_score`,
- `incumbent.objective_score`,
- `RoundSummary.rel_gain_vs_prev_best`,
- `AttemptRecord` history,
- `TabuEntry` history,
- `BottleneckAssessment.primary_tag` plus `evidence`.

### Corrected Derived Signals

```python
avg_relative_gain = mean(last_n.rel_gain_vs_prev_best)
is_plateau = avg_relative_gain < plateau_threshold
is_regress = avg_relative_gain < 0
near_target = incumbent.objective_score.value <= target_metric_value / target_threshold
```

The important change is that all thresholded gates use relative gain or objective ratios, not raw microsecond deltas.

---

## 7. Profile Interpreter Contract Changes

The Profile Interpreter should output more than tags.

It should produce:

1. normalized metrics,
2. derived metrics,
3. bottleneck tags,
4. a primary bottleneck,
5. the rule trace used to classify the bottleneck.

Example:

```python
BottleneckAssessment(
    tags=["memory_bandwidth", "low_coalescing_efficiency"],
    primary_tag="memory_bandwidth",
    evidence={
        "dram_throughput_pct_of_peak": 34.2,
        "l1_sector_utilization_pct": 51.0,
        "arithmetic_intensity_flop_per_byte": 0.42,
    },
    rule_trace=[
        "dram_throughput_pct_of_peak < 60",
        "arithmetic_intensity below roofline crossover",
        "sector utilization indicates poor coalescing",
    ],
)
```

This keeps the system aligned with first-principles optimization.

---

## 8. Minimal Migration Plan

This is the smallest implementation order that makes the architecture materially more correct.

### Phase A: Baseline Seeding

Implement first:

1. `BaselineArtifact`
2. bootstrap step before round 0
3. incumbent initialized from baseline

This closes finding 1 immediately.

### Phase B: Unit-Correct State

Implement next:

1. `rel_gain_vs_prev_best`
2. objective-based round summaries
3. plateau and near-target gates rewritten against relative metrics

This closes finding 2.

### Phase C: Search Memory Repair

Implement next:

1. `AttemptRecord`
2. `TabuEntry`
3. pairwise tabu enforcement

This closes finding 3.

### Phase D: Profile Evidence Preservation

Implement next:

1. `ProfileMetrics`
2. `BottleneckAssessment`
3. `rule_trace`

This closes finding 4.

### Phase E: Workload Objective

Implement next:

1. `ShapeCase`
2. `PerformanceObjective`
3. per-shape ranking and aggregate score

This closes finding 5.

---

## 9. Bottom Line

The current skeleton already has the right control-plane decomposition:

- Orchestrator for sequencing,
- Navigator for search policy,
- Coding Agent for synthesis,
- deterministic services for evaluation.

The main issue is not module count. The issue is contract strength.

The corrected design keeps the architecture simple, but makes it much harder for the system to optimize against the wrong baseline, reason over the wrong units, forget why it made a profiling judgment, or overfit to a single shape.
