# Orchestrator Module Specification

## §1 Overview

The Orchestrator is the global control loop of the Kerlever optimization system. It drives the full kernel optimization cycle: request a strategy, generate kernel candidates, evaluate them on GPU hardware, analyze results, update state, and repeat until the target performance is met or the round budget is exhausted.

The Orchestrator does not contain optimization intelligence. It is a state machine that sequences calls to four downstream services (all accessed via async Protocols), manages global optimization state, enforces early-exit rules for failed or regressing candidates, and persists every round's state to a workdir for auditability.

**V1 scope:** The Orchestrator runs end-to-end with stub implementations of all four downstream Protocols. No real LLM calls, no real GPU evaluation. The goal is to prove the control flow, state management, early-exit paths, and persistence are correct.

---

## §2 Requirements

### Functional Requirements

**REQ-ORCH-001: Loop Termination** [traces SC-1]
The optimization loop must always terminate. It runs until either (a) the target performance is met or (b) the maximum round count is reached. The loop must not hang, deadlock, or crash under normal operation, including when downstream services return errors or empty results.

**REQ-ORCH-002: State Persistence** [traces SC-2]
After each round completes, the Orchestrator must persist the round state to the workdir. Upon loop termination, it must write a final result file. The workdir must contain:
- `state.json` — full optimization state snapshot (overwritten each round)
- `rounds/round_NNN.json` — per-round state (one file per round, zero-padded)
- `kernels/<hash>.cu` — source code for every generated kernel candidate
- `decision_log.jsonl` — append-only log of strategy decisions and round outcomes
- `result.json` — final optimization result (written once at termination)

**REQ-ORCH-003: Global Best Tracking** [traces SC-3]
The Orchestrator must maintain a global best kernel across all rounds. After evaluating candidates in a round, if any candidate achieves better latency than the current global best (and passes compile + correctness), the global best must be updated. The global best latency must be monotonically non-increasing across rounds (a new best is only set when strictly better).

**REQ-ORCH-004: Candidate Discard on Failure** [traces SC-4]
Candidates that fail compilation or correctness checks must be discarded immediately. They must not participate in best-kernel comparison, cross-candidate analysis, or any downstream processing. Their outcome must be recorded as COMPILE_FAIL or CORRECTNESS_FAIL respectively.

**REQ-ORCH-005: Candidate Discard on Regression** [traces SC-4]
Candidates whose benchmark latency is worse than the current global best must be discarded from the "improving" pool. Their outcome must be recorded as REGRESSION. They must not participate in cross-candidate analysis. (Note: the regression threshold is determined by the evaluation pipeline, not the Orchestrator — the Orchestrator acts on the outcome field returned by the pipeline.)

**REQ-ORCH-006: Concurrent Candidate Evaluation** [traces SC-1]
Multiple kernel candidates produced in a single round must be evaluated concurrently using asyncio.TaskGroup. A failure in one candidate's evaluation must not abort or corrupt the evaluation of other candidates in the same round.

**REQ-ORCH-007: Cross-Candidate Analysis Gate** [traces SC-1]
Cross-candidate analysis must only be invoked when a round produces two or more candidates that passed evaluation (outcome is IMPROVED or BASELINE_MATCH). If fewer than two candidates pass, cross-candidate analysis is skipped for that round.

**REQ-ORCH-008: Problem Spec Loading** [traces SC-1]
The Orchestrator must accept a ProblemSpec (loaded from YAML) that defines the optimization target: operation name, shapes, dtype, target GPU, baseline performance, target performance, tolerance, and maximum rounds.

### Quality Gates

**QG-ORCH-001: Type Safety** [traces SC-5]
All source code must pass `mypy --strict` with no errors.

**QG-ORCH-002: Lint** [traces SC-5]
All source code must pass `ruff check` with no errors.

---

## §3 Scenarios

**SCN-ORCH-001-01: Normal loop to target met**
- GIVEN: a ProblemSpec with target_perf_us = 1.0 and max_rounds = 10
- AND: the evaluation pipeline returns progressively improving latencies across rounds
- WHEN: a candidate achieves latency <= target_perf_us (within tolerance)
- THEN: the loop terminates with status TARGET_MET
- AND: result.json contains the best kernel and its latency
- AND: the number of completed rounds is <= max_rounds

**SCN-ORCH-001-02: Loop exhausts max_rounds**
- GIVEN: a ProblemSpec with max_rounds = 3
- AND: no candidate ever meets target_perf_us
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

**SCN-ORCH-003-01: Global best updates when improvement found**
- GIVEN: current global best has latency 2.0 us
- WHEN: a round produces a candidate with latency 1.5 us that passes correctness
- THEN: global best is updated to that candidate
- AND: global best latency becomes 1.5 us

**SCN-ORCH-003-02: Global best unchanged when no improvement**
- GIVEN: current global best has latency 2.0 us
- WHEN: all candidates in a round have latency >= 2.0 us
- THEN: global best remains unchanged

**SCN-ORCH-004-01: Compile-failed candidate is discarded**
- GIVEN: a round with 3 candidates
- AND: candidate B fails compilation
- WHEN: evaluation completes for all candidates
- THEN: candidate B's outcome is COMPILE_FAIL
- AND: candidate B is not considered for global best update
- AND: candidate B is not included in cross-candidate analysis input

**SCN-ORCH-004-02: Correctness-failed candidate is discarded**
- GIVEN: a round with 3 candidates
- AND: candidate C passes compilation but fails correctness
- WHEN: evaluation completes for all candidates
- THEN: candidate C's outcome is CORRECTNESS_FAIL
- AND: candidate C is not considered for global best update
- AND: candidate C is not included in cross-candidate analysis input

**SCN-ORCH-005-01: Regressing candidate is discarded**
- GIVEN: current global best has latency 2.0 us
- AND: candidate A evaluates to latency 3.0 us
- WHEN: evaluation completes
- THEN: candidate A's outcome is REGRESSION
- AND: candidate A is not included in cross-candidate analysis input

**SCN-ORCH-006-01: Concurrent evaluation isolates failures**
- GIVEN: a round with 4 candidates
- AND: the evaluation pipeline raises an unexpected exception for candidate 2
- WHEN: evaluation runs concurrently
- THEN: candidates 1, 3, and 4 still complete evaluation normally
- AND: candidate 2 is recorded with outcome ERROR

**SCN-ORCH-007-01: Cross-analysis skipped with fewer than 2 passing candidates**
- GIVEN: a round produces 3 candidates
- AND: 2 fail compilation, 1 passes with IMPROVED outcome
- WHEN: the round reaches the analysis phase
- THEN: cross-candidate analysis is not invoked
- AND: the round proceeds directly to state update

**SCN-ORCH-008-01: ProblemSpec loaded from YAML**
- GIVEN: a YAML file with valid fields (op_name, shapes, dtype, target_gpu, baseline_perf_us, target_perf_us, tolerance, max_rounds)
- WHEN: the Orchestrator is initialized with this file path
- THEN: a ProblemSpec is constructed with all fields populated
- AND: the loop uses these values for termination checks

---

## §4 Invariants

**INV-ORCH-001: Global best latency is monotonically non-increasing**
Once a global best is set, its latency can only decrease (improve) or stay the same. A candidate with worse latency must never replace the current best.
*Enforcement:* The global best update logic compares candidate latency < current best latency before assignment. This check is the sole path to updating the global best.

**INV-ORCH-002: Round counter is strictly increasing**
The round counter increments by exactly 1 after each round. No round number is skipped or repeated.
*Enforcement:* The round counter is incremented at exactly one point in the loop — after all round processing (evaluation, analysis, persistence) completes. It is not modified elsewhere.

**INV-ORCH-003: Every generated kernel is persisted**
Every kernel candidate's source code is written to `kernels/<hash>.cu` before evaluation begins, regardless of whether it later fails compilation or is discarded.
*Enforcement:* Kernel source files are written immediately after receiving candidates from the Coding Agent, before dispatching evaluation.

**INV-ORCH-004: Failed candidates never reach analysis**
Candidates with outcome COMPILE_FAIL, CORRECTNESS_FAIL, REGRESSION, or ERROR are filtered out before being passed to cross-candidate analysis.
*Enforcement:* The cross-candidate analysis input is constructed by filtering evaluation results to only those with outcome IMPROVED or BASELINE_MATCH.

**INV-ORCH-005: State file writes are atomic**
State files (state.json, round files, result.json) must not be left in a partially written state after a crash.
*Enforcement:* All state writes use a write-to-temporary-then-rename pattern. Write to `<target>.tmp`, then `os.replace()` to the final path.

---

## §5 Interfaces

### Protocol Signatures

The Orchestrator depends on exactly four downstream Protocols. All methods are async.

```
StrategyNavigatorProtocol:
    decide(problem_spec, optimization_state, round_summary, cross_analysis) -> StrategyDirective

CodingAgentProtocol:
    generate(problem_spec, directive, current_best_source) -> list[KernelCandidate]

GPUPipelineProtocol:
    evaluate(candidate, problem_spec, current_best_latency_us) -> EvaluationResult

CrossCandidateAnalyzerProtocol:
    analyze(top_k_results, problem_spec) -> CrossCandidateAnalysis
```

Parameter semantics:
- `optimization_state`: Full accumulated state (rounds history, tabu list, bottleneck history, global best).
- `round_summary`: Compressed summary of the most recent round, consumed by Strategy Navigator.
- `cross_analysis`: Output from the previous round's cross-candidate analysis, or None on the first round.
- `current_best_source`: Source code of the current best kernel, or None on the first round.
- `current_best_latency_us`: Latency of the current best kernel in microseconds, or None on the first round. Used by the pipeline to determine regression.
- `top_k_results`: List of (candidate, evaluation_result) pairs for candidates that passed evaluation in the current round.

### Data Types

All types are Pydantic BaseModel (for serialization) or Python enums.

**ProblemSpec** (from YAML):
- op_name: str
- op_semantics: str
- shapes: list of shape tuples
- dtype: str
- target_gpu: str
- baseline_perf_us: float
- target_perf_us: float
- tolerance: float (fractional, e.g. 0.05 means 5%)
- max_rounds: int
- reference_kernel: str (source code of the reference/baseline kernel)

**KernelCandidate**:
- code_hash: str (content hash of source_code)
- source_code: str
- intent_tag: str (what this candidate is trying to optimize)
- parent_hash: str | None (hash of the kernel it was derived from)
- mode: Mode
- sub_mode: SubMode | None

**EvaluationResult**:
- candidate_hash: str
- compile_result: CompileResult
- bench_result: BenchResult | None (None if compile failed)
- profile_result: ProfileResult | None (None if bench failed or regressed)
- outcome: CandidateOutcome

**CompileResult**:
- status: CompileStatus (SUCCESS, COMPILE_ERROR, CORRECTNESS_FAIL)
- error_message: str | None
- register_count: int | None
- smem_bytes: int | None

**BenchResult**:
- latency_us: float
- p50_us: float
- p95_us: float

**ProfileResult**:
- bottleneck_tags: list[str]
- metrics: dict[str, float]

**StrategyDirective**:
- mode: Mode
- direction: str
- reason: str
- base_kernel_hash: str | None
- num_candidates: int
- tabu: list[str]

**CrossCandidateAnalysis**:
- insights: list[str]
- winning_genes: list[str]
- recombination_suggestions: list[str]

**RoundState**:
- round_number: int
- phase: Phase
- directive: StrategyDirective
- candidates: list[KernelCandidate]
- evaluation_results: list[EvaluationResult]
- cross_analysis: CrossCandidateAnalysis | None
- best_candidate_hash: str | None (round-level best)
- best_latency_us: float | None

**RoundSummary**:
- round_number: int
- mode: Mode
- direction: str
- num_candidates: int
- num_improved: int
- best_latency_us: float | None
- improvement_over_prev_best: float | None

**OptimizationState**:
- problem_spec: ProblemSpec
- current_round: int
- global_best_hash: str | None
- global_best_latency_us: float | None
- global_best_source: str | None
- rounds: list[RoundSummary]
- tabu_list: list[str] (list of intent_tags recently tried)
- bottleneck_history: list[list[str]] (per-round bottleneck tags)
- decision_log: list[dict] (structured log entries)

**OptimizationResult**:
- status: str (TARGET_MET or MAX_ROUNDS_REACHED)
- best_kernel_hash: str | None
- best_latency_us: float | None
- best_kernel_source: str | None
- total_rounds: int
- total_candidates_evaluated: int

**Enums:**
- Phase: AWAITING_STRATEGY, AWAITING_CODING, AWAITING_EVALUATION, ANALYSIS, ROUND_COMPLETE
- Mode: EXPLOIT, EXPLORE
- SubMode: PARAM_SEARCH, LOCAL_REWRITE, PATTERN_APPLY, DE_NOVO, RECOMBINATION
- CompileStatus: SUCCESS, COMPILE_ERROR, CORRECTNESS_FAIL
- CandidateOutcome: IMPROVED, BASELINE_MATCH, REGRESSION, COMPILE_FAIL, CORRECTNESS_FAIL, ERROR

### YAML Input Format

```yaml
op_name: matmul
op_semantics: "C[M,N] = A[M,K] @ B[K,N]"
shapes:
  - [4096, 4096, 4096]  # M, N, K
dtype: float16
target_gpu: A100
baseline_perf_us: 5.0
target_perf_us: 1.0
tolerance: 0.05
max_rounds: 20
reference_kernel: |
  __global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K) {
      // naive reference implementation
  }
```

---

## §6 Behavioral Specification

### 6.1 Main Loop

The Orchestrator operates as a single async function that runs a bounded loop. Each iteration of the loop is one "round" of optimization. The loop structure is:

```
Initialize: load ProblemSpec, create empty OptimizationState, create workdir
For each round from 0 to max_rounds - 1:
    1. Request strategy directive from Strategy Navigator
    2. Request kernel candidates from Coding Agent
    3. Persist all candidate source files to workdir
    4. Evaluate all candidates concurrently
    5. Classify outcomes and filter results
    6. Update global best if any candidate improved
    7. Check termination: if target met, break
    8. Run cross-candidate analysis (if >= 2 passing candidates)
    9. Update bottleneck history and tabu list
    10. Build round summary, persist round state, append to decision log
    11. Increment round counter
Build and persist OptimizationResult
```

**First round special case:** On round 0, there is no prior round_summary or cross_analysis. The Orchestrator passes None for both when calling the Strategy Navigator. There is no global best yet, so current_best_source and current_best_latency_us are also None.

### 6.2 State Machine

The Orchestrator tracks its current phase within each round for observability and debugging. The phase transitions are strictly sequential within a round:

```
AWAITING_STRATEGY
    → Strategy Navigator returns directive
AWAITING_CODING
    → Coding Agent returns candidates
AWAITING_EVALUATION
    → All concurrent evaluations complete
ANALYSIS
    → Cross-candidate analysis completes (or is skipped)
ROUND_COMPLETE
    → State persisted, round counter incremented
    → Back to AWAITING_STRATEGY (next round) or loop exit
```

The phase is informational — it records where the Orchestrator is in the current round. There are no conditional phase transitions or branches in the state machine. Every round passes through every phase in order (though the ANALYSIS phase may be a no-op when skipped per REQ-ORCH-007).

### 6.3 Early-Exit and Candidate Outcome Classification

After concurrent evaluation completes, each candidate has an EvaluationResult with an outcome field set by the GPU pipeline. The Orchestrator uses these outcomes to partition candidates:

| Outcome | Meaning | Enters best-update? | Enters cross-analysis? |
|---------|---------|---------------------|----------------------|
| IMPROVED | Latency better than current best | Yes | Yes |
| BASELINE_MATCH | Latency within tolerance of current best | No (not strictly better) | Yes |
| REGRESSION | Latency worse than current best | No | No |
| COMPILE_FAIL | Compilation failed | No | No |
| CORRECTNESS_FAIL | Compiled but produced wrong results | No | No |
| ERROR | Unexpected error during evaluation | No | No |

**Best-update rule:** Only candidates with outcome IMPROVED are eligible to become the new global best. Among multiple IMPROVED candidates in the same round, the one with the lowest latency wins.

**Cross-analysis input:** Candidates with outcome IMPROVED or BASELINE_MATCH form the input to cross-candidate analysis. If this set has fewer than 2 members, cross-analysis is skipped.

**Handling evaluation exceptions:** If a candidate's evaluation raises an unexpected exception (not a compile/correctness failure, but an infrastructure error), the Orchestrator catches it, records the candidate with outcome ERROR, and continues evaluating other candidates. The round proceeds normally with the remaining results.

### 6.4 Global Best Update

After classifying outcomes:

1. Filter candidates to those with outcome IMPROVED.
2. If the filtered set is empty, skip update.
3. Among filtered candidates, select the one with the lowest bench_result.latency_us.
4. If no global best exists yet, set it unconditionally.
5. If a global best exists, set the new one only if its latency is strictly less than the current global best latency.
6. When updating, store the new best's hash, latency, and source code in OptimizationState.

### 6.5 Termination Check

After updating the global best, check if the target has been met:

```
target_met = (global_best_latency_us is not None)
             and (global_best_latency_us <= target_perf_us * (1 + tolerance))
```

The tolerance allows a small margin. For example, if target_perf_us = 1.0 and tolerance = 0.05, then any latency <= 1.05 counts as target met.

If target_met is true, the loop breaks immediately. The current round's state is still persisted before exiting.

If the round counter reaches max_rounds without meeting the target, the loop exits naturally. This is a safety limit, not a goal.

### 6.6 Tabu and Bottleneck Management

**Tabu list:** After each round, the intent_tags of all candidates generated in that round are appended to the tabu list. The tabu list is passed to the Strategy Navigator in subsequent rounds so it can avoid repeating recently tried directions. The Orchestrator itself does not enforce tabu — it merely tracks and passes the information. Tabu management (windowing, expiry) is the Strategy Navigator's responsibility.

**Bottleneck history:** After each round, the bottleneck_tags from the ProfileResults of passing candidates are appended to the bottleneck_history (one entry per round, which is a list of all bottleneck tags seen in that round). If no candidates have profile results (all failed or regressed before profiling), an empty list is recorded for that round.

### 6.7 Round Summary Construction

After each round, the Orchestrator builds a RoundSummary containing:
- Round number
- Mode and direction from the directive
- Number of candidates generated
- Number with outcome IMPROVED
- Best latency achieved in this round (from the best passing candidate, or None)
- Improvement delta over the previous global best (or None if no improvement)

This summary is stored in OptimizationState.rounds and passed to the Strategy Navigator in the next round.

### 6.8 Decision Log

After each round, the Orchestrator appends a structured entry to decision_log.jsonl. Each entry is a single JSON line containing:
- round_number
- directive (mode, direction, reason, num_candidates)
- outcomes summary (count per outcome type)
- best_latency_this_round
- global_best_latency_after_round
- improvement (boolean: did global best improve this round?)

### 6.9 State Persistence

**StateManager** handles all filesystem operations. The Orchestrator calls it at defined points:

| When | What is written | File |
|------|----------------|------|
| After candidates received (step 3) | Each candidate's source code | `kernels/<hash>.cu` |
| After round completes (step 10) | Full optimization state snapshot | `state.json` |
| After round completes (step 10) | This round's complete state | `rounds/round_NNN.json` |
| After round completes (step 10) | Decision log entry (append) | `decision_log.jsonl` |
| After loop exits | Final result | `result.json` |

All writes to state.json, round files, and result.json use the atomic write pattern: write to `<path>.tmp`, then `os.replace(<path>.tmp, <path>)`. The decision_log.jsonl is append-only (open in append mode, write line, flush).

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
2. Creates stub implementations of all four Protocols
3. Instantiates the Orchestrator with the ProblemSpec, stubs, and workdir path
4. Runs the async main loop via `asyncio.run()`
5. Exits with code 0 on success

---

## §7 Production Path Trace

This traces the data and decision flow for a single round of the optimization loop, following the exact sequence an operator would observe.

**Trigger:** The loop enters a new round iteration (round N).

1. **Strategy request.** The Orchestrator sends the current optimization state, previous round summary (None if N=0), and previous cross-analysis (None if N=0) to the Strategy Navigator. It receives back a StrategyDirective specifying mode (exploit/explore), direction, and number of candidates to generate.

2. **Candidate generation.** The Orchestrator sends the ProblemSpec, the directive, and the current best kernel source (None if N=0) to the Coding Agent. It receives back a list of KernelCandidate objects, each with a unique code_hash, source code, and intent tag.

3. **Kernel persistence.** Each candidate's source code is written to `kernels/<hash>.cu` in the workdir. This happens before evaluation so that even if evaluation crashes, the source code is preserved.

4. **Concurrent evaluation.** All candidates are dispatched concurrently to the GPU Pipeline via asyncio.TaskGroup. Each evaluation runs independently. If one raises an exception, the Orchestrator catches it and marks that candidate as ERROR. All other evaluations continue. The result is a list of EvaluationResults, one per candidate.

5. **Outcome classification.** Each EvaluationResult carries an outcome set by the pipeline: IMPROVED, BASELINE_MATCH, REGRESSION, COMPILE_FAIL, CORRECTNESS_FAIL, or ERROR. The Orchestrator partitions results by outcome.

6. **Global best update.** Among IMPROVED candidates, the one with the lowest latency is compared against the current global best. If it is strictly better (or no global best exists yet), it becomes the new global best.

7. **Termination check.** If the global best latency is at or below the target (within tolerance), the loop sets status TARGET_MET and proceeds to persist the round state before exiting.

8. **Cross-candidate analysis.** If two or more candidates have outcome IMPROVED or BASELINE_MATCH, they are sent to the Cross-Candidate Analyzer, which returns insights about code patterns. If fewer than two pass, this step is skipped.

9. **State update.** The tabu list is extended with intent_tags from this round's candidates. Bottleneck history is extended with the profile tags from passing candidates. A RoundSummary is constructed and appended to the state.

10. **Persistence.** The full RoundState is written to `rounds/round_NNN.json`. The OptimizationState snapshot is written to `state.json`. A decision log entry is appended to `decision_log.jsonl`.

11. **Next round.** The round counter increments. Control returns to step 1, unless the loop is exiting.

**On loop exit:** The Orchestrator builds an OptimizationResult (status, best kernel, total rounds, total candidates evaluated) and writes it to `result.json`.

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | Partial state on crash | Writing state.json directly without atomic rename | Corrupted state file if process killed mid-write | Use tmp-then-rename for all state writes (INV-ORCH-005) |
| 2 | Lost kernel source | Writing kernel files after evaluation instead of before | If evaluation crashes, the candidate source code is lost | Write kernels immediately after generation, before evaluation (INV-ORCH-003) |
| 3 | Cascading evaluation failure | Using a single try/except around TaskGroup instead of per-task handling | One failed candidate kills the entire round | Catch exceptions inside each evaluation task; record ERROR outcome per candidate |
| 4 | False global best | Not checking compile/correctness status before best-update comparison | A candidate with wrong results could become the "best" | Filter to IMPROVED-only before best comparison (INV-ORCH-004) |
| 5 | Stale tabu data | Forgetting to update tabu list when a round has no improving candidates | Strategy Navigator re-suggests the same direction repeatedly | Always extend tabu with all intent_tags from the round, regardless of outcomes |

---

## §9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|-------------|-----------|
| SC-1: Loop runs to termination with stubs, no hang/crash | REQ-ORCH-001, REQ-ORCH-006, REQ-ORCH-007, REQ-ORCH-008 | SCN-ORCH-001-01, SCN-ORCH-001-02, SCN-ORCH-006-01, SCN-ORCH-007-01, SCN-ORCH-008-01 |
| SC-2: Workdir produces complete artifacts | REQ-ORCH-002 | SCN-ORCH-002-01 |
| SC-3: Global best updates correctly | REQ-ORCH-003 | SCN-ORCH-003-01, SCN-ORCH-003-02 |
| SC-4: Failed/regressing candidates discarded | REQ-ORCH-004, REQ-ORCH-005 | SCN-ORCH-004-01, SCN-ORCH-004-02, SCN-ORCH-005-01 |
| SC-5: mypy --strict and ruff check pass | QG-ORCH-001, QG-ORCH-002 | (quality gate, verified by CI) |
