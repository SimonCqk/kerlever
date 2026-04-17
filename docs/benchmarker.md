# Benchmarker

The Benchmarker is the deterministic measurement service of the optimization
loop. It decides **how fast a compiled and verified kernel is on the target
workload**, whether the result is stable enough to trust, which candidates
survive the regression guard, and which survivors deserve deep profiling.

It is a **measurement-producing service**, not a search-policy module. It does
not generate kernels, prove correctness, interpret bottlenecks, update the
incumbent, or choose the next optimization direction.

---

## Inputs

```text
+----------------------+  +----------------------+  +----------------------+
| Compiler Service      |  | Problem Spec /       |  | Baseline /          |
| Success Artifact      |  | Operation Adapter    |  | Incumbent Context   |
| artifact_id, cubin,   |  | shapes, weights,     |  | objective scores,   |
| launch spec, static   |  | dtype, objective,    |  | prior measured      |
| analysis, toolchain   |  | useful work model    |  | shape results       |
+----------+-----------+  +----------+-----------+  +----------+-----------+
           |                         |                         |
           +-----------+-------------+-------------+-----------+
                       |                           |
                       v                           v
              +------------------------------------------------+
              |       GPU Pipeline Adapter / Benchmark Request |
              | request id, run id, candidate hash, batch id   |
              +----------------------+-------------------------+
                                     |
                                     v
        =========================================================
         Phase 1 - Request Normalization / Measurement Envelope
        =========================================================
```

The Benchmarker only receives candidates that passed the Compiler Service
compile, correctness, and sanitizer gates. A kernel that has not passed
correctness must never be timed for optimization scoring.

---

## Core Responsibilities

| Responsibility | Meaning |
|---|---|
| Fast benchmark | Measure kernel latency on every objective shape using a controlled GPU timing path |
| Measurement quality | Detect noise, drift, throttling, competing work, unstable samples, and invalid timing episodes |
| Objective scoring | Aggregate per-shape measurements into the `PerformanceObjective` score |
| Regression guard | Compare candidate score against the incumbent under a unit-consistent threshold |
| Candidate ranking | Select score winners and diagnostic candidates for deep profiling |
| Deep profiling | Collect raw Nsight Compute / Nsight Systems metrics for selected candidates and profile shapes |
| Evidence preservation | Return raw measurements, normalized metrics, run identity, and artifact references without flattening them into prose |

---

## Design Principles

### 1. Timing and Profiling Are Separate

Benchmark timing produces optimization scores. Profiling produces diagnostic
counters. Nsight Compute and Nsight Systems can add overhead, replay kernels, or
alter execution behavior, so profiler durations are not objective scores.

### 2. Compare Candidates Under The Same Measurement Episode

Stored incumbent latency is useful history, but it is not enough for a fair
regression decision when the current GPU may be hotter, clocked differently, or
running on a different pod. The incumbent should be remeasured as an anchor in
the same benchmark episode whenever a candidate may replace it.

### 3. Benchmark The Workload, Not A Convenient Shape

Every `ShapeCase` in `ProblemSpec.shape_cases` participates in scoring. Profile
shapes are a subset for diagnostics; they are not a replacement for the full
multi-shape objective.

### 4. Treat Measurement Quality As Data

Temperature, power throttling, clock state, active processes, variance,
coefficient of variation, and profiler availability are not side notes. They
are structured fields that determine whether a benchmark result is usable.

### 5. Distributed Speed Does Not Override Statistical Fairness

Compiles can run in parallel. Timed benchmark runs on a single GPU must be
exclusive. Cross-pod ranking is allowed only when hardware and software identity
match and each pod has fresh anchor measurements.

### 6. Deterministic Service, No Natural-Language Decisions

The Benchmarker returns structured facts and deterministic routing outcomes:
passed, regressed, unstable, timed out, profile unavailable, or infrastructure
error. It does not emit optimization advice.

---

## Phase 1 - Request Normalization / Measurement Envelope

Normalize the request, bind it to concrete artifacts, and create the envelope
that makes the measurement reproducible.

```text
    +-------------------------------------------------------------+
    |              Normalize Benchmark Request                    |
    |                                                             |
    |  1. Validate Compiler Service status is success             |
    |  2. Resolve candidate artifact id and launch metadata       |
    |  3. Resolve objective shapes and profile shapes             |
    |  4. Bind baseline and incumbent references                  |
    |  5. Select operation adapter timing harness                 |
    |  6. Assign benchmark batch id and request id                |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                Create Measurement Envelope                  |
    |                                                             |
    |  identity:                                                   |
    |    run_id, round_id, batch_id, request_id, candidate_hash    |
    |                                                             |
    |  artifact identity:                                          |
    |    artifact_id, source_hash, launch_spec_hash, toolchain     |
    |                                                             |
    |  workload identity:                                          |
    |    problem_spec_hash, objective_hash, shape_ids, adapter     |
    |                                                             |
    |  device identity:                                            |
    |    target_gpu, gpu_uuid, pci_bus_id, mig_profile, sm_arch    |
    |                                                             |
    |  timing policy:                                              |
    |    warmup policy, repeat policy, cache policy, clock policy  |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 2 ]
```

### Batch-Level Preferred Interface

The current Orchestrator abstraction can evaluate candidates individually, but
the real Benchmarker should prefer a batch request for candidates from the same
round:

```text
BenchmarkBatchRequest
  run_envelope
  problem_spec
  baseline_ref
  incumbent_ref
  candidate_module_artifact_refs[]
  objective_shape_cases[]
  profile_shape_cases[]
  operation_adapter_abi
  top_k_profile
```

Batching enables fair interleaving, shared incumbent anchor runs, and top-K
selection across the round. A per-candidate adapter can still exist for V1, but
it should internally serialize timed GPU access and record that ranking quality
is weaker than a true batch episode.

### Artifact Execution Model

Long-term, the Benchmarker should use **cubin/module artifacts loaded by a
common benchmark harness**. The Compiler Service produces cubin/fatbin artifacts
plus structured launch metadata; the Benchmarker runs a stable host harness,
loads modules through the CUDA Driver API, resolves entrypoints, packs arguments
through the operation adapter, and launches all candidates in one controlled
CUDA context.

The common harness should run inside a disposable worker process for each
benchmark batch. If a generated kernel poisons the CUDA context, crashes the
process, or triggers an ambiguous device fault, the entire measurement episode
is marked ambiguous and retried according to fault-attribution policy. This
keeps the clean one-context ranking model without losing process-level fault
containment.

Other artifact models remain useful only as transitional or fallback paths:

| Model | What It Means | Strength | Weakness | Interleaving Policy |
|---|---|---|---|---|
| Cubin/module loaded by common harness | Stable harness loads candidate cubins through the CUDA driver API | Cleanest long-term model: one context, stable host code, no generated-source linking, fair ranking | Requires explicit entrypoint, launch metadata, and adapter ABI | Required for true batch interleaving |
| Shared object loaded by common harness | Candidate device code is linked into a stable benchmark harness process | Same process/context, easier shared allocation, fairer interleaving | Requires symbol isolation and stable host ABI | Interleaving allowed if all candidates use the same adapter ABI |
| Generated harness per batch | The service generates one harness containing incumbent plus all candidates | Can interleave fairly and keep adapter-specific code simple | More complex build step and symbol collision risk | Allowed if the generated harness proves one process/context and unique symbols |
| Separate executable per candidate | The Compiler Service builds one host executable per candidate | Maximum isolation and simplest early fallback | Separate process/context effects make fine-grained comparisons weaker | V1 fallback only; do not interleave individual repetitions; run candidate block with incumbent anchors before/after |

True interleaving means repeated launches happen inside one benchmark process,
one CUDA context, one leased device, and one operation adapter harness. If the
service cannot satisfy that, it must fall back to sequential candidate blocks
with incumbent anchors and mark the measurement episode as non-interleaved.

### Measurement Envelope Requirements

The measurement envelope must include every variable that can change runtime:

| Field | Why It Matters |
|---|---|
| `gpu_uuid` / `pci_bus_id` | GPU enumeration can change across reboots; UUID or PCI bus id is stable enough to audit |
| `target_gpu` / `sm_arch` | Cross-architecture timings are not comparable |
| `driver_version`, `cuda_runtime_version` | CUDA driver/runtime changes can alter launch behavior and generated code compatibility |
| `toolchain_hash` | Benchmark must match the compile artifact that was actually built |
| `launch_spec_hash` | Block size, dynamic smem, and ABI change performance |
| `module_artifact_hash` | Cubin/fatbin identity determines the executable device code |
| `artifact_execution_model` | Interleaving validity depends on common-harness versus separate-process execution |
| `metric_mode` | CUDA event, host launch, end-to-end, and CUDA graph replay timings answer different questions |
| `function_attribute_policy` | Dynamic shared-memory limit, cache config, shared-memory/L1 carveout, and cluster attributes change launch/runtime behavior |
| `objective_hash` | Shape weights and primary metric define ranking semantics |
| `clock_policy` / observed clocks | GPU boost, locked clocks, and throttling can dominate microbenchmarks |
| `cache_policy` | Warm-cache and cold-cache timings answer different questions |
| `batch_id` | Candidate comparisons are only fair inside a known measurement episode |

---

## Phase 2 - Device Lease / Measurement Hygiene

Acquire an exclusive timed-measurement lease on a physical GPU or MIG instance.
The service can use multiple GPUs in parallel, but only one timed benchmark job
may run per leased device.

```text
    +-------------------------------------------------------------+
    |                 Acquire GPU Measurement Lease               |
    |                                                             |
    |  - select GPU matching target_gpu and sm_arch               |
    |  - enforce one timed benchmark per GPU or MIG instance      |
    |  - verify no foreign active compute processes               |
    |  - record compute mode, MIG mode, and visible device id     |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                   Preflight Device State                    |
    |                                                             |
    |  - run known-good probe kernel if pod state is suspect      |
    |  - sample clocks, temperature, power, ECC/Xid state         |
    |  - check clock throttle / clock event reasons               |
    |  - apply clock policy if privileged and configured          |
    |  - fail or mark unstable if hygiene gates fail              |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 3 ]
```

### Device Hygiene Gates

The Benchmarker should reject or mark a timing episode unstable when any hard
hygiene condition is present:

| Condition | Result |
|---|---|
| GPU does not match target architecture | `infra_error`, do not benchmark |
| Foreign compute process on the same GPU | `unstable_measurement`, retry later |
| Clock throttle reason indicates thermal or power slowdown | `unstable_measurement` unless objective explicitly accepts throttled state |
| GPU temperature is above configured steady-state limit | warm/cooldown retry, then unstable |
| ECC double-bit error, Xid error, lost GPU, or failed probe | `infra_error`, quarantine pod |
| Profiler counter permission missing | fast benchmark may continue; profiling returns `profile_unavailable` |
| MIG profile differs from requested profile | `infra_error`, do not compare against full-GPU results |

### Clock Policy

Default policy is **observe and enforce hygiene**, because managed cloud GPU
environments often do not permit clock locking. Locked clocks are a deployment
hardening option, not the baseline requirement.

Preferred order:

1. Observe and record SM/memory clocks, power, temperature, and throttle or
   clock-event reasons before, during, and after each batch.
2. Reject or mark unstable any episode with active thermal, power, HW slowdown,
   or unexpected clock drift beyond tolerance.
3. If the pod has administrator/root permission and the GPU supports it, lock
   GPU and memory clocks to configured supported values for the benchmark pool.
4. Record whether clocks were `observed_only`, `locked`, or
   `lock_requested_unavailable`.

Clock locking is useful on dedicated benchmark hosts, but it must not be the
assumed production path. The hard requirement is that clock state and throttle
reasons are measured and used in the measurement-quality decision.

### Concurrency Rule

```text
CPU compile / artifact extraction:        parallel
Correctness execution:                    bounded by GPU health policy
Fast timed benchmark on one GPU:          serial
Deep profile on one GPU:                  serial
Different same-class GPUs in one pool:    parallel only with per-GPU anchors
Different GPU SKUs or MIG profiles:       never directly ranked together
```

---

## Phase 3 - Fast Benchmark Plan Calibration

Create a timing plan for each shape. The plan decides warmup count, iterations
per timing sample, repetition count, and cache policy before collecting final
samples.

```text
    +-------------------------------------------------------------+
    |                 Prepare Shape Benchmark Plan                |
    |                                                             |
    |  for each ShapeCase:                                        |
    |    - allocate inputs and outputs outside timed region       |
    |    - initialize deterministic inputs from adapter seed      |
    |    - run untimed warmup launches                            |
    |    - calibrate iterations per sample                        |
    |    - choose repetitions and cache policy                    |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |               Calibrate Iterations Per Sample               |
    |                                                             |
    |  elapsed_ms = timed repeated launches                       |
    |  if elapsed_ms < MIN_TIMED_BATCH_MS:                        |
    |      double iterations until threshold or max cap           |
    |  if elapsed_ms > MAX_TIMED_BATCH_MS:                        |
    |      reduce iterations or mark calibration warning          |
    |  per_launch_us = elapsed_ms * 1000 / iterations             |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 4 ]
```

### Metric Modes / Timing Scope

The Benchmarker does not choose the metric mode opportunistically. Metric mode
is part of `PerformanceObjective` and the operation adapter contract. The
Benchmarker validates support for the requested mode and returns
`not_comparable` or `unsupported_metric_mode` if the mode cannot be measured for
the candidate or adapter.

V1 optimizes device-side kernel execution latency by default. The default metric
name should be explicit:

```text
device_kernel_us
```

For `device_kernel_us`, the timed region should include:

- candidate kernel launch work on the target stream;
- optional device-side loop only if the adapter defines it as part of the
  kernel contract.

The timed region should exclude:

- host allocation;
- host-to-device and device-to-host copies;
- random input generation;
- reference kernel execution;
- correctness comparison;
- profiler startup;
- context creation.

If a future objective optimizes end-to-end operator latency including transfers
or host overhead, that must be a distinct `PerformanceObjective` and timing
harness mode. Kernel-only and end-to-end timings must not share the same metric
name.

Useful future metric modes should be named separately:

| Metric Mode | Includes | Excludes | Use Case |
|---|---|---|---|
| `device_kernel_us` | Device elapsed time from CUDA events around kernel launches | Host launch overhead, allocation, transfers, correctness checks | Default kernel optimization loop |
| `host_launch_us` | CPU-side enqueue/synchronization cost for launch path | Device execution work when separately measured | Very short kernels where launch overhead dominates |
| `operator_end_to_end_us` | Adapter-defined full operator path, including transfers or host work if specified | Nothing outside the adapter contract | User-visible latency objectives |
| `cuda_graph_replay_us` | CUDA graph replay path for supported adapters | Graph capture/build cost unless objective includes it | Microsecond kernels where graph replay changes launch overhead |

CUDA Graph replay is not a Benchmarker heuristic. If the objective requests
`cuda_graph_replay_us`, the adapter must define graph capture/update/replay
support and whether graph construction cost is excluded or included. If the
objective requests `device_kernel_us`, the Benchmarker must use ordinary stream
launch timing even when a CUDA Graph path would be faster.

### CUDA Event Timing Path

Use CUDA events in the benchmark stream for kernel elapsed time:

```text
record start event
repeat N kernel launches on the same stream
record stop event
synchronize stop event
elapsed_ms = cudaEventElapsedTime(start, stop)
per_launch_us = elapsed_ms * 1000 / N
```

For very short kernels, repeat enough launches per sample to make event and
launch overhead negligible relative to measured work. The selected `N` must be
stored in the per-shape result.

CUDA event elapsed time is returned as a floating-point millisecond value, so
calibration needs both lower and upper bounds:

```text
MIN_TIMED_BATCH_MS <= elapsed_ms <= MAX_TIMED_BATCH_MS
```

The lower bound keeps event resolution and launch overhead from dominating. The
upper bound keeps one sample from becoming so long that clock drift, thermal
state, and floating-point elapsed-time precision become hidden sources of error.
If a shape cannot satisfy both bounds within the iteration cap, mark the
measurement quality as `valid_with_warning` or `unstable` according to policy.

### Adapter Iteration Semantics

Repeated-launch timing is only valid when the operation adapter defines what
happens between launches. The Benchmarker must not assume every kernel can be
launched repeatedly against the same buffers.

```text
AdapterIterationSemantics
  overwrite_pure:             repeated launches are valid without reset
  requires_output_reset:      output buffers must be restored between launches
  requires_full_input_reset:  inputs and outputs must be restored between launches
  not_repeatable:             repeated-launch timing is disallowed
```

For `overwrite_pure`, calibration may repeat the kernel N times and divide by N.
For reset-required modes, reset work must either be outside the measured region
with a documented steady-state contract, or the adapter must use buffer rotation
so each measured launch sees equivalent state. For `not_repeatable`, the harness
must collect one launch per sample and mark launch overhead sensitivity in the
measurement quality artifact.

### Function Attribute Policy

Runtime function attributes are part of the executable measurement contract. The
same cubin can behave differently if the harness sets a different dynamic shared
memory limit or shared-memory/L1 carveout preference.

The measurement envelope should record:

```text
function_attribute_policy:
  max_dynamic_shared_memory_size: int | null
  preferred_shared_memory_carveout_pct: int | null
  cache_config: "prefer_none" | "prefer_shared" | "prefer_l1" | "prefer_equal" | null
  cluster_dims: [x, y, z] | null
  non_portable_cluster_size_allowed: bool | null
```

If the harness calls `cudaFuncSetAttribute`, `cudaFuncSetCacheConfig`, or driver
API equivalents, the requested values and the observed function attributes must
both be retained. A result measured under one carveout policy is not replayable
as the same benchmark episode under a different policy.

### Warmup Policy

Warmup is required before collecting samples because first-use effects can
include context creation, JIT behavior, cache state, memory page touch, clock
ramp, and allocator initialization. Warmup samples are not reported as
benchmark samples.

Default cache policy depends on measurement shape:

- single-candidate steady-state measurement: `warm_same_buffers`;
- interleaved batch measurement: `warm_rotating_buffers`.

Cold-cache benchmarking is valid only when the problem objective explicitly
requests it, and then the harness must define the cache-flush method as part of
the measurement envelope.

Cache policy should be an explicit enum, not a note:

| Cache Policy | Meaning |
|---|---|
| `warm_same_buffers` | Reuse the same allocated buffers after warmup; default steady-state path |
| `warm_rotating_buffers` | Rotate through equivalent buffers to avoid candidate-to-candidate cache coupling while keeping warm allocation state |
| `cold_flush_buffer` | Touch a configured eviction buffer between measured samples |
| `reset_persisting_l2` | Clear CUDA persisting L2 state before the measured block when the adapter or candidate uses persisting access policy |

The cache policy must be recorded in the measurement envelope because changing
it changes what the latency means.

If interleaving is enabled and the configured policy is `warm_same_buffers`, the
Benchmarker should automatically promote the effective policy to
`warm_rotating_buffers` and record:

```text
requested_cache_policy: "warm_same_buffers"
effective_cache_policy: "warm_rotating_buffers"
cache_policy_reason: "interleaved_batch_requires_rotation"
```

---

## Phase 4 - Fast Benchmark Execution / Quality Checks

Collect timed samples, compute robust latency statistics, and decide whether the
measurement is usable.

```text
    +-------------------------------------------------------------+
    |                  Execute Timed Samples                      |
    |                                                             |
    |  for each shape:                                            |
    |    - run repetitions in interleaved batch order             |
    |    - collect per-sample per-launch latency                  |
    |    - check CUDA launch and synchronization errors           |
    |    - record pre/post device telemetry                       |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                 Compute Shape Statistics                    |
    |                                                             |
    |  p50_us  = median(samples)                                  |
    |  p95_us  = percentile(samples, 95)                          |
    |  stdev   = standard_deviation(samples)                      |
    |  cv_pct  = stdev / mean(samples) * 100                      |
    |  derived = bandwidth, FLOP/s, arithmetic intensity          |
    |  samples = raw bounded sample list or artifact ref          |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                 Evaluate Measurement Quality                |
    |                                                             |
    |  - enough repetitions?                                      |
    |  - coefficient of variation under threshold?                |
    |  - p95/p50 ratio under threshold?                           |
    |  - no clock, thermal, power, ECC, or Xid issue?             |
    |  - incumbent anchor stable?                                 |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 5 ]
```

### Interleaving Within A Batch

When a round contains multiple candidates, benchmark repetitions should be
interleaved:

```text
shape_1:
  warmup all candidates and incumbent
  repeat randomized blocks:
    incumbent anchor
    candidate B
    candidate A
    candidate C
    incumbent anchor
```

This reduces bias from thermal drift and clock ramp. The random seed must be
stable and recorded:

```text
interleave_seed = hash(run_id, batch_id, shape_id, "kerlever_benchmark_order")
```

Interleaving is a measurement technique, not a search policy decision.

Interleaving is valid only for the chosen long-term execution model: candidate
cubins/modules loaded by a common benchmark harness in one worker process and
one CUDA context. If the service is using separate executables, it must run
sequential candidate blocks with incumbent anchors before and after each block.

The block size must be configured, not hand-picked in the harness:

```text
ANCHOR_EVERY_N_SAMPLES
MAX_INTERLEAVE_BLOCK_LEN
```

Short blocks increase sensitivity to drift but add anchor overhead. Long blocks
reduce anchor overhead but make drift harder to detect. The selected values and
actual realized block order must be stored in the measurement artifact.

### ShapeBenchResult

Current shared types can carry the minimum required fields:

```text
ShapeBenchResult
  shape_id
  latency_p50_us
  latency_p95_us
  stdev_us
  run_count
```

The real service should also preserve richer measurement metadata in its own
artifact, even if the Orchestrator-visible V1 type remains small:

```text
ShapeMeasurementArtifact
  shape_id
  samples_us[]
  warmup_count
  iterations_per_sample
  min_samples_required
  p50_us
  p95_us | null
  mean_us
  stdev_us
  cv_pct
  min_us
  max_us
  cache_policy
  requested_cache_policy
  effective_cache_policy
  interleave_block_len
  anchor_every_n_samples
  anchor_pre_us
  anchor_post_us
  anchor_drift_pct
  artifact_execution_model
  adapter_iteration_semantics
  metric_mode
  max_timed_batch_ms
  function_attribute_policy
  useful_bytes
  actual_bytes | null
  algorithmic_flops
  effective_bandwidth_gbps
  achieved_flops
  arithmetic_intensity_flop_per_byte
  measurement_quality
  telemetry_before
  telemetry_after
```

The Orchestrator needs the compact shape result. Debugging, audit, and future
statistical rules need the richer artifact.

The operation adapter owns the work model: useful bytes, algorithmic FLOPs, and
expected tensor shapes. The Benchmarker derives effective bandwidth and achieved
FLOP/s from measured latency. Hardware-counter bytes from profiling can later
fill `actual_bytes`, which lets the Profile Interpreter compare useful work to
traffic actually requested from memory.

### Sample Count And Quantiles

Quantiles must not be reported as if they were equally meaningful at every
sample count. The service should define minimum sample counts per statistic:

| Statistic | Minimum Requirement |
|---|---|
| `p50_us` | available when the valid sample count meets the basic benchmark minimum |
| `p95_us` | available only when `run_count >= MIN_P95_SAMPLES` |
| `cv_pct` | available only when at least two valid samples exist |

If the sample count is too small for p95, return `p95_us = null` in the rich
artifact and avoid using `weighted_p95_us` objectives until enough samples are
collected. The compact `ShapeBenchResult` can keep its current fields for
compatibility, but the real service must not pretend an under-sampled p95 is a
high-confidence tail estimate.

### Measurement Quality Status

| Status | Meaning | Routing |
|---|---|---|
| `valid` | Samples and device state pass quality gates | usable for scoring |
| `valid_with_warning` | Minor noise but below guard threshold | usable, warning preserved |
| `unstable` | Variance or device telemetry invalidates confidence | rerun or record non-signal failure |
| `runtime_fault` | Candidate faults during timed execution | candidate failure unless pod health is suspect |
| `infra_fault` | Device, profiler, artifact, or scheduler failure | no optimization signal |

Unstable timing must not count as a clean regression. A noisy pod is not
evidence that an optimization direction is bad.

---

## Phase 5 - Objective Scoring / Regression Guard / Ranking

Aggregate per-shape statistics into the configured objective score and decide
which candidates survive.

```text
    +-------------------------------------------------------------+
    |                Compute Objective Score                      |
    |                                                             |
    |  metric source:                                             |
    |    weighted_p50_us     -> shape.latency_p50_us              |
    |    weighted_p95_us     -> shape.latency_p95_us              |
    |    worst_case_p50_us   -> shape.latency_p50_us              |
    |                                                             |
    |  aggregation:                                               |
    |    weighted_mean = sum(w_i * metric_i) / sum(w_i)           |
    |    max           = max(metric_i)                            |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |             Compare Against Baseline / Incumbent            |
    |                                                             |
    |  relative_to_baseline  = score / baseline_score             |
    |  relative_to_incumbent = score / incumbent_anchor_score     |
    |                                                             |
    |  incumbent_comparison:                                      |
    |    improved / statistical_tie / regressed / unstable /      |
    |    not_comparable                                           |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                    Select Profile Candidates                |
    |                                                             |
    |  candidates eligible for profile:                           |
    |    - measurement valid                                      |
    |    - correctness already passed                             |
    |    - not regressed beyond guard                             |
    |                                                             |
    |  profile set:                                               |
    |    top_k_by_score UNION top_m_by_shift_potential            |
    |                                                             |
    |  tie-break by: objective score, measurement noise, hash      |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 6 ]
```

### Unit-Consistent Regression

`regression_guard_pct` is a relative threshold. It must not be compared against
an absolute microsecond delta.

For latency objectives, lower is better:

```text
candidate_regressed =
  candidate_score > incumbent_anchor_score * (1 + guard_pct + noise_margin_pct)
```

If `guard_pct = 0.02`, a candidate may be up to 2% slower than the incumbent
before it is marked as a regression, subject to the noise guard.

The service also needs a separate improvement gate:

```text
candidate_improved =
  candidate_score < incumbent_anchor_score * (1 - noise_margin_pct)
```

Scores inside the noise band are statistical ties. They may be useful search
evidence, but they must not replace the incumbent.

### Noise Margin

The Benchmarker should add a measurement-derived noise margin so that tiny
differences are not treated as meaningful improvements or regressions. A simple
V1-compatible rule:

```text
noise_margin_pct = max(
  NOISE_FLOOR_PCT,
  candidate_objective_cv_pct,
  incumbent_anchor_objective_cv_pct,
  anchor_drift_pct
)
```

Where:

```text
anchor_drift_pct = abs(anchor_post_score - anchor_pre_score) / anchor_pre_score
```

Sample CV captures within-episode jitter. Anchor drift captures systematic
movement across the episode from temperature, clock ramp, power state, or other
time-correlated effects. If pre/post incumbent anchors differ by 3%, then any
candidate "improvement" under 3% in that episode is a statistical tie.

This is not a statistical proof. It is a conservative guardrail that prevents
the search loop from overfitting noise. If a candidate appears to beat the
incumbent only inside the noise margin, it must be reported as
`statistical_tie`, not as an improvement.

The Orchestrator-visible result model should be extended beyond a single
`regressed_vs_incumbent` boolean:

```text
IncumbentComparison
  improved
  statistical_tie
  regressed
  unstable
  not_comparable
```

Until shared types support this directly, the GPU pipeline adapter should map
`statistical_tie` to a non-improving outcome rather than allowing a strict but
noise-sized latency decrease to update the incumbent.

### Anchor Policy

| Anchor | Required When | Purpose |
|---|---|---|
| Incumbent anchor | Any candidate may become new incumbent | fair regression and improvement decision |
| Baseline anchor | New pod, new GPU, long-running job, or drift detected | stable `relative_to_baseline` |
| Known-good probe | Pod transitions from suspect to healthy | infra health, not optimization scoring |

Stored baseline and incumbent artifacts remain durable search state. Anchor
measurements are episode-local fairness checks.

### Profile Candidate Selection

Deep profiling only top score winners can starve the search loop of evidence
from slower but structurally promising candidates. A candidate may regress on
latency while proving that a major bottleneck has moved, for example much higher
tensor-pipe activity or a large memory-traffic reduction. That evidence is
valuable for the Strategy Navigator even when the candidate is not an incumbent.

Profile selection should therefore be:

```text
profile_candidates =
  top_k_by_objective_score
  union top_m_by_bottleneck_shift_potential
```

`top_m_by_bottleneck_shift_potential` cannot rely on deep-profile counters that
have not been collected yet. It should use pre-profile signals:

- candidate intent direction and sub-mode;
- static analysis deltas, such as registers, shared memory, spills, occupancy estimate;
- fast-benchmark derived throughput, arithmetic intensity, useful bytes, and score shape;
- novelty relative to the incumbent and recent attempted directions;
- Cross-Candidate Analyzer hints from earlier rounds, if available.

After profiling, the Benchmarker records measured `bottleneck_shift` against the
incumbent or previous profiled baseline. The Profile Interpreter can then decide
whether the shift is actionable.

---

## Phase 6 - Deep Profiling Plan

Deep profiling is run only after fast benchmark ranking. Its job is to produce
raw metrics and artifacts for the Profile Interpreter.

```text
    +-------------------------------------------------------------+
    |                    Select Profile Targets                   |
    |                                                             |
    |  profile candidates: top-K score winners plus diagnostic    |
    |                      candidates with shift potential         |
    |  profile shapes: ShapeCase.profile == true                  |
    |  fallback: best objective shape if no profile shape exists  |
    |  include incumbent profile when comparison is useful        |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |                 Run Nsight Compute Collection               |
    |                                                             |
    |  default: focused metric set                                |
    |  collect: occupancy, SM/DRAM throughput, cache, stalls,     |
    |           tensor pipes, memory traffic, instruction mix     |
    |  output: raw report artifact + normalized metric snapshot   |
    +---------------------------+---------------------------------+
                                |
                                v
    +-------------------------------------------------------------+
    |            Optional Nsight Systems Timeline Collection      |
    |                                                             |
    |  run only when timeline evidence is relevant:               |
    |    - objective includes transfers or host overhead          |
    |    - overlap/stream behavior is suspected                   |
    |    - profiler counters conflict or are insufficient         |
    +---------------------------+---------------------------------+
                                |
                                v
                         [ enter Phase 7 ]
```

### Nsight Compute Metric Families

The Benchmarker should collect raw counters and normalized percentages when
available. Metric availability varies by GPU architecture and profiler version,
so missing values must be represented as missing, not fabricated.

| Family | Examples | Why It Matters |
|---|---|---|
| Launch / occupancy | achieved occupancy, theoretical occupancy, active warps | latency hiding and resource residency |
| SM throughput | SM busy / throughput percent of peak | compute pipeline utilization |
| DRAM throughput | DRAM bytes, throughput, percent of peak | memory bandwidth pressure |
| L2 / cache | L2 hit rate, L2 sectors, cache throughput | locality and redundant traffic |
| Warp stalls | memory dependency, execution dependency, barrier, not selected | scheduler and dependency bottlenecks |
| Tensor / math pipes | tensor pipe utilization, FP32/FP64/INT pipe activity | whether intended compute units are used |
| Memory access efficiency | global load/store sectors and useful bytes | coalescing and overfetch |
| Control flow | branch efficiency, predication, divergent branch counters | warp divergence |
| Instruction mix | arithmetic, memory, control, address-calculation instructions | useful work versus overhead |
| Roofline inputs | FLOPs, bytes, arithmetic intensity, achieved FLOP/s | compute-bound versus bandwidth-bound analysis |

The Profile Interpreter turns these measurements into bottleneck tags. The
Benchmarker only supplies the evidence.

### V1 Replay Coverage

Nsight Compute may replay kernels to collect metric groups. V1 deep-profile
coverage should be explicit:

| Adapter Iteration Semantics | V1 Deep Profile Status |
|---|---|
| `overwrite_pure` | supported |
| `requires_output_reset` | supported if the adapter provides a safe reset or buffer-rotation path outside the profiled launch |
| `requires_full_input_reset` | `profile_unavailable` unless the adapter provides a safe restore path |
| `not_repeatable` | `profile_unavailable` |

When profile is unavailable because of adapter semantics, return:

```text
profile_status: "profile_unavailable"
profile_unavailable_reason: "adapter_not_repeatable"
```

This keeps agents from interpreting missing profile evidence as absence of a
bottleneck.

### Nsight Compute Target Selection

The profiler must target the measured kernel, not warmup, anchors, unrelated
runtime probes, or every repeated launch in the timing loop. The common harness
should mark each candidate/shape launch region with deterministic NVTX names:

```text
kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile
```

The profiling command should select one measured launch per candidate and shape
using NVTX range filtering or explicit launch skip/count. The selected range,
launch index, metric set, replay mode, profiler version, and raw report path
must be stored in the profile artifact.

Nsight Compute may replay kernels to collect metric groups. That is acceptable
for diagnostic counters, but it strengthens the requirement that adapter
iteration semantics are explicit. Non-repeatable kernels cannot be deep-profiled
with replay-based collection unless the adapter provides a safe restore path.

### Metric Portability

Profile metric names and semantics vary across GPU architecture and Nsight
Compute version. The Benchmarker must preserve raw metrics rather than reducing
them directly into one architecture-specific field.

```text
RawProfileMetric
  metric_name: string
  value: float | int | null
  unit: string | null
  architecture: string
  profiler_name: "ncu"
  profiler_version: string
  collection_section: string | null
```

Normalized aliases such as `tensor_core_utilization_pct` are compact
convenience fields only. They must include provenance:

```text
normalized_metrics:
  tensor_core_utilization_pct:
    value: float | null
    source_metrics: [raw metric names]
    architecture: "sm_80" | "sm_90" | ...
    profiler_version: string
    comparable_across_arch: false
```

The Profile Interpreter should not assume an Ampere tensor metric maps directly
to Hopper or Blackwell tensor-pipe behavior. Cross-architecture comparability
requires explicit normalization rules.

### Nsight Systems Trigger Policy

Nsight Systems is not a default per-candidate profiler for kernel-only search.
Use it when the question is timeline-shaped:

- Are host API calls dominating an end-to-end objective?
- Are transfers accidentally inside the timed path?
- Are streams overlapping or serializing as expected?
- Are multiple contexts or foreign processes interfering?
- Is a CUDA graph or library call producing hidden kernels?

Timeline artifacts are large and should be retained by reference.

### Profiling Overhead Rule

Profiler output can include durations, but those durations are diagnostic. The
`BenchmarkBundle.objective_score` must always come from Phase 4 fast benchmark
timing, not from Nsight Compute or Nsight Systems.

---

## Phase 7 - Output To GPU Pipeline Adapter / Profile Interpreter

The Benchmarker returns structured measurement facts and routing outcomes.

```text
    +-------------------------------------------------------------+
    |                    Benchmark Batch Result                   |
    |                                                             |
    |  {                                                          |
    |    "status": "success" | "partial" | "unstable" |          |
    |              "timeout" | "infra_error",                    |
    |    "run_envelope": { ... },                                |
    |    "measurement_context": { ... },                         |
    |    "incumbent_anchor": { ... },                            |
    |    "candidate_results": [                                  |
    |      {                                                      |
    |        "candidate_hash": "...",                            |
    |        "benchmark": {                                      |
    |          "shape_results": [ ... ],                         |
    |          "objective_score": { ... },                       |
    |          "incumbent_comparison": "improved",              |
    |          "regressed_vs_incumbent": false                   |
    |        },                                                   |
    |        "measurement_quality": { ... },                     |
    |        "profile_artifacts": [ ... ],                       |
    |        "raw_profile_metrics": { ... } | null,              |
    |        "fault_class": null | "candidate_fault" |           |
    |                       "infra_fault" | "ambiguous_fault"    |
    |      }                                                      |
    |    ],                                                       |
    |    "top_k_profiled": ["hash_A", "hash_B"]                  |
    |  }                                                          |
    +---------------------------+---------------------------------+
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
    +--------------------------+    +----------------------------+
    | Profile Interpreter      |    | Orchestrator / GPU Adapter |
    | raw metrics -> tags      |    | result mapping + state     |
    +--------------------------+    +----------------------------+
```

### Orchestrator-Visible Mapping

The GPU pipeline adapter maps Benchmarker output into shared types. The current
V1 `regressed_vs_incumbent` boolean is not expressive enough for noise-aware
incumbent promotion, so the contract should be extended with
`incumbent_comparison` while preserving the boolean as a compatibility field.

```text
BenchmarkBundle
  shape_results
  objective_score
  incumbent_comparison
  regressed_vs_incumbent  # compatibility: true only when comparison == regressed

ProfileMetrics
  achieved_occupancy_pct
  dram_throughput_pct_of_peak
  sm_throughput_pct_of_peak
  l2_hit_rate_pct
  warp_stall_memory_dependency_pct
  warp_stall_exec_dependency_pct
  tensor_core_utilization_pct  # compact alias with raw metric provenance
  arithmetic_intensity_flop_per_byte
  raw_profile_metrics_ref
  profiler_version
  profile_arch
```

The richer artifacts stay pod-local or artifact-store-local and are referenced
by id. The Orchestrator should not need raw profiler reports in memory.

### Routing Semantics

| Benchmarker Outcome | Orchestrator Meaning | Search-Memory Signal |
|---|---|---|
| `success`, improved | eligible for incumbent comparison | positive evidence |
| `success`, statistical tie | valid attempt, do not update incumbent | neutral evidence |
| `success`, baseline match | valid attempt, no improvement against baseline or incumbent | neutral evidence |
| `success`, regression | discard from improving pool | negative evidence |
| `unstable_measurement` | do not update incumbent | no optimization signal unless repeated on healthy pods |
| `candidate_runtime_fault` | candidate failed during timed execution | candidate negative signal |
| `profile_unavailable` | benchmark can still be valid | no bottleneck evidence for this candidate |
| `infra_error` | retry elsewhere or mark pod unhealthy | no optimization signal |

---

## Full Phase Flow Summary

```text
    INPUTS
      |
      v
    Phase 1: Request Normalization / Measurement Envelope
      |
      v
    Phase 2: Device Lease / Measurement Hygiene
      |
      +--> hygiene fail -> unstable / infra_error
      |
      v
    Phase 3: Fast Benchmark Plan Calibration
      |
      v
    Phase 4: Fast Benchmark Execution / Quality Checks
      |
      +--> unstable samples -> rerun once or mark unstable
      +--> runtime fault    -> candidate_fault or ambiguous_fault
      |
      v
    Phase 5: Objective Scoring / Regression Guard / Ranking
      |
      +--> regression -> discard before deep profile
      |
      v
    Phase 6: Deep Profiling Plan
      |
      +--> profiler unavailable -> keep benchmark, omit profile
      |
      v
    Phase 7: Output Structured Results
      |
      +--> Profile Interpreter
      +--> GPU Pipeline Adapter / Orchestrator
```

---

## Distributed Benchmarking Rules

### Rule 1 - Fair Ranking Requires Comparable Devices

Candidates in the same ranking batch may be distributed only across devices
with matching:

- GPU SKU and SM architecture;
- MIG profile, if MIG is enabled;
- driver and CUDA runtime compatibility class;
- benchmark harness version;
- artifact execution model and module loader version;
- clock policy;
- objective hash;
- operation adapter version.

If any of these differ, the result can be recorded but should not directly rank
against candidates measured elsewhere.

### Rule 2 - Anchors Normalize Within A Pod, Not Across Architectures

Per-pod baseline and incumbent anchors help detect drift and local noise. They
do not make A100 and H100 numbers comparable. Different target GPUs are
different optimization problems.

### Rule 3 - Timed GPU Work Is Serialized Per Device

Multiple CUDA contexts on the same GPU can be time-sliced and can reduce
utilization. The Benchmarker should prevent competing benchmark contexts rather
than trying to statistically repair the damage afterward.

### Rule 4 - Remote Failures Need Fault Attribution

The result must distinguish:

| Fault Class | Examples | Search Meaning |
|---|---|---|
| `candidate_fault` | illegal memory access reproduced on healthy pod, kernel timeout with healthy device, launch config runtime error | valid negative signal |
| `infra_fault` | GPU lost, Xid, artifact store failure, profiler permission denied for profile phase, pod disk full | no optimization signal |
| `ambiguous_fault` | process killed during GPU execution, CUDA context poisoned, one-off timeout on suspect pod | retry before using as signal |

### Rule 5 - Scheduler Must Prefer Information Per GPU-Second

Fast benchmark all passing candidates before deep profiling. Deep profile only
score winners, diagnostic bottleneck-shift candidates, and designated profile
shapes. Profiling every candidate wastes GPU time and slows the search loop
without improving ranking quality.

---

## Boundary Decisions

### Benchmarker vs. Compiler Service

The Compiler Service decides whether a kernel is executable and correct enough
to measure. The Benchmarker assumes that gate has passed and focuses on
controlled timing and profiling. It may detect runtime faults during benchmark
execution, but it does not rerun correctness comparisons.

### Benchmarker vs. Profile Interpreter

The Benchmarker collects raw profiler metrics and normalized percentages. The
Profile Interpreter owns rule-based bottleneck tags, evidence-to-direction
mapping, and bottleneck assessments.

### Benchmarker vs. Strategy Navigator

The Benchmarker can say "candidate A is 7.2% faster on the weighted p50
objective" or "DRAM throughput was 82% of peak during the profiled shape." It
cannot say "explore a new memory layout next." That is policy.

### Benchmarker vs. Orchestrator

The Orchestrator owns durable baseline, incumbent, attempt records, and round
state. The Benchmarker returns measurement results and artifact references. It
does not update global state.

### Benchmarker vs. Cross-Candidate Analyzer

The Benchmarker ranks candidates by objective score. It does not infer semantic
genes, reusable code changes, or recombination hints.

---

## Design Rationale

### Why Cubin/Module Loaded By A Common Harness?

Kerlever needs fair candidate ranking more than it needs convenient executable
packaging. A stable common harness gives the Benchmarker one process, one CUDA
context, one stream policy, one allocation policy, and one timing path for the
incumbent and all candidates in the batch. Loading cubins/modules avoids linking
LLM-generated CUDA sources into one translation unit, avoids symbol-collision
problems, and keeps host benchmarking code deterministic. Running that harness
inside a disposable worker process gives enough fault containment for bad
generated kernels without giving up one-context measurement.

### Why Remeasure The Incumbent?

GPU latency is sensitive to clock state, thermal state, driver behavior, and
contention. A candidate measured on a hot pod should not replace an incumbent
measured earlier on a cool pod unless the incumbent is remeasured under the
same episode or the system has enough anchor evidence to normalize the
comparison.

### Why Use Median And p95?

Median is robust for central tendency under occasional outliers. p95 preserves
tail behavior that can matter for production workloads. Kerlever stores both
because the objective may optimize either central latency or tail-sensitive
latency.

### Why Not Profile Every Candidate?

Deep profiling is expensive and can replay kernels or collect many hardware
counters. Regressed candidates are not useful enough to justify profiler cost in
the common path. The search loop needs fast ranking first, then diagnostic depth
for the few candidates that matter.

### Why Keep Raw Samples?

Aggregates are convenient, but they hide whether a result was stable. Raw or
artifact-referenced samples let the system later diagnose noisy pods, refine
thresholds, and audit false incumbent updates.

### Why Exclude Data Transfers By Default?

The current `ProblemSpec` describes CUDA kernel optimization. If the objective
is kernel latency, including host-device copies would reward or punish behavior
outside the kernel implementation. End-to-end operator timing is useful, but it
needs a separate metric contract.

---

## Configuration Parameters

| Parameter | Default Direction | Description |
|---|---|---|
| `ARTIFACT_EXECUTION_MODEL` | cubin/module common harness | Long-term benchmark execution model; separate executables are V1 fallback only |
| `DEFAULT_METRIC_MODE` | `device_kernel_us` | Kernel-only CUDA-event timing unless the objective explicitly chooses another mode |
| `GPU_BENCH_CONCURRENCY` | 1 per GPU or MIG instance | Timed benchmark jobs are serialized per leased device |
| `FAST_BENCH_WARMUP_MIN_RUNS` | small fixed minimum | Minimum untimed launches before sampling |
| `FAST_BENCH_MIN_TIMED_BATCH_MS` | enough to dwarf event overhead | Minimum elapsed time for one repeated-launch sample |
| `FAST_BENCH_MAX_TIMED_BATCH_MS` | bounded service default | Maximum elapsed time for one repeated-launch sample before recalibration |
| `FAST_BENCH_REPETITIONS` | enough for p50 and p95 | Number of timed samples after warmup |
| `FAST_BENCH_MAX_ITERATIONS_PER_SAMPLE` | bounded cap | Prevents pathological tiny kernels from creating huge loops |
| `MIN_P95_SAMPLES` | explicit service default | Minimum valid samples before p95 may be treated as available |
| `MEASUREMENT_CV_WARN_PCT` | low single digits | Warning threshold for coefficient of variation |
| `MEASUREMENT_CV_FAIL_PCT` | service default | Mark measurement unstable above this threshold |
| `P95_P50_RATIO_WARN` | service default | Detects heavy tail behavior in timing samples |
| `NOISE_FLOOR_PCT` | nonzero conservative floor | Minimum noise margin for improvement/regression decisions |
| `ANCHOR_DRIFT_WARN_PCT` | service default | Warning threshold for pre/post incumbent anchor drift |
| `ANCHOR_DRIFT_FAIL_PCT` | service default | Mark measurement unstable above this anchor drift |
| `ANCHOR_EVERY_N_SAMPLES` | service default | Controls incumbent anchor cadence inside interleaved batches |
| `MAX_INTERLEAVE_BLOCK_LEN` | service default | Maximum randomized candidate block length between anchors |
| `BENCH_RERUN_LIMIT` | 1 or 2 | Retry unstable timing before returning unstable |
| `ANCHOR_INCUMBENT_POLICY` | same episode | Remeasure incumbent in replacement-eligible batches |
| `ANCHOR_BASELINE_POLICY` | drift/new-pod triggered | Remeasure baseline when comparability is uncertain |
| `CACHE_POLICY` | single-candidate `warm_same_buffers`; interleaved `warm_rotating_buffers` | Default steady-state cache behavior by measurement mode |
| `CACHE_FLUSH_BYTES` | adapter/pool specific | Eviction buffer size when cold-cache policy is selected |
| `CLOCK_POLICY` | observe + throttle enforcement | Clock telemetry requirement; locking is optional hardening when privileged |
| `CLOCK_LOCK_POLICY` | disabled unless privileged pool config enables it | Optional GPU/memory clock locking for dedicated benchmark hosts |
| `THERMAL_STEADY_STATE_LIMIT` | hardware/pool specific | Temperature threshold for valid timing |
| `TOP_K_PROFILE` | 1-3 | Number of objective-score winners to deep profile |
| `TOP_M_PROFILE_SHIFT_POTENTIAL` | 1-2 | Diagnostic candidates selected for possible bottleneck migration despite lower score |
| `NCU_PROFILE_SET` | focused/basic first | Avoid full profiling unless needed |
| `NCU_TARGET_SELECTION` | NVTX range + launch count | Select the measured candidate/shape launch, not warmup or anchors |
| `NCU_REPLAY_ADAPTER_POLICY` | overwrite/reset only | V1 profile coverage by adapter iteration semantics |
| `PROFILE_METRIC_PROVENANCE` | required | Store raw metric names, architecture, profiler version, and normalized aliases |
| `NSYS_PROFILE_POLICY` | trigger-based | Timeline profiling only for timeline-shaped questions |
| `PROFILE_TIMEOUT` | fixed service default | Bound profiler runs |
| `ARTIFACT_RETENTION` | keep raw early | Retain samples and profiler reports while rules mature |
| `POD_HEALTH_PROBE` | known-good kernel | Probe used before benchmarking after suspect state |

---

## Reference Basis

This design is grounded in primary CUDA, profiler, and benchmarking sources:

- CUDA C++ Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA C++ Programming Guide, CUDA events and elapsed timing:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA Nsight Compute Profiling Guide:
  https://archive.docs.nvidia.com/nsight-compute/2025.3/ProfilingGuide/index.html
- NVIDIA Nsight Systems User Guide:
  https://docs.nvidia.com/nsight-systems/UserGuide/
- NVIDIA System Management Interface documentation:
  https://docs.nvidia.com/deploy/nvidia-smi/index.html
- NVIDIA NVML API reference:
  https://docs.nvidia.com/deploy/nvml-api/
- NVIDIA Data Center GPU Manager documentation:
  https://docs.nvidia.com/datacenter/dcgm/
- Google Benchmark User Guide:
  https://google.github.io/benchmark/user_guide.html
- Georges, Buytaert, and Eeckhout, "Statistically Rigorous Java Performance
  Evaluation":
  https://biblio.ugent.be/publication/417084
