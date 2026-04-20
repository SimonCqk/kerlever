# Benchmarker Module Specification

## §1 Overview

The Benchmarker is the deterministic measurement service of the Kerlever optimization loop. It receives batches of correctness-passing, sanitizer-clean candidate artifacts produced by the Compiler Service, measures their latency on every objective shape under a controlled CUDA timing path, decides improvement, statistical tie, and regression against a freshly measured incumbent anchor, selects survivors for deep profiling, runs Nsight Compute against selected candidates/shapes, and returns a structured batch result plus pod-local artifact references.

The Benchmarker is a **measurement-producing service**, not a search-policy module. It does not:

- generate or mutate kernel source;
- compile or rerun correctness;
- update the incumbent, baseline, or search memory;
- interpret Nsight metrics into bottleneck tags or direction hints;
- emit natural-language advice about what optimization to try next.

It is a standalone FastAPI service running on a remote GPU pod, Docker-packaged, with real CUDA tooling (`cuda-python` driver API for module load/launch/events/function attributes, `pynvml` for device telemetry, and `compute-sanitizer`'s sibling tool `ncu` invoked as a subprocess for deep profiling).

**Scope boundary — this task is Benchmarker-only.** The Benchmarker exposes its own HTTP API (`POST /benchmark`, `GET /healthz`, `GET /info`) and its own request/response contract (`BenchmarkBatchRequest` / `BenchmarkBatchResult`). It is deliberately **not** bound to the existing `GPUPipelineProtocol.evaluate(candidate, problem_spec, baseline, incumbent) -> EvaluationResult` signature in `kerlever/protocols.py`. That protocol is per-candidate and coarse-grained; the real Benchmarker prefers a per-round batch call so that fair interleaving, shared incumbent anchors, and top-K selection can all happen inside one measurement episode on one leased device. A future **GPU Pipeline Adapter** (out of scope here) will bridge batch Benchmarker output into the orchestrator-visible `EvaluationResult` per candidate. The Benchmarker's spec must be read with that split in mind: the orchestrator does not call the Benchmarker directly.

**Shared types reuse.** The Benchmarker reuses `ShapeCase`, `PerformanceObjective`, `ProblemSpec`, `StaticAnalysis`, `ShapeBenchResult`, `ObjectiveScore`, `ProfileMetrics`, `BottleneckAssessment`, `ProfileBundle`, `CorrectnessResult`, and `BenchmarkBundle` from `kerlever/types.py` verbatim. Any richer measurement record lives in `kerlever/benchmarker/types.py` and is referenced from compact shared types by id.

**Design reference.** `docs/benchmarker.md` is the narrative design doc (1326 lines) that this spec formalizes. Every numbered phase, configuration parameter, boundary decision, and rationale in that doc is preserved here as a REQ-BENCH-*, SCN-BENCH-*, INV-BENCH-*, or §6 subsection.

---

## §2 Requirements

### Success Criteria

**SC-BENCH-001: Fast kernel benchmark on every objective shape**
Given a batch of correctness-passing candidate module artifacts and a `ProblemSpec` with `N` `ShapeCase`s, the Benchmarker produces per-shape `ShapeBenchResult` (`latency_p50_us`, `latency_p95_us`, `stdev_us`, `run_count`) for every (candidate, shape) pair, using the CUDA event timing path, `device_kernel_us` metric mode by default, and deterministic iteration/warmup/repetition calibration.

**SC-BENCH-002: Measurement hygiene gates**
Before any timed work, the Benchmarker leases one GPU or MIG instance, serializes timed execution per device, and refuses or marks unstable episodes that violate any hygiene condition (architecture mismatch, foreign compute process, thermal/power/HW-slowdown throttle, temperature above steady-state limit, ECC/Xid/lost-GPU, MIG profile mismatch, known-good probe failure). Clock state is observed and recorded (`observed_only` / `locked` / `lock_requested_unavailable`), never assumed.

**SC-BENCH-003: Fair interleaving with anchored drift detection**
When the artifact execution model is cubin/module + common harness, the Benchmarker interleaves candidate launches and incumbent anchor launches inside one worker process and one CUDA context using a stable `interleave_seed = hash(run_id, batch_id, shape_id, "kerlever_benchmark_order")`. Pre/post incumbent anchors are recorded per shape; `anchor_drift_pct = |anchor_post_score - anchor_pre_score| / anchor_pre_score` feeds the noise margin.

**SC-BENCH-004: Noise-margin-gated improvement and regression decision**
The Benchmarker computes `noise_margin_pct = max(NOISE_FLOOR_PCT, candidate_cv_pct, anchor_cv_pct, anchor_drift_pct)` per shape and aggregates per objective. A candidate is `improved` only when `candidate_score < incumbent_anchor_score * (1 - noise_margin_pct)`, `regressed` only when `candidate_score > incumbent_anchor_score * (1 + guard_pct + noise_margin_pct)`, `statistical_tie` otherwise. `IncumbentComparison` is the sole decision path; the legacy boolean `regressed_vs_incumbent` is mapped from it.

**SC-BENCH-005: Top-K ∪ top-M deep profile selection**
After scoring, the Benchmarker selects a bounded set of candidates for deep profiling: the top `TOP_K_PROFILE` by objective score (excluding regressed and unstable candidates, including the incumbent if its profile is still useful) UNION the top `TOP_M_PROFILE_SHIFT_POTENTIAL` by pre-profile bottleneck-shift signals (intent direction, static-analysis delta, fast-benchmark throughput/arithmetic-intensity/useful-bytes shape, novelty). Selection is deterministic and tie-broken by objective score, then measurement noise, then candidate hash.

**SC-BENCH-006: Nsight Compute collection with NVTX targeting and metric provenance**
For each (selected candidate, profile shape), the Benchmarker runs a sub-process `ncu` collection filtered to an NVTX range named `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile`, with a focused metric set, replay behavior determined by `AdapterIterationSemantics`. Raw metrics are returned with provenance (`metric_name`, `value`, `unit`, `architecture`, `profiler_name`, `profiler_version`, `collection_section`); normalized aliases (e.g., `tensor_core_utilization_pct`) carry `source_metrics`, `architecture`, `comparable_across_arch: false`. Metrics are never fabricated: an unavailable metric is `null`, and an unavailable profile is a `profile_unavailable` envelope with a typed reason.

**SC-BENCH-007: Typed fault attribution and profile-unavailable reasons**
Every candidate result carries a `FaultClass` (`null`, `candidate_fault`, `infra_fault`, `ambiguous_fault`) and — when profile collection did not run or failed — a `ProfileStatus` with `profile_unavailable_reason` drawn from a closed enum (`profiler_permission_denied`, `adapter_not_repeatable`, `arch_mismatch`, `profiler_timeout`, `profiler_binary_missing`, `profiler_replay_refused`, `mig_profile_mismatch`). Ambiguous faults never flow as optimization signal and trigger pod-health state transitions.

**SC-BENCH-008: Structured batch output reusing shared types**
The Benchmarker returns `BenchmarkBatchResult` with a top-level `status` (`success` / `partial` / `unstable` / `timeout` / `infra_error`), a `run_envelope`, a per-batch `measurement_context`, an `incumbent_anchor` block, a `candidate_results[]` list containing — per candidate — a `BenchmarkBundle` (shared type, mapped to `IncumbentComparison` + legacy boolean), a rich `MeasurementEnvelope`, an optional `ProfileBundle` (shared type) plus a `raw_profile_metrics_ref` to a `ShapeMeasurementArtifact` held in the artifact store, a `FaultClass`, and artifact ids for samples and ncu reports. The `top_k_profiled` list names the candidates that were deep-profiled.

**SC-BENCH-009: Self-contained HTTP request contract**
The Benchmarker accepts the full measurement context (`run_envelope`, `problem_spec`, `baseline_ref`, `incumbent_ref`, `candidate_module_artifact_refs[]`, `objective_shape_cases[]`, `profile_shape_cases[]`, `operation_adapter_abi`, `top_k_profile`) in a single `BenchmarkBatchRequest` and returns one `BenchmarkBatchResult`. The contract is designed for a future GPU Pipeline Adapter to consume; the Benchmarker itself does not implement `GPUPipelineProtocol.evaluate` and does not import the shared protocol module.

**SC-BENCH-010: FastAPI + Docker deployment shape**
The Benchmarker ships as a FastAPI app with `POST /benchmark`, `GET /healthz`, `GET /info`, runs under Uvicorn, and builds from a Dockerfile based on `nvidia/cuda:*-devel` with `cuda-python`, `pynvml`, and `compute-sanitizer`/`ncu` available. `/healthz` performs a real readiness probe (driver present, at least one visible GPU matching a configured target, `ncu` binary resolvable, artifact root writable, pod health not `quarantined`) and fails non-zero during container startup when any required tool is absent.

**SC-BENCH-011: Static quality gates**
The new package under `kerlever/benchmarker/` passes `ruff check .` and `mypy .` under the repository configuration (`mypy.strict = true`, ruff selection `["E","F","W","I","N","UP","B","A","SIM"]`, line length 88).

**SC-BENCH-012: Operation Adapter contract is a named, versioned plugin point**
The Benchmarker owns a registry of operation adapters keyed by `(operation_adapter_abi, operation_adapter_version)`. Every request must reference a registered adapter. The adapter is the sole owner of: per-shape input/output device buffer allocation; deterministic input seeding from `(run_id, batch_id, shape_id)`; grid-dim derivation from shape and block-dim; launch-argument tuple construction (device pointers plus scalar dimensions in the ABI-declared order); useful-bytes and algorithmic-flops computation per shape; reset hooks invoked between timed iterations for non-`OVERWRITE_PURE` semantics; buffer-rotation strategy for `WARM_ROTATING_BUFFERS`. V1 ships exactly two concrete adapters: `elementwise_add_fp32_v1` (trivially correct, minimal ABI) and `matmul_fp16_v1` (minimal matmul ABI). Unknown `operation_adapter_abi` or mismatched `operation_adapter_version` → `status = infra_error`, no GPU lease, structured reason `adapter_unregistered` or `adapter_version_mismatch`.

**SC-BENCH-013: NCU target process architecture is the dedicated profile child**
The Benchmarker never asks NCU to attach to its own worker process. For each `(selected candidate, profile_shape)`, the worker spawns a dedicated **profile child** (same Python executable, a distinct `python -m kerlever.benchmarker.profile_child` entry point or equivalent invocation mode) that: loads one CUDA context on the leased device; loads the same cubin resolved in Phase 3; uses the same operation adapter to allocate buffers and seed inputs identically to the measurement; performs `FAST_BENCH_WARMUP_MIN_RUNS` warmup launches outside any NVTX range; performs `iterations_per_sample` launches inside a single NVTX range `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile` such that exactly ONE measured launch inside that range is profiled by NCU (`--launch-count 1` plus `--nvtx --nvtx-include <range>`); exits cleanly. NCU wraps the profile child. The Benchmarker worker process itself is never an NCU target.

**SC-BENCH-014: Cubin transport contract and artifact store lifecycle**
V1 cubin transport is **local file path only**: `CandidateArtifactRef.cubin_uri` must be an absolute POSIX path readable by the Benchmarker process on a shared mount between caller and Benchmarker. Non-absolute paths, scheme-prefixed URIs (`s3:`, `http:`, `https:`, `file:`, etc.), and paths that are not readable at Phase 1 → `status = infra_error`, reason `cubin_uri_unsupported_scheme` or `cubin_uri_not_readable`. Inline bytes transport is explicitly **out of scope for V1** (see §9 Non-Goals); a `cubin_bytes_b64` field MUST NOT be accepted even if present in the request body. The pod-local artifact store rooted at `cfg.artifact.root` is the Benchmarker's own write-only area for NCU reports, raw metrics JSON, sample JSON, and `ShapeMeasurementArtifact` JSON. The service does not read previously-written artifacts after the response is emitted; `artifact_refs` in the response are opaque ids for downstream consumers. Retention and rotation of this store are **operator-owned**; the Benchmarker only guarantees that artifacts referenced by a still-being-emitted response are not deleted mid-read (see REQ-BENCH-021).

**SC-BENCH-015: Rich artifact fidelity — every field derived from real runtime state**
Every field of `ShapeMeasurementArtifact` that could plausibly be populated from real runtime state MUST be derived from that state, not hardcoded to a spec default. Specifically: `warmup_count` from the actual count of warmup launches executed; `iterations_per_sample` from Phase 3 calibration output for that `(candidate, shape)`; `min_samples_required` from the effective config (e.g., `MIN_P95_SAMPLES`); `interleave_block_len` and `interleave_block_order` from the realized interleave block generator output; `anchor_pre_us` / `anchor_post_us` / `anchor_drift_pct` from the actual incumbent anchor samples collected in this batch for this shape (null only when `interleave_enabled = False`); `cache_policy` / `requested_cache_policy` / `effective_cache_policy` from Phase 3 cache-policy resolution; `artifact_execution_model` from the request envelope; `adapter_iteration_semantics` from the matched adapter registry entry (NOT from the request — the adapter is the source of truth); `metric_mode` from the effective envelope; `max_timed_batch_ms` from the effective config; `function_attribute_policy` from `function_attribute_policy_observed` in the envelope (the values read back after `cuFuncSetAttribute`); `measurement_quality` from the quality classifier in §6.4; `telemetry_before` and `telemetry_after` from real `pynvml` samples taken immediately before the first measured launch and immediately after the last measured launch for that shape. A hardcoded default that does not match the real runtime state is a spec violation even if all static gates pass.

**SC-BENCH-016: Batch staging artifacts are cleaned up after batch completion**
The Supervisor stages per-batch input and intermediate files under `<cfg.artifact.root>/staging/<batch_id>/` (typically `request.json`, `config.json`, per-shape result JSON, and any `.ncu-rep` files emitted under that batch's directory). These staging files are transient by design and MUST be removed after the batch's `BenchmarkBatchResult` has been returned to the caller — regardless of whether the batch completed `success`, `partial`, `unstable`, `timeout`, or `infra_error`. Durably-referenced artifacts (the ones named in `CandidateResult.shape_measurement_artifact_refs`, `profile_artifact_refs`, and `raw_profile_metrics_ref`) live outside the staging directory (under `<cfg.artifact.root>/samples/`, `<cfg.artifact.root>/ncu/`, `<cfg.artifact.root>/raw_metrics/`, `<cfg.artifact.root>/artifacts/`) and are NOT affected by staging cleanup; retention of those is operator-owned per REQ-BENCH-021. An operator-controlled escape hatch (`KEEP_STAGED_ARTIFACTS=1` env var) disables staging cleanup for forensic debugging. Long-running pods MUST NOT accumulate `<cfg.artifact.root>/staging/<batch_id>/` directories after their batches complete.

### Functional Requirements

**REQ-BENCH-001: Batch request normalization and envelope creation** [traces SC-BENCH-001, SC-BENCH-008, SC-BENCH-009]
Phase 1 must validate that every candidate artifact reference reports Compiler Service `status=success`, resolve launch metadata (entrypoint, block_dim, dynamic_smem_bytes, abi_name, abi_version), bind baseline and incumbent references, select the operation adapter matching `operation_adapter_abi`, and assemble a `MeasurementEnvelope` carrying the identity, artifact identity, workload identity, device identity, and timing policy fields listed in §6.1. A request that fails any of these checks returns `status=infra_error` or `status=partial` without leasing a GPU.

**REQ-BENCH-002: Exclusive timed device lease with MIG awareness** [traces SC-BENCH-002]
Phase 2 must lease exactly one GPU or MIG instance whose `sm_arch` and MIG profile match the requested target before timed execution, and must hold the lease with an asyncio.Semaphore(1) per `gpu_uuid`. Parallel use across different GPUs is allowed only when they share SKU, sm_arch, MIG profile, driver, runtime, harness version, artifact execution model, clock policy, objective hash, and adapter version.

**REQ-BENCH-003: Device hygiene preflight gates** [traces SC-BENCH-002, SC-BENCH-007]
Before sampling, Phase 2 must sample SM/memory clocks, power, temperature, ECC aggregate counts, Xid event state, and `nvmlDeviceGetCurrentClocksEventReasons` (or pynvml equivalent). Each gate in §6.2 Table maps a condition to a routing outcome; failing gates short-circuit to `infra_error`, `unstable_measurement`, or `profile_unavailable` (`profiler_permission_denied`) as applicable.

**REQ-BENCH-004: Warmup and per-sample iteration calibration** [traces SC-BENCH-001]
Phase 3 must perform at least `FAST_BENCH_WARMUP_MIN_RUNS` untimed launches per (candidate, shape) with adapter-specified reset semantics, then calibrate iterations-per-sample such that `FAST_BENCH_MIN_TIMED_BATCH_MS <= elapsed_ms <= FAST_BENCH_MAX_TIMED_BATCH_MS` within `FAST_BENCH_MAX_ITERATIONS_PER_SAMPLE`. The selected iteration count is recorded per shape.

**REQ-BENCH-005: Adapter iteration semantics dispatch** [traces SC-BENCH-001, SC-BENCH-006, SC-BENCH-007]
The Benchmarker must honor the adapter's declared `AdapterIterationSemantics` (`overwrite_pure` / `requires_output_reset` / `requires_full_input_reset` / `not_repeatable`) for both fast benchmark repeated-launch timing and deep-profile replay. `not_repeatable` forbids repeated-launch timing (one launch per sample, launch-overhead sensitivity recorded) and forces `profile_unavailable` with reason `adapter_not_repeatable`. `requires_full_input_reset` forces `profile_unavailable` unless the adapter provides a safe pre-launch restore hook.

**REQ-BENCH-006: CUDA event timing path with N-launch batching** [traces SC-BENCH-001]
Timed samples must be collected via `cudaEventRecord(start)` / N kernel launches on the benchmark stream / `cudaEventRecord(stop)` / `cudaEventSynchronize(stop)` / `cudaEventElapsedTime`. `per_launch_us = elapsed_ms * 1000 / N`. The timed region must exclude host allocation, host↔device copies, random input generation, reference kernel execution, correctness comparison, profiler startup, and context creation.

**REQ-BENCH-007: Metric mode validation** [traces SC-BENCH-001]
The Benchmarker must validate that the objective's `primary_metric` is supported by the adapter for the target harness. If the requested `metric_mode` is not `device_kernel_us`, the adapter must declare support for the requested mode; otherwise the candidate result carries `measurement_quality.status=not_comparable` with reason `unsupported_metric_mode` and contributes no score.

**REQ-BENCH-008: Function attribute policy recording** [traces SC-BENCH-001, SC-BENCH-008]
If the harness calls `cudaFuncSetAttribute`, `cudaFuncSetCacheConfig`, or driver-API equivalents for `max_dynamic_shared_memory_size`, `preferred_shared_memory_carveout_pct`, `cache_config`, `cluster_dims`, or `non_portable_cluster_size_allowed`, the requested values and the observed function attributes must both be stored in the envelope. Mismatched policies across batches are never considered the same measurement episode.

**REQ-BENCH-009: Cache policy selection and auto-promotion** [traces SC-BENCH-001, SC-BENCH-003]
Default cache policy is `warm_same_buffers` for single-candidate batches and `warm_rotating_buffers` for interleaved batches. When interleaving is enabled and the requested policy is `warm_same_buffers`, the effective policy is promoted to `warm_rotating_buffers` and the envelope records `requested_cache_policy`, `effective_cache_policy`, and `cache_policy_reason="interleaved_batch_requires_rotation"`. `cold_flush_buffer` and `reset_persisting_l2` are honored when explicitly requested.

**REQ-BENCH-010: Interleaved batch order generation** [traces SC-BENCH-003]
When `artifact_execution_model = "common_harness_cubin"` and the batch contains ≥2 candidates, Phase 4 must interleave candidate launches and incumbent anchors using a deterministic pseudo-random permutation seeded by `interleave_seed`. Anchor cadence obeys `ANCHOR_EVERY_N_SAMPLES`; the randomized block between anchors has length at most `MAX_INTERLEAVE_BLOCK_LEN`. The realized block order and the seed are stored in every per-shape artifact.

**REQ-BENCH-011: Per-shape statistics and minimum sample gating** [traces SC-BENCH-001, SC-BENCH-004]
Per shape, the Benchmarker must compute `p50_us`, `mean_us`, `stdev_us`, `cv_pct = stdev/mean*100`, `min_us`, `max_us` and — when `run_count >= MIN_P95_SAMPLES` — `p95_us`. Below `MIN_P95_SAMPLES` the rich artifact must report `p95_us = null`; the compact `ShapeBenchResult.latency_p95_us` may be populated as `p95_us` only when the minimum is met.

**REQ-BENCH-012: Measurement quality classification** [traces SC-BENCH-001, SC-BENCH-002, SC-BENCH-004, SC-BENCH-007]
Each shape measurement is classified `valid` / `valid_with_warning` / `unstable` / `runtime_fault` / `infra_fault` per the decision table in §6.4. Unstable measurements are retried at most `BENCH_RERUN_LIMIT` times before being reported as `unstable`. Unstable measurements never count as regressions.

**REQ-BENCH-013: Incumbent anchor policy** [traces SC-BENCH-003, SC-BENCH-004]
For any batch where a candidate may replace the incumbent, the incumbent is measured inside the same batch as both pre-block and post-block anchors (`ANCHOR_INCUMBENT_POLICY = same_episode`). Baseline is re-anchored when `ANCHOR_BASELINE_POLICY` triggers (new pod, new GPU, long-running job, or detected drift). Known-good probe is distinct from anchors and is used only for pod health, never for scoring.

**REQ-BENCH-014: Objective scoring from fast benchmark only** [traces SC-BENCH-001, SC-BENCH-004]
`ObjectiveScore.value` must be computed from Phase 4 fast-benchmark `ShapeBenchResult` fields only, using `PerformanceObjective.primary_metric` and `aggregation`. Nsight Compute timings and Nsight Systems timings must never contribute to `objective_score`.

**REQ-BENCH-015: Noise margin and incumbent comparison decision** [traces SC-BENCH-004]
The Benchmarker must compute `noise_margin_pct` per §6.5 and populate `IncumbentComparison` per the decision table in §6.5. Results inside the noise band are `statistical_tie`; results outside are `improved` or `regressed`; unstable inputs produce `unstable`; mismatched metric/arch/adapter produce `not_comparable`. `regressed_vs_incumbent` is set `true` only when `IncumbentComparison = regressed`.

**REQ-BENCH-016: Profile candidate selection** [traces SC-BENCH-005]
`top_k_profiled = top_k_by_objective_score ∪ top_m_by_bottleneck_shift_potential`, deduplicated by candidate hash, with the incumbent optionally included when profile comparison is useful. Shift-potential score is computed from intent direction/sub-mode, static-analysis delta (registers, smem, spills, occupancy), fast-benchmark throughput shape (useful bytes, arithmetic intensity, achieved FLOP/s), and novelty relative to incumbent. Shift-potential must not rely on NCU counters that have not yet been collected.

**REQ-BENCH-017: Deep profile target selection via NVTX** [traces SC-BENCH-006]
Each measured launch inside the profile phase is wrapped in an NVTX range named `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile`. The `ncu` command line must use NVTX range filtering and/or explicit launch skip/count to profile exactly one measured launch per (candidate, shape), not warmup, anchors, unrelated probes, or every repeated launch in the timing loop. The selected range, launch index, metric set, replay mode, profiler version, and raw report path are stored in the profile artifact.

**REQ-BENCH-018: Metric portability and provenance** [traces SC-BENCH-006]
Raw NCU metrics are stored as `RawProfileMetric{metric_name, value, unit, architecture, profiler_name, profiler_version, collection_section}`. Compact `ProfileMetrics` normalized fields include `source_metrics`, `architecture`, `profiler_version`, and `comparable_across_arch: false`. Missing metrics are `null`, never fabricated. `ProfileBundle.metrics` (shared type) and the richer raw-metric artifact are linked by `raw_profile_metrics_ref`.

**REQ-BENCH-019: Fault attribution** [traces SC-BENCH-007]
Faults are classified by `faults.attribute(exception, pod_health, repeated)`:
- `candidate_fault`: reproducible illegal-memory-access on a healthy pod, kernel timeout on a healthy device, runtime launch config error;
- `infra_fault`: GPU lost, Xid error, artifact store failure, profiler permission denied, pod disk full, driver version mismatch, `ncu` binary missing at runtime;
- `ambiguous_fault`: process killed during GPU execution, CUDA context poisoned, one-off timeout on a suspect pod.

**REQ-BENCH-020: Pod health state machine and probe** [traces SC-BENCH-002, SC-BENCH-007]
The Benchmarker maintains `pod_health ∈ {healthy, suspect, quarantined}`. A clean batch keeps health `healthy`; an ambiguous Phase 4 outcome moves it to `suspect` and runs a known-good probe kernel before the next batch; probe failure moves it to `quarantined`; repeated ambiguous faults above `AMBIGUOUS_FAILURE_LIMIT` also trigger `quarantined`. Every response envelope carries `pod_health` and the ambiguous-failure counter.

**REQ-BENCH-021: Artifact store and retention** [traces SC-BENCH-006, SC-BENCH-008]
Pod-local artifact store holds raw samples JSON, NCU `.ncu-rep` reports, NVTX range logs, telemetry snapshots, and `ShapeMeasurementArtifact` JSON. Each artifact is addressable by id, retained per `ARTIFACT_RETENTION`, and GC'd on disk-high-watermark and on batch completion. Artifacts referenced by an active response are never GC'd mid-read.

**REQ-BENCH-022: Nsight Systems trigger policy** [traces SC-BENCH-006]
Nsight Systems (`nsys`) is not a default per-candidate profiler. It runs only when `NSYS_PROFILE_POLICY` fires on a timeline-shaped trigger: objective includes transfers or host overhead, overlap/stream behavior is suspected, profiler counters conflict or are insufficient, multiple contexts or foreign processes are suspected, or a CUDA graph / library call produces hidden kernels. Timeline artifacts are stored by reference only.

**REQ-BENCH-023: FastAPI surface** [traces SC-BENCH-010]
The service exposes:
- `POST /benchmark` — body `BenchmarkBatchRequest`, response `BenchmarkBatchResult`.
- `GET /healthz` — returns 200 + `{status: "ready", ...}` only when all readiness sub-checks pass (driver, GPU, `ncu`, artifact root, pod health ≠ quarantined); 503 + structured error otherwise; startup hook exits non-zero on any missing required tool.
- `GET /info` — returns service identity, build hash, toolchain versions (driver, CUDA runtime, `cuda-python`, `pynvml`, `ncu`), list of visible GPUs with `sm_arch` / `gpu_uuid` / MIG profile, configured `DEFAULT_METRIC_MODE`, `ARTIFACT_EXECUTION_MODEL`, and configured supported operation adapters.

**REQ-BENCH-024: Artifact execution model is cubin/module + common harness** [traces SC-BENCH-003, SC-BENCH-008]
V1 supports exactly one artifact execution model: cubin/module loaded by a common benchmark harness running inside a disposable worker subprocess, with one CUDA context per batch. Separate-executable-per-candidate mode is not implemented; the request schema accepts only `ARTIFACT_EXECUTION_MODEL = "common_harness_cubin"`.

**REQ-BENCH-025: Disposable worker subprocess per batch** [traces SC-BENCH-003, SC-BENCH-007]
The harness runs in a subprocess launched once per batch. A poisoned CUDA context, a crashed candidate, or an ambiguous device fault collapses that subprocess; the supervising service marks the batch `ambiguous_fault`, transitions pod health to `suspect`, and returns `candidate_results[]` for any candidates whose measurements were already durably recorded before the crash. One candidate's crash must not corrupt measurements for later candidates because the subprocess is terminated rather than reused.

**REQ-BENCH-026: Shared type reuse without mutation** [traces SC-BENCH-008]
The Benchmarker reuses `ShapeCase`, `PerformanceObjective`, `ProblemSpec`, `StaticAnalysis`, `ShapeBenchResult`, `ObjectiveScore`, `ProfileMetrics`, `BottleneckAssessment`, `ProfileBundle`, `CorrectnessResult`, and `BenchmarkBundle` from `kerlever/types.py` verbatim. If a shared type does not have a field the Benchmarker needs (e.g., `IncumbentComparison`), the field lives in a Benchmarker-local type and `BenchmarkBundle.regressed_vs_incumbent` is populated as the compatibility mapping. No modification of `kerlever/types.py` is required for V1.

**REQ-BENCH-027: Deterministic service** [traces SC-BENCH-001..SC-BENCH-008]
The Benchmarker must be fully deterministic given identical inputs, pod health, and device identity. No LLM calls, no natural-language routing outcomes. Every decision — hygiene gates, thresholds, measurement quality, noise margin, incumbent comparison, profile selection, fault attribution — is implemented as code.

**REQ-BENCH-028: Operation Adapter registry and plugin contract** [traces SC-BENCH-012]
Phase 1 must consult an in-process Operation Adapter registry keyed by `(operation_adapter_abi, operation_adapter_version)`. Exactly one adapter per key is registered at service startup. Phase 1 resolution: if the key is unknown → `status = infra_error`, reason `adapter_unregistered`; if `operation_adapter_abi` matches but `operation_adapter_version` does not → `status = infra_error`, reason `adapter_version_mismatch`. The resolved adapter is bound to every candidate in the batch — mixed adapters inside one batch are not supported (batches must homogenize adapter upstream). The adapter instance is the sole source of: `allocate` (per-shape input/output device buffers using `cuda-python` driver-API device memory), `seed_inputs` (deterministic seeding from `hash(run_id, batch_id, shape_id)`), `build_launch_args` (ordered tuple of device pointers plus ABI-declared scalar dimensions, returned as `tuple[object, ...]` suitable for `cuLaunchKernel`), `grid_dim` (per-shape grid derivation from shape and block-dim), `useful_bytes` and `algorithmic_flops` (work-model computation for `effective_bandwidth_gbps` / `achieved_flops` / `arithmetic_intensity_flop_per_byte`), `reset_between_iterations` (reset hook invoked between timed iterations for non-`OVERWRITE_PURE` semantics), `rotate_buffers` (buffer-rotation strategy for `WARM_ROTATING_BUFFERS`), `free` (buffer teardown). Adapter calls that return default/empty structures in V1 are a spec violation — `build_launch_args` MUST NOT return `tuple()` when the adapter ABI declares any operand. V1 ships exactly two concrete adapters: `elementwise_add_fp32_v1` and `matmul_fp16_v1`.

**REQ-BENCH-029: Function attribute policy applied at module load** [traces SC-BENCH-001, SC-BENCH-008]

**V1 SCOPE (NARROWED).** In V1 the request schema — specifically `CandidateArtifactRef.launch_spec.dynamic_smem_bytes` inherited from the shared Compiler Service artifact contract — only carries the **dynamic shared memory size** dimension of the function-attribute policy. The richer `FunctionAttributePolicy` fields (`preferred_shared_memory_carveout_pct`, `cache_config`, `cluster_dims`, `non_portable_cluster_size_allowed`) are NOT populated by any V1 caller because `CandidateArtifactRef` does not yet expose fields to carry them. V1 therefore enforces application of ONLY `max_dynamic_shared_memory_size`. The remaining fields are V2 scope (see §9 Non-Goals) and will require a coordinated shared-type extension before they can be applied. A V1 harness that tries to apply `cache_config` / `cluster_dims` / `carveout_pct` from the request has no source of truth for those values — the fields do not flow end-to-end.

**V1 normative rule.** Before the first measured launch of every candidate, the harness MUST apply the request's effective `max_dynamic_shared_memory_size` to the loaded function via the driver API:

- When `launch_spec.dynamic_smem_bytes > 0`, call `cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, launch_spec.dynamic_smem_bytes)`.
- After the apply, call `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func)` and record the observed value in `MeasurementEnvelope.function_attribute_policy_observed.max_dynamic_shared_memory_size`.
- `MeasurementEnvelope.function_attribute_policy_requested.max_dynamic_shared_memory_size` is populated from `launch_spec.dynamic_smem_bytes`.
- If the `cuFuncSetAttribute` call returns a driver error (e.g., `launch_spec.dynamic_smem_bytes` exceeds the device maximum), the candidate is rejected in Phase 3 with `fault_class = infra_fault`, `measurement_quality.status = infra_fault`, reason `function_attribute_policy_apply_failed` and the driver error code; other candidates in the batch proceed.

**V2 fields (NOT applied in V1; documented here for forward-compatibility).** `preferred_shared_memory_carveout_pct`, `cache_config`, `cluster_dims`, `non_portable_cluster_size_allowed` — the V1 harness MUST NOT attempt to apply these because no request field carries them end-to-end. The `FunctionAttributePolicy` Pydantic model retains all fields (shape parity with V2) but only `max_dynamic_shared_memory_size` is guaranteed populated in V1 — the other four fields default to `None` in both `function_attribute_policy_requested` and `function_attribute_policy_observed` in V1 batches. A V1 implementation that silently applies a hardcoded value for any of these fields (e.g., "prefer_shared" as a harness default) is a spec violation — the envelope would then claim a policy was "requested" that no caller actually supplied. When V2 extends the request schema to carry these fields, REQ-BENCH-029 will be broadened in the same commit that updates the schema.

**Enforcement vs. recording.** The requirement is enforcement, not recording — a candidate whose `max_dynamic_shared_memory_size` was "recorded" into the envelope but not actually applied via `cuFuncSetAttribute` has violated REQ-BENCH-029 even if the envelope is syntactically complete.

**REQ-BENCH-030: NCU target is the dedicated profile child subprocess** [traces SC-BENCH-013, SC-BENCH-006]
For each `(selected candidate, profile_shape)` pair in Phase 6, the worker MUST spawn a dedicated profile child subprocess and invoke NCU as the parent of that child. The profile child command line MUST carry enough context — cubin path, launch_spec, operation adapter abi/version, shape dimensions, function_attribute_policy, `iterations_per_sample`, and an NVTX range name — to reproduce exactly one measured launch inside the NVTX range and then exit. The NCU command line MUST use `--nvtx --nvtx-include <range> --launch-count 1` and MUST target the profile child's executable path (as `--` positional). The profile child MUST NOT attempt to run Phase 2 hygiene, Phase 4 statistics, or Phase 5 scoring — its sole responsibility is the deterministic replay of one NVTX-ranged launch. The worker process is never invoked with `target_cmd=[sys.executable, "-c", "pass"]` or any other placeholder; such a target is a spec violation even if the command succeeds. On profile-child exit code ≠ 0, NCU timeout, or NCU exit code ≠ 0, the result is `ProfileStatus.profile_unavailable` with the mapped reason (`profiler_timeout`, `profiler_replay_refused`, `profiler_binary_missing`, etc.).

**REQ-BENCH-031: Cubin transport V1 contract** [traces SC-BENCH-014]
Phase 1 MUST validate that every `CandidateArtifactRef.cubin_uri` is an absolute POSIX path (`str.startswith("/")`) and that the file exists and is readable by the Benchmarker process. A `cubin_uri` that matches the regex `^[a-zA-Z][a-zA-Z0-9+.-]*:` (URI scheme prefix) → `status = infra_error`, reason `cubin_uri_unsupported_scheme`. A `cubin_uri` that is not absolute or does not resolve to a readable file → `status = infra_error`, reason `cubin_uri_not_readable`. The request schema MUST NOT declare a `cubin_bytes_b64` field; if the HTTP body contains such a field it is ignored (Pydantic `extra='ignore'`) and a warning is recorded in the batch envelope. The `incumbent_ref.cubin_uri` is validated under the same rules. V1 assumes the caller and Benchmarker share a volume where cubin blobs are written by the Compiler Service and read by the Benchmarker; any remote transport (object store download, HTTP fetch, inline bytes) is out of scope and will be added in V2 under a separate SC.

**REQ-BENCH-032: Rich artifact fidelity enforcement** [traces SC-BENCH-015]
The code path that constructs `ShapeMeasurementArtifact` MUST populate every field from the real runtime sources listed in SC-BENCH-015. Specifically:
- `warmup_count` from the actual warmup-loop iteration count for this `(candidate, shape)` (not from config directly — from the executed count).
- `iterations_per_sample` from Phase 3 calibration output for this `(candidate, shape)`.
- `min_samples_required` from the effective `MIN_P95_SAMPLES` (or objective-specific minimum) for this batch.
- `interleave_block_len` = `len(interleave_block_order)` when `interleave_enabled = True`, else `None`.
- `interleave_block_order` = the realized token sequence emitted by the block generator.
- `anchor_pre_us`, `anchor_post_us`, `anchor_drift_pct` from the pre/post-block anchor samples; when `interleave_enabled = False`, all three MUST be `None` (not `0.0`).
- `cache_policy` is the effective cache policy; `requested_cache_policy` and `effective_cache_policy` mirror `MeasurementEnvelope.cache_policy.requested` and `.effective`.
- `artifact_execution_model` = `MeasurementEnvelope.artifact_execution_model`.
- `adapter_iteration_semantics` = the matched adapter's declared semantics (adapter is authoritative; request field is a hint).
- `metric_mode` = `MeasurementEnvelope.metric_mode`.
- `max_timed_batch_ms` = effective `FAST_BENCH_MAX_TIMED_BATCH_MS`.
- `function_attribute_policy` = `MeasurementEnvelope.function_attribute_policy_observed`.
- `measurement_quality` from the Phase 4 classifier output for this `(candidate, shape)`.
- `telemetry_before` ← the Phase 2 preflight `pynvml` `DeviceTelemetrySnapshot` for the leased `gpu_uuid`, OR a fresh `pynvml` sample taken in Phase 4 immediately before the first measured launch for this shape (whichever is closer to the first measured launch). Real `pynvml` readout with real `taken_at_ms`, `sm_clock_mhz`, `mem_clock_mhz`, `gpu_temp_c`, `power_w`, `throttle_reasons`, `ecc_sbe_total`, `ecc_dbe_total`, `xid_events_since_last`. A `DeviceTelemetrySnapshot()` (no-arg default construction — `taken_at_ms=0`, every other field `None` or empty) is a spec violation unless `pynvml` initialization failed at startup and `MeasurementEnvelope.warnings` explicitly records `pynvml_unavailable`.
- `telemetry_after` ← the Phase 4 postflight `pynvml` sample taken immediately after the last measured launch for this shape (after `cudaEventSynchronize(stop)` returned for the final sample of the shape). Same realness requirements as `telemetry_before`. A default-constructed `DeviceTelemetrySnapshot` here is a spec violation.

Any field assigned a literal default that does not match the real runtime source above is a spec violation. Static typing cannot catch this; enforcement is via code review, behavioral tests, and §4 INV-BENCH-014.

**REQ-BENCH-033: p95 sentinel when gated** [traces SC-BENCH-004, SC-BENCH-001]
When `run_count < MIN_P95_SAMPLES` for a shape:
- `ShapeMeasurementArtifact.p95_us = None` (rich artifact).
- `ShapeBenchResult.latency_p95_us = -1.0` (compact artifact; the shared type `ShapeBenchResult.latency_p95_us` is `float` non-optional per `kerlever/types.py`; `-1.0` is the p95-unavailable sentinel because latencies are strictly non-negative in real measurements). Populating `ShapeBenchResult.latency_p95_us = latency_p50_us` as a fallback is **forbidden** and a spec violation — that shim silently promotes p50 to p95 and breaks `weighted_p95_us` and `worst_case_p95_us` objective aggregation.
- When the objective's `primary_metric` is `weighted_p95_us` (or any p95-based metric), the scoring aggregator in Phase 5 MUST exclude any shape whose `latency_p95_us == -1.0` (or `ShapeMeasurementArtifact.p95_us is None`) from the weighted mean / max. If after exclusion no shape contributes, `ObjectiveScore.value` MUST NOT be computed; the candidate's `measurement_quality.status = not_comparable`, reason `p95_samples_below_minimum_for_p95_metric`, and it is excluded from `top_k_profiled`.
- Downstream consumers are expected to treat `latency_p95_us == -1.0` as "p95 not available, consult `ShapeMeasurementArtifact.p95_us`"; the sentinel is documented in the compact shared-type comment in code via `REQ-BENCH-033`.

**REQ-BENCH-034: Incumbent anchor CV fed into noise margin** [traces SC-BENCH-004]
When `decide_incumbent_comparison` is invoked, `incumbent_cv_pct` MUST be computed from the actual incumbent anchor samples collected in this batch for the relevant shape (via `stats.cv_pct(anchor_samples)`) whenever `len(anchor_samples) >= 2`. Passing `incumbent_cv_pct = None` or `0.0` when anchor samples exist is a spec violation; the noise margin `max(NOISE_FLOOR_PCT, candidate_cv_pct, anchor_cv_pct, anchor_drift_pct)` requires `anchor_cv_pct` to be a real measurement. `anchor_cv_pct = None` is permitted only when `interleave_enabled = False` AND no in-episode anchor was collected for the shape (rare; usually implies `NOT_COMPARABLE`).

**REQ-BENCH-035: Supervisor staging lifecycle** [traces SC-BENCH-016]
The Supervisor MUST clean up the per-batch staging directory `<cfg.artifact.root>/staging/<batch_id>/` after the batch's `BenchmarkBatchResult` has been handed back to the HTTP caller. "Clean up" means removing every file the Supervisor wrote under that directory during the batch: `request.json`, `config.json`, per-shape intermediate result JSON, any `.ncu-rep` files the Supervisor chose to emit under the staging subtree, and finally the `<batch_id>/` directory itself. Cleanup MUST run regardless of the batch's terminal `status` (`success`, `partial`, `unstable`, `timeout`, `infra_error`) and regardless of whether a Python exception was raised during Phase 1..Phase 7 — a `finally`-equivalent code path owns cleanup. Cleanup MUST NOT touch any file outside `<cfg.artifact.root>/staging/<batch_id>/`: the durable artifact tree referenced by `CandidateResult.shape_measurement_artifact_refs`, `profile_artifact_refs`, and `raw_profile_metrics_ref` (typically under `<cfg.artifact.root>/samples/`, `/ncu/`, `/raw_metrics/`, `/artifacts/`) is out of scope for staging cleanup and retained per REQ-BENCH-021. **Escape hatch.** If the environment variable `KEEP_STAGED_ARTIFACTS=1` is set at Supervisor startup, staging cleanup is skipped for all batches in this pod lifetime and a one-time warning `keep_staged_artifacts_enabled_staging_cleanup_skipped` is recorded in the pod's service-level log; the pod operator is then responsible for manual cleanup. This flag is intended only for short-duration forensic debugging and MUST NOT be the default in production deployments. Any other value (including unset or `0`) means cleanup is active.

### Quality Gates

**QG-BENCH-001: Type Safety** [traces SC-BENCH-011]
All source under `kerlever/benchmarker/` passes `mypy .` with `strict = true`.

**QG-BENCH-002: Lint** [traces SC-BENCH-011]
All source under `kerlever/benchmarker/` passes `ruff check .` with repository configuration.

---

## §3 Scenarios

**SCN-BENCH-001-01: Happy path, three candidates, two shapes, one profile shape**
- GIVEN: a `BenchmarkBatchRequest` with three correctness-passing candidate artifact refs, two `objective_shape_cases` (`shape_small`, `shape_large`) and one `profile_shape_cases` (`shape_large`), `artifact_execution_model = "common_harness_cubin"`, `top_k_profile = 2`
- AND: pod health is `healthy`, GPU matches `sm_arch`, no throttle, no foreign process
- WHEN: the Benchmarker runs Phase 1..7
- THEN: the response `status = success`
- AND: each of three `candidate_results[]` contains a `BenchmarkBundle` with `shape_results` for both shapes, an `ObjectiveScore`, and an `IncumbentComparison`
- AND: two candidates appear in `top_k_profiled` with ncu reports attached
- AND: `pod_health = healthy` in the envelope
Implements: REQ-BENCH-001, REQ-BENCH-011, REQ-BENCH-014, REQ-BENCH-015, REQ-BENCH-016, REQ-BENCH-017

**SCN-BENCH-002-01: Architecture mismatch short-circuits to infra_error**
- GIVEN: the request names `target_gpu` with `sm_arch = "sm_90"` but the only visible GPU has `sm_arch = "sm_80"`
- WHEN: Phase 2 attempts to lease a matching device
- THEN: the response `status = infra_error` with `reason = "arch_mismatch"`, no timed work is done, `pod_health` is unchanged
Implements: REQ-BENCH-002, REQ-BENCH-003, INV-BENCH-003

**SCN-BENCH-002-02: Foreign compute process marks episode unstable**
- GIVEN: `nvmlDeviceGetComputeRunningProcesses` reports a non-Benchmarker PID holding the target GPU
- WHEN: Phase 2 preflight runs
- THEN: the measurement is marked `measurement_quality.status = unstable` with reason `foreign_compute_process`
- AND: the Benchmarker may retry up to `BENCH_RERUN_LIMIT` times
- AND: if still unstable, `status = unstable` at the batch level
Implements: REQ-BENCH-003, REQ-BENCH-012

**SCN-BENCH-002-03: Thermal throttle marks episode unstable**
- GIVEN: `nvmlDeviceGetCurrentClocksEventReasons` returns `HW_SLOWDOWN | SW_THERMAL_SLOWDOWN`
- WHEN: Phase 2 preflight runs
- THEN: measurement_quality is `unstable` with reason `thermal_throttle`
- AND: the candidate result carries `measurement_quality.status = unstable` and contributes no optimization signal
Implements: REQ-BENCH-003, REQ-BENCH-012

**SCN-BENCH-002-04: Temperature above steady-state limit triggers cooldown retry**
- GIVEN: `nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU) > THERMAL_STEADY_STATE_LIMIT` at preflight
- WHEN: hygiene runs
- THEN: a single bounded cooldown wait is performed; if temperature then satisfies the limit the batch proceeds, else `status = unstable` with reason `thermal_above_steady_state`
Implements: REQ-BENCH-003

**SCN-BENCH-002-05: ECC double-bit / Xid error quarantines pod**
- GIVEN: a nonzero double-bit ECC aggregate or an Xid event since last probe
- WHEN: hygiene runs
- THEN: `status = infra_error` with reason `ecc_xid`, `pod_health = quarantined`, subsequent batches short-circuit without touching GPU
Implements: REQ-BENCH-003, REQ-BENCH-019, REQ-BENCH-020, INV-BENCH-003

**SCN-BENCH-002-06: MIG profile mismatch blocks ranking**
- GIVEN: the request asks for a MIG profile `mig_1g.10gb` and the lease candidate is `mig_3g.40gb`
- WHEN: Phase 2 runs
- THEN: `status = infra_error` with reason `mig_profile_mismatch`, no timed work
Implements: REQ-BENCH-002, REQ-BENCH-003

**SCN-BENCH-002-07: Profiler permission missing does not block fast benchmark**
- GIVEN: `ncu` exists but the container/user lacks `CAP_PERFMON` / `--add-cap SYS_ADMIN` privileges
- WHEN: Phase 2 preflight detects no performance-counter permission
- THEN: fast benchmark continues normally
- AND: every selected profile candidate returns `ProfileStatus.profile_unavailable` with `profile_unavailable_reason = profiler_permission_denied`
Implements: REQ-BENCH-003, REQ-BENCH-006, SC-BENCH-007

**SCN-BENCH-002-08: Clock state is recorded even when locking is unavailable**
- GIVEN: the container lacks `nvidia-smi -ac` privileges
- WHEN: Phase 2 resolves `CLOCK_POLICY`
- THEN: `clock_policy.mode = "lock_requested_unavailable"` in the envelope and observed clocks are still sampled and recorded
Implements: REQ-BENCH-003, REQ-BENCH-008

**SCN-BENCH-003-01: Warmup + iteration calibration hits lower bound**
- GIVEN: a very fast kernel where one launch elapses < `FAST_BENCH_MIN_TIMED_BATCH_MS` even at the cap
- WHEN: Phase 3 runs
- THEN: iterations-per-sample doubles until either the lower bound is met or `FAST_BENCH_MAX_ITERATIONS_PER_SAMPLE` is hit
- AND: if the cap is hit without meeting the lower bound, `measurement_quality = valid_with_warning` with reason `calibration_lower_bound_unmet`
Implements: REQ-BENCH-004

**SCN-BENCH-003-02: Iteration calibration hits upper bound**
- GIVEN: a slow kernel where the initial calibration produces elapsed > `FAST_BENCH_MAX_TIMED_BATCH_MS`
- WHEN: Phase 3 runs
- THEN: iterations-per-sample is reduced (down to 1 if necessary)
- AND: if the upper bound still cannot be met, `measurement_quality = valid_with_warning` with reason `calibration_upper_bound_unmet`
Implements: REQ-BENCH-004

**SCN-BENCH-003-03: Metric mode not supported by adapter**
- GIVEN: the objective declares `primary_metric = weighted_p95_us` with `metric_mode = cuda_graph_replay_us` but the adapter declares no graph capture support
- WHEN: Phase 3 validates the metric mode
- THEN: the candidate result carries `measurement_quality.status = not_comparable`, reason `unsupported_metric_mode`, and no `ObjectiveScore.value`
Implements: REQ-BENCH-007

**SCN-BENCH-003-04: Adapter not_repeatable forbids repeated-launch timing**
- GIVEN: the adapter declares `AdapterIterationSemantics = not_repeatable`
- WHEN: Phase 3 runs
- THEN: iterations-per-sample = 1, launch-overhead sensitivity is recorded in `ShapeMeasurementArtifact`
- AND: deep profile selection returns `profile_unavailable` with reason `adapter_not_repeatable` for this candidate
Implements: REQ-BENCH-005

**SCN-BENCH-003-05: Cache policy auto-promoted in interleaved batch**
- GIVEN: `cache_policy = warm_same_buffers` requested and `len(candidate_module_artifact_refs) >= 2` with `artifact_execution_model = common_harness_cubin`
- WHEN: Phase 3 resolves effective cache policy
- THEN: `requested_cache_policy = warm_same_buffers`, `effective_cache_policy = warm_rotating_buffers`, `cache_policy_reason = interleaved_batch_requires_rotation`
Implements: REQ-BENCH-009, INV-BENCH-005

**SCN-BENCH-003-06: Single-candidate batch uses warm_same_buffers**
- GIVEN: one candidate in the batch
- WHEN: Phase 3 resolves cache policy with no explicit override
- THEN: `effective_cache_policy = warm_same_buffers`, no promotion, interleaving is not used
Implements: REQ-BENCH-009, REQ-BENCH-010

**SCN-BENCH-004-01: Interleaved batch uses stable seed and records block order**
- GIVEN: three candidates, one shape, `ANCHOR_EVERY_N_SAMPLES = 4`, `MAX_INTERLEAVE_BLOCK_LEN = 6`
- AND: `interleave_seed = hash(run_id, batch_id, shape_id, "kerlever_benchmark_order")` is the same across retries
- WHEN: Phase 4 generates block order
- THEN: the realized `[A, anchor, B, C, A, anchor, ...]` sequence is deterministic for that seed
- AND: the sequence is stored in `ShapeMeasurementArtifact.interleave_block_order`
Implements: REQ-BENCH-010, INV-BENCH-004

**SCN-BENCH-004-02: Pre/post anchors and drift recorded per shape**
- GIVEN: a shape measured in an interleaved batch
- WHEN: Phase 4 completes
- THEN: `anchor_pre_us` and `anchor_post_us` are both populated
- AND: `anchor_drift_pct = |anchor_post - anchor_pre| / anchor_pre` is computed
- AND: if `anchor_drift_pct > ANCHOR_DRIFT_FAIL_PCT`, `measurement_quality.status = unstable` with reason `anchor_drift_exceeded`
Implements: REQ-BENCH-013, INV-BENCH-002

**SCN-BENCH-004-03: p95 gated by minimum sample count**
- GIVEN: `run_count = MIN_P95_SAMPLES - 1` valid samples
- WHEN: per-shape stats are computed
- THEN: `ShapeMeasurementArtifact.p95_us = null`
- AND: `ShapeBenchResult.latency_p95_us` is set to `latency_p50_us` as a safe fallback or flagged per policy; objective `weighted_p95_us` is not used in scoring unless samples meet the minimum
Implements: REQ-BENCH-011

**SCN-BENCH-004-04: CV above MEASUREMENT_CV_FAIL_PCT marks unstable**
- GIVEN: a candidate whose shape samples yield `cv_pct > MEASUREMENT_CV_FAIL_PCT`
- WHEN: Phase 4 classifies quality
- THEN: `measurement_quality.status = unstable`, reason `cv_above_fail_threshold`
- AND: the Benchmarker retries at most `BENCH_RERUN_LIMIT` times before reporting `unstable`
Implements: REQ-BENCH-012

**SCN-BENCH-005-01: Improvement beyond noise margin is `improved`**
- GIVEN: `incumbent_anchor_score = 100.0 µs`, `candidate_score = 88.0 µs`, `noise_margin_pct = 0.03`
- WHEN: Phase 5 runs
- THEN: `IncumbentComparison = improved`, `regressed_vs_incumbent = false`, `relative_to_incumbent = 0.88`
Implements: REQ-BENCH-015, INV-BENCH-006

**SCN-BENCH-005-02: Improvement inside noise margin is `statistical_tie`**
- GIVEN: `incumbent_anchor_score = 100.0 µs`, `candidate_score = 98.5 µs`, `anchor_drift_pct = 0.04`, `NOISE_FLOOR_PCT = 0.01`
- WHEN: Phase 5 runs
- THEN: `noise_margin_pct = 0.04`, candidate is not below `100 * (1 - 0.04) = 96.0`
- AND: `IncumbentComparison = statistical_tie`, `regressed_vs_incumbent = false`
Implements: REQ-BENCH-015, INV-BENCH-006

**SCN-BENCH-005-03: Regression beyond guard + noise is `regressed`**
- GIVEN: `incumbent_anchor_score = 100.0`, `candidate_score = 110.0`, `guard_pct = 0.02`, `noise_margin_pct = 0.03`
- WHEN: Phase 5 runs
- THEN: candidate exceeds `100 * (1 + 0.02 + 0.03) = 105`, `IncumbentComparison = regressed`, `regressed_vs_incumbent = true`
Implements: REQ-BENCH-015

**SCN-BENCH-005-04: Regression is not profile-eligible**
- GIVEN: a candidate with `IncumbentComparison = regressed`
- WHEN: Phase 6 selects profile targets
- THEN: the candidate is excluded from `top_k_profiled`
- AND: it may still appear in a shift-potential list if allowed by policy, but V1 excludes regressed candidates from `top_k_profiled`
Implements: REQ-BENCH-016

**SCN-BENCH-005-05: Unstable measurement is `unstable` comparison**
- GIVEN: a candidate whose shape measurement is classified `unstable` for any shape contributing to the objective
- WHEN: Phase 5 runs
- THEN: `IncumbentComparison = unstable`, `regressed_vs_incumbent = false`, the candidate is excluded from `top_k_profiled`
Implements: REQ-BENCH-012, REQ-BENCH-015, INV-BENCH-006

**SCN-BENCH-005-06: Mismatched metric / arch / adapter is `not_comparable`**
- GIVEN: the candidate was measured under a different `function_attribute_policy` than the incumbent anchor
- WHEN: Phase 5 runs
- THEN: `IncumbentComparison = not_comparable`, reason `envelope_mismatch`
Implements: REQ-BENCH-008, REQ-BENCH-015

**SCN-BENCH-006-01: top_k ∪ top_m deduplicates**
- GIVEN: candidate A is #1 by score and also in the top-M by shift potential
- WHEN: Phase 6 builds `top_k_profiled`
- THEN: A appears exactly once; |top_k_profiled| ≤ `TOP_K_PROFILE + TOP_M_PROFILE_SHIFT_POTENTIAL`
Implements: REQ-BENCH-016

**SCN-BENCH-006-02: Shift-potential uses pre-profile signals only**
- GIVEN: no NCU metrics have been collected yet in this batch
- WHEN: shift-potential is computed
- THEN: the score depends only on intent direction/sub-mode, static-analysis delta, fast-benchmark throughput shape, and novelty — never on uncollected NCU counters
Implements: REQ-BENCH-016, INV-BENCH-007

**SCN-BENCH-007-01: NCU targets the marked launch via NVTX**
- GIVEN: a selected (candidate, profile_shape)
- WHEN: Phase 6 runs `ncu`
- THEN: the `ncu` command includes NVTX range filter `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile`
- AND: the collected report contains exactly one profiled launch for that candidate
Implements: REQ-BENCH-017, INV-BENCH-008

**SCN-BENCH-007-02: Raw metrics carry provenance**
- GIVEN: NCU returns sections including `SpeedOfLight`, `LaunchStats`, `Occupancy`, `MemoryWorkloadAnalysis`
- WHEN: the Benchmarker extracts metrics
- THEN: each entry is a `RawProfileMetric{metric_name, value, unit, architecture, profiler_name="ncu", profiler_version, collection_section}`
- AND: normalized `tensor_core_utilization_pct` carries `source_metrics`, `architecture`, `comparable_across_arch = false`
Implements: REQ-BENCH-018, INV-BENCH-009

**SCN-BENCH-007-03: Missing metric is null, never fabricated**
- GIVEN: the architecture does not expose `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`
- WHEN: normalization runs
- THEN: `ProfileMetrics.tensor_core_utilization_pct = null`, the raw metric list records `value = null`, and no synthetic value is produced
Implements: REQ-BENCH-018, INV-BENCH-009

**SCN-BENCH-007-04: requires_full_input_reset without restore hook is profile_unavailable**
- GIVEN: adapter declares `requires_full_input_reset` and provides no safe restore hook
- WHEN: Phase 6 decides replay coverage
- THEN: `ProfileStatus = profile_unavailable`, reason `profiler_replay_refused`
Implements: REQ-BENCH-005

**SCN-BENCH-007-05: NCU timeout is profile_unavailable, not infra_error**
- GIVEN: `ncu` exceeds `PROFILE_TIMEOUT`
- WHEN: the subprocess is terminated
- THEN: `ProfileStatus = profile_unavailable`, reason `profiler_timeout`; the fast benchmark result is preserved
Implements: REQ-BENCH-021, REQ-BENCH-022

**SCN-BENCH-008-01: Candidate runtime fault on healthy pod is candidate_fault**
- GIVEN: pod health = healthy, candidate X raises `CUDA_ERROR_ILLEGAL_ADDRESS` on the first launch after warmup
- WHEN: Phase 4 catches the exception
- THEN: `candidate_results[X].fault_class = candidate_fault`, `measurement_quality.status = runtime_fault`, `measurement_quality.reason = illegal_memory_access`
- AND: the subprocess is terminated and the batch resumes for remaining candidates in a fresh subprocess if possible
Implements: REQ-BENCH-019, REQ-BENCH-025

**SCN-BENCH-008-02: Ambiguous fault on suspect pod does not produce negative signal**
- GIVEN: pod health = suspect, candidate Y times out once during timed execution
- WHEN: Phase 4 attributes the fault
- THEN: `fault_class = ambiguous_fault`, measurement_quality is `infra_fault`, no `IncumbentComparison` decision
- AND: pod health remains `suspect` and a known-good probe runs before the next batch
Implements: REQ-BENCH-019, REQ-BENCH-020, INV-BENCH-010

**SCN-BENCH-008-03: Repeated ambiguous faults quarantine pod**
- GIVEN: three ambiguous faults within `AMBIGUOUS_FAILURE_LIMIT` batches
- WHEN: the third fault is recorded
- THEN: `pod_health = quarantined`, every subsequent batch returns `status = infra_error` with `fault_class = infra_fault` and `pod_health = quarantined` without touching the GPU
Implements: REQ-BENCH-020

**SCN-BENCH-009-01: Request is accepted without `GPUPipelineProtocol`**
- GIVEN: a `BenchmarkBatchRequest` JSON body
- WHEN: `POST /benchmark` is called
- THEN: the service accepts it without importing `kerlever.protocols`
- AND: the response `BenchmarkBatchResult` contains `candidate_results[]` shaped for a future GPU Pipeline Adapter, not `EvaluationResult`
Implements: REQ-BENCH-026, INV-BENCH-001

**SCN-BENCH-010-01: /healthz ready on GPU host**
- GIVEN: the container runs on a GPU host with driver, visible GPU matching configured target, `ncu` present, artifact root writable
- WHEN: `GET /healthz` is called
- THEN: 200 with `{status: "ready", toolchain: {...}, gpus: [...], pod_health: "healthy"}`
Implements: REQ-BENCH-023

**SCN-BENCH-010-02: /healthz returns 503 on missing ncu**
- GIVEN: the container image is built without `ncu`
- WHEN: `GET /healthz` is called
- THEN: 503 with `{status: "not_ready", missing: ["ncu"], ...}`
- AND: the startup hook has already exited non-zero so the container is visibly unhealthy
Implements: REQ-BENCH-023

**SCN-BENCH-010-03: /info exposes toolchain and configured adapters**
- GIVEN: the service is ready
- WHEN: `GET /info` is called
- THEN: the response includes driver/runtime/cuda-python/pynvml/ncu versions, visible GPUs with `sm_arch` and MIG profile, `DEFAULT_METRIC_MODE`, `ARTIFACT_EXECUTION_MODEL`, and configured supported adapters
Implements: REQ-BENCH-023

**SCN-BENCH-011-01: Unknown adapter ABI returns infra_error before GPU lease**
- GIVEN: a `BenchmarkBatchRequest` with `operation_adapter_abi = "conv2d_bf16_v7"` and no such adapter is registered
- WHEN: Phase 1 looks up the adapter
- THEN: the response `status = infra_error`, reason `adapter_unregistered`
- AND: no GPU lease is acquired and no `pod_health` transition happens
Implements: REQ-BENCH-028

**SCN-BENCH-011-02: Known adapter with wrong version returns adapter_version_mismatch**
- GIVEN: `operation_adapter_abi = "matmul_fp16_v1"` is registered at version `0.3.1`, but request carries `operation_adapter_version = "0.2.0"`
- WHEN: Phase 1 resolves the adapter
- THEN: `status = infra_error`, reason `adapter_version_mismatch`
Implements: REQ-BENCH-028

**SCN-BENCH-011-03: Adapter build_launch_args returns non-empty tuple for declared ABI**
- GIVEN: the `matmul_fp16_v1` adapter is bound for candidate C_X on `shape_large = [4096, 4096, 4096]`
- WHEN: Phase 3 invokes `adapter.build_launch_args(buffers, shape_large)`
- THEN: the returned tuple has exactly the ABI-declared operand count (at minimum: `(d_A, d_B, d_C, M, N, K)` for matmul) and is passed unchanged to `cuLaunchKernel`
- AND: `tuple()` as the launch-args return is a spec violation
Implements: REQ-BENCH-028

**SCN-BENCH-011-04: adapter_iteration_semantics source of truth is the adapter, not the request**
- GIVEN: request's `CandidateArtifactRef.adapter_iteration_semantics = OVERWRITE_PURE`, but the resolved adapter declares `REQUIRES_OUTPUT_RESET`
- WHEN: Phase 3 reads iteration semantics
- THEN: the harness uses `REQUIRES_OUTPUT_RESET`, invoking the adapter's `reset_between_iterations` hook between timed iterations
- AND: `ShapeMeasurementArtifact.adapter_iteration_semantics = REQUIRES_OUTPUT_RESET` (adapter-declared value)
- AND: if the two disagree, a warning is recorded in the batch envelope; the adapter wins
Implements: REQ-BENCH-028, REQ-BENCH-032

**SCN-BENCH-012-01: NCU wraps the dedicated profile child, not the worker**
- GIVEN: Phase 6 selected (C_B, shape_large) for profiling
- WHEN: the worker launches the NCU subprocess
- THEN: the NCU argv contains `--` followed by the path to `python -m kerlever.benchmarker.profile_child` (or equivalent dedicated entry point) with arguments carrying `cubin_path`, `entrypoint`, `block_dim`, `grid_dim`, `dynamic_smem_bytes`, adapter abi/version, `shape_dims`, `function_attribute_policy`, `iterations_per_sample`, and the NVTX range name
- AND: NCU argv does NOT contain `[sys.executable, "-c", "pass"]` or any other placeholder target
- AND: the profile child process emits the NVTX range exactly once per run
Implements: REQ-BENCH-030, INV-BENCH-008

**SCN-BENCH-012-02: Profile child loads same cubin and reuses adapter for buffer setup**
- GIVEN: a profile child invocation
- WHEN: the profile child starts
- THEN: it creates one CUDA context on the leased GPU, loads the cubin via `cuModuleLoadDataEx`, applies the `function_attribute_policy` via `cuFuncSetAttribute`, resolves the adapter from the registry by `(abi, version)`, calls `adapter.allocate` + `adapter.seed_inputs` with the same `(run_id, batch_id, shape_id)` seed, and runs `iterations_per_sample` launches of which exactly one is inside the NVTX range
- AND: the profile child exits cleanly after the last launch
Implements: REQ-BENCH-030

**SCN-BENCH-012-03: Profile child exit ≠ 0 maps to profile_unavailable**
- GIVEN: the profile child raises `CUDA_ERROR_LAUNCH_FAILED` during the profiled launch
- WHEN: NCU captures the failure
- THEN: `ProfileStatus = profile_unavailable`, reason `profiler_replay_refused`, `fault_class` remains the candidate's Phase 4 attribution; fast-benchmark results are preserved
Implements: REQ-BENCH-030, INV-BENCH-008

**SCN-BENCH-012-04: Multi-profile-shape — NCU invoked once per (candidate, profile_shape) pair**
- GIVEN: `top_k_profiled = [C_A, C_B]` and `profile_shape_cases = [shape_medium, shape_large]`
- WHEN: Phase 6 runs
- THEN: NCU is invoked exactly 4 times (one per cartesian product element)
- AND: no early `break` after the first profile shape
- AND: each invocation carries its own NVTX range name tagged with the `(candidate_hash, shape_id)` pair
Implements: REQ-BENCH-017, REQ-BENCH-030

**SCN-BENCH-013-01: Relative cubin_uri rejected as infra_error**
- GIVEN: `CandidateArtifactRef.cubin_uri = "candidates/C_X.cubin"` (relative, no leading slash)
- WHEN: Phase 1 validates
- THEN: `status = infra_error`, reason `cubin_uri_not_readable` or `cubin_uri_unsupported_scheme`
- AND: no GPU lease is acquired
Implements: REQ-BENCH-031

**SCN-BENCH-013-02: URI scheme cubin_uri rejected**
- GIVEN: `cubin_uri = "s3://kerlever-artifacts/round7/C_X.cubin"`
- WHEN: Phase 1 validates
- THEN: `status = infra_error`, reason `cubin_uri_unsupported_scheme`
Implements: REQ-BENCH-031

**SCN-BENCH-013-03: Absolute path on shared mount accepted**
- GIVEN: `cubin_uri = "/shared/kerlever/artifacts/run_xyz/r7_b1/C_X.cubin"` and the file is readable
- WHEN: Phase 1 validates
- THEN: validation passes; Phase 2 proceeds to lease
Implements: REQ-BENCH-031

**SCN-BENCH-013-04: Absolute path not readable is infra_error**
- GIVEN: `cubin_uri = "/shared/kerlever/artifacts/missing.cubin"` where the file does not exist
- WHEN: Phase 1 validates
- THEN: `status = infra_error`, reason `cubin_uri_not_readable`
Implements: REQ-BENCH-031

**SCN-BENCH-013-05: cubin_bytes_b64 in request body is ignored with warning**
- GIVEN: the HTTP body contains an unexpected `cubin_bytes_b64` field alongside `cubin_uri`
- WHEN: Pydantic parses the request
- THEN: `cubin_bytes_b64` is silently dropped (model has `extra='ignore'`)
- AND: a warning `unsupported_cubin_bytes_b64_field_ignored` is recorded in the batch envelope
- AND: validation proceeds using `cubin_uri` alone
Implements: REQ-BENCH-031

**SCN-BENCH-014-01: Artifact store is write-only from service perspective post-response**
- GIVEN: a completed batch whose response contains artifact ids for ShapeMeasurementArtifact, NCU reports, raw metrics JSON
- WHEN: a subsequent `/benchmark` call arrives for a different batch
- THEN: the service does not read the previous batch's artifacts (they are referenced by id only for downstream consumers)
- AND: retention/rotation is operator-owned; the spec makes no guarantee beyond "not GC'd while the current response write is in flight"
Implements: REQ-BENCH-021, SC-BENCH-014

**SCN-BENCH-015-01: warmup_count reflects executed warmup loop count**
- GIVEN: Phase 3 executed 5 warmup launches for (C_A, shape_small)
- WHEN: the `ShapeMeasurementArtifact` for that pair is constructed
- THEN: `warmup_count = 5`
- AND: a hardcoded value (e.g., `warmup_count = 0` or `warmup_count = FAST_BENCH_WARMUP_MIN_RUNS` when no warmup was actually run) is a spec violation
Implements: REQ-BENCH-032

**SCN-BENCH-015-02: telemetry_before and telemetry_after are real pynvml samples**
- GIVEN: a shape measurement for (C_B, shape_large)
- WHEN: the artifact is constructed
- THEN: `telemetry_before.taken_at_ms` ≤ the timestamp of the first `cudaEventRecord(start)` for this shape, and `telemetry_after.taken_at_ms` ≥ the timestamp of the last `cudaEventSynchronize(stop)` for this shape
- AND: both snapshots contain real `sm_clock_mhz`, `gpu_temp_c`, `power_w`, `throttle_reasons` values from `pynvml`, not zero defaults
Implements: REQ-BENCH-032

**SCN-BENCH-015-03: anchor_pre_us and anchor_post_us populated only in interleaved batch**
- GIVEN: a single-candidate batch (`interleave_enabled = False`)
- WHEN: the artifact is constructed
- THEN: `anchor_pre_us = None`, `anchor_post_us = None`, `anchor_drift_pct = None`
- AND: setting any of them to `0.0` is a spec violation (None communicates "not measured", 0.0 would imply "measured and zero drift")
Implements: REQ-BENCH-032

**SCN-BENCH-015-04: function_attribute_policy field carries observed readback values**
- GIVEN: request `function_attribute_policy.max_dynamic_shared_memory_size = 65536` on a device whose max is `98304`
- WHEN: Phase 3 applies via `cuFuncSetAttribute` and reads back
- THEN: `MeasurementEnvelope.function_attribute_policy_requested.max_dynamic_shared_memory_size = 65536` and `function_attribute_policy_observed.max_dynamic_shared_memory_size = 65536`
- AND: `ShapeMeasurementArtifact.function_attribute_policy = function_attribute_policy_observed`
Implements: REQ-BENCH-029, REQ-BENCH-032

**SCN-BENCH-015-05: measurement_quality in artifact mirrors Phase 4 classifier output**
- GIVEN: a shape measurement whose CV was classified `valid_with_warning` with reason `p95_p50_ratio_warn`
- WHEN: the artifact is constructed
- THEN: `ShapeMeasurementArtifact.measurement_quality.status = valid_with_warning`, `.reason = "p95_p50_ratio_warn"`, and `.warnings` list matches the classifier output
- AND: hardcoding `measurement_quality.status = valid` regardless of classifier output is a spec violation
Implements: REQ-BENCH-032

**SCN-BENCH-015-06: telemetry_before and telemetry_after carry real pynvml readouts, not default snapshots**
- GIVEN: a batch with two candidates C_A and C_B and one shape `shape_large`
- AND: Phase 2 preflight `pynvml` sampling on `gpu_uuid = "GPU-abc"` records `sm_clock_mhz = 1980`, `mem_clock_mhz = 2619`, `gpu_temp_c = 44.5`, `power_w = 182.3`, `throttle_reasons = []`, `ecc_sbe_total = 0`, `ecc_dbe_total = 0`, `xid_events_since_last = 0`
- AND: Phase 4 postflight `pynvml` sampling, taken immediately after the last measured launch for `shape_large`, records `sm_clock_mhz = 1905`, `mem_clock_mhz = 2619`, `gpu_temp_c = 61.2`, `power_w = 279.6`, `throttle_reasons = ["SW_THERMAL_SLOWDOWN"]`
- WHEN: the `ShapeMeasurementArtifact` entries for `(C_A, shape_large)` and `(C_B, shape_large)` are constructed
- THEN: both artifacts report `telemetry_before.sm_clock_mhz == 1980`, `telemetry_before.gpu_temp_c == 44.5`, `telemetry_before.taken_at_ms > 0` (real monotonic-ms timestamp from Phase 2)
- AND: both artifacts report `telemetry_after.sm_clock_mhz == 1905`, `telemetry_after.gpu_temp_c == 61.2`, `telemetry_after.throttle_reasons == ["SW_THERMAL_SLOWDOWN"]`, `telemetry_after.taken_at_ms >= telemetry_before.taken_at_ms`
- AND: constructing either artifact with `telemetry_before = DeviceTelemetrySnapshot()` (the no-arg default, `taken_at_ms=0`, every field `None` or empty) is a spec violation per INV-BENCH-014 even if `pynvml` is available and the rest of the artifact is correct
- AND: constructing either artifact with `telemetry_after = DeviceTelemetrySnapshot()` is the same spec violation
- AND: the ONLY permitted default-constructed snapshot is when `pynvml` initialization failed at startup AND `MeasurementEnvelope.warnings` explicitly records `pynvml_unavailable` — absent that warning, default snapshots are a violation regardless of anything else
Implements: REQ-BENCH-032, INV-BENCH-014

**SCN-BENCH-016-01: Function attribute policy apply failure rejects candidate in Phase 3**
- GIVEN: request `function_attribute_policy.max_dynamic_shared_memory_size = 131072` on a device whose maximum is `98304`
- WHEN: Phase 3 calls `cuFuncSetAttribute` and receives a driver error
- THEN: candidate is rejected with `fault_class = infra_fault`, `measurement_quality.status = infra_fault`, reason `function_attribute_policy_apply_failed`
- AND: the driver error code is recorded in the envelope
- AND: remaining candidates in the batch proceed normally
Implements: REQ-BENCH-029

**SCN-BENCH-016-02: Function attribute policy applied before first measured launch**
- GIVEN: a candidate C_H requires 80 KiB dynamic shared memory on Hopper (default per-function limit is 48 KiB)
- WHEN: Phase 3 runs
- THEN: `cuFuncSetAttribute(func, MAX_DYNAMIC_SHARED_SIZE_BYTES, 81920)` is called before any warmup or measured launch for C_H
- AND: `function_attribute_policy_observed.max_dynamic_shared_memory_size = 81920` is read back successfully
- AND: the first measured launch succeeds (kernel can access 80 KiB of dynamic smem)
Implements: REQ-BENCH-029

**SCN-BENCH-016-03: V1 does not apply cache_config / carveout / cluster_dims even if caller wanted them**
- GIVEN: a hypothetical caller that wants the candidate function to run with `cache_config = prefer_shared`, `preferred_shared_memory_carveout_pct = 100`, and `cluster_dims = [2, 2, 1]`
- AND: the V1 `BenchmarkBatchRequest` / `CandidateArtifactRef` schema has no field that carries any of these three values end-to-end (only `launch_spec.dynamic_smem_bytes` flows)
- WHEN: Phase 3 applies the function attribute policy for that candidate
- THEN: the harness calls ONLY `cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, launch_spec.dynamic_smem_bytes)` and reads it back
- AND: the harness does NOT call `cuFuncSetCacheConfig`, does NOT call `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, ...)`, does NOT call `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH/HEIGHT/DEPTH, ...)`, and does NOT call `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, ...)` — there is no request field to source the values from
- AND: `MeasurementEnvelope.function_attribute_policy_requested.cache_config = None`, `.preferred_shared_memory_carveout_pct = None`, `.cluster_dims = None`, `.non_portable_cluster_size_allowed = None`
- AND: `MeasurementEnvelope.function_attribute_policy_observed` mirrors the same four `None` values (nothing was applied, so nothing distinct was observed)
- AND: a V1 harness that hardcodes `cache_config = prefer_shared` (or any other non-None default) into the requested/observed envelope is a spec violation — the envelope would falsely claim a policy was requested that no caller supplied
- AND: broadening V1 to apply these four fields requires extending the shared `CandidateArtifactRef` schema (V2 change tracked as a Non-Goal in §9)
Implements: REQ-BENCH-029

**SCN-BENCH-017-01: p95 below gate uses sentinel, not p50 shim**
- GIVEN: `run_count = MIN_P95_SAMPLES - 1` for shape S on candidate C
- WHEN: the compact `ShapeBenchResult` is assembled
- THEN: `ShapeBenchResult.latency_p95_us = -1.0` (sentinel)
- AND: `ShapeBenchResult.latency_p50_us` is the real p50
- AND: `ShapeMeasurementArtifact.p95_us = None`
- AND: `ShapeBenchResult.latency_p95_us = ShapeBenchResult.latency_p50_us` is a spec violation
Implements: REQ-BENCH-033

**SCN-BENCH-017-02: weighted_p95_us objective excludes sentinel shapes**
- GIVEN: objective `primary_metric = weighted_p95_us`, aggregation `weighted_mean`; candidate X has shape S1 with p95 available and S2 with `latency_p95_us = -1.0`
- WHEN: Phase 5 computes `ObjectiveScore.value`
- THEN: only S1 contributes to the weighted mean; S2's weight is skipped
- AND: if every contributing shape has the sentinel, `measurement_quality.status = not_comparable`, reason `p95_samples_below_minimum_for_p95_metric`, no `ObjectiveScore.value`
Implements: REQ-BENCH-033

**SCN-BENCH-017-03: Staging directories do not accumulate across batches**
- GIVEN: a Benchmarker service pod freshly started with `<cfg.artifact.root>/staging/` empty and `KEEP_STAGED_ARTIFACTS` unset (or `0`)
- AND: 10 `BenchmarkBatchRequest` calls arrive sequentially with `batch_id ∈ {"b1", "b2", ..., "b10"}`; each completes before the next arrives
- AND: the batches terminate in a mix of `status` values (for example: 7 × `success`, 1 × `unstable`, 1 × `infra_error` caused by unreadable `cubin_uri`, 1 × `timeout`)
- WHEN: the tenth response has been returned to the caller
- THEN: `os.listdir(<cfg.artifact.root>/staging/)` returns an empty list (modulo at most one in-flight `<batch_id>/` if an 11th batch has begun before observation)
- AND: for every terminated `batch_id ∈ {"b1", ..., "b10"}`, the path `<cfg.artifact.root>/staging/<batch_id>/` does NOT exist (removed by the Supervisor's `finally` cleanup)
- AND: the durable artifact subtrees (`<cfg.artifact.root>/samples/`, `/ncu/`, `/raw_metrics/`, `/artifacts/`) MAY contain any number of artifact files — those are retention-policy territory (REQ-BENCH-021) and MUST NOT be removed by staging cleanup
- AND: if the pod is restarted with `KEEP_STAGED_ARTIFACTS=1` and the same sequence is replayed, then all 10 `<batch_id>/` directories remain on disk after the tenth response, AND a one-time warning `keep_staged_artifacts_enabled_staging_cleanup_skipped` is recorded in the pod's service-level log
- AND: a Supervisor implementation that leaves any `<cfg.artifact.root>/staging/<batch_id>/` directory on disk after its batch completed (with `KEEP_STAGED_ARTIFACTS` unset or `0`) is a spec violation even if all other behavior is correct
Implements: REQ-BENCH-035, INV-BENCH-016

**SCN-BENCH-018-01: incumbent_cv_pct computed from anchor_samples, not passed as None**
- GIVEN: an interleaved batch with anchor samples `[4.05, 4.11, 4.09, 4.12, 4.10, 4.08, 4.11, 4.09]` for shape_small
- WHEN: `decide_incumbent_comparison` is called for any candidate on shape_small
- THEN: `anchor_cv_pct = stats.cv_pct([4.05, ..., 4.09])` (a real positive float, not None and not 0.0)
- AND: `noise_margin_pct = max(NOISE_FLOOR_PCT, candidate_cv_pct, anchor_cv_pct, anchor_drift_pct)` uses that value
- AND: passing `incumbent_cv_pct = None` or `0.0` when ≥ 2 anchor samples exist is a spec violation
Implements: REQ-BENCH-034, INV-BENCH-006

**SCN-BENCH-018-02: envelope-mismatch comparison uses incumbent's measured envelope**
- GIVEN: candidate C_X with envelope `{metric_mode: DEVICE_KERNEL_US, sm_arch: sm_90, function_attribute_policy: {smem: 81920}}`; incumbent measured in-episode with envelope `{metric_mode: DEVICE_KERNEL_US, sm_arch: sm_90, function_attribute_policy: {smem: 49152}}`
- WHEN: `decide_incumbent_comparison` is called
- THEN: envelope hash mismatch is detected against the **incumbent's own measured envelope** (observed values, not requested values), `IncumbentComparison = not_comparable`, reason `envelope_mismatch`
- AND: synthesizing an envelope for the incumbent by copying the candidate's envelope and tweaking fields is forbidden
Implements: REQ-BENCH-015, INV-BENCH-011
---

## §4 Invariants

**INV-BENCH-001: Request contract is self-contained**
The Benchmarker's only orchestrator-visible contract is `BenchmarkBatchRequest → BenchmarkBatchResult` over HTTP. It does not implement `GPUPipelineProtocol` in V1 and does not import the shared `kerlever.protocols` module.
*Enforcement:* `kerlever/benchmarker/` has no import of `kerlever.protocols`; the FastAPI layer only references local Pydantic models in `kerlever/benchmarker/types.py` and shared data types in `kerlever/types.py`.

**INV-BENCH-002: Interleaving is only valid in one-CUDA-context-per-worker**
Repeated, randomized cross-candidate launches are permitted only when all candidates in the batch share one process, one CUDA context, one stream policy, and one operation adapter. Any fallback must run sequential candidate blocks with incumbent anchors before/after and mark the episode as non-interleaved.
*Enforcement:* The harness is launched inside a single disposable worker subprocess per batch. V1 refuses `artifact_execution_model != "common_harness_cubin"`. Interleaving is only scheduled when the adapter is uniform across candidates and the subprocess holds the one context. The realized interleave is stored alongside `interleave_seed`.

**INV-BENCH-003: Timed GPU access is serialized per leased device**
At most one timed benchmark job may run on a given `gpu_uuid` or MIG instance at any time. Parallelism across different same-class GPUs is allowed only with per-GPU anchors; different SKUs or MIG profiles are never directly ranked together.
*Enforcement:* `asyncio.Semaphore(1)` keyed by `gpu_uuid`/MIG id held across Phase 2..Phase 6 of each batch. Cross-device comparability requires matching `MeasurementEnvelope` fields; mismatch sets `IncumbentComparison = not_comparable`.

**INV-BENCH-004: Interleave permutations are deterministic**
For a given `(run_id, batch_id, shape_id, "kerlever_benchmark_order")`, the realized interleave block order is reproducible. Retries of the same batch use the same seed unless a new `batch_id` is issued.
*Enforcement:* `interleave_seed` is computed by a pure hash; the block-order generator is a deterministic function of the seed, candidate set, `ANCHOR_EVERY_N_SAMPLES`, and `MAX_INTERLEAVE_BLOCK_LEN`. The block order is recorded in `ShapeMeasurementArtifact.interleave_block_order`.

**INV-BENCH-005: Cache policy cannot be silently incompatible with interleaving**
Interleaving with `warm_same_buffers` would cause candidate-to-candidate cache coupling that invalidates ranking. The service must not silently use `warm_same_buffers` when interleaving is active.
*Enforcement:* Phase 3 promotes `warm_same_buffers` → `warm_rotating_buffers` when interleaving applies and records `requested_cache_policy`, `effective_cache_policy`, `cache_policy_reason` in the envelope. Any other cache policy (`cold_flush_buffer`, `reset_persisting_l2`) is accepted as-is without promotion.

**INV-BENCH-006: Noise-margin-filtered comparison is the sole improvement/regression decision path**
Improvement and regression are decided exclusively through `noise_margin_pct` against the in-episode incumbent anchor. Raw latency deltas or absolute microsecond comparisons cannot set `IncumbentComparison`.
*Enforcement:* `phase5_score.decide_incumbent_comparison(...)` is the single function that produces `IncumbentComparison`. Unit-level code paths route only through it. `regressed_vs_incumbent` is derived, never set independently.

**INV-BENCH-007: Objective score comes only from fast benchmark**
`ObjectiveScore.value` is computed from Phase 4 `ShapeBenchResult` fields only. Nsight Compute and Nsight Systems durations are diagnostic and never contribute to scoring.
*Enforcement:* Profile artifacts cannot be written into `BenchmarkBundle.objective_score`; the scoring function reads only `ShapeBenchResult`s. Code paths that construct `BenchmarkBundle` import from `phase5_score`, not from `phase6_profile`.

**INV-BENCH-008: Deep-profile targets the marked launch, not warmup or anchors**
`ncu` must profile one measured launch per (candidate, shape); warmup, anchor, or bulk repeated-launch profiling is forbidden because it would conflate anchor/candidate traffic.
*Enforcement:* The common harness wraps exactly the measured-launch region in NVTX range `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile`; `ncu` is invoked with `--nvtx --nvtx-include` filter for that range and `--launch-count 1` or equivalent explicit skip/count to isolate one launch.

**INV-BENCH-009: Profiler metrics are never fabricated**
An unavailable NCU metric is `null`. Missing architecture support, missing section, failed collection, or stale profiler version must not produce a fabricated or inferred numeric value.
*Enforcement:* The NCU parser returns `None` for missing metrics. Normalization to compact `ProfileMetrics` copies only present raw metrics; absent inputs yield `null` outputs. Provenance fields are mandatory and validated at Pydantic layer.

**INV-BENCH-010: Ambiguous faults produce no optimization signal**
A candidate whose fault is `ambiguous_fault` must never contribute to `IncumbentComparison`, `top_k_profiled`, or cross-candidate scoring. The search loop must not treat ambiguous faults as either negative or positive evidence.
*Enforcement:* Scoring, ranking, and profile-selection code paths filter by `fault_class in {null}` for positive contribution, and do not consult `ambiguous_fault` candidates for any decision beyond pod-health bookkeeping.

**INV-BENCH-011: `not_comparable` is a terminal outcome for that candidate in that batch**
A candidate whose envelope disagrees with the incumbent anchor's envelope (metric mode, arch, adapter, function-attribute policy) returns `IncumbentComparison = not_comparable`. It is not silently downgraded to `regressed` or `improved`.
*Enforcement:* `decide_incumbent_comparison(...)` takes both envelopes; envelope-hash mismatch short-circuits to `not_comparable` before any score comparison.

**INV-BENCH-012: Disposable worker subprocess contains GPU faults**
A poisoned CUDA context in the benchmark subprocess must not leak into the next batch.
*Enforcement:* The supervising service launches the harness via `asyncio.create_subprocess_exec` once per batch and terminates the subprocess on completion or on any ambiguous GPU fault; no context is reused across batches.

**INV-BENCH-013: Adapter iteration semantics dispatch is honored for every timed iteration loop**
Every timed iteration loop — calibration (Phase 3), measurement (Phase 4), and profile child (Phase 6) — must consult the matched adapter's `adapter_iteration_semantics` and invoke the adapter's `reset_between_iterations` hook (or equivalent buffer-rotation path) between timed iterations whenever the semantics is not `OVERWRITE_PURE`. For `REQUIRES_OUTPUT_RESET`, the output buffer is reset between iterations; for `REQUIRES_FULL_INPUT_RESET`, the full input + output state is restored via buffer rotation (a dedicated buffer pool of at least `iterations_per_sample` equivalent-state buffers); for `NOT_REPEATABLE`, `iterations_per_sample = 1` and a fresh full reset happens between samples (not inside a sample). Running a timed loop that issues N `cuLaunchKernel` calls back-to-back without invoking the dispatch when semantics ≠ `OVERWRITE_PURE` is forbidden.
*Enforcement:* `harness.run_sample(candidate, shape, *, reset_hook: Callable[..., None] | None, rotate_hook: Callable[..., AdapterBuffers] | None, semantics: AdapterIterationSemantics)` accepts the hooks and dispatches based on `semantics`. Coding wires both hooks from the resolved adapter. A `run_sample` invocation with `reset_hook = None` when `semantics != OVERWRITE_PURE` is a spec violation. The profile child honors the same dispatch.

**INV-BENCH-014: Rich artifact fields are derived, never hardcoded defaults**
`ShapeMeasurementArtifact` fields enumerated in SC-BENCH-015 / REQ-BENCH-032 MUST be populated from the real runtime sources listed in those IDs. Hardcoding a literal default that does not match the runtime source (e.g., `warmup_count=0` when warmup ran, `interleave_block_len=0` when a block was generated, `function_attribute_policy=FunctionAttributePolicy()` when a policy was applied) is a spec violation even if static gates pass.

**Explicit call-out for `telemetry_before` and `telemetry_after`.** These two fields are the most at risk of silent partial compliance — both have well-formed default-constructed `DeviceTelemetrySnapshot` values (`taken_at_ms=0`, every other field `None` or `0`) that pass static type checks and pass any "is populated" presence check but carry zero real runtime information. To close that gap:

- `ShapeMeasurementArtifact.telemetry_before` MUST be the `DeviceTelemetrySnapshot` produced by the **Phase 2 preflight `pynvml` sample** (the same sample already fed into `HygieneReport` and `MeasurementEnvelope` initial telemetry) OR a fresh `pynvml` sample taken immediately before the first measured launch for this shape in Phase 4 — whichever is closer to the first measured launch, as long as it is a real `pynvml` readout with a real `taken_at_ms` monotonic timestamp, non-`None` `sm_clock_mhz`/`mem_clock_mhz` on supported drivers, and a real `throttle_reasons` list.
- `ShapeMeasurementArtifact.telemetry_after` MUST be a fresh `pynvml` sample taken in Phase 4 **immediately after the last measured launch** for this shape (`cudaEventSynchronize(stop)` has returned, ambient pynvml readout follows). Same realness criteria apply.
- `DeviceTelemetrySnapshot(taken_at_ms=0, sm_clock_mhz=None, mem_clock_mhz=None, gpu_temp_c=None, power_w=None, throttle_reasons=[], ecc_sbe_total=None, ecc_dbe_total=None, xid_events_since_last=0)` — the all-default/all-None snapshot — is explicitly forbidden when `pynvml` is available and the service has passed its startup readiness probe. An empty snapshot here is a spec violation even if every other `ShapeMeasurementArtifact` field is correctly derived, because it erases the thermal/clock/throttle provenance that downstream analysis and forensic triage depend on.
- The ONLY permitted all-default/all-None snapshot is the explicit degraded path where `pynvml` initialization failed at startup (unusual; the service should not have been reporting `/healthz = ready`). In that case `telemetry_before` / `telemetry_after` are both default-constructed and `MeasurementEnvelope` records `warnings += ["pynvml_unavailable"]`; absent this explicit warning, default snapshots are a violation.

*Enforcement:* The `ShapeMeasurementArtifact` constructor is centralized in the harness code path that owns Phase 4 output. The harness receives explicit arguments for every enumerated field and wires them from `calibration_output`, `interleave_output`, `anchor_samples`, `envelope`, `adapter.semantics`, `config.thresholds`, `quality_classifier.output`, `telemetry.before`, `telemetry.after`. The `telemetry.before` and `telemetry.after` arguments are constructed by a dedicated `telemetry.sample_device(gpu_uuid, pynvml_handle)` helper that returns a real `DeviceTelemetrySnapshot` or raises — the helper MUST NOT return a default-constructed snapshot silently. Static review checks that no `ShapeMeasurementArtifact(...)` call-site uses `DeviceTelemetrySnapshot()` (no-arg constructor) or an equivalent literal for an enumerated field.

**INV-BENCH-015: Incumbent comparison uses the incumbent's own measured envelope**
The envelope-mismatch check in `decide_incumbent_comparison` compares the candidate's `MeasurementEnvelope` against the **incumbent's own measured envelope from this batch** (the envelope produced by loading the incumbent cubin in Phase 3, applying function attributes, observing actual clocks, and running anchor samples). Constructing a synthetic envelope for the incumbent by copying the candidate's envelope, or by reading the `incumbent_ref.objective_score` metadata without remeasurement, is forbidden because it trivially passes the envelope check by construction and defeats INV-BENCH-011.
*Enforcement:* `decide_incumbent_comparison(candidate_envelope: MeasurementEnvelope, candidate_score: ObjectiveScore, incumbent_envelope: MeasurementEnvelope, incumbent_anchor_score: ObjectiveScore, anchor_samples: list[float], noise_cfg: NoiseConfig)` has an explicit `incumbent_envelope` parameter that must be the `MeasurementEnvelope` produced by measuring the incumbent in this batch. Callers pass `incumbent_anchor.envelope`, never a derived or synthesized envelope.

**INV-BENCH-016: No staged artifact outlives its batch unless explicitly retained**
After `Supervisor.run_batch(request)` returns its `BenchmarkBatchResult` to the HTTP layer, no file that the Supervisor wrote under `<cfg.artifact.root>/staging/<batch_id>/` may remain on disk. This holds for every terminal batch `status` and for every Python control-flow path (normal return, exception, asyncio cancellation). A long-running pod that serves N batches must, at steady state with all batches terminal, have `<cfg.artifact.root>/staging/` containing zero `<batch_id>/` directories (modulo any one transient in-flight batch). Durable artifacts referenced by `CandidateResult.*_refs` are outside staging and unaffected — their retention is operator-owned per REQ-BENCH-021.
*Enforcement:* `Supervisor.run_batch` wraps its entire Phase 1..Phase 7 work in a `try / finally` (or `async with contextlib.AsyncExitStack()` with a registered cleanup callback) such that the cleanup callback `shutil.rmtree(<cfg.artifact.root>/staging/<batch_id>/, ignore_errors=True)` (or equivalent) runs after the response is prepared, regardless of whether the `try` block returned normally or raised. The callback is guarded by `if os.environ.get("KEEP_STAGED_ARTIFACTS", "") == "1": return` — when the escape hatch is set, cleanup is skipped. The cleanup target path is constructed from `cfg.artifact.root` and `batch_id` (both validated in Phase 1) and MUST NOT escape `<cfg.artifact.root>/staging/` (no traversal via `..`, no symlink following). A Supervisor implementation that writes staging files but has no `finally`-equivalent cleanup path is a spec violation even if no test has yet caught the leak.

---

## §5 Interfaces

### HTTP Surface

**`POST /benchmark`**
- Request body: `BenchmarkBatchRequest` (Pydantic).
- Response body: `BenchmarkBatchResult` (Pydantic).
- Semantics: Phase 1..Phase 7 for exactly one batch. The call blocks for the duration of the batch, but batches are `async` and cooperate on the per-GPU semaphore. Failure is always reported as a structured response; HTTP 200 is the normal channel even for `status = infra_error`. HTTP 4xx/5xx is used only for malformed requests and fatal service errors (e.g., `/healthz` would also return 503 in the same state).

**`GET /healthz`**
- Response 200 with JSON `{status: "ready", toolchain: {...}, gpus: [...], pod_health: "healthy"|"suspect"}` when all sub-checks pass and pod health is not `quarantined`.
- Response 503 with JSON `{status: "not_ready", missing: [...], reason: "..."}` otherwise.
- Startup hook runs the same check; non-zero exit on failure.

**`GET /info`**
- Response 200 with JSON containing service version, build hash, `DEFAULT_METRIC_MODE`, `ARTIFACT_EXECUTION_MODEL`, toolchain versions (driver, CUDA runtime, `cuda-python`, `pynvml`, `ncu`), visible GPUs (with `gpu_uuid`, `pci_bus_id`, `sm_arch`, `mig_profile`), supported operation adapters, configured thresholds (noise floor, guard, CV warn/fail, anchor drift warn/fail).

### Shared Types (reused verbatim from `kerlever/types.py`)

- `ShapeCase`, `PerformanceObjective`, `ProblemSpec` — inputs.
- `StaticAnalysis`, `CorrectnessResult` — passed through from Compiler Service artifact metadata.
- `ShapeBenchResult`, `ObjectiveScore` — per-shape and objective scoring output.
- `ProfileMetrics`, `BottleneckAssessment`, `ProfileBundle` — profile output (note: `BottleneckAssessment` is emitted empty by the Benchmarker; bottleneck tagging is the Profile Interpreter's job).
- `BenchmarkBundle` — per-candidate scoring bundle, `regressed_vs_incumbent` derived from `IncumbentComparison`.

### Benchmarker-Local Types (`kerlever/benchmarker/types.py`)

```text
ArtifactExecutionModel (StrEnum)
  COMMON_HARNESS_CUBIN = "common_harness_cubin"   # V1 only

MetricMode (StrEnum)
  DEVICE_KERNEL_US = "device_kernel_us"           # V1 default
  HOST_LAUNCH_US = "host_launch_us"
  OPERATOR_END_TO_END_US = "operator_end_to_end_us"
  CUDA_GRAPH_REPLAY_US = "cuda_graph_replay_us"

AdapterIterationSemantics (StrEnum)
  OVERWRITE_PURE = "overwrite_pure"
  REQUIRES_OUTPUT_RESET = "requires_output_reset"
  REQUIRES_FULL_INPUT_RESET = "requires_full_input_reset"
  NOT_REPEATABLE = "not_repeatable"

CachePolicy (StrEnum)
  WARM_SAME_BUFFERS = "warm_same_buffers"
  WARM_ROTATING_BUFFERS = "warm_rotating_buffers"
  COLD_FLUSH_BUFFER = "cold_flush_buffer"
  RESET_PERSISTING_L2 = "reset_persisting_l2"

ClockPolicyMode (StrEnum)
  OBSERVED_ONLY = "observed_only"
  LOCKED = "locked"
  LOCK_REQUESTED_UNAVAILABLE = "lock_requested_unavailable"

IncumbentComparison (StrEnum)
  IMPROVED = "improved"
  STATISTICAL_TIE = "statistical_tie"
  REGRESSED = "regressed"
  UNSTABLE = "unstable"
  NOT_COMPARABLE = "not_comparable"

MeasurementQualityStatus (StrEnum)
  VALID = "valid"
  VALID_WITH_WARNING = "valid_with_warning"
  UNSTABLE = "unstable"
  RUNTIME_FAULT = "runtime_fault"
  INFRA_FAULT = "infra_fault"

ProfileStatus (StrEnum)
  PRESENT = "present"
  PROFILE_UNAVAILABLE = "profile_unavailable"

ProfileUnavailableReason (StrEnum)
  PROFILER_PERMISSION_DENIED = "profiler_permission_denied"
  ADAPTER_NOT_REPEATABLE = "adapter_not_repeatable"
  ARCH_MISMATCH = "arch_mismatch"
  PROFILER_TIMEOUT = "profiler_timeout"
  PROFILER_BINARY_MISSING = "profiler_binary_missing"
  PROFILER_REPLAY_REFUSED = "profiler_replay_refused"
  MIG_PROFILE_MISMATCH = "mig_profile_mismatch"

FaultClass (StrEnum)
  CANDIDATE_FAULT = "candidate_fault"
  INFRA_FAULT = "infra_fault"
  AMBIGUOUS_FAULT = "ambiguous_fault"

PodHealth (StrEnum)
  HEALTHY = "healthy"
  SUSPECT = "suspect"
  QUARANTINED = "quarantined"

BatchStatus (StrEnum)
  SUCCESS = "success"
  PARTIAL = "partial"
  UNSTABLE = "unstable"
  TIMEOUT = "timeout"
  INFRA_ERROR = "infra_error"
```

```text
CandidateArtifactRef
  candidate_hash: str
  artifact_id: str                # opaque pointer into pod-local or object-store-addressable cubin blob
  cubin_uri: str                  # pod-local path or URL; service must be able to fetch/resolve
  launch_spec:
    entrypoint: str
    block_dim: [int, int, int]
    grid_dim: [int, int, int] | null   # if null, adapter derives from shape
    dynamic_smem_bytes: int
    abi_name: str
    abi_version: str
    metadata_mode: str | null
  launch_spec_hash: str
  source_hash: str
  toolchain_hash: str
  static_analysis: StaticAnalysis | null
  correctness: CorrectnessResult | null
  adapter_iteration_semantics: AdapterIterationSemantics

FunctionAttributePolicy
  max_dynamic_shared_memory_size: int | null
  preferred_shared_memory_carveout_pct: int | null
  cache_config: "prefer_none" | "prefer_shared" | "prefer_l1" | "prefer_equal" | null
  cluster_dims: [int, int, int] | null
  non_portable_cluster_size_allowed: bool | null

ClockPolicy
  mode: ClockPolicyMode
  requested_sm_clock_mhz: int | null
  requested_mem_clock_mhz: int | null

MeasurementEnvelope
  # identity
  run_id: str
  round_id: str | null
  batch_id: str
  request_id: str
  candidate_hash: str
  # artifact identity
  artifact_id: str
  source_hash: str
  launch_spec_hash: str
  toolchain_hash: str
  module_artifact_hash: str
  artifact_execution_model: ArtifactExecutionModel
  # workload identity
  problem_spec_hash: str
  objective_hash: str
  shape_ids: list[str]
  operation_adapter_abi: str
  operation_adapter_version: str
  # device identity
  target_gpu: str
  gpu_uuid: str
  pci_bus_id: str
  mig_profile: str | null
  sm_arch: str
  driver_version: str
  cuda_runtime_version: str
  # timing policy
  metric_mode: MetricMode
  function_attribute_policy_requested: FunctionAttributePolicy
  function_attribute_policy_observed: FunctionAttributePolicy
  warmup_policy:
    min_runs: int
    cache_state: "untouched" | "touched"
  repeat_policy:
    repetitions: int
    iterations_per_sample: int
    min_timed_batch_ms: float
    max_timed_batch_ms: float
  cache_policy:
    requested: CachePolicy
    effective: CachePolicy
    reason: str | null
  clock_policy: ClockPolicy
  interleave_seed: int | null

DeviceTelemetrySnapshot
  taken_at_ms: int
  sm_clock_mhz: int | null
  mem_clock_mhz: int | null
  gpu_temp_c: float | null
  power_w: float | null
  throttle_reasons: list[str]
  ecc_sbe_total: int | null
  ecc_dbe_total: int | null
  xid_events_since_last: int

HygieneReport
  gpu_uuid: str
  sm_arch: str
  mig_profile: str | null
  compute_mode: str
  foreign_processes: list[str]
  clocks_event_reasons: list[str]
  gpu_temp_c: float | null
  power_w: float | null
  ecc_ok: bool
  xid_ok: bool
  probe_ok: bool | null                 # null when probe not required
  profiler_counter_permission: bool
  reason_on_fail: str | null

ShapeMeasurementArtifact
  shape_id: str
  samples_us: list[float]               # may be empty for non_repeatable with 1 launch/sample
  warmup_count: int
  iterations_per_sample: int
  min_samples_required: int
  p50_us: float
  p95_us: float | null
  mean_us: float
  stdev_us: float
  cv_pct: float
  min_us: float
  max_us: float
  cache_policy: CachePolicy
  requested_cache_policy: CachePolicy
  effective_cache_policy: CachePolicy
  interleave_block_len: int | null
  anchor_every_n_samples: int | null
  anchor_pre_us: float | null
  anchor_post_us: float | null
  anchor_drift_pct: float | null
  interleave_block_order: list[str]     # sequence of candidate_hash / "anchor" tokens
  artifact_execution_model: ArtifactExecutionModel
  adapter_iteration_semantics: AdapterIterationSemantics
  metric_mode: MetricMode
  max_timed_batch_ms: float
  function_attribute_policy: FunctionAttributePolicy
  useful_bytes: int | null
  actual_bytes: int | null              # null unless back-filled from profile
  algorithmic_flops: int | null
  effective_bandwidth_gbps: float | null
  achieved_flops: float | null
  arithmetic_intensity_flop_per_byte: float | null
  measurement_quality:
    status: MeasurementQualityStatus
    reason: str | null
    warnings: list[str]
  telemetry_before: DeviceTelemetrySnapshot
  telemetry_after: DeviceTelemetrySnapshot

RawProfileMetric
  metric_name: str
  value: float | int | null
  unit: str | null
  architecture: str
  profiler_name: Literal["ncu", "nsys"]
  profiler_version: str
  collection_section: str | null

NormalizedProfileMetricProvenance
  source_metrics: list[str]
  architecture: str
  profiler_version: str
  comparable_across_arch: Literal[False]

ProfileArtifactRef
  artifact_id: str
  kind: Literal["ncu_report", "nsys_report", "raw_metrics_json", "samples_json"]
  uri: str
  size_bytes: int
  created_at_ms: int

IncumbentAnchor
  incumbent_artifact_id: str
  shape_results: list[ShapeBenchResult]
  objective_score: ObjectiveScore
  anchor_drift_pct_per_shape: dict[str, float]
  measurement_quality_per_shape: dict[str, MeasurementQualityStatus]

CandidateResult
  candidate_hash: str
  envelope: MeasurementEnvelope
  benchmark: BenchmarkBundle | null                       # null only if no scoreable measurement produced
  incumbent_comparison: IncumbentComparison
  measurement_quality: MeasurementQualityStatus
  measurement_quality_reason: str | null
  shape_measurement_artifact_refs: dict[str, str]         # shape_id -> artifact_id of ShapeMeasurementArtifact
  profile_status: ProfileStatus
  profile_unavailable_reason: ProfileUnavailableReason | null
  profile_bundle: ProfileBundle | null                    # shared type; null when profile unavailable
  raw_profile_metrics_ref: str | null                     # artifact id
  profile_artifact_refs: list[ProfileArtifactRef]
  fault_class: FaultClass | null
  failure_reason: str | null

BenchmarkBatchRequest
  request_id: str
  run_id: str
  round_id: str | null
  batch_id: str
  problem_spec: ProblemSpec
  objective_shape_cases: list[ShapeCase]
  profile_shape_cases: list[ShapeCase]
  baseline_ref:
    artifact_id: str
    objective_score: ObjectiveScore
  incumbent_ref:
    artifact_id: str
    objective_score: ObjectiveScore
  candidate_module_artifact_refs: list[CandidateArtifactRef]
  operation_adapter_abi: str
  operation_adapter_version: str
  artifact_execution_model: ArtifactExecutionModel       # V1 must be COMMON_HARNESS_CUBIN
  metric_mode: MetricMode                                # typically DEVICE_KERNEL_US
  function_attribute_policy: FunctionAttributePolicy
  cache_policy: CachePolicy                              # may be auto-promoted
  clock_policy: ClockPolicy
  top_k_profile: int                                     # TOP_K_PROFILE override (bounded)
  top_m_profile_shift_potential: int                     # TOP_M_PROFILE_SHIFT_POTENTIAL override (bounded)
  anchor_every_n_samples: int | null
  max_interleave_block_len: int | null
  bench_rerun_limit: int | null

BenchmarkBatchResult
  status: BatchStatus
  run_envelope:
    run_id: str
    round_id: str | null
    batch_id: str
    request_id: str
    pod_id: str
    pod_health: PodHealth
    ambiguous_failure_count: int
    toolchain:
      driver_version: str
      cuda_runtime_version: str
      cuda_python_version: str
      pynvml_version: str
      ncu_version: str | null
    visible_gpu:
      gpu_uuid: str
      pci_bus_id: str
      sm_arch: str
      mig_profile: str | null
  measurement_context:
    artifact_execution_model: ArtifactExecutionModel
    metric_mode: MetricMode
    cache_policy_requested: CachePolicy
    cache_policy_effective: CachePolicy
    clock_policy: ClockPolicy
    interleave_enabled: bool
    anchor_every_n_samples: int | null
    max_interleave_block_len: int | null
    noise_floor_pct: float
    guard_pct: float
  hygiene: HygieneReport
  incumbent_anchor: IncumbentAnchor
  candidate_results: list[CandidateResult]
  top_k_profiled: list[str]                              # candidate hashes
  failure_reason: str | null                             # populated when status in {infra_error, timeout}
```

### Operation Adapter Protocol (kerlever/benchmarker/adapter.py)

Adapters are plugins registered at service startup. Each adapter is a concrete class implementing the `OperationAdapter` Protocol below. V1 ships two built-in adapters; custom adapters are registered by operators via configuration. No adapter implementation runs LLM calls, does kernel compilation, or rewrites ABI.

```python
from typing import ClassVar, Protocol
from dataclasses import dataclass

@dataclass
class AdapterBuffers:
    """Opaque, adapter-defined buffer record for one (shape, candidate) pair.

    The adapter is the only code that interprets these; the harness passes
    them through.
    """
    device_ptrs: tuple[int, ...]       # device pointer integers (from cuMemAlloc)
    host_side: dict[str, object]       # adapter-private host state
    shape_dims: tuple[int, ...]
    dtype: str

class OperationAdapter(Protocol):
    abi_name: ClassVar[str]            # e.g. "matmul_fp16_v1"
    abi_version: ClassVar[str]         # semantic version, e.g. "0.3.1"

    def allocate(
        self, shape: "ShapeCase", dtype: str, device: "DeviceLease"
    ) -> AdapterBuffers: ...

    def seed_inputs(
        self, buffers: AdapterBuffers, shape: "ShapeCase", seed: int
    ) -> None: ...

    def build_launch_args(
        self, buffers: AdapterBuffers, shape: "ShapeCase"
    ) -> tuple[object, ...]: ...
    # tuple of: (d_ptr_0, d_ptr_1, ..., d_ptr_M-1, scalar_0, scalar_1, ...)
    # as required by the kernel ABI declared by abi_name. Must never return ()
    # when the ABI declares any operand.

    def grid_dim(
        self, shape: "ShapeCase", block_dim: tuple[int, int, int]
    ) -> tuple[int, int, int]: ...

    def useful_bytes(self, shape: "ShapeCase") -> int: ...

    def algorithmic_flops(self, shape: "ShapeCase") -> int: ...

    def reset_between_iterations(
        self, buffers: AdapterBuffers, semantics: "AdapterIterationSemantics"
    ) -> None: ...
    # called between timed iterations when semantics != OVERWRITE_PURE.
    # For REQUIRES_OUTPUT_RESET: zeroes or re-initializes the output buffer.
    # For REQUIRES_FULL_INPUT_RESET: re-seeds all buffers; harness additionally
    # uses rotate_buffers if a buffer pool is configured.

    def rotate_buffers(
        self, buffers_pool: list[AdapterBuffers]
    ) -> AdapterBuffers: ...
    # for WARM_ROTATING_BUFFERS cache policy: returns the next buffer in the
    # rotation. Deterministic order.

    def free(self, buffers: AdapterBuffers) -> None: ...
```

**Adapter registry.**

```python
class AdapterRegistry:
    def register(self, adapter: OperationAdapter) -> None: ...
    def resolve(self, abi_name: str, abi_version: str) -> OperationAdapter: ...
    # raises AdapterUnregistered if (abi_name, abi_version) not found
    # raises AdapterVersionMismatch if abi_name is registered at a different version
```

**V1 built-in adapters.**

- `elementwise_add_fp32_v1` — ABI `(const float* A, const float* B, float* C, int N)`; `allocate` makes three `N*sizeof(float)` buffers; `build_launch_args` returns `(d_A, d_B, d_C, N)`; `grid_dim` returns `((N + block_dim.x - 1)//block_dim.x, 1, 1)`; `useful_bytes = 3 * N * 4`; `algorithmic_flops = N`; `adapter_iteration_semantics = OVERWRITE_PURE`.
- `matmul_fp16_v1` — ABI `(const half* A, const half* B, half* C, int M, int N, int K)`; `allocate` makes three buffers for M×K, K×N, M×N in fp16; `build_launch_args` returns `(d_A, d_B, d_C, M, N, K)`; `grid_dim` derived from block-dim and `(M, N)` tiling; `useful_bytes = (M*K + K*N + M*N) * 2`; `algorithmic_flops = 2 * M * N * K`; `adapter_iteration_semantics = OVERWRITE_PURE`.

Both V1 adapters trust the cubin for correctness (Compiler Service already closed that gate); the adapter's own correctness responsibility is confined to buffer layout and deterministic seeding.

### Profile Child Subprocess Surface

The profile child is a dedicated Python entry point invoked by NCU. Its responsibility is minimal: replay exactly one measured launch inside one NVTX range.

```
python -m kerlever.benchmarker.profile_child \
    --cubin-path /shared/.../C_B.cubin \
    --entrypoint matmul_B \
    --block-dim 16,16,1 \
    --grid-dim 256,256,1 \
    --dynamic-smem-bytes 16384 \
    --adapter-abi matmul_fp16_v1 \
    --adapter-version 0.3.1 \
    --shape-dims 4096,4096,4096 \
    --shape-dtype fp16 \
    --function-attr-max-smem 81920 \
    --function-attr-carveout-pct 100 \
    --function-attr-cache-config prefer_shared \
    --iterations-per-sample 16 \
    --warmup-count 5 \
    --nvtx-range "kerlever/run_xyz/r7_b1/C_B/shape_large/profile" \
    --seed 1234567890
```

Behavior:
1. Create one CUDA context on GPU 0 (the leased device is passed via `CUDA_VISIBLE_DEVICES`).
2. Load the cubin via `cuModuleLoadDataEx`; resolve entrypoint.
3. Apply function-attribute policy via `cuFuncSetAttribute` / `cuFuncSetCacheConfig` and read back; exit non-zero if apply fails.
4. Resolve adapter from the registry by `(abi, version)`; allocate + seed buffers.
5. Run `warmup_count` launches outside any NVTX range.
6. Push NVTX range `nvtx_range`; run `iterations_per_sample` launches; pop NVTX range.
7. Free buffers; destroy context; exit 0.

The profile child never reports measurements back to the parent; all data lives inside the `.ncu-rep` produced by NCU around it. The parent worker reads the `.ncu-rep` via `ncu --import --print-metrics-json` after the NCU subprocess exits.

### Module Layout (public API surface only)

```
kerlever/benchmarker/
  __init__.py                # BenchmarkerService, BenchmarkBatchRequest, BenchmarkBatchResult
  types.py                   # All Benchmarker-local Pydantic/StrEnum types above
  adapter.py                 # OperationAdapter Protocol + AdapterRegistry + V1 built-ins
  profile_child.py           # python -m kerlever.benchmarker.profile_child entry point
  api/
    app.py                   # FastAPI factory: POST /benchmark, GET /healthz, GET /info
  # remainder (phases/, harness/, etc.) belongs to design.md, not spec.md
```

---

## §6 Behavioral Specification

### 6.1 Phase 1 — Request Normalization and Measurement Envelope

**Input.** A `BenchmarkBatchRequest` as defined in §5. The service is not configured to trust the request blindly — every field is validated.

**Validation steps (in order, fail-fast except where noted):**

1. `request.artifact_execution_model == COMMON_HARNESS_CUBIN`, else `status = infra_error`, reason `unsupported_artifact_execution_model`.
2. `len(candidate_module_artifact_refs) >= 1`, else `status = infra_error`, reason `empty_batch`.
3. For each `CandidateArtifactRef`:
   - `correctness is not None and correctness.passed is True`; else the candidate is ignored with `fault_class = infra_fault` (the Compiler Service should never have forwarded a failing candidate, but we defend).
   - `launch_spec` fields `entrypoint`, `block_dim`, `dynamic_smem_bytes`, `abi_name`, `abi_version` are all present.
   - `launch_spec_hash`, `source_hash`, `toolchain_hash` are present and non-empty.
   - `adapter_iteration_semantics` is a known enum value.
   - `cubin_uri` resolves to a readable file (or the service can fetch it from the configured artifact gateway).
4. `operation_adapter_abi` matches exactly one registered adapter; `operation_adapter_version` matches the registered adapter version.
5. `problem_spec.target_gpu` / `sm_arch` match at least one visible GPU (architecture only — lease happens in Phase 2).
6. `metric_mode` is supported by the adapter; otherwise every affected candidate is pre-marked `not_comparable`.
7. `top_k_profile` and `top_m_profile_shift_potential` are bounded by hard limits (2 × configured defaults) to prevent runaway profile cost.

**Derivations:**

- `problem_spec_hash = sha256(canonical_json(problem_spec))`.
- `objective_hash = sha256(canonical_json(problem_spec.objective))`.
- `interleave_seed_per_shape[shape_id] = hash((run_id, batch_id, shape_id, "kerlever_benchmark_order"))` (stable across retries of the same batch id).
- `function_attribute_policy_requested` defaults from the request; `function_attribute_policy_observed` is filled in Phase 3 after module load.
- `cache_policy.effective` is provisionally set from `cache_policy.requested`; interleaving promotion happens in Phase 3.

**Output of Phase 1:** a fully populated `MeasurementEnvelope` per candidate and a batch-level `measurement_context`. No GPU work has happened yet.

**Edge cases.**
- `run_id` missing → `status = infra_error`, reason `missing_run_id`. The service cannot form a stable interleave seed without it.
- Two candidates with the same `candidate_hash` → deduplicate (first wins), warn in the batch envelope.
- A candidate whose `toolchain_hash` differs from another candidate's `toolchain_hash` but shares the adapter → allowed; they still run in the same interleaved batch if the other envelope fields match. If they do not match, that candidate gets its own non-interleaved slot and `IncumbentComparison` will resolve to `not_comparable` for any envelope-mismatched shape.

### 6.2 Phase 2 — Device Lease and Measurement Hygiene

**Lease selection.** The service iterates visible GPUs (resolved at startup from `cuda-python` + `pynvml`) and picks the first whose `sm_arch`, MIG profile, driver version compatibility class, and CUDA runtime compatibility class match the request and whose per-GPU semaphore is free. If multiple GPUs match, the scheduler prefers the least-recently-used device.

**Preflight telemetry (via `pynvml`).** For the leased device, sample:
- `nvmlDeviceGetClockInfo(SM)` and `(MEM)`.
- `nvmlDeviceGetCurrentClocksEventReasons` — mapped to a list of reason strings.
- `nvmlDeviceGetTemperature(GPU)`.
- `nvmlDeviceGetPowerUsage`.
- `nvmlDeviceGetComputeRunningProcesses` — any PID not belonging to the Benchmarker worker is a "foreign compute process".
- `nvmlDeviceGetMigDeviceHandleByIndex` (if MIG) and compare profile.
- ECC aggregate counts (`nvmlDeviceGetTotalEccErrors`), Xid events since last probe.
- `compute-sanitizer`-parented permissions for performance counters (test by running a trivial `ncu` probe once at service startup; cache result).

**Decision table (Phase 2):**

| Condition | Outcome | `IncumbentComparison` applicable to affected candidates |
|---|---|---|
| GPU arch ≠ target_gpu.sm_arch | `status = infra_error`, reason `arch_mismatch` | n/a |
| MIG profile mismatch | `status = infra_error`, reason `mig_profile_mismatch` | n/a |
| Foreign compute process present | retry up to `BENCH_RERUN_LIMIT`; if still present, `unstable` | `unstable` |
| `HW_SLOWDOWN`/`SW_THERMAL_SLOWDOWN`/`SW_POWER_CAP` in event reasons and objective does not accept throttled state | `unstable` | `unstable` |
| Temp > `THERMAL_STEADY_STATE_LIMIT` | cooldown retry (bounded); still above → `unstable` | `unstable` |
| ECC DBE > 0 or Xid event seen | `infra_error`, pod quarantined | n/a |
| Probe required and probe failed | `infra_error`, pod quarantined | n/a |
| Profiler counter permission missing | fast benchmark continues; profile phase returns `profile_unavailable` w/ `profiler_permission_denied` | unaffected by permission alone |
| Clock-lock requested but unavailable | `clock_policy.mode = lock_requested_unavailable`; proceed | proceed |
| All clear | proceed to Phase 3 | decided downstream |

**Clock policy resolution.**
- Default `CLOCK_POLICY = observe_and_enforce_hygiene`.
- If `CLOCK_LOCK_POLICY = enabled_when_privileged` and the container has `CAP_SYS_ADMIN` + permission to call `nvidia-smi --lock-gpu-clocks`, attempt the lock; record `clock_policy.mode = locked`.
- Otherwise `clock_policy.mode = observed_only`.
- Throttle reasons discovered later in Phase 4 invalidate samples for that shape regardless of initial clock policy.

**Output of Phase 2:** a `HygieneReport`, an initial telemetry snapshot, a held per-GPU semaphore, and a `ClockPolicy` resolution. If any hard gate failed, Phase 3..7 do not run for that batch; the service still returns a `BenchmarkBatchResult` with `status` set accordingly.

### 6.3 Phase 3 — Fast Benchmark Plan Calibration

Phase 3 runs inside the disposable worker subprocess, which:
1. Creates exactly one CUDA context on the leased device.
2. Loads every candidate cubin via `cuModuleLoadDataEx`.
3. Resolves entrypoints via `cuModuleGetFunction`.
4. Applies function-attribute policy via `cuFuncSetAttribute` / `cuFuncSetCacheConfig`. Observed values are read back and stored in `function_attribute_policy_observed`.
5. Allocates shape-specific input/output buffers via the operation adapter (`adapter.allocate_inputs(shape, dtype)`), outside any timed region.
6. Initializes inputs deterministically from `adapter.deterministic_seed(batch_id, shape_id)`.

**Warmup.** For each (candidate, shape), run `FAST_BENCH_WARMUP_MIN_RUNS` untimed launches. For adapters with reset semantics other than `overwrite_pure`, the adapter's reset hook runs between warmup launches.

**Iteration calibration.** For each (candidate, shape):
1. Start with `iterations_per_sample = 1`.
2. Record `start_event`, issue `iterations_per_sample` launches on the benchmark stream, record `stop_event`, synchronize, read `cudaEventElapsedTime → elapsed_ms`.
3. If `elapsed_ms < FAST_BENCH_MIN_TIMED_BATCH_MS` and `iterations_per_sample * 2 <= FAST_BENCH_MAX_ITERATIONS_PER_SAMPLE`: double `iterations_per_sample` and retry.
4. If `elapsed_ms > FAST_BENCH_MAX_TIMED_BATCH_MS`: halve `iterations_per_sample` (down to 1). If still too long at 1, mark `measurement_quality.status = valid_with_warning`, reason `calibration_upper_bound_unmet`.
5. If `iterations_per_sample` cap is reached without meeting the lower bound, mark `valid_with_warning`, reason `calibration_lower_bound_unmet`.
6. `per_launch_us = elapsed_ms * 1000 / iterations_per_sample`.

**Adapter iteration semantics dispatch.**

```
if adapter_iteration_semantics == OVERWRITE_PURE:
    allow repeated-launch timing; no reset between launches
elif adapter_iteration_semantics == REQUIRES_OUTPUT_RESET:
    insert adapter.reset_output() outside the timed region between samples; use buffer rotation within a sample
elif adapter_iteration_semantics == REQUIRES_FULL_INPUT_RESET:
    use buffer rotation with an equivalent-state buffer per measured launch; reset work outside the timed region
elif adapter_iteration_semantics == NOT_REPEATABLE:
    iterations_per_sample = 1; collect REPETITIONS samples independently, each preceded by adapter.reset_full()
    mark launch_overhead_sensitive = True in artifact
```

**Cache policy resolution.**

```
if batch has multiple candidates and artifact_execution_model == COMMON_HARNESS_CUBIN:
    interleave_enabled = True
else:
    interleave_enabled = False

if interleave_enabled and request.cache_policy == WARM_SAME_BUFFERS:
    effective_cache_policy = WARM_ROTATING_BUFFERS
    reason = "interleaved_batch_requires_rotation"
else:
    effective_cache_policy = request.cache_policy
    reason = None
```

**Metric mode dispatch.**
- `DEVICE_KERNEL_US`: CUDA events around kernel launches on the stream; this is the default.
- `HOST_LAUNCH_US`: host-side `time.perf_counter_ns` deltas around `cuLaunchKernel` without waiting for completion; requires adapter support.
- `OPERATOR_END_TO_END_US`: adapter-defined region inclusive of transfers/host work.
- `CUDA_GRAPH_REPLAY_US`: requires adapter to provide graph capture/update/replay hooks.

If the adapter does not support the requested metric mode, every affected (candidate, shape) gets `measurement_quality.status = not_comparable`, reason `unsupported_metric_mode`, and scoring is skipped for those shapes.

**Output of Phase 3:** a per-(candidate, shape) plan containing `warmup_count`, `iterations_per_sample`, `repetitions`, `effective_cache_policy`, `metric_mode`, reset policy. No timed samples yet.

### 6.4 Phase 4 — Fast Benchmark Execution and Quality Checks

**Interleave block generation (pseudocode).**

```
def generate_block_order(candidates, anchor_every_n, max_block_len, seed):
    rng = PCG64(seed)
    order = []
    total_repetitions = FAST_BENCH_REPETITIONS
    emitted = 0
    while emitted < total_repetitions * len(candidates):
        order.append("anchor")
        block = []
        for i in range(min(max_block_len, anchor_every_n)):
            # pick next candidate with remaining samples, weighted to keep counts even
            c = pick_underfull_candidate(rng, candidates, order)
            if c is None:
                break
            block.append(c)
            emitted += 1
        order.extend(block)
    order.append("anchor")   # post-block anchor
    return order
```

The realized `order` is stored in `ShapeMeasurementArtifact.interleave_block_order` per shape.

**Per-sample loop (per shape).**

```
pre_telemetry = sample_telemetry(gpu_uuid)
anchor_pre_samples = []
for token in order:
    if token == "anchor":
        anchor_pre_samples.append(run_sample(incumbent, shape))
    else:
        candidate_samples[token].append(run_sample(token, shape))
post_telemetry = sample_telemetry(gpu_uuid)
anchor_pre_score = objective_score([anchor_pre_samples[0..N_pre]])
anchor_post_score = objective_score([anchor_pre_samples[-N_post..]])
anchor_drift_pct = abs(anchor_post_score - anchor_pre_score) / anchor_pre_score
```

`run_sample(candidate, shape)`:

```
ensure adapter reset outside timed region per adapter_iteration_semantics
select buffer rotation index if applicable
cudaEventRecord(start)
for _ in range(iterations_per_sample):
    cuLaunchKernel(entrypoint, grid, block, smem, stream, args)
cudaEventRecord(stop)
cudaEventSynchronize(stop)
elapsed_ms = cudaEventElapsedTime(start, stop)
check for cudaGetLastError; surface as runtime_fault on error
return elapsed_ms * 1000 / iterations_per_sample
```

**Per-shape statistics.**
- `p50_us = median(samples)`.
- `p95_us = percentile(samples, 95)` if `len(samples) >= MIN_P95_SAMPLES`, else `null`.
- `mean_us`, `stdev_us`, `cv_pct = stdev/mean*100`, `min_us`, `max_us`.
- Derived (via adapter work model): `effective_bandwidth_gbps = useful_bytes / (p50_us * 1e3)`, `achieved_flops = algorithmic_flops / (p50_us * 1e-6)`, `arithmetic_intensity_flop_per_byte = algorithmic_flops / useful_bytes` when `useful_bytes > 0`.

**p95 sentinel in the compact shared type.** The shared `ShapeBenchResult.latency_p95_us` is `float` (non-optional) per `kerlever/types.py` and cannot be modified in V1 (Non-Goal). When `run_count < MIN_P95_SAMPLES`:
- `ShapeMeasurementArtifact.p95_us = None` (rich artifact, authoritative).
- `ShapeBenchResult.latency_p95_us = -1.0` (compact, sentinel value; latencies are strictly non-negative so `-1.0` is unambiguous).
- `ShapeBenchResult.latency_p95_us = latency_p50_us` is **forbidden** (REQ-BENCH-033). The p50→p95 shim silently breaks `weighted_p95_us` / `worst_case_p95_us` objective aggregation because it promotes p50 to a fake p95.
- Phase 5 scoring excludes shapes with `latency_p95_us == -1.0` from p95-based objective aggregation.

**Measurement quality decision table.**

| Condition | `measurement_quality.status` | Reason |
|---|---|---|
| Valid samples + all hygiene + CV ≤ `MEASUREMENT_CV_WARN_PCT` + anchor_drift ≤ `ANCHOR_DRIFT_WARN_PCT` + p95/p50 ≤ `P95_P50_RATIO_WARN` | `valid` | — |
| CV in (`MEASUREMENT_CV_WARN_PCT`, `MEASUREMENT_CV_FAIL_PCT`] or anchor_drift in (`WARN`, `FAIL`) or p95/p50 > `P95_P50_RATIO_WARN` | `valid_with_warning` | specific tag |
| CV > `MEASUREMENT_CV_FAIL_PCT` or anchor_drift > `ANCHOR_DRIFT_FAIL_PCT` | `unstable` | `cv_above_fail_threshold` / `anchor_drift_exceeded` |
| Runtime fault (CUDA error in launch/sync) | `runtime_fault` | CUDA error name |
| Hygiene fail mid-episode (e.g., Xid during sampling) | `infra_fault` | `xid_mid_episode` |
| Foreign process appeared mid-episode | `unstable` | `foreign_compute_process` |

**Retry logic.** `unstable` shapes are retried up to `BENCH_RERUN_LIMIT` times. `runtime_fault` and `infra_fault` are never retried in the same batch (ambiguity triggers pod-health transition instead).

**Fault attribution.**

| Event | Pod health | Attribution |
|---|---|---|
| `CUDA_ERROR_ILLEGAL_ADDRESS` during candidate launch | healthy | `candidate_fault` + `runtime_fault` |
| `CUDA_ERROR_ILLEGAL_ADDRESS` during candidate launch | suspect | `ambiguous_fault`; probe next batch |
| Kernel timeout (>` KERNEL_TIMEOUT_MS`) | healthy | `candidate_fault` (one-shot) |
| Kernel timeout | suspect | `ambiguous_fault` |
| Xid during sampling | any | `infra_fault`; quarantine pod |
| Subprocess killed by OS (OOM / SIGKILL) | any | `ambiguous_fault` |
| `ncu` launcher missing at runtime | any | `infra_fault` (profile-only; fast benchmark unaffected) |

**Output of Phase 4:** per (candidate, shape) `ShapeMeasurementArtifact`, per-candidate runtime-fault map, `IncumbentAnchor` pre/post scores per shape, updated device telemetry snapshots.

### 6.5 Phase 5 — Objective Scoring, Noise Margin, Incumbent Comparison

**Objective score aggregation.**

```
let metric_src = primary_metric_to_src(problem_spec.objective.primary_metric)
# primary_metric -> shape-level source:
#   weighted_p50_us   -> shape.latency_p50_us
#   weighted_p95_us   -> shape.latency_p95_us (requires run_count >= MIN_P95_SAMPLES per shape)
#   worst_case_p50_us -> shape.latency_p50_us

per_shape_value[shape_id] = metric_src(shape_result)

if problem_spec.objective.aggregation == "weighted_mean":
    value = sum(w_i * per_shape_value[i]) / sum(w_i) across shapes
elif problem_spec.objective.aggregation == "max":
    value = max(per_shape_value[i])

ObjectiveScore:
  metric_name = problem_spec.objective.primary_metric
  value = value
  relative_to_baseline  = value / baseline_ref.objective_score.value
  relative_to_incumbent = value / incumbent_anchor.objective_score.value
```

**Noise margin.**

`anchor_cv_pct` is required input to the noise margin and MUST be computed from the actual incumbent anchor samples collected in this batch via `stats.cv_pct(anchor_samples)` whenever `len(anchor_samples) >= 2` (REQ-BENCH-034). Passing `None` or `0.0` for `anchor_cv_pct` when anchor samples exist is forbidden. The `incumbent_envelope` argument into `decide_incumbent_comparison` MUST be the envelope produced by measuring the incumbent in this batch (INV-BENCH-015); envelopes synthesized from the candidate's envelope are forbidden.

```
candidate_cv_pct = aggregated CV across shapes contributing to objective
                 (weighted the same way as value; max aggregation uses the max-shape's CV)
anchor_cv_pct   = same aggregation over incumbent anchor samples
anchor_drift_pct = aggregated |anchor_post - anchor_pre| / anchor_pre across shapes

noise_margin_pct = max(
    NOISE_FLOOR_PCT,
    candidate_cv_pct / 100.0,
    anchor_cv_pct / 100.0,
    anchor_drift_pct,
)
```

**`IncumbentComparison` decision table (applies per candidate).**

| Condition | Result |
|---|---|
| Any shape contributing to objective has `measurement_quality.status == infra_fault` | `NOT_COMPARABLE` (reason: `envelope_invalid`) |
| Any shape contributing to objective has `measurement_quality.status == runtime_fault` | `REGRESSED` only if the fault is `candidate_fault`; `NOT_COMPARABLE` if `ambiguous_fault` |
| Any shape contributing to objective has `measurement_quality.status == unstable` | `UNSTABLE` |
| Envelope hash mismatch vs. incumbent anchor envelope (metric_mode, sm_arch, adapter, function_attribute_policy) | `NOT_COMPARABLE` (reason: `envelope_mismatch`) |
| `value < incumbent_anchor.value * (1 - noise_margin_pct)` | `IMPROVED` |
| `value > incumbent_anchor.value * (1 + guard_pct + noise_margin_pct)` where `guard_pct = problem_spec.objective.regression_guard_pct` | `REGRESSED` |
| Otherwise | `STATISTICAL_TIE` |

`BenchmarkBundle.regressed_vs_incumbent = (IncumbentComparison == REGRESSED)`.

**Example noise-margin arithmetic.**

```
incumbent_anchor.value = 100.0 µs
candidate.value        = 98.5 µs
candidate_cv_pct       = 1.2%
anchor_cv_pct          = 1.0%
anchor_drift_pct       = 4%
NOISE_FLOOR_PCT        = 1%

noise_margin_pct = max(0.01, 0.012, 0.010, 0.04) = 0.04
improved_threshold  = 100 * (1 - 0.04) = 96.0
regressed_threshold = 100 * (1 + 0.02 + 0.04) = 106.0  (guard_pct = 0.02)

candidate.value (98.5) > 96.0 and < 106.0 → STATISTICAL_TIE
```

### 6.6 Phase 6 — Deep Profile Planning and Collection

**Profile-target selection (pseudocode).**

```
scoreable = [c for c in candidates if c.incumbent_comparison in {IMPROVED, STATISTICAL_TIE}]
# REGRESSED, UNSTABLE, NOT_COMPARABLE are excluded from top_k_profiled in V1

# top-K by objective score (ascending, since lower is better)
top_k = sorted(scoreable, key=lambda c: c.objective_score.value)[:request.top_k_profile]

# top-M by bottleneck-shift potential
shift_scores = compute_shift_potential(candidates, incumbent)
top_m = sorted(scoreable - top_k, key=lambda c: -shift_scores[c.hash])[:request.top_m_profile_shift_potential]

top_k_profiled = dedup(top_k + top_m, key=candidate_hash)
# if comparison against incumbent profile is useful and incumbent is known, include incumbent
if policy.include_incumbent_profile: top_k_profiled.append(incumbent)
```

`compute_shift_potential` — pre-profile signals only:
- intent direction / sub-mode novelty vs. recent rounds;
- static-analysis delta vs. incumbent: `|registers_delta|`, `|smem_delta|`, `spills_delta`, `|occupancy_delta|`;
- fast-benchmark throughput shape: `effective_bandwidth_gbps`, `achieved_flops`, `arithmetic_intensity_flop_per_byte` movement vs. incumbent;
- useful-bytes ratio vs. actual-bytes if available (actual_bytes is filled only after profile, so this input is null in V1);
- Cross-Candidate Analyzer hints from earlier rounds, if carried in the request (optional, not required in V1).

Shift potential never reads NCU counters from this batch (they don't exist yet — INV-BENCH-007/INV-BENCH-008 guard).

**Profile shapes.** `problem_spec.shape_cases[i].profile == True` collected from `profile_shape_cases`; if empty, fallback to `best objective shape` (the shape with the smallest `p50_us` that contributed to objective).

**Multi-profile-shape iteration.** Phase 6 iterates over the **full cartesian product** `top_k_profiled × profile_shape_cases`. For each `(candidate, profile_shape)` tuple, NCU is invoked exactly once against a dedicated profile child subprocess (see §6.12 and REQ-BENCH-030). No early `break` after the first profile shape is allowed; a Phase 6 loop that stops after the first profile shape for a candidate is a spec violation. If a specific `(candidate, profile_shape)` invocation fails, its `ProfileStatus = profile_unavailable` is recorded independently of the other pairs; other pairs are still attempted.

**NVTX harnessing.** The common harness wraps the measured launch in NVTX range `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile` via the driver API NVTX library. Warmup and anchor launches are not wrapped in that range (they may carry a different NVTX name for debugging only).

**Replay coverage per adapter iteration semantics.**

| Semantics | V1 replay status |
|---|---|
| `OVERWRITE_PURE` | supported |
| `REQUIRES_OUTPUT_RESET` | supported if adapter provides a safe reset or buffer-rotation path outside the profiled launch |
| `REQUIRES_FULL_INPUT_RESET` | `profile_unavailable` unless adapter provides a safe restore path; V1 default is unavailable |
| `NOT_REPEATABLE` | `profile_unavailable` reason `adapter_not_repeatable` |

**NCU invocation.**

```
ncu
  --target-processes all
  --nvtx --nvtx-include "kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile"
  --launch-count 1
  --set <NCU_PROFILE_SET>                     # focused/basic first
  --replay-mode <application|kernel>          # chosen by adapter iteration semantics
  --export <artifact_root>/ncu/<artifact_id>.ncu-rep
  -- <benchmark_harness_executable> ...
```

Timeout: `PROFILE_TIMEOUT`. On timeout, kill subprocess, return `profile_unavailable` reason `profiler_timeout`.

**Metric extraction.** Parse the `.ncu-rep` via `ncu --import --print-metrics-json` (or equivalent). Each metric becomes a `RawProfileMetric`. Compact `ProfileMetrics` normalization:

| Compact field | Source metrics (examples, subject to arch/profiler version) |
|---|---|
| `achieved_occupancy_pct` | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| `dram_throughput_pct_of_peak` | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| `sm_throughput_pct_of_peak` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| `l2_hit_rate_pct` | `lts__t_sectors_hit_rate.pct` |
| `warp_stall_memory_dependency_pct` | `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` |
| `warp_stall_exec_dependency_pct` | `smsp__average_warps_issue_stalled_execution_dependency_per_issue_active.ratio` |
| `tensor_core_utilization_pct` | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` |
| `arithmetic_intensity_flop_per_byte` | derived from flop counters + dram bytes |

Every normalized entry carries `NormalizedProfileMetricProvenance` with `source_metrics`, `architecture`, `profiler_version`, `comparable_across_arch = False`. `BottleneckAssessment` fields in the emitted `ProfileBundle` are empty — the Profile Interpreter owns bottleneck tagging; the Benchmarker only supplies the evidence.

**Nsight Systems policy.** Not invoked in V1 unless `NSYS_PROFILE_POLICY` trigger fires. When fired, the harness runs `nsys profile -o <artifact_root>/nsys/<artifact_id>.nsys-rep ...` separately from the NCU run (they must not share the same launch). Output is stored by reference.

**Output of Phase 6:** per-candidate `ProfileStatus`, `ProfileBundle` (shared type, with empty `BottleneckAssessment`), `raw_profile_metrics_ref`, list of `ProfileArtifactRef`.

### 6.7 Phase 7 — Output Assembly

**Batch `status` resolution.**

```
if hygiene failed hard: status = infra_error
elif any subprocess crashed in a way that lost measurements: status = partial
elif any shape for any candidate was classified unstable after BENCH_RERUN_LIMIT: status = unstable
elif total elapsed > BATCH_TIMEOUT: status = timeout
else: status = success
```

`partial` is valid: some candidates may have durably-recorded measurements while others were lost when the subprocess crashed. Their `candidate_results[]` entries are returned for the surviving candidates; missing candidates appear with `benchmark = null`, `fault_class = ambiguous_fault`.

**Assembly order:**
1. `run_envelope` (includes `pod_health`, `ambiguous_failure_count`, toolchain, visible GPU identity).
2. `measurement_context` (batch-wide).
3. `hygiene` report.
4. `incumbent_anchor` (`ShapeBenchResult` list, `ObjectiveScore`, per-shape drift and quality).
5. `candidate_results[]` — one entry per requested candidate (including those short-circuited). For each:
   - `envelope` (the full `MeasurementEnvelope` for this candidate);
   - `benchmark` (`BenchmarkBundle` shared type) — `shape_results` (compact `ShapeBenchResult`), `objective_score`, `regressed_vs_incumbent` (derived from `IncumbentComparison`);
   - `incumbent_comparison`;
   - `measurement_quality` (worst-case over shapes contributing to objective);
   - `shape_measurement_artifact_refs` mapping shape_id → artifact id of rich `ShapeMeasurementArtifact`;
   - `profile_status`, `profile_unavailable_reason`, `profile_bundle`, `raw_profile_metrics_ref`, `profile_artifact_refs`;
   - `fault_class`, `failure_reason`.
6. `top_k_profiled` — list of candidate hashes.
7. `failure_reason` — set only when `status ∈ {infra_error, timeout}`.

**Artifact durability.** All `ShapeMeasurementArtifact`, NCU reports, raw-metric JSON, and samples JSON are written to the pod-local artifact store before the HTTP response is emitted. Artifact ids referenced by the response survive until `ARTIFACT_RETENTION` expires, and are never GC'd mid-read.

### 6.8 Configuration Parameters

Every parameter from `docs/benchmarker.md` §Configuration Parameters is supported. Non-exhaustive summary (defaults are indicative, not policy for this spec):

| Parameter | Default direction |
|---|---|
| `ARTIFACT_EXECUTION_MODEL` | `COMMON_HARNESS_CUBIN` (V1 only option) |
| `DEFAULT_METRIC_MODE` | `DEVICE_KERNEL_US` |
| `GPU_BENCH_CONCURRENCY` | 1 per GPU or MIG instance |
| `FAST_BENCH_WARMUP_MIN_RUNS` | small fixed minimum (e.g., 5) |
| `FAST_BENCH_MIN_TIMED_BATCH_MS` | enough to dwarf event overhead (e.g., 1.0) |
| `FAST_BENCH_MAX_TIMED_BATCH_MS` | bounded service default (e.g., 200.0) |
| `FAST_BENCH_REPETITIONS` | enough for p50 and p95 (e.g., 20–50) |
| `FAST_BENCH_MAX_ITERATIONS_PER_SAMPLE` | bounded cap (e.g., 1024) |
| `MIN_P95_SAMPLES` | explicit default (e.g., 20) |
| `MEASUREMENT_CV_WARN_PCT` | low single digits (e.g., 2.0) |
| `MEASUREMENT_CV_FAIL_PCT` | e.g., 5.0 |
| `P95_P50_RATIO_WARN` | e.g., 1.5 |
| `NOISE_FLOOR_PCT` | conservative nonzero floor (e.g., 0.01 = 1%) |
| `ANCHOR_DRIFT_WARN_PCT` | e.g., 0.02 |
| `ANCHOR_DRIFT_FAIL_PCT` | e.g., 0.05 |
| `ANCHOR_EVERY_N_SAMPLES` | service default (e.g., 4) |
| `MAX_INTERLEAVE_BLOCK_LEN` | service default (e.g., 6) |
| `BENCH_RERUN_LIMIT` | 1 or 2 |
| `ANCHOR_INCUMBENT_POLICY` | `same_episode` |
| `ANCHOR_BASELINE_POLICY` | drift/new-pod triggered |
| `CACHE_POLICY` | single `WARM_SAME_BUFFERS`; interleaved `WARM_ROTATING_BUFFERS` |
| `CACHE_FLUSH_BYTES` | adapter/pool specific |
| `CLOCK_POLICY` | observe + throttle enforcement |
| `CLOCK_LOCK_POLICY` | disabled unless privileged |
| `THERMAL_STEADY_STATE_LIMIT` | hardware/pool specific |
| `TOP_K_PROFILE` | 1–3 |
| `TOP_M_PROFILE_SHIFT_POTENTIAL` | 1–2 |
| `NCU_PROFILE_SET` | focused/basic first |
| `NCU_TARGET_SELECTION` | NVTX range + launch count |
| `NCU_REPLAY_ADAPTER_POLICY` | overwrite/reset only |
| `PROFILE_METRIC_PROVENANCE` | required |
| `NSYS_PROFILE_POLICY` | trigger-based |
| `PROFILE_TIMEOUT` | fixed service default (e.g., 300s) |
| `ARTIFACT_RETENTION` | keep raw early |
| `POD_HEALTH_PROBE` | known-good kernel (e.g., vec_add) |
| `KERNEL_TIMEOUT_MS` | service default (e.g., 10000) |
| `AMBIGUOUS_FAILURE_LIMIT` | small integer (e.g., 3) |
| `BATCH_TIMEOUT` | bounded (e.g., 30 min) |

### 6.9 Distributed Benchmarking Rules (normative)

Rules 1..5 from `docs/benchmarker.md` are binding:

1. **Fair ranking requires comparable devices.** Batches distributed across devices may rank only when SKU, sm_arch, MIG profile, driver class, runtime class, harness version, artifact execution model, loader version, clock policy, objective hash, and adapter version all match.
2. **Anchors normalize within a pod, not across architectures.** A100 ≠ H100 for ranking.
3. **Timed GPU work is serialized per device.** Per-GPU semaphore is mandatory.
4. **Remote failures need fault attribution.** `candidate_fault` / `infra_fault` / `ambiguous_fault` are distinct; ambiguous faults never contribute to signal.
5. **Scheduler prefers information per GPU-second.** Fast-benchmark all passing candidates; deep-profile only score winners + diagnostic shift-potential candidates.

### 6.10 Boundary Enforcement (normative)

- **Benchmarker vs. Compiler Service.** The Benchmarker assumes the correctness gate is closed; it does not rerun correctness. Runtime faults during timed execution are classified by fault attribution, not by correctness re-check.
- **Benchmarker vs. Profile Interpreter.** The Benchmarker emits raw metrics and normalized aliases with provenance; `BottleneckAssessment.tags`, `primary_tag`, `evidence`, and `rule_trace` are empty in Benchmarker output.
- **Benchmarker vs. Strategy Navigator / Cross-Candidate Analyzer.** The Benchmarker can say "A is 7.2% faster on weighted_p50" (measurement); it cannot say "try a new memory layout next" (policy) or "this candidate has a reusable gene" (semantics).
- **Benchmarker vs. Orchestrator.** The Benchmarker does not mutate `OptimizationState` and does not persist incumbent updates. It returns structured facts; state mutation is the Orchestrator's job.

### 6.11 Operation Adapter Lifecycle

**Registration.** At service startup, the adapter module imports the V1 built-ins (`elementwise_add_fp32_v1`, `matmul_fp16_v1`) and registers them into a process-local `AdapterRegistry` keyed by `(abi_name, abi_version)`. Additional adapters may be registered by operators via configuration; each registration is an explicit code call (no plugin auto-discovery from filesystem).

**Resolution.** Phase 1 calls `registry.resolve(request.operation_adapter_abi, request.operation_adapter_version)`. `AdapterUnregistered` → `status = infra_error`, reason `adapter_unregistered`. `AdapterVersionMismatch` → `status = infra_error`, reason `adapter_version_mismatch`. The resolved adapter instance is bound to the batch and passed to every Phase 3 / Phase 4 / Phase 6 code path that performs buffer management or launch-argument construction.

**Per-shape buffer lifecycle (inside the worker subprocess).**

```
for shape in objective_shape_cases ∪ profile_shape_cases:
    buffers[shape_id] = adapter.allocate(shape, dtype, device)
    seed = hash((run_id, batch_id, shape_id))
    adapter.seed_inputs(buffers[shape_id], shape, seed)
    if effective_cache_policy == WARM_ROTATING_BUFFERS:
        # build a pool of >= iterations_per_sample equivalent buffers
        buffer_pool[shape_id] = [
            adapter.allocate(shape, dtype, device)
            for _ in range(iterations_per_sample + 1)
        ]
        for b in buffer_pool[shape_id]:
            adapter.seed_inputs(b, shape, seed)
# ... Phase 4 uses buffers; Phase 6 profile child allocates its own identical copy ...
# teardown at end of batch
for shape_id, b in buffers.items():
    adapter.free(b)
for shape_id, pool in buffer_pool.items():
    for b in pool:
        adapter.free(b)
```

**Iteration semantics dispatch (inside `run_sample`).**

```
def run_sample(cand, shape, *, adapter, semantics, buffers, buffer_pool_for_shape,
               iterations_per_sample, entrypoint, grid_dim, block_dim, smem, stream):
    if semantics == OVERWRITE_PURE:
        args = adapter.build_launch_args(buffers, shape)
        cudaEventRecord(start)
        for _ in range(iterations_per_sample):
            cuLaunchKernel(entrypoint, grid_dim, block_dim, smem, stream, args)
        cudaEventRecord(stop); cudaEventSynchronize(stop)
    elif semantics == REQUIRES_OUTPUT_RESET:
        # rotate buffers to avoid cross-iteration coupling
        b = adapter.rotate_buffers(buffer_pool_for_shape)
        args = adapter.build_launch_args(b, shape)
        cudaEventRecord(start)
        for i in range(iterations_per_sample):
            cuLaunchKernel(...)
            if i < iterations_per_sample - 1:
                # out-of-timed-region reset is impossible here; the reset
                # happens implicitly via rotation at sample boundary. The
                # per-iteration reset for output-only kernels is elided
                # because output is overwritten; for kernels that accumulate,
                # use REQUIRES_FULL_INPUT_RESET instead.
        cudaEventRecord(stop); cudaEventSynchronize(stop)
        adapter.reset_between_iterations(b, semantics)  # outside timed region
    elif semantics == REQUIRES_FULL_INPUT_RESET:
        # one sample = one launch with a fresh buffer from the pool
        assert iterations_per_sample == 1
        b = adapter.rotate_buffers(buffer_pool_for_shape)
        adapter.reset_between_iterations(b, semantics)  # outside timed region
        args = adapter.build_launch_args(b, shape)
        cudaEventRecord(start)
        cuLaunchKernel(...)
        cudaEventRecord(stop); cudaEventSynchronize(stop)
    elif semantics == NOT_REPEATABLE:
        assert iterations_per_sample == 1
        adapter.reset_between_iterations(buffers, semantics)  # full restore
        args = adapter.build_launch_args(buffers, shape)
        cudaEventRecord(start)
        cuLaunchKernel(...)
        cudaEventRecord(stop); cudaEventSynchronize(stop)
```

The harness always consults `semantics` before issuing the inner launch loop. INV-BENCH-013 enforces that `reset_hook = None` is only valid for `OVERWRITE_PURE`.

**Adapter authoritative over `adapter_iteration_semantics`.** The `CandidateArtifactRef.adapter_iteration_semantics` field in the request is a hint. When the resolved adapter declares a different value, the adapter wins, a warning `adapter_semantics_overridden_request_hint` is recorded in the batch envelope, and `ShapeMeasurementArtifact.adapter_iteration_semantics` is set to the adapter's declared value (REQ-BENCH-032).

### 6.12 Profile Child Subprocess

**Invocation contract.** For each `(candidate, profile_shape)` in the Phase 6 cartesian product, the worker constructs an NCU command line of the form:

```
ncu \
    --target-processes all \
    --nvtx --nvtx-include "kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile" \
    --launch-count 1 \
    --set <NCU_PROFILE_SET> \
    --replay-mode <application|kernel> \
    --export <artifact_root>/ncu/<artifact_id>.ncu-rep \
    -- \
    <python_executable> -m kerlever.benchmarker.profile_child \
        --cubin-path <cubin_path> \
        --entrypoint <entrypoint> \
        --block-dim <bx,by,bz> \
        --grid-dim <gx,gy,gz> \
        --dynamic-smem-bytes <bytes> \
        --adapter-abi <abi_name> \
        --adapter-version <abi_version> \
        --shape-dims <d0,d1,...> \
        --shape-dtype <dtype> \
        --function-attr-max-smem <bytes> \
        --function-attr-carveout-pct <pct> \
        --function-attr-cache-config <prefer_*> \
        --iterations-per-sample <N> \
        --warmup-count <W> \
        --nvtx-range "<range_name>" \
        --seed <seed>
```

The target command is **never** `[sys.executable, "-c", "pass"]` or any other placeholder — REQ-BENCH-030 enforces this. The profile child binary path is provided via `sys.executable` at worker invocation time; `-m kerlever.benchmarker.profile_child` is the module entrypoint.

**Profile child execution steps.**
1. Parse argv; validate all fields present.
2. Create one CUDA context on GPU 0 (leased device is selected via parent-set `CUDA_VISIBLE_DEVICES`).
3. `cuModuleLoadDataEx(cubin_bytes)`; `cuModuleGetFunction(module, entrypoint)`.
4. Apply function-attribute policy via `cuFuncSetAttribute` / `cuFuncSetCacheConfig`; read back; exit with code 11 (`EX_FUNC_ATTR_APPLY_FAILED`) and error message on stderr if any apply fails.
5. `registry = AdapterRegistry(); registry.register_builtin_adapters()`; `adapter = registry.resolve(adapter_abi, adapter_version)`.
6. `buffers = adapter.allocate(shape, dtype, device)`; `adapter.seed_inputs(buffers, shape, seed)`.
7. For `i in range(warmup_count)`: `cuLaunchKernel(...)` outside any NVTX range.
8. Push NVTX range `nvtx_range` (via `nvtx.push_range(range_name)` or equivalent). For `i in range(iterations_per_sample)`: `cuLaunchKernel(...)`. Pop NVTX range.
9. `cuCtxSynchronize()`; `adapter.free(buffers)`; destroy context; exit 0.

**Failure modes and mapping.**

| Profile child exit condition | Parent worker interpretation |
|---|---|
| Exit 0, NCU exit 0, `.ncu-rep` present | `ProfileStatus = present`; parse metrics |
| Exit 11 (func_attr_apply_failed) | `ProfileStatus = profile_unavailable`, reason `profiler_replay_refused` |
| Exit ≠ 0 (CUDA launch failed during measured launch) | `ProfileStatus = profile_unavailable`, reason `profiler_replay_refused` |
| NCU exit ≠ 0, child exited cleanly | `ProfileStatus = profile_unavailable`, reason `profiler_permission_denied` (if EACCES) or `profiler_timeout` (if killed) |
| NCU command timeout (> `PROFILE_TIMEOUT`) | Kill child; `ProfileStatus = profile_unavailable`, reason `profiler_timeout` |
| `ncu` binary missing at runtime | `ProfileStatus = profile_unavailable`, reason `profiler_binary_missing` |

The profile child does not emit measurements back to the parent; all telemetry lives inside the `.ncu-rep`. The parent reads the report via `ncu --import --print-metrics-json` after the subprocess exits.

**Isolation.** The profile child holds its own CUDA context, distinct from the worker's Phase 3/4 context. A profile child crash does not affect the worker's pre-computed Phase 4 results; those are already durably recorded before Phase 6 begins.

---

## §7 Production Path Trace

**Scenario.** A future GPU Pipeline Adapter submits a round-level batch from round 7. The problem is a matmul `op`, fp16, target `sm_90`, three candidate kernels `C_A` / `C_B` / `C_C`, two objective shapes `shape_small` (`[512, 512, 512]`, weight 1.0) and `shape_large` (`[4096, 4096, 4096]`, weight 2.0), one profile shape `shape_large`, `top_k_profile = 2`, `top_m_profile_shift_potential = 1`. The incumbent is kernel `C_inc` from round 4 with `ObjectiveScore.value = 100.0 µs`.

### Trigger

The Adapter issues `POST /benchmark` with a `BenchmarkBatchRequest`:
- `request_id = "req_r7_b1"`, `run_id = "run_xyz"`, `round_id = "r7"`, `batch_id = "r7_b1"`.
- `problem_spec`: `op_name = "matmul"`, `dtype = "fp16"`, `target_gpu = "H100-SXM5-80G"`, `objective.primary_metric = "weighted_p50_us"`, `objective.aggregation = "weighted_mean"`, `objective.regression_guard_pct = 0.02`.
- `candidate_module_artifact_refs` = [C_A, C_B, C_C], each with its cubin URI, `launch_spec_hash`, `source_hash`, `toolchain_hash`, `adapter_iteration_semantics = OVERWRITE_PURE`, `correctness.passed = True`.
- `incumbent_ref` = {artifact_id = "inc_r4", objective_score.value = 100.0}.
- `operation_adapter_abi = "matmul_fp16_abi_v1"`, `operation_adapter_version = "0.3.1"`.
- `artifact_execution_model = COMMON_HARNESS_CUBIN`, `metric_mode = DEVICE_KERNEL_US`.
- `cache_policy = WARM_SAME_BUFFERS` (will be auto-promoted), `clock_policy = observe+enforce`.
- `top_k_profile = 2`, `top_m_profile_shift_potential = 1`, `anchor_every_n_samples = 4`, `max_interleave_block_len = 6`, `bench_rerun_limit = 1`.

### Phase 1 — Normalize

- Validation: all 3 candidates have `correctness.passed = True`, `launch_spec.entrypoint ∈ {"matmul_A", "matmul_B", "matmul_C"}`, `block_dim = [16, 16, 1]`, `dynamic_smem_bytes = 16384`, `abi_name = "matmul_fp16_abi_v1"`.
- Adapter lookup: `matmul_fp16_abi_v1 @ 0.3.1` registered ✓.
- Hashes derived: `problem_spec_hash = sha256(...)`, `objective_hash = sha256(...)`. Per-shape `interleave_seed` = `hash(("run_xyz","r7_b1","shape_small","kerlever_benchmark_order"))`, same for `shape_large`.
- `MeasurementEnvelope` populated for each candidate.

### Phase 2 — Lease + Hygiene

- Visible GPUs from startup enumeration: `[GPU#0 (H100-SXM5, gpu_uuid="GPU-abc", no MIG)]`.
- Match `sm_90` → lease GPU#0, acquire semaphore.
- `pynvml` preflight: SM clock 1980 MHz, mem clock 2619 MHz, temp 42 °C, power 180 W, throttle reasons `[]`, foreign processes `[]`, ECC DBE 0, Xid 0, profiler counter permission `true`.
- `HygieneReport.reason_on_fail = None`.
- `ClockPolicy.mode = observed_only` (no lock privileges configured).
- Proceed.

### Phase 3 — Plan

- Disposable subprocess spawned. One CUDA context on GPU#0.
- Cubins for C_A / C_B / C_C / C_inc loaded via `cuModuleLoadDataEx`; entrypoints resolved.
- Function attributes applied: `cuFuncSetAttribute(MAX_DYNAMIC_SMEM, 16384)` — observed readback matches.
- Adapter allocates input buffers for `shape_small` and `shape_large`, seeds deterministically.
- Warmup: 5 launches per (candidate, shape) untimed.
- Calibration:
  - `shape_small`: start at 1 launch, `elapsed_ms` = 0.2 ms < 1.0 ms → double to 2, 4, 8 → elapsed 1.6 ms ✓. `iterations_per_sample_small = 8`.
  - `shape_large`: start at 1, `elapsed_ms` = 9.8 ms ✓. `iterations_per_sample_large = 1`.
- Cache policy: interleaving enabled (3 candidates, common harness) + requested `WARM_SAME_BUFFERS` → promote to `WARM_ROTATING_BUFFERS`; record reason.

### Phase 4 — Execute

- For `shape_small`, `shape_large` each:
  - Generate interleave block order via `PCG64(interleave_seed)`:
    ```
    ["anchor", "C_A", "C_B", "C_C", "C_A", "anchor",
     "C_B", "C_C", "C_A", "C_B", "anchor",
     ... FAST_BENCH_REPETITIONS reached ..., "anchor"]
    ```
  - Loop `run_sample` for each token; anchors run incumbent kernel.
- Per-shape samples:
  - `shape_small` — C_A p50 = 3.4 µs, C_B p50 = 3.7 µs, C_C p50 = 4.0 µs, anchor p50 = 4.1 µs. CV ≈ 1.2%. `anchor_pre_score = 4.1`, `anchor_post_score = 4.12`, drift ≈ 0.5%. p95: `run_count = 30 ≥ MIN_P95_SAMPLES`, populated.
  - `shape_large` — C_A p50 = 220 µs, C_B p50 = 210 µs, C_C p50 = 260 µs, anchor p50 = 240 µs. CV ≈ 1.0%. drift ≈ 1.3%. p95 populated.
- Quality classification: all `valid`.

### Phase 5 — Score

- Per-shape values using `primary_metric = weighted_p50_us`:

| Candidate | small p50 | large p50 | weighted value = (1·small + 2·large)/3 |
|---|---|---|---|
| C_A | 3.4 | 220 | 147.8 |
| C_B | 3.7 | 210 | 141.2 |
| C_C | 4.0 | 260 | 174.7 |
| incumbent_anchor | 4.1 | 240 | 161.4 |

- `relative_to_incumbent`:
  - C_A = 147.8 / 161.4 = 0.916
  - C_B = 141.2 / 161.4 = 0.875
  - C_C = 174.7 / 161.4 = 1.082

- Noise margin (aggregated): `NOISE_FLOOR_PCT = 0.01`, candidate_cv ≈ 0.012, anchor_cv ≈ 0.010, anchor_drift ≈ 0.013 → `noise_margin_pct = 0.013`.
- Guard: 0.02.
- Thresholds: improved_threshold = 161.4 · (1 − 0.013) = 159.3; regressed_threshold = 161.4 · (1 + 0.02 + 0.013) = 166.7.
- Decisions:
  - C_A 147.8 < 159.3 → `IMPROVED`, regressed_vs_incumbent=false.
  - C_B 141.2 < 159.3 → `IMPROVED`.
  - C_C 174.7 > 166.7 → `REGRESSED`, regressed_vs_incumbent=true.

### Phase 6 — Profile

- `scoreable = [C_A, C_B]` (C_C excluded as regressed).
- top_k (k=2, ascending by score): `[C_B (141.2), C_A (147.8)]`.
- top_m (m=1 by shift potential) computed from intent direction + static-analysis delta + fast-bench shape vs. incumbent. Suppose C_A shows the largest static-analysis delta (larger smem, same occupancy) → included; already in top_k → dedup leaves `top_k_profiled = [C_B, C_A]`.
- Profile shape: `shape_large` (the declared profile shape).
- For C_B: NVTX range `kerlever/run_xyz/r7_b1/C_B/shape_large/profile` wraps exactly one measured launch. `ncu --nvtx --nvtx-include <range> --launch-count 1 --set focused --export <artifact>/ncu/C_B_shape_large.ncu-rep -- <harness>`.
- For C_A: same pattern.
- Parsed metrics (example, provenance recorded):

| metric_name | value | unit | architecture | profiler_version |
|---|---|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 78.4 | pct | sm_90 | NCU 2025.3 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 62.1 | pct | sm_90 | NCU 2025.3 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 71.7 | pct | sm_90 | NCU 2025.3 |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | null | pct | sm_90 | NCU 2025.3 |

- Normalized:
  - `ProfileMetrics.achieved_occupancy_pct = 78.4` (source: `sm__warps_active...`, provenance recorded).
  - `ProfileMetrics.tensor_core_utilization_pct = null` (metric absent).
  - `BottleneckAssessment.tags = []`, `primary_tag = null`, `evidence = {}`, `rule_trace = []` — Benchmarker does not interpret.

### Phase 7 — Assemble

- `status = success`.
- `run_envelope.pod_health = healthy`, `ambiguous_failure_count = 0`, toolchain versions populated.
- `measurement_context.cache_policy_requested = WARM_SAME_BUFFERS`, `cache_policy_effective = WARM_ROTATING_BUFFERS`, `interleave_enabled = true`.
- `hygiene` report from Phase 2.
- `incumbent_anchor`: ShapeBenchResults (shape_small p50=4.1, shape_large p50=240.0), `ObjectiveScore.value = 161.4`, per-shape drift map.
- `candidate_results`:
  - C_A: `benchmark = BenchmarkBundle{shape_results=[ShapeBenchResult(small, p50=3.4, p95, stdev), ShapeBenchResult(large, p50=220, p95, stdev)], objective_score=ObjectiveScore(value=147.8, metric_name="weighted_p50_us", relative_to_incumbent=0.916, relative_to_baseline=...), regressed_vs_incumbent=false}`, `incumbent_comparison=IMPROVED`, `measurement_quality=valid`, `profile_status=present`, `profile_bundle=ProfileBundle{shape_id="shape_large", metrics=ProfileMetrics{...}, assessment=BottleneckAssessment{tags=[], ...}}`, `raw_profile_metrics_ref="art_raw_CA"`, `fault_class=null`.
  - C_B: analogous with `IMPROVED`.
  - C_C: `incumbent_comparison=REGRESSED`, `profile_status=profile_unavailable`? No — in V1 C_C is simply not selected for profile (excluded); profile_status=profile_unavailable would apply if selected and failed. Here C_C's `profile_status` is set to `profile_unavailable` with reason `not_selected` OR we represent it as `profile_status = profile_unavailable` with reason `not_selected_top_k`. (Spec choice: `not_selected` is not a `ProfileUnavailableReason`; we leave `profile_bundle = null` and `profile_status = profile_unavailable` with `profile_unavailable_reason = null`, matching the "not selected" case.)
- `top_k_profiled = ["C_B", "C_A"]`.
- All rich artifacts (ShapeMeasurementArtifact × 6, NCU reports × 2, raw metrics JSON × 2, samples JSON × 6) written to pod-local artifact store and referenced by id.

### Downstream

The GPU Pipeline Adapter now maps the batch result into three `EvaluationResult` records — one per candidate — for the Orchestrator. That mapping is outside the Benchmarker's scope (Adapter task).

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Countermeasure |
|---|---|---|---|---|
| 1 | Ranking on stale incumbent latency without re-anchoring | Read `incumbent.objective_score.value` from shared state and compare directly to in-episode candidate scores | Warm-vs-cold pod, different clock state, or thermal drift produces fake improvements or regressions; wrong kernel becomes incumbent | Enforce `ANCHOR_INCUMBENT_POLICY = same_episode`; `IncumbentComparison` is always computed against `incumbent_anchor.objective_score`, never against `incumbent_ref.objective_score` (INV-BENCH-006, REQ-BENCH-013) |
| 2 | Absolute microsecond comparison instead of noise-margin gate | "3 µs is definitely faster than 4 µs, mark improved" — forget the noise margin computation | Candidates inside the noise band replace the incumbent, the loop overfits to GPU jitter and thermal drift, regression direction hides real wins | `decide_incumbent_comparison(...)` is the only function that sets `IncumbentComparison`; it always computes `noise_margin_pct` from CV + anchor drift + floor and applies it (INV-BENCH-006, REQ-BENCH-015) |
| 3 | Profiling warmup/anchor launches instead of the measured candidate launch | Wrap the entire repeated-launch loop or the whole shape in one NVTX range and let `ncu` profile whatever it picks | Metrics conflate anchor and candidate behavior; tensor-pipe utilization is inflated by anchor traffic; Profile Interpreter tags wrong bottleneck | NVTX range is `kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile` wrapping exactly one measured launch; `ncu` invoked with `--nvtx --nvtx-include <range> --launch-count 1` (INV-BENCH-008, REQ-BENCH-017) |
| 4 | Using `WARM_SAME_BUFFERS` during interleaved batch | Accept the requested cache policy literally; no promotion logic | Candidate-to-candidate cache coupling — whichever candidate runs first primes the L2 for the next; ranking depends on order, not kernel quality | Phase 3 auto-promotes `WARM_SAME_BUFFERS` → `WARM_ROTATING_BUFFERS` for interleaved batches and records `requested_cache_policy`, `effective_cache_policy`, `cache_policy_reason` (INV-BENCH-005, REQ-BENCH-009) |
| 5 | Fabricating missing NCU metrics as 0 or "not available" strings | When `sm__pipe_tensor_cycles_active...` isn't present on this arch, emit 0.0 "for compatibility" | Profile Interpreter reads 0% tensor utilization as evidence of compute underutilization on a kernel that in fact uses tensor pipes fine | Missing metrics are `null`; normalized compact aliases also `null`; provenance mandatory; Pydantic validation (INV-BENCH-009, REQ-BENCH-018) |
| 6 | Running the benchmark harness in-process with the FastAPI server | Skip the disposable subprocess to save startup cost | A poisoned CUDA context from one bad candidate kills the entire service; subsequent batches cannot run until the pod is restarted | Harness runs in a disposable `asyncio.create_subprocess_exec` subprocess per batch; subprocess is terminated on completion or fault (INV-BENCH-012, REQ-BENCH-025) |
| 7 | Treating an ambiguous fault as a `candidate_fault` because the kernel crashed | Same try/except branch for all exceptions; classify everything as candidate's fault | Real pod-health issues (Xid, context poisoning) become negative search signal; Strategy Navigator avoids correct directions because "they crashed before" | `faults.attribute(exception, pod_health, repeated)` distinguishes candidate / infra / ambiguous; ambiguous faults trigger probe + pod-health transition and never contribute to signal (INV-BENCH-010, REQ-BENCH-019, REQ-BENCH-020) |
| 8 | Computing `top_k_profiled` from uncollected NCU metrics | "Let's pick by DRAM throughput" — but DRAM throughput comes from the profile that hasn't run yet | Selection depends on an accidentally-cached stale profile or fails silently when the cache is empty; starves the shift-potential set | Shift-potential uses pre-profile signals only (intent, static analysis, fast-bench throughput shape, novelty); NCU counters from this batch are not available at selection time (INV-BENCH-007, REQ-BENCH-016) |
| 9 | Using `p95_us` below `MIN_P95_SAMPLES` | Compute percentile from 5 samples and populate the field anyway | Downstream weighted_p95 objectives rank on noise, not on measured tail latency | `p95_us = null` below the minimum; compact `ShapeBenchResult.latency_p95_us` respects the gate; weighted_p95 objectives do not score shapes below the gate (REQ-BENCH-011) |
| 10 | Mapping `statistical_tie` to `regressed_vs_incumbent = true` | Because `value` is not strictly below incumbent, "obviously it's a regression" | Orchestrator discards benign neutral attempts from search memory; exploration loop loses evidence | `regressed_vs_incumbent = (IncumbentComparison == REGRESSED)`; statistical_tie is not regression (REQ-BENCH-015) |
| 11 | Silently promoting `profile_unavailable` candidates into `top_k_profiled` | Ignore ProfileStatus; keep whoever "should" be profiled | `top_k_profiled` includes candidates with no profile data; downstream Profile Interpreter receives empty bundles and either skips or fabricates | `top_k_profiled` contains only candidates with `profile_status = present`; `profile_unavailable` reasons are typed and never confused with "collected" (REQ-BENCH-016, INV-BENCH-009) |
| 12 | Nsight Systems on every candidate "because it's cheap" | Add nsys by default alongside NCU | GPU-second budget explodes, large timeline artifacts fill disk, no improvement to ranking quality | Nsight Systems is trigger-based per `NSYS_PROFILE_POLICY`; never default per-candidate (REQ-BENCH-022) |

---

## §9 Non-Goals

This task explicitly excludes:

- **Orchestrator wiring.** The Benchmarker is not plugged into `GPUPipelineProtocol.evaluate` in V1. A separate GPU Pipeline Adapter task will bridge batch responses into `EvaluationResult` records. The Benchmarker must not depend on or import `kerlever.protocols`.
- **Compiler Service.** Compilation, correctness validation, sanitizer gating, and artifact-key generation remain the Compiler Service's responsibility. The Benchmarker trusts the forwarded `correctness.passed = True` gate and does not rerun correctness.
- **Profile Interpreter.** The Benchmarker emits raw + normalized metrics with provenance; bottleneck tags, evidence-to-direction mapping, and `BottleneckAssessment.tags`/`primary_tag`/`evidence`/`rule_trace` are the Profile Interpreter's job. Benchmarker output always sets `BottleneckAssessment` fields to empty.
- **Search-memory persistence.** The Benchmarker is stateless across batches except for pod-health and artifact-store bookkeeping. It does not persist incumbent history, tabu entries, round state, or decision logs.
- **Incumbent updates.** The Benchmarker decides `IncumbentComparison` per candidate; the Orchestrator decides whether to update the incumbent based on `IncumbentComparison == IMPROVED`. The Benchmarker does not mutate baseline or incumbent.
- **Unit tests.** Per the manager's explicit instruction, no unit tests are written in this task. Static gates (`ruff check .`, `mypy .`) are the only quality gates.
- **Default Nsight Systems collection.** Nsight Systems is trigger-based only; no default `nsys` on every candidate.
- **Clock locking as baseline requirement.** Clock locking is optional hardening for privileged pools. Default is observe+enforce-hygiene; throttled or thermally unstable episodes are marked unstable.
- **Cross-architecture ranking.** A100 / H100 / Blackwell numbers are never directly ranked against each other. Baseline/incumbent anchors normalize within a pod, not across architectures.
- **Separate-executable artifact mode.** V1 supports only cubin/module + common harness. Separate-executable-per-candidate is explicitly excluded.
- **LLM decision-making.** The Benchmarker is fully deterministic. No LLM call, no natural-language classification or advice.
- **Cross-pod shared state.** Pod-health, idempotency, and artifact store are pod-local. Distribution across pods is handled by the orchestrator/adapter layer, not the Benchmarker.
- **Kernel compilation inside the Benchmarker.** V1 does not compile source; it consumes cubin blobs produced by the Compiler Service. PTX compilation, JIT, and any build-time toolchain are out of scope.
- **Nsight Compute metric interpretation.** The Benchmarker does not tag bottlenecks, interpret counter patterns, or emit optimization direction hints from NCU metrics. Raw and normalized metrics with provenance are the Benchmarker's output; interpretation is the Profile Interpreter's job.
- **Inline cubin bytes transport (`cubin_bytes_b64`).** V1 supports only absolute-POSIX-path cubin transport on a shared mount (REQ-BENCH-031). Inline base64 cubin bytes in the request body are explicitly deferred to V2; if present they are ignored with a warning.
- **Remote cubin URIs (`s3:`, `http:`, `https:`, `file:` schemes).** V1 refuses these schemes. Remote blob fetching, object-store integration, and signed-URL resolution are out of scope.
- **Artifact store retention policy and rotation.** The Benchmarker writes to `cfg.artifact.root`; retention, rotation, disk-quota management, and backup are operator-owned. The Benchmarker does not run a GC daemon, does not enforce TTL-based deletion, and does not provide a tenant-namespaced directory layout.
- **Adapter plugin auto-discovery.** V1 ships exactly two built-in adapters registered at startup by explicit code. No filesystem-scan or `pkg_resources`-based plugin discovery; operators who want custom adapters add an explicit registration call.
- **Mutating shared types in `kerlever/types.py`.** The compact `ShapeBenchResult.latency_p95_us` remains `float` non-optional. The p95-gated case is encoded via the sentinel value `-1.0` (REQ-BENCH-033), not via a type change. Making `latency_p95_us: float | None` is a deferred V2 change requiring a coordinated update across all modules that consume `ShapeBenchResult`.
- **Applying `FunctionAttributePolicy` fields beyond `max_dynamic_shared_memory_size`.** V1 applies ONLY `max_dynamic_shared_memory_size` (sourced from `launch_spec.dynamic_smem_bytes`). The other four fields — `preferred_shared_memory_carveout_pct`, `cache_config`, `cluster_dims`, `non_portable_cluster_size_allowed` — are NOT applied in V1 because the V1 request schema (shared `CandidateArtifactRef` / `BenchmarkBatchRequest`) has no field that carries them end-to-end. Extending the schema to carry these values, and correspondingly broadening REQ-BENCH-029's enforcement to include them, is a V2 change tracked here as an explicit Non-Goal. The V1 `FunctionAttributePolicy` Pydantic model retains all five fields for forward-compatibility; only `max_dynamic_shared_memory_size` is guaranteed populated in V1 envelopes (the other four are `None` in both `_requested` and `_observed`).

---

## §10 Traceability Matrix

| Success Criterion | Requirements | Invariants | Scenarios |
|---|---|---|---|
| SC-BENCH-001: Fast bench on every shape | REQ-BENCH-001, REQ-BENCH-004, REQ-BENCH-005, REQ-BENCH-006, REQ-BENCH-007, REQ-BENCH-008, REQ-BENCH-011, REQ-BENCH-012, REQ-BENCH-014, REQ-BENCH-027, REQ-BENCH-029, REQ-BENCH-033 | INV-BENCH-007, INV-BENCH-013 | SCN-BENCH-001-01, SCN-BENCH-003-01, SCN-BENCH-003-02, SCN-BENCH-003-03, SCN-BENCH-003-04, SCN-BENCH-004-03, SCN-BENCH-004-04, SCN-BENCH-011-04, SCN-BENCH-017-01 |
| SC-BENCH-002: Hygiene gates | REQ-BENCH-002, REQ-BENCH-003, REQ-BENCH-020, REQ-BENCH-027 | INV-BENCH-003 | SCN-BENCH-002-01 .. SCN-BENCH-002-08 |
| SC-BENCH-003: Fair interleaving | REQ-BENCH-009, REQ-BENCH-010, REQ-BENCH-013, REQ-BENCH-024, REQ-BENCH-025 | INV-BENCH-002, INV-BENCH-004, INV-BENCH-005, INV-BENCH-012, INV-BENCH-013 | SCN-BENCH-003-05, SCN-BENCH-003-06, SCN-BENCH-004-01, SCN-BENCH-004-02 |
| SC-BENCH-004: Noise-margin decision | REQ-BENCH-012, REQ-BENCH-013, REQ-BENCH-014, REQ-BENCH-015, REQ-BENCH-033, REQ-BENCH-034 | INV-BENCH-006, INV-BENCH-011, INV-BENCH-015 | SCN-BENCH-005-01, SCN-BENCH-005-02, SCN-BENCH-005-03, SCN-BENCH-005-05, SCN-BENCH-005-06, SCN-BENCH-017-02, SCN-BENCH-018-01, SCN-BENCH-018-02 |
| SC-BENCH-005: Top-K ∪ top-M profile selection | REQ-BENCH-016 | INV-BENCH-007 | SCN-BENCH-005-04, SCN-BENCH-006-01, SCN-BENCH-006-02 |
| SC-BENCH-006: NCU with NVTX + provenance | REQ-BENCH-005, REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-021, REQ-BENCH-022, REQ-BENCH-030 | INV-BENCH-008, INV-BENCH-009 | SCN-BENCH-007-01, SCN-BENCH-007-02, SCN-BENCH-007-03, SCN-BENCH-007-04, SCN-BENCH-007-05, SCN-BENCH-012-01, SCN-BENCH-012-02, SCN-BENCH-012-03, SCN-BENCH-012-04 |
| SC-BENCH-007: Fault attribution + profile_unavailable | REQ-BENCH-003, REQ-BENCH-005, REQ-BENCH-012, REQ-BENCH-019, REQ-BENCH-020, REQ-BENCH-025 | INV-BENCH-010, INV-BENCH-012 | SCN-BENCH-002-05, SCN-BENCH-002-07, SCN-BENCH-007-04, SCN-BENCH-007-05, SCN-BENCH-008-01, SCN-BENCH-008-02, SCN-BENCH-008-03 |
| SC-BENCH-008: Structured output reusing shared types | REQ-BENCH-001, REQ-BENCH-008, REQ-BENCH-018, REQ-BENCH-021, REQ-BENCH-024, REQ-BENCH-026, REQ-BENCH-029 | INV-BENCH-001 | SCN-BENCH-001-01, SCN-BENCH-009-01 |
| SC-BENCH-009: Self-contained HTTP contract | REQ-BENCH-023, REQ-BENCH-026 | INV-BENCH-001 | SCN-BENCH-009-01 |
| SC-BENCH-010: FastAPI + Docker | REQ-BENCH-023 | — | SCN-BENCH-010-01, SCN-BENCH-010-02, SCN-BENCH-010-03 |
| SC-BENCH-011: Static gates pass | QG-BENCH-001, QG-BENCH-002 | — | (verified by Manager running `ruff check .` + `mypy .`) |
| SC-BENCH-012: Operation Adapter contract | REQ-BENCH-028 | INV-BENCH-013 | SCN-BENCH-011-01, SCN-BENCH-011-02, SCN-BENCH-011-03, SCN-BENCH-011-04 |
| SC-BENCH-013: NCU target = profile child | REQ-BENCH-030 | INV-BENCH-008 | SCN-BENCH-012-01, SCN-BENCH-012-02, SCN-BENCH-012-03, SCN-BENCH-012-04 |
| SC-BENCH-014: Cubin transport + artifact store lifecycle | REQ-BENCH-031, REQ-BENCH-021 | — | SCN-BENCH-013-01, SCN-BENCH-013-02, SCN-BENCH-013-03, SCN-BENCH-013-04, SCN-BENCH-013-05, SCN-BENCH-014-01 |
| SC-BENCH-015: Rich artifact fidelity | REQ-BENCH-032, REQ-BENCH-029 | INV-BENCH-014 | SCN-BENCH-015-01, SCN-BENCH-015-02, SCN-BENCH-015-03, SCN-BENCH-015-04, SCN-BENCH-015-05, SCN-BENCH-015-06, SCN-BENCH-016-01, SCN-BENCH-016-02, SCN-BENCH-016-03 |
| SC-BENCH-016: Batch staging artifacts cleaned up | REQ-BENCH-035, REQ-BENCH-021 | INV-BENCH-016 | SCN-BENCH-017-03 |
