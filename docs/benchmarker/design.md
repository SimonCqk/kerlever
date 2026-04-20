# Benchmarker Module Design

## §1 Purpose and Scope

The Benchmarker is the deterministic measurement service specified in `docs/benchmarker/spec.md`. That spec is the behavior of record — it defines the seven phases, the measurement envelope, interleaving rules, noise-margin decision, hygiene gates, profile selection, fault taxonomy, request/response contracts, invariants, and shortcut risks. This design document owns **composition, lifecycle, and wiring**: how the files under `kerlever/benchmarker/` are arranged, which function calls which, what process holds which CUDA/NVML handle, how the FastAPI service, the batch supervisor, and the disposable worker subprocess communicate, what the Dockerfile looks like, and what the startup/shutdown ordering is. Every behavioral rule — hygiene gates, regression formula, incumbent-comparison decision table, cache-policy auto-promotion, NVTX range shape, fault attribution — is owned by spec §6 and is referenced from here rather than restated. When a reader needs to know *what* a phase decides, they read the spec; when they need to know *which module holds the code that decides it*, they read this design.

---

## §2 Module Composition

All module paths are under `kerlever/benchmarker/`. The file list matches the plan's "Module layout" exactly.

```
kerlever/benchmarker/
├── __init__.py         # public exports
├── service.py          # FastAPI app, endpoints, request validation
├── supervisor.py       # batch supervisor: spawn worker, collect result, fault attribution
├── worker.py           # worker subprocess entrypoint (phases 1 → 7)
├── config.py           # BenchmarkerConfig + env parsing
├── types.py            # Pydantic models + StrEnum (spec §5)
├── normalize.py        # Phase 1: validate request + build MeasurementEnvelope
├── lease.py            # Phase 2: GPU lease + per-gpu asyncio.Semaphore
├── telemetry.py        # Phase 2: pynvml wrapper + HygieneReport construction
├── plan.py             # Phase 3: warmup + iteration calibration + cache policy
├── cuda_driver.py      # cuda-python (Driver API) wrapper: ctx, module, function, events, device memory
├── adapter.py          # Operation Adapter Protocol + AdapterRegistry + V1 built-ins (spec §6.11)
├── harness.py          # Phase 4: common harness, interleave, run_sample, anchors
├── stats.py            # p50/p95/mean/stdev/cv/noise_margin/anchor_drift
├── scoring.py          # Phase 5: ObjectiveScore + decide_incumbent_comparison
├── selection.py        # Phase 5/6 boundary: top-K ∪ top-M, shift-potential
├── profiler.py         # Phase 6: ncu subprocess wrapper, cmdline builder, NVTX filter, metric parsing
├── profile_child.py    # Phase 6 NCU target: one measured launch inside one NVTX range (spec §6.12)
├── fault.py            # fault attribution + pod-health state machine
└── main.py             # uvicorn entrypoint

docker/benchmarker/
├── Dockerfile
└── entrypoint.sh

docs/benchmarker/
├── spec.md             # behavior of record
└── design.md           # this file
```

Twenty Python modules plus Docker assets and docs.

### 2.1 Module responsibilities and public surface

**`__init__.py`** — re-exports the public surface used by tests/tooling only:
- `BenchmarkerConfig` from `.config`.
- `create_app` from `.service`.
- Pydantic request/response types from `.types` (`BenchmarkBatchRequest`, `BenchmarkBatchResult`, `MeasurementEnvelope`, `CandidateResult`, `HygieneReport`, `ShapeMeasurementArtifact`, `IncumbentAnchor`, and every StrEnum from spec §5).
- No GPU-touching symbols (those live under modules whose import itself does not call NVML or cuda-python; import-time side effects are forbidden).

**`service.py`** — FastAPI factory and endpoint handlers. Owns the uvicorn-visible surface:
- `create_app(config: BenchmarkerConfig) -> FastAPI` — builds the app, registers handlers, installs the exception middleware per REQ-BENCH-026, stores an initialized `ServiceEnv` on `app.state`.
- Handlers: `handle_benchmark(req)`, `handle_healthz()`, `handle_info()`.
- Holds **no CUDA context, no pynvml handles beyond the cached info inventory**. Its only GPU coupling is the cached `DeviceInventory` built once at startup for `/info` and hygiene pre-check.
- Private: `_ServiceEnv` dataclass bundling `config`, `supervisor`, `lease_manager`, `device_inventory`, `pod_health_store`, `artifact_store`.

**`supervisor.py`** — batch-level orchestrator in the service process. Owns:
- `run_batch(req: BenchmarkBatchRequest, env: ServiceEnv) -> BenchmarkBatchResult`.
- `_spawn_worker(req_path: Path, res_path: Path, cfg_path: Path, target_device: LeasedDevice) -> asyncio.subprocess.Process`.
- `_await_worker(proc, timeout_s: float) -> WorkerExit`.
- `_read_worker_result(res_path: Path) -> BenchmarkBatchResult | WorkerFailure`.
- `_attribute_worker_exit(exit: WorkerExit, partial_result: BenchmarkBatchResult | None) -> BatchStatus + FaultClass`.
- Fault attribution for supervisor-visible faults (non-zero exit code, signal, timeout, unparseable result file) — candidate-level fault attribution is done inside the worker and serialized into the result.

**`worker.py`** — disposable subprocess entrypoint. This is the **only** module that imports `cuda_driver`, `adapter`, and `profiler` at module-scope. Owns:
- `main(argv: list[str]) -> NoReturn` — parses argv, performs NVML init, creates CUDA context, runs phases 1b..7 inside the subprocess, writes result file, calls `os._exit(0)` on controlled completion or `os._exit(2)` on uncaught exception after attempting to flush a `BenchmarkBatchResult` with `status=infra_error`.
- Never returns to Python's interpreter shutdown path — the `os._exit` is the INV-BENCH-012 enforcement seam.
- Private: `_run_phases(req, cfg, device)` — orchestrates `normalize` → (adapter resolve) → `_load_candidates` → (per-shape adapter allocate+seed) → `plan` → `harness` → `scoring` → `selection` → `_run_profile_phase` in order, catching per-candidate exceptions and routing them through `fault.attribute`.
- Private: `_resolve_adapter(req) -> OperationAdapter` — single call to `adapter.get_adapter(req.operation_adapter_abi, req.operation_adapter_version)`. On `AdapterUnregistered` / `AdapterVersionMismatch` → writes an `INFRA_ERROR` result with `failure_reason ∈ {"adapter_unregistered", "adapter_version_mismatch"}` and `os._exit(1)`. The adapter is resolved **before** any cubin load (§8.2).
- Private: `_build_shape_buffers(adapter, shapes, device, effective_cache_policy, iterations_per_sample_per_shape) -> (buffers_per_shape, buffer_pool_per_shape)` — allocates adapter buffers per shape; when `effective_cache_policy == WARM_ROTATING_BUFFERS` it also builds a pool sized `iterations_per_sample + 1` for each shape (spec §6.11). Teardown is symmetric: `adapter.free` for every allocated buffer on the success or candidate-fault path.
- Private: `_build_adapter_build_args(adapter)` and `_build_adapter_reset_hook(adapter)` / `_build_adapter_rotate_hook(adapter)` — small closure helpers wired into `harness.execute_batch` so the harness never imports `adapter` directly (keeps the harness adapter-agnostic).
- Private: `_load_candidates(admit_refs: list[CandidateArtifactRef], ctx: CudaContext, policy: FunctionAttributePolicy) -> list[LoadedCandidate]` — reads each cubin, calls `cuda_driver.load_module` + `cuda_driver.get_function`, then applies the function-attribute policy via `cuda_driver.set_function_attribute` (and `set_cache_config` when required); read-back values populate `LoadedCandidate.function_attribute_policy_observed` for the envelope (REQ-BENCH-029). Apply failures raise `FunctionAttributePolicyApplyFailed`; the caller converts to a per-candidate `CandidateRuntimeFault(fault_class=infra_fault)` and the other candidates continue.
- Private: `_run_profile_phase(profile_set, profile_shapes, plan_out, loaded, req, cfg) -> dict[str, ProfileOutcome]` — drives the Phase 6 cartesian product by building an NCU command line per `(candidate, profile_shape)` whose `-- <target_cmd>` points at `[sys.executable, "-m", "kerlever.benchmarker.profile_child", ...flags...]` (see §3.4 and spec §6.12).
- The previous `_build_default_args` helper (which returned an empty tuple of launch arguments) is removed — launch arguments now always come from `adapter.build_launch_args`. An empty tuple from an adapter is explicitly rejected by `REQ-BENCH-028` when the ABI declares operands.

**`config.py`** — `BenchmarkerConfig` dataclass loaded from env (see §9 for the split). Public:
- `BenchmarkerConfig.from_env() -> BenchmarkerConfig`.
- `ClockPolicyConfig`, `ArtifactStoreConfig`, `ProfilerConfig`, `LeaseConfig`, `HygieneThresholds` as nested dataclasses.
- No Pydantic here — the config is an internal dataclass, not a request schema.

**`types.py`** — all Pydantic models and StrEnum from spec §5, verbatim names. Imports `ShapeBenchResult`, `ObjectiveScore`, `ProfileMetrics`, `ProfileBundle`, `BottleneckAssessment`, `BenchmarkBundle`, `CorrectnessResult`, `StaticAnalysis`, `ProblemSpec`, `ShapeCase`, `PerformanceObjective` from `kerlever.types` and re-uses them without redefinition. Adds the local StrEnum + envelope/result types.

**`normalize.py`** — Phase 1 logic. Public:
- `normalize_request(req: BenchmarkBatchRequest, cfg: BenchmarkerConfig, device: LeasedDevice) -> NormalizedBatch`.
- `NormalizedBatch` is an internal dataclass holding `envelope_per_candidate: dict[str, MeasurementEnvelope]`, `measurement_context: MeasurementContext`, `admit_candidates: list[CandidateArtifactRef]`, `reject_candidates: dict[str, InfraFault]`, `interleave_seed_per_shape: dict[str, int]`.
- Pure: no I/O, no GPU touch. Cubin URI resolution to a readable `Path` happens here (local filesystem stat only), but cubin bytes are not read until `cuda_driver.load_modules` in Phase 3.

**`lease.py`** — GPU lease manager in the service process. Public:
- `LeaseManager(cfg: LeaseConfig, inventory: DeviceInventory)` — constructor.
- `LeaseManager.acquire(target: TargetGpuSpec) -> AsyncIterator[LeasedDevice]` — async context manager, holds an `asyncio.Semaphore(1)` keyed on `gpu_uuid` for the duration of Phase 2..7 on that device (INV-BENCH-003).
- `DeviceInventory.list() -> list[DeviceInventoryEntry]` — cached at startup, not called per-request.

**`telemetry.py`** — pynvml wrapper. Public:
- `telemetry.init() -> None` — wraps `pynvml.nvmlInit`; idempotent.
- `telemetry.shutdown() -> None`.
- `telemetry.info_inventory() -> list[DeviceInventoryEntry]` — called once at service startup.
- `telemetry.preflight(lease: LeasedDevice, policy: ClockPolicyConfig) -> HygieneReport` — returns a populated `HygieneReport` plus a first `DeviceTelemetrySnapshot`. See spec §6.2 decision table for the outcome routing.
- `telemetry.snapshot(lease: LeasedDevice) -> DeviceTelemetrySnapshot` — single sample, used before/after Phase 4.
- `telemetry.postflight(lease: LeasedDevice, pre: DeviceTelemetrySnapshot) -> tuple[DeviceTelemetrySnapshot, AnchorDriftTelemetry]` — second sample + drift annotation.
- All functions are synchronous; pynvml is not asyncio-aware. They are called from the worker (except `info_inventory` and the per-request preflight done under lease, which run on the asyncio event loop thread via `asyncio.to_thread` — see §5).
- Internal: dict-based translation of `nvmlDeviceGetCurrentClocksEventReasons` bitmask to the spec §6.2 reason list (`HW_SLOWDOWN`, `SW_THERMAL_SLOWDOWN`, `SW_POWER_CAP`, etc.).

**`plan.py`** — Phase 3 logic. Public:
- `plan.calibrate(candidates: list[LoadedCandidate], shapes: list[ShapeCase], cfg: PlanConfig, adapter: OperationAdapter) -> CalibratedPlan`.
- `CalibratedPlan` holds: `warmup_count_per_candidate_shape`, `iterations_per_sample_per_candidate_shape`, `repetitions`, `effective_cache_policy`, `requested_cache_policy`, `cache_policy_reason`, `adapter_iteration_semantics_per_candidate`, `metric_mode`, `function_attribute_policy_observed_per_candidate`.
- Depends on `cuda_driver` for launching calibration probes (events + launches) but does **not** do scoring or anchors.

**`cuda_driver.py`** — minimal cuda-python (Driver API) wrapper. Public:
- `cuda_driver.init() -> None` — wraps `cuInit(0)`.
- `cuda_driver.create_primary_context(device_ordinal: int) -> CudaContext`.
- `cuda_driver.destroy_primary_context(ctx: CudaContext) -> None`.
- `cuda_driver.load_module(cubin: bytes, options: ModuleLoadOptions | None) -> CudaModule` — wraps `cuModuleLoadDataEx`.
- `cuda_driver.get_function(module: CudaModule, entrypoint: str) -> CudaFunction`.
- `cuda_driver.set_function_attribute(fn: CudaFunction, attr: FunctionAttribute, value: int) -> int` — wraps `cuFuncSetAttribute` then `cuFuncGetAttribute`; returns the observed read-back value for envelope recording (REQ-BENCH-029).
- `cuda_driver.get_function_attribute(fn: CudaFunction, attr: FunctionAttribute) -> int` — wraps `cuFuncGetAttribute` (read-only; used during envelope capture and for compliance checks).
- `cuda_driver.set_cache_config(fn: CudaFunction, cfg: CacheConfig) -> None` — wraps `cuFuncSetCacheConfig`.
- `cuda_driver.mem_alloc(bytes_: int) -> DevicePtr` — wraps `cuMemAlloc`; returned `DevicePtr` is an opaque int-like handle that adapters carry in `AdapterBuffers.device_ptrs`.
- `cuda_driver.mem_free(ptr: DevicePtr) -> None` — wraps `cuMemFree`; idempotent against already-freed pointers only via explicit adapter discipline (no double-free detection in the wrapper).
- `cuda_driver.memcpy_htod(ptr: DevicePtr, host_bytes: bytes) -> None` — wraps `cuMemcpyHtoD`; synchronous copy used by adapters during `seed_inputs` / `reset_between_iterations`.
- `cuda_driver.memcpy_dtoh(ptr: DevicePtr, nbytes: int) -> bytes` — wraps `cuMemcpyDtoH`; used by adapters for correctness cross-checks (not by harness hot-path).
- `cuda_driver.launch(fn: CudaFunction, grid: tuple[int, int, int], block: tuple[int, int, int], smem: int, stream: CudaStream, args: tuple[Any, ...]) -> None` — wraps `cuLaunchKernel`.
- `cuda_driver.event_record(event: CudaEvent, stream: CudaStream) -> None`.
- `cuda_driver.event_elapsed_ms(start: CudaEvent, stop: CudaEvent) -> float` — synchronizes on stop, returns `cuEventElapsedTime` delta.
- `cuda_driver.stream_synchronize(stream: CudaStream) -> None` — wraps `cuStreamSynchronize`; used by the profile child after its single timed launch and by the adapter teardown path.
- Plus lifecycle-only helpers (`create_event`, `create_stream`, `destroy_*`) that are constructor-and-destroy symmetric.

Memory functions (`mem_alloc` / `mem_free` / `memcpy_htod` / `memcpy_dtoh`) and the function-attribute pair (`set_function_attribute` / `get_function_attribute`) are called by `adapter.py` and by `worker._load_candidates`. No other module reaches into device memory — the harness hot path only sees opaque `AdapterBuffers`.

**`adapter.py`** — Operation Adapter Protocol, registry, and V1 built-ins. Spec §6.11 owns the behavioral contract (allocation, seeding, launch-arg tuple construction, grid derivation, iteration-semantics reset hook, buffer rotation, teardown). Public:
- `OperationAdapter` (Protocol) — the 9-method interface defined in spec §5 Operation Adapter Protocol: `allocate`, `seed_inputs`, `build_launch_args`, `grid_dim`, `useful_bytes`, `algorithmic_flops`, `reset_between_iterations`, `rotate_buffers`, `free`. Class-level attributes `abi_name: ClassVar[str]` and `abi_version: ClassVar[str]` identify the plugin key.
- `AdapterBuffers` (dataclass) — `device_ptrs: tuple[int, ...]`, `host_side: dict[str, object]`, `shape_dims: tuple[int, ...]`, `dtype: str`. Opaque to every module except the owning adapter and `cuda_driver`.
- `AdapterRegistry` — process-local registry. Methods: `register(adapter: OperationAdapter) -> None`, `resolve(abi_name: str, abi_version: str) -> OperationAdapter`, `register_builtin_adapters() -> None` (imports + registers the two V1 built-ins; invoked by both `service.create_app` and `profile_child.main`).
- `register_adapter(adapter: OperationAdapter) -> None` — module-level convenience that delegates to the process-global registry singleton.
- `get_adapter(abi_name: str, abi_version: str) -> OperationAdapter` — module-level convenience that delegates to the singleton; raises `AdapterUnregistered` or `AdapterVersionMismatch`.
- `ElementwiseAddFp32V1` — V1 built-in adapter for the `elementwise_add_fp32_v1` ABI (`(const float* A, const float* B, float* C, int N)`). Uses `cuda_driver.mem_alloc` + `memcpy_htod` for allocation/seeding; `adapter_iteration_semantics = OVERWRITE_PURE`.
- `MatmulFp16V1` — V1 built-in adapter for the `matmul_fp16_v1` ABI (`(const half* A, const half* B, half* C, int M, int N, int K)`). Same allocation model scaled to fp16 operand sizes; `adapter_iteration_semantics = OVERWRITE_PURE`.
- Module-level import side effect: both V1 built-ins auto-register on import. Third-party adapters are loaded via the `ADAPTER_REGISTRY_MODULES` env var (see §9).

**`profile_child.py`** — dedicated Phase 6 NCU target. Invoked by NCU as the `-- <target_cmd>` child; not the worker. Public:
- `main(argv: list[str]) -> NoReturn` — parses argv (shape listed in spec §6.12), opens CUDA context, loads the cubin, applies `function_attribute_policy`, resolves the adapter, allocates + seeds buffers, runs warmup outside any NVTX range, wraps exactly `iterations_per_sample` launches inside one NVTX push/pop, synchronizes, frees buffers, destroys the context, and exits via `os._exit(returncode)`.
- Exit-code discipline (spec §6.12 "Failure modes"):
  - `os._exit(0)` — success.
  - `os._exit(11)` — `EX_FUNC_ATTR_APPLY_FAILED`: `cuFuncSetAttribute` returned a driver error.
  - `os._exit(12)` — `EX_ADAPTER_UNREGISTERED`: adapter lookup failed (registry mis-seeded in child).
  - `os._exit(1)` — generic runtime fault (launch failure, copy failure).
  - `os._exit(2)` — uncaught exception path.
- Never returns to CPython interpreter shutdown — `os._exit` is mandatory (same INV-BENCH-012 rationale as the worker).
- Imports: `cuda_driver`, `adapter`, `nvtx` (via `cuda-python` or `nvidia-nvtx` binding). Does **not** import `telemetry`, `plan`, `harness`, `scoring`, `selection`, `profiler`, `service`, `supervisor`, `worker`, or `fault` — by construction, the profile child does not perform hygiene, statistics, scoring, or fault attribution (REQ-BENCH-030).

**`harness.py`** — Phase 4 common harness + interleaved batch execution. Public:
- `harness.execute_batch(plan: CalibratedPlan, candidates: list[LoadedCandidate], incumbent: LoadedCandidate, shapes: list[ShapeCase], seeds: dict[str, int], cfg: HarnessConfig, telem: TelemetryRecorder, build_args: Callable[[AdapterBuffers, ShapeCase], tuple[object, ...]], reset_hook: Callable[[AdapterBuffers, AdapterIterationSemantics], None] | None, rotate_hook: Callable[[list[AdapterBuffers]], AdapterBuffers] | None, buffers_per_shape: dict[str, AdapterBuffers], buffer_pool_per_shape: dict[str, list[AdapterBuffers]] | None) -> BatchMeasurement`.
- `harness.generate_block_order(candidates: list[str], anchor_every_n: int, max_block_len: int, seed: int, total_repetitions: int) -> list[str]` — deterministic PCG64-seeded permutation per spec §6.4.
- `harness.run_sample(fn: CudaFunction, shape: ShapeCase, plan: SamplePlan, *, semantics: AdapterIterationSemantics, buffers: AdapterBuffers, buffer_pool: list[AdapterBuffers] | None, build_args: Callable[[AdapterBuffers, ShapeCase], tuple[object, ...]], reset_hook: Callable[[AdapterBuffers, AdapterIterationSemantics], None] | None, rotate_hook: Callable[[list[AdapterBuffers]], AdapterBuffers] | None, nvtx: NvtxRange | None) -> float` — single-sample timed measurement, returns per-launch microseconds. Wraps NVTX range only when `nvtx` is not None (only the Phase 6 profiled launch in the profile child gets wrapped; REQ-BENCH-017, INV-BENCH-008). Dispatches on `semantics` per the spec §6.11 `run_sample` dispatch table: `OVERWRITE_PURE` runs the straight inner launch loop; `REQUIRES_OUTPUT_RESET` rotates from `buffer_pool` per sample and calls `reset_hook` outside the timed region; `REQUIRES_FULL_INPUT_RESET` asserts `iterations_per_sample == 1` and performs a fresh reset + rotation per sample; `NOT_REPEATABLE` asserts `iterations_per_sample == 1` and calls `reset_hook` for a full restore. `reset_hook = None` is valid only for `OVERWRITE_PURE`; any other combination is caught by an assertion and escalates to `CandidateRuntimeFault` (INV-BENCH-013 enforcement seam).
- `BatchMeasurement` holds per-(candidate, shape) raw samples, anchor pre/post lists, realized `interleave_block_order` per shape, `warmup_count` per (candidate, shape), and the resolved `adapter_iteration_semantics` (for artifact fidelity — REQ-BENCH-032 / INV-BENCH-014).

**`stats.py`** — pure numeric functions. Public:
- `stats.p50(samples: list[float]) -> float`.
- `stats.p95(samples: list[float], min_required: int) -> float | None`.
- `stats.mean(samples: list[float]) -> float`.
- `stats.stdev(samples: list[float]) -> float`.
- `stats.cv_pct(mean: float, stdev: float) -> float`.
- `stats.aggregate_noise_margin(candidate_cv_pct: float, anchor_cv_pct: float, anchor_drift_pct: float, floor: float) -> float` — spec §6.5 formula.
- `stats.anchor_drift_pct(pre: float, post: float) -> float`.
- No GPU, no I/O, no asyncio.

**`scoring.py`** — Phase 5. Public:
- `scoring.compute_objective_score(shape_results: list[ShapeBenchResult], objective: PerformanceObjective, baseline_value: float, incumbent_anchor_value: float) -> ObjectiveScore`.
- `scoring.decide_incumbent_comparison(candidate_envelope: MeasurementEnvelope, candidate_score: float, candidate_quality: list[MeasurementQualityStatus], candidate_cv_pct: float, incumbent_envelope: MeasurementEnvelope, incumbent_score: float, incumbent_cv_pct: float, anchor_drift_pct: float, guard_pct: float, noise_floor_pct: float) -> IncumbentComparison` — the **single** function that produces `IncumbentComparison` (INV-BENCH-006). `regressed_vs_incumbent` in the emitted `BenchmarkBundle` is always `(IncumbentComparison == REGRESSED)`.

**`selection.py`** — Phase 6 pre-work. Public:
- `selection.top_k_by_score(candidates: list[ScoredCandidate], k: int) -> list[ScoredCandidate]`.
- `selection.shift_potential_score(candidate: ScoredCandidate, incumbent: ScoredCandidate, hints: ShiftPotentialHints) -> float` — pre-profile signals only (INV-BENCH-007).
- `selection.top_m_by_shift_potential(candidates: list[ScoredCandidate], incumbent: ScoredCandidate, m: int, hints_per_candidate: dict[str, ShiftPotentialHints]) -> list[ScoredCandidate]`.
- `selection.build_profile_set(scoreable: list[ScoredCandidate], k: int, m: int, incumbent: ScoredCandidate, include_incumbent: bool) -> list[ScoredCandidate]` — dedup + ordering per spec §6.6.

**`profiler.py`** — ncu subprocess adapter. Public:
- `profiler.run_ncu(target_cmd: list[str], nvtx_range: str, set_name: str, replay_mode: ReplayMode, report_out: Path, timeout_s: float) -> NcuRunResult`. `target_cmd` is always `[sys.executable, "-m", "kerlever.benchmarker.profile_child", <flags...>]` (REQ-BENCH-030); the caller in `worker._run_profile_phase` is responsible for constructing it. `run_ncu` itself does not know about the profile child — it still treats `target_cmd` as an opaque argv — but the public contract pins the invariant that placeholders like `[sys.executable, "-c", "pass"]` are a spec violation and callers must not pass them.
- `profiler.parse_report(report_path: Path) -> list[RawProfileMetric]` — wraps `ncu --import --print-metrics-json <file>`.
- `profiler.normalize(raw: list[RawProfileMetric], arch: str, profiler_version: str) -> ProfileMetrics` — applies the §6.6 compact-field mapping; missing metrics → `null`, never fabricated (INV-BENCH-009).
- `profiler.resolve_unavailable_reason(err: NcuRunResult | None, hygiene: HygieneReport, semantics: AdapterIterationSemantics) -> ProfileUnavailableReason | None`.
- Internal: `_build_cmdline(...)` assembles the `ncu --target-processes all --nvtx --nvtx-include <range> --launch-count 1 --set <set> --replay-mode <mode> --export <path> -- <target_cmd...>` tuple. Behavior is unchanged; the only delta from V0 is that `<target_cmd>` is now the profile-child argv rather than the worker binary or a placeholder.

**`fault.py`** — classification + pod-health machine. Public:
- `fault.attribute(exc: BaseException | None, exit_signal: int | None, exit_code: int | None, pod_health: PodHealth) -> FaultClass`.
- `fault.update_pod_health(current: PodHealth, outcome: BatchOutcomeSignal, ambiguous_counter: int, limit: int) -> tuple[PodHealth, int]`.
- `fault.known_good_probe(lease: LeasedDevice, cfg: ProbeConfig) -> bool` — runs the in-process known-good kernel (vec_add-style) when pod is in `suspect` state before the next batch.

**`main.py`** — `python -m kerlever.benchmarker.main`. Owns the uvicorn boot:
- `main()` — parses argv, builds config, calls `create_app`, runs uvicorn. Accepts `--help`. No GPU/NVML init happens at import time of `main` itself — NVML init runs inside the FastAPI lifespan.

---

## §3 Call Graph

Only public entry-point signatures are shown; private helper calls are elided unless load-bearing. "→" means "calls".

### 3.1 Service process (FastAPI + supervisor)

```
service.handle_benchmark(req: BenchmarkBatchRequest, request: fastapi.Request) -> BenchmarkBatchResult
  ├─ env: ServiceEnv = request.app.state.env
  ├─ target = lease.parse_target(req.problem_spec.target_gpu, req.problem_spec.sm_arch)
  └─ async with env.lease_manager.acquire(target) as device:
       → supervisor.run_batch(req, device, env)
         ├─ pre_hygiene: HygieneReport = await asyncio.to_thread(
         │      telemetry.preflight, device, env.config.clock_policy)
         ├─ if pre_hygiene.reason_on_fail in HARD_GATE_SET:
         │      return supervisor._hard_gate_result(req, pre_hygiene, env)
         │
         ├─ req_path, res_path, cfg_path = env.artifact_store.stage_worker_inputs(req, env.config)
         ├─ proc: asyncio.subprocess.Process = await supervisor._spawn_worker(
         │      req_path, res_path, cfg_path, device, env.config)
         ├─ exit: WorkerExit = await supervisor._await_worker(proc, env.config.batch_timeout_s)
         ├─ result: BenchmarkBatchResult | WorkerFailure = supervisor._read_worker_result(res_path)
         ├─ post_hygiene: HygieneReport = await asyncio.to_thread(
         │      telemetry.postflight, device, pre_hygiene.telemetry)
         ├─ fault_class, batch_status = supervisor._attribute_worker_exit(exit, result)
         ├─ env.pod_health_store.update(fault_class, batch_outcome_signal(exit, result))
         └─ return supervisor._finalize(result, pre_hygiene, post_hygiene, batch_status,
                                        env.pod_health_store.snapshot())

service.handle_healthz() -> JSONResponse
  → env.health_check.run() -> HealthReport
    ├─ telemetry.probe_ready()                 # pynvml init already happened
    ├─ profiler.ncu_ready(env.config.profiler) # shell `ncu --version`, cached
    ├─ env.artifact_store.writable()
    └─ env.pod_health_store.snapshot()

service.handle_info() -> JSONResponse
  → env.device_inventory.list() -> list[DeviceInventoryEntry]
  → env.toolchain_identity.snapshot() -> ToolchainIdentity
```

### 3.2 Worker subprocess (phases 1b..7)

```
worker.main(argv: list[str]) -> NoReturn
  ├─ args: WorkerArgs = worker._parse_argv(argv)
  ├─ cfg: BenchmarkerConfig = BenchmarkerConfig.from_file(args.config_path)
  ├─ req: BenchmarkBatchRequest = worker._read_request(args.request_path)
  ├─ telemetry.init()                           # fresh NVML init inside subprocess
  ├─ cuda_driver.init()
  ├─ device: LeasedDevice = worker._resolve_leased_device(args.device_uuid)
  ├─ ctx: CudaContext = cuda_driver.create_primary_context(device.ordinal)
  │
  ├─ try:
  │    normalized: NormalizedBatch = normalize.normalize_request(req, cfg, device)
  │    loaded: list[LoadedCandidate] = worker._load_candidates(normalized.admit_candidates, ctx)
  │    incumbent: LoadedCandidate = worker._load_incumbent(req.incumbent_ref, ctx)
  │    plan_out: CalibratedPlan = plan.calibrate(loaded, req.objective_shape_cases,
  │                                              cfg.plan, worker._adapter(req))
  │    batch_meas: BatchMeasurement = harness.execute_batch(
  │         plan_out, loaded, incumbent,
  │         req.objective_shape_cases, normalized.interleave_seed_per_shape,
  │         cfg.harness, worker._telem_recorder(device))
  │    scored: list[ScoredCandidate] = worker._score_all(batch_meas, normalized, req)
  │           ├─ for each candidate:
  │           │    stats.cv_pct / p50 / p95 / mean / stdev / anchor_drift_pct
  │           │    scoring.compute_objective_score(...)
  │           │    scoring.decide_incumbent_comparison(...)
  │    profile_set: list[ScoredCandidate] = selection.build_profile_set(
  │           scored, req.top_k_profile, req.top_m_profile_shift_potential,
  │           incumbent_scored, include_incumbent=cfg.profiler.include_incumbent)
  │    profiles: dict[str, ProfileOutcome] = worker._run_profiles(
  │           profile_set, req.profile_shape_cases, plan_out, loaded)
  │           → profiler.run_ncu(...)
  │           → profiler.parse_report(...)
  │           → profiler.normalize(...)
  │    result: BenchmarkBatchResult = worker._assemble_result(
  │           req, normalized, scored, profiles, batch_meas, cfg)
  │    worker._write_result(args.result_path, result)
  │    telemetry.shutdown()
  │    cuda_driver.destroy_primary_context(ctx)
  │    os._exit(0)
  │
  ├─ except CandidateRuntimeFault as e:
  │    result = worker._assemble_partial_result(req, e)
  │    worker._write_result(args.result_path, result)
  │    os._exit(1)        # controlled candidate_fault; supervisor reads result_path
  │
  └─ except BaseException as e:
       # Uncaught — write a best-effort infra_error result, then hard-exit
       worker._write_result_best_effort(args.result_path, req, e)
       os._exit(2)        # ambiguous or infra_fault; supervisor inspects exit code
```

Key points:

- `os._exit` on every path (0 / 1 / 2). The worker never returns to the CPython atexit chain — that is the INV-BENCH-012 seam (no Python-level finalizer can observe the tainted CUDA context of this process once it exits; no PyModule unload can attempt `cuCtxDestroy` on a poisoned context).
- `_load_candidates` reads the cubin bytes from the URI, calls `cuda_driver.load_module`, then `cuda_driver.get_function`, and applies `function_attribute_policy_requested` via `cuda_driver.set_function_attribute`. Observed values come back from the same wrapper and are stored in `MeasurementEnvelope.function_attribute_policy_observed`.
- `_run_profiles` invokes the worker harness **as a separate subprocess** under `ncu` (see §4.3 IPC note). The profile subprocess is **not** the same as the benchmark worker subprocess — `ncu` re-launches the harness binary so it can attach the profiler. The measured NVTX range is still owned by the harness code; ncu just filters. Returning `ProfileStatus.PROFILE_UNAVAILABLE` with a typed reason is routed through `profiler.resolve_unavailable_reason`.

### 3.3 Adapter integration in the worker (Phase 3 / Phase 4)

Adapter calls are interleaved with cubin load and with every per-shape buffer lifecycle step. The worker owns the adapter object for the whole subprocess lifetime; the harness receives adapter-derived callables but never the adapter instance itself.

```
worker._run_phases(req, cfg, device)
  ├─ adapter: OperationAdapter = worker._resolve_adapter(req)
  │   → adapter.get_adapter(req.operation_adapter_abi, req.operation_adapter_version)
  │       → AdapterRegistry.resolve(abi_name, abi_version) -> OperationAdapter
  │       # AdapterUnregistered → write INFRA_ERROR result, os._exit(1)
  │       # AdapterVersionMismatch → write INFRA_ERROR result, os._exit(1)
  │
  ├─ loaded: list[LoadedCandidate] = worker._load_candidates(admit, ctx, policy)
  │   # see §3.4 for the function-attribute apply subgraph
  │
  ├─ plan_out: CalibratedPlan = plan.calibrate(loaded, req.objective_shape_cases,
  │                                             cfg.plan, adapter)
  │
  ├─ # Per-shape buffer allocation (spec §6.11)
  ├─ buffers_per_shape, buffer_pool_per_shape = worker._build_shape_buffers(
  │       adapter, req.objective_shape_cases ∪ req.profile_shape_cases, device,
  │       plan_out.effective_cache_policy, plan_out.iterations_per_sample_per_shape)
  │   → for shape in shapes:
  │       bufs: AdapterBuffers = adapter.allocate(shape, dtype, device)
  │       seed: int = hash((req.run_id, req.batch_id, shape.shape_id))
  │       adapter.seed_inputs(bufs, shape, seed)
  │       if plan_out.effective_cache_policy == WARM_ROTATING_BUFFERS:
  │           pool: list[AdapterBuffers] = []
  │           for _ in range(plan_out.iterations_per_sample[shape.shape_id] + 1):
  │               b = adapter.allocate(shape, dtype, device)
  │               adapter.seed_inputs(b, shape, seed)
  │               pool.append(b)
  │
  ├─ # Harness receives closures, not the adapter itself
  ├─ batch_meas = harness.execute_batch(
  │       plan_out, loaded, incumbent, req.objective_shape_cases,
  │       normalized.interleave_seed_per_shape, cfg.harness, telem,
  │       build_args=adapter.build_launch_args,
  │       reset_hook=(lambda b, sem: adapter.reset_between_iterations(b, sem))
  │                  if any_non_overwrite_pure(plan_out.adapter_iteration_semantics)
  │                  else None,
  │       rotate_hook=(lambda pool: adapter.rotate_buffers(pool))
  │                   if plan_out.effective_cache_policy == WARM_ROTATING_BUFFERS
  │                   else None,
  │       buffers_per_shape=buffers_per_shape,
  │       buffer_pool_per_shape=buffer_pool_per_shape)
  │   # Inside execute_batch → run_sample, the semantics dispatch runs per spec §6.11.
  │
  ├─ # Teardown — symmetric to allocation
  ├─ for shape_id, b in buffers_per_shape.items():
  │       adapter.free(b)
  ├─ for shape_id, pool in (buffer_pool_per_shape or {}).items():
  │       for b in pool:
  │           adapter.free(b)
  │
  └─ # `useful_bytes` and `algorithmic_flops` are consulted during Phase 7 assembly
     # via `adapter.useful_bytes(shape)` / `adapter.algorithmic_flops(shape)` to fill
     # `ShapeBenchResult.effective_bandwidth_gbps`, `achieved_flops`, and
     # `arithmetic_intensity_flop_per_byte`. No adapter call happens inside the timed
     # region — all work-model values are computed from shape + measured time.
```

### 3.4 Function-attribute apply on candidate load

Applied once per candidate at module-load time, before any calibration launch. Each apply is immediately read back; the read-back value populates `MeasurementEnvelope.function_attribute_policy_observed` (REQ-BENCH-029). Observed ≠ requested is allowed (driver may clamp) but is recorded — the envelope hash captures the observed value so two pods with different clamp behavior will not share an envelope.

```
worker._load_candidates(admit: list[CandidateArtifactRef], ctx: CudaContext,
                        policy: FunctionAttributePolicy) -> list[LoadedCandidate]
  for ref in admit:
    cubin: bytes = worker._read_cubin(ref.cubin_uri)
        # REQ-BENCH-031: cubin_uri must be absolute POSIX path, readable.
        # Relative path / URI scheme / missing file → CubinUriRejected in Phase 1
        # before this helper is even called.

    module: CudaModule = cuda_driver.load_module(cubin, ref.launch_spec.module_load_options)
    fn:     CudaFunction = cuda_driver.get_function(module, ref.launch_spec.entrypoint)

    observed: FunctionAttributePolicyObserved = {}
    try:
      if policy.max_dynamic_shared_memory_size > 0:
        v = cuda_driver.set_function_attribute(
              fn, FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
              policy.max_dynamic_shared_memory_size)
        observed["max_dynamic_shared_memory_size"] = v

      if policy.preferred_shared_memory_carveout_pct is not None:
        v = cuda_driver.set_function_attribute(
              fn, FunctionAttribute.PREFERRED_SHARED_MEMORY_CARVEOUT,
              policy.preferred_shared_memory_carveout_pct)
        observed["preferred_shared_memory_carveout_pct"] = v

      if policy.cache_config is not None:
        cuda_driver.set_cache_config(fn, policy.cache_config)
        observed["cache_config"] = cuda_driver.get_function_attribute(
              fn, FunctionAttribute.PREFERRED_CACHE_CONFIG)

      if policy.cluster_dims is not None:
        for (axis, attr) in [(policy.cluster_dims.x, REQUIRED_CLUSTER_WIDTH),
                             (policy.cluster_dims.y, REQUIRED_CLUSTER_HEIGHT),
                             (policy.cluster_dims.z, REQUIRED_CLUSTER_DEPTH)]:
          cuda_driver.set_function_attribute(fn, attr, axis)
        observed["cluster_dims"] = policy.cluster_dims   # + read-back

    except CudaDriverError as e:
      raise FunctionAttributePolicyApplyFailed(
            candidate_hash=ref.candidate_hash, driver_error=e)
      # Caller converts to CandidateRuntimeFault(fault_class=infra_fault,
      #   reason="function_attribute_policy_apply_failed"); other candidates proceed.

    loaded.append(LoadedCandidate(
        candidate_hash=ref.candidate_hash,
        module=module, fn=fn,
        function_attribute_policy_requested=policy,
        function_attribute_policy_observed=observed))

  return loaded
```

### 3.5 Profile-child lifecycle (Phase 6)

The profile child is a separate Python process spawned by NCU, not by the worker. The worker's role is to build the NCU command line whose `-- <target_cmd>` slot points at `python -m kerlever.benchmarker.profile_child …`. NCU then forks the profile child, attaches its profiler to it, and times exactly one NVTX-ranged launch.

```
# Parent side — runs inside the worker subprocess
worker._run_profile_phase(profile_set, profile_shape_cases, plan_out, loaded, req, cfg)
  for (candidate, profile_shape) in cartesian(profile_set, profile_shape_cases):
    range_name: str = worker._build_nvtx_range(
        req.run_id, req.batch_id, candidate.candidate_hash, profile_shape.shape_id)

    target_cmd: list[str] = [
        sys.executable, "-m", "kerlever.benchmarker.profile_child",
        "--cubin-path",            candidate.cubin_path,
        "--entrypoint",            candidate.launch_spec.entrypoint,
        "--block-dim",             _fmt_triple(candidate.launch_spec.block_dim),
        "--grid-dim",              _fmt_triple(plan_out.grid_dim_per_shape[profile_shape.shape_id]),
        "--dynamic-smem-bytes",    str(candidate.launch_spec.dynamic_smem_bytes),
        "--adapter-abi",           req.operation_adapter_abi,
        "--adapter-version",       req.operation_adapter_version,
        "--shape-dims",            _fmt_dims(profile_shape.dims),
        "--shape-dtype",           profile_shape.dtype,
        "--function-attr-max-smem",
            str(candidate.envelope.function_attribute_policy_observed.max_dynamic_shared_memory_size),
        "--function-attr-carveout-pct",
            str(candidate.envelope.function_attribute_policy_observed.preferred_shared_memory_carveout_pct or 0),
        "--function-attr-cache-config",
            candidate.envelope.function_attribute_policy_observed.cache_config or "prefer_none",
        "--iterations-per-sample", str(plan_out.iterations_per_sample_per_shape[profile_shape.shape_id]),
        "--warmup-count",          str(plan_out.warmup_count_per_shape[profile_shape.shape_id]),
        "--nvtx-range",            range_name,
        "--seed",                  str(hash((req.run_id, req.batch_id, profile_shape.shape_id))),
    ]

    ncu_res: NcuRunResult = profiler.run_ncu(
        target_cmd=target_cmd,
        nvtx_range=range_name,
        set_name=cfg.profiler.set_name,
        replay_mode=cfg.profiler.replay_mode,
        report_out=cfg.artifact.ncu_path(req, candidate, profile_shape),
        timeout_s=cfg.profiler.profile_timeout_s,
    )
    # profiler.run_ncu returns; NCU has already reaped the profile child.

    if ncu_res.returncode != 0 or ncu_res.timed_out or ncu_res.report_path is None:
      reason = profiler.resolve_unavailable_reason(
          ncu_res, hygiene, plan_out.adapter_iteration_semantics_per_candidate[candidate.candidate_hash])
      outcomes[candidate.candidate_hash] = ProfileOutcome.unavailable(reason)
      continue

    raw: list[RawProfileMetric] = profiler.parse_report(ncu_res.report_path)
    metrics: ProfileMetrics     = profiler.normalize(raw, arch, profiler_version)
    outcomes[candidate.candidate_hash] = ProfileOutcome.present(metrics, ncu_res.report_path)

  return outcomes

# Child side — runs inside the NCU-supervised profile child
profile_child.main(argv: list[str]) -> NoReturn
  args = profile_child._parse_argv(argv)
  try:
    cuda_driver.init()
    ctx:    CudaContext = cuda_driver.create_primary_context(device_ordinal=0)
        # CUDA_VISIBLE_DEVICES is set by the parent to pin GPU#0 to the leased device.
    cubin:  bytes       = Path(args.cubin_path).read_bytes()
    module: CudaModule  = cuda_driver.load_module(cubin, options=None)
    fn:     CudaFunction = cuda_driver.get_function(module, args.entrypoint)

    # Apply function-attribute policy (same apply sequence as worker, no read-back storage).
    if args.function_attr_max_smem > 0:
      cuda_driver.set_function_attribute(
          fn, FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, args.function_attr_max_smem)
    if args.function_attr_carveout_pct:
      cuda_driver.set_function_attribute(
          fn, FunctionAttribute.PREFERRED_SHARED_MEMORY_CARVEOUT, args.function_attr_carveout_pct)
    if args.function_attr_cache_config:
      cuda_driver.set_cache_config(fn, args.function_attr_cache_config)
    # On any CudaDriverError above → os._exit(11)  (EX_FUNC_ATTR_APPLY_FAILED)

    # Registry is re-seeded inside the child; V1 built-ins auto-register on adapter import.
    adapter.AdapterRegistry.register_builtin_adapters()
    op = adapter.get_adapter(args.adapter_abi, args.adapter_version)
    # On AdapterUnregistered → os._exit(12)  (EX_ADAPTER_UNREGISTERED)

    shape:  ShapeCase     = ShapeCase(dims=args.shape_dims, dtype=args.shape_dtype)
    bufs:   AdapterBuffers = op.allocate(shape, args.shape_dtype, device=<leased>)
    op.seed_inputs(bufs, shape, args.seed)

    launch_args = op.build_launch_args(bufs, shape)
    stream: CudaStream    = cuda_driver.create_stream()

    # Warmup — OUTSIDE any NVTX range
    for _ in range(args.warmup_count):
      cuda_driver.launch(fn, args.grid_dim, args.block_dim,
                         args.dynamic_smem_bytes, stream, launch_args)
    cuda_driver.stream_synchronize(stream)

    # Timed range — exactly `iterations_per_sample` launches wrapped in ONE NVTX push/pop
    nvtx.push_range(args.nvtx_range)
    for _ in range(args.iterations_per_sample):
      cuda_driver.launch(fn, args.grid_dim, args.block_dim,
                         args.dynamic_smem_bytes, stream, launch_args)
    nvtx.pop_range()

    cuda_driver.stream_synchronize(stream)
    op.free(bufs)
    cuda_driver.destroy_primary_context(ctx)
    os._exit(0)

  except CudaDriverError:
    os._exit(1)                       # launch / sync failed inside measured loop
  except BaseException:
    os._exit(2)                       # uncaught; ambiguous fault
```

Notes:
- The profile child honors INV-BENCH-013 — if the adapter semantics for this candidate is not `OVERWRITE_PURE`, the child must invoke `op.reset_between_iterations` / `op.rotate_buffers` between timed iterations. For V1 adapters (`elementwise_add_fp32_v1`, `matmul_fp16_v1`), semantics is `OVERWRITE_PURE` and no reset hook runs in the measured loop. The child code path still carries the same dispatch machinery for forward compatibility with future adapters.
- The profile child's CUDA context is distinct from the worker's Phase 3/4 context; they share the physical GPU, not the context. That isolation is what allows an NCU replay failure to surface as `profile_unavailable` without corrupting the fast-bench result.
- No measurement value travels from the child back to the parent. The NVTX range + `.ncu-rep` file are the entire communication channel, and the parent reads the `.ncu-rep` via `profiler.parse_report`.

### 3.6 Supervisor-side helpers

```
supervisor._spawn_worker(req_path, res_path, cfg_path, device, config) -> asyncio.subprocess.Process
  → asyncio.create_subprocess_exec(
       sys.executable, "-m", "kerlever.benchmarker.worker",
       "--config", str(cfg_path),
       "--request-file", str(req_path),
       "--result-file", str(res_path),
       "--device-uuid", device.gpu_uuid,
       "--device-ordinal", str(device.ordinal),
       env=supervisor._child_env(config, device),
       stdin=asyncio.subprocess.DEVNULL,
       stdout=asyncio.subprocess.PIPE,   # log-only; result does NOT travel here
       stderr=asyncio.subprocess.PIPE,   # log-only
    )

supervisor._await_worker(proc, timeout_s) -> WorkerExit
  → async with asyncio.timeout(timeout_s):
       stdout, stderr = await proc.communicate()
  # On timeout:
  #   proc.terminate(); await sleep 5s; proc.kill(); return WorkerExit(timed_out=True, ...)

supervisor._read_worker_result(res_path) -> BenchmarkBatchResult | WorkerFailure
  → text = res_path.read_text() if res_path.exists() else ""
  → BenchmarkBatchResult.model_validate_json(text)   # Pydantic parse
  # If res_path missing or parse fails → WorkerFailure with classified reason.

supervisor._attribute_worker_exit(exit, result) -> tuple[BatchStatus, FaultClass]
  → fault.attribute(None, exit.signal, exit.returncode, env.pod_health_store.current())
  → BatchStatus per spec §6.7:
       exit.timed_out                   → BatchStatus.TIMEOUT, FaultClass.AMBIGUOUS_FAULT
       exit.returncode == 0             → from result.status
       exit.returncode == 1             → BatchStatus.PARTIAL, FaultClass from result
       exit.returncode == 2 or signal   → BatchStatus.INFRA_ERROR, FaultClass.AMBIGUOUS_FAULT
```

---

## §4 Process Lifecycle and IPC

### 4.1 FastAPI service process

Holds:
- `BenchmarkerConfig` (frozen dataclass).
- `DeviceInventory` (list of `DeviceInventoryEntry` cached from a single `telemetry.info_inventory()` call at startup).
- `LeaseManager` (dict of `asyncio.Semaphore(1)` keyed on `gpu_uuid`).
- `PodHealthStore` (in-memory: `(pod_health, ambiguous_counter, last_probe_at)`).
- `ArtifactStore` (wraps `ARTIFACT_ROOT` on disk — writable directory mounted on the pod).
- `ToolchainIdentity` (driver/runtime/cuda-python/pynvml/ncu versions, captured once at startup).

Does **not** hold:
- Any CUDA context. `cuda-python` is **not imported** from `service.py`; it is imported only from `worker.py` (and transitively `cuda_driver.py`). This is an INV-BENCH-012 defense at the import level.
- Any live-per-request pynvml handle. The inventory is captured once; per-request preflight opens/uses/closes its own handles under `asyncio.to_thread` (pynvml is not thread-safe across arbitrary concurrent use, but our preflight is always serialized behind the per-GPU semaphore, so at most one `asyncio.to_thread` preflight is active per device at any time).

Lifecycle:
1. Container start → uvicorn starts → FastAPI lifespan startup hook runs (see §8 init order).
2. Lifespan shutdown hook runs on SIGTERM: stop accepting new `/benchmark` requests (uvicorn's default graceful drain), wait for in-flight batches up to `SHUTDOWN_GRACE_S`, then signal any still-running worker subprocesses SIGTERM → SIGKILL past grace, finally `telemetry.shutdown()`.

### 4.2 Batch supervisor

Lives in the service process. Its lifetime is one FastAPI request: it creates exactly one worker subprocess, awaits its exit, collects the result, and returns. It does **not** pool or reuse workers (REQ-BENCH-025, INV-BENCH-012).

Worker process creation uses `asyncio.create_subprocess_exec` with the Python module entry `python -m kerlever.benchmarker.worker` so that `sys.argv[0]` is a real module path rather than a script path — this keeps the Dockerfile simple (no separate entry script for the worker) and makes the entire worker importable as a normal module for linting.

The supervisor's `asyncio.timeout(batch_timeout_s)` wraps only the `await proc.communicate()` call; it does **not** wrap the pre/post hygiene or the result-file read, because those are already bounded (`telemetry.preflight` is a small fixed set of NVML calls, and reading a result file is a single I/O).

On timeout, the supervisor escalates SIGTERM → SIGKILL with a 5-second grace. After SIGKILL it still attempts to read `res_path` in case the worker flushed a partial result before being killed (best-effort PARTIAL).

Signal handling:
- Supervisor installs **no** signal handlers of its own; uvicorn owns SIGTERM/SIGINT at the process level, and the lifespan shutdown hook is the drain coordination point.
- Child process reaping is done by `asyncio.subprocess`, which reads `returncode` and `signal` after `communicate()`.

Fault attribution from child exit:

| Observable | Attribution | BatchStatus |
|---|---|---|
| `returncode == 0` and result parses | from `result.status` | inherit |
| `returncode == 0` and result missing/unparseable | `ambiguous_fault` | `INFRA_ERROR` |
| `returncode == 1` and result parses (partial) | inherit per-candidate `fault_class` | `PARTIAL` |
| `returncode == 2` or any nonzero besides 1 | `ambiguous_fault` | `INFRA_ERROR` |
| Killed by signal (SIGSEGV/SIGBUS/SIGKILL from OOM-kill) | `ambiguous_fault` | `INFRA_ERROR` |
| `asyncio.TimeoutError` from `communicate()` | `ambiguous_fault` | `TIMEOUT` |

### 4.3 Worker subprocess

Argv format:

```
python -m kerlever.benchmarker.worker
   --config           /tmp/bench/<request_id>/config.json
   --request-file     /tmp/bench/<request_id>/request.json
   --result-file      /tmp/bench/<request_id>/result.json
   --device-uuid      GPU-abc-...
   --device-ordinal   0
```

Why files and not pipes:
- The worker later re-launches itself (or a harness binary) under `ncu`, whose stdout/stderr get mixed with profiler diagnostics. Parsing a typed `BenchmarkBatchResult` JSON out of that mixed stream is fragile. A dedicated result file is unambiguous and survives even if ncu prints warnings to stdout.
- The result file also allows "best effort PARTIAL" on abnormal exit: if the worker had time to flush before being killed, the supervisor still reads it.
- Framing alternative (BEGIN_RESULT/END_RESULT sentinels on stdout) was considered and rejected: ncu can emit arbitrary text to stdout including lines that start with arbitrary tokens. A file gives us correctness for free.

IPC contract:
- **Input**: `request.json` is `BenchmarkBatchRequest.model_dump_json()`; `config.json` is `BenchmarkerConfig.to_dict()` serialized. Both are written by the supervisor before `_spawn_worker` and read by the worker at startup.
- **Output**: `result.json` is `BenchmarkBatchResult.model_dump_json()` written atomically (write to `result.json.tmp`, then rename) just before `os._exit`.
- **Diagnostics**: stdout and stderr of the worker are captured via `proc.communicate()` and logged at the supervisor with structured key/value annotations (run_id, batch_id, returncode). They never carry result payload.

Lifecycle inside the worker:
1. Parse argv.
2. NVML init (fresh in this process — pynvml keeps per-process state; the service-process NVML init does not carry over).
3. `cuda_driver.init()` → `cuCtxCreate` on the leased device ordinal. Exactly one primary CUDA context for this subprocess (INV-BENCH-002).
4. Load each candidate cubin via `cuModuleLoadDataEx`; resolve entrypoint via `cuModuleGetFunction`.
5. Apply function-attribute policy; read observed values back.
6. Phases 3 → 4 → 5 → 6 as per call graph §3.2.
7. On success or caught `CandidateRuntimeFault`: write `result.json`, `cuCtxDestroy`, `telemetry.shutdown`, `os._exit(0 or 1)`.
8. On uncaught `BaseException`: best-effort write of an `INFRA_ERROR` result, skip clean CUDA teardown, `os._exit(2)`. Skipping clean teardown here is intentional: if the context is poisoned, `cuCtxDestroy` can hang; we want the OS to reap the process.

### 4.4 Profile child subprocess

The profile child is a Python process spawned by NCU during Phase 6, one invocation per `(candidate, profile_shape)`. It sits two process boundaries away from the service:

```
service (PID A) → worker (PID B) → ncu (PID C) → profile_child (PID D)
                     holds lease         supervises
```

- PID A (service): holds the per-GPU semaphore via `LeaseManager`; no CUDA context.
- PID B (worker): holds the fast-bench CUDA context; runs `subprocess.run("ncu …")` sequentially per profile target.
- PID C (ncu): NVIDIA Nsight Compute CLI binary. NCU is the **supervisor** of the profile child — it forks PID D, attaches its profiler, watches for timeout, and collects the `.ncu-rep`. The worker (PID B) is NCU's parent, but the worker is **not** the parent of the profile child.
- PID D (profile_child): runs `kerlever.benchmarker.profile_child.main(argv)`; holds its own short-lived CUDA context distinct from PID B's.

**Argv contract** for the profile child (consumed by `profile_child._parse_argv`; spec §6.12 is the behavior of record):

```
python -m kerlever.benchmarker.profile_child
   --cubin-path                  <abs posix path to the cubin>
   --entrypoint                  <kernel entrypoint symbol>
   --block-dim                   <bx,by,bz>
   --grid-dim                    <gx,gy,gz>
   --dynamic-smem-bytes          <int>
   --adapter-abi                 <abi_name>
   --adapter-version             <abi_version>
   --shape-dims                  <d0,d1,...>
   --shape-dtype                 <str>
   --function-attr-max-smem      <int>
   --function-attr-carveout-pct  <int 0..100>
   --function-attr-cache-config  <prefer_none|prefer_shared|prefer_l1|prefer_equal>
   --iterations-per-sample       <int>
   --warmup-count                <int>
   --nvtx-range                  <str>
   --seed                        <int>
```

Design rationale for flat argv rather than file-based IPC:

- The profile child is invoked through `ncu -- <target_cmd>`. NCU's argument layer is a simple argv passthrough; it does not propagate files for us. A flat argv lets NCU serve as the supervisor without any file-shuttling logic in the worker.
- The same `config.json` / `request.json` files that the worker already owns remain the worker's private inputs; we do **not** introduce a new IPC file for the profile child. The worker extracts the subset of fields the child needs and embeds them as argv flags. This keeps the child dependency-free of the full request schema and keeps its startup under 100 ms.
- All argv values are serializable scalars. No bytes traverse argv — the cubin is read from the filesystem by the child (REQ-BENCH-031 pins `cubin_uri` to an absolute POSIX path that both the worker and child can open).

**Supervisor topology.** NCU is the supervisor of the profile child; the worker is NCU's parent. The worker invokes NCU via `subprocess.run(...)` (blocking, synchronous), not `asyncio.create_subprocess_exec`, because the worker itself is synchronous (§5.3) and Phase 6 is strictly serial per profile target. The worker does not wait on the profile child's PID directly and does not install a signal handler for it; NCU owns that relationship end-to-end.

**Exit discipline.** The profile child uses `os._exit(returncode)` on every termination path (same INV-BENCH-012 rationale as the worker): the child's CUDA context is tainted after any driver error, and running Python interpreter finalizers would risk `cuCtxDestroy` on a poisoned context. Exit codes map to the spec §6.12 failure table:

| Exit | Meaning | Parent (worker) interpretation |
|---|---|---|
| 0 | success; `.ncu-rep` present | `ProfileStatus.PRESENT`; proceed to `parse_report` |
| 1 | CUDA launch or copy failure during measured loop | `profile_unavailable`, reason `profiler_replay_refused` |
| 2 | uncaught Python exception | `profile_unavailable`, reason `profiler_replay_refused` |
| 11 | `EX_FUNC_ATTR_APPLY_FAILED` | `profile_unavailable`, reason `profiler_replay_refused` |
| 12 | `EX_ADAPTER_UNREGISTERED` | `profile_unavailable`, reason `profiler_replay_refused` (child is mis-seeded) |
| 137 / 139 / signal | killed by NCU (timeout) or SIGSEGV | NCU surfaces `timed_out=True` or nonzero `returncode`; worker maps via `profiler.resolve_unavailable_reason` |

**Timeout.** The profile child runs inside the `timeout_s` budget passed to `profiler.run_ncu` (env var `KERLEVER_BENCH_PROFILE_TIMEOUT_S`, default 300 s). NCU is the enforcer: on timeout NCU kills the child and returns nonzero. The worker does not install its own timeout on the child PID.

**Isolation from fast-bench state.** The profile child has no access to the worker's Phase 4 sample arrays, calibration plan, or scoring state. All Phase 4 measurements are durably computed before Phase 6 begins (spec §6.6); a profile-child crash therefore cannot corrupt fast-bench results — the worst case is `profile_unavailable` for that `(candidate, shape)`.

### 4.5 Handoff summary

```
service.handle_benchmark
  → supervisor.run_batch
      → (writes request.json, config.json)
      → spawn worker subprocess    ─── child process ────┐
      → await proc.communicate()                         │
      ← result.json (Pydantic-parsed BenchmarkBatchResult)
      → finalize, return to FastAPI
  ← BenchmarkBatchResult (HTTP 200)
```

Nothing from the worker's in-memory Python state flows back to the service process except via `result.json`. This is the isolation seam.

---

## §5 Concurrency Model

### 5.1 Per-GPU serialization

Rule: at most one timed benchmark batch is active per leased physical GPU at any time (INV-BENCH-003).

Implementation:
- `LeaseManager` holds a `dict[str, asyncio.Semaphore]` keyed on `gpu_uuid` (MIG instances use the MIG id as the key; bare GPUs use the physical UUID).
- `LeaseManager.acquire(target)` is an async context manager: `await semaphore.acquire()` on enter, `semaphore.release()` on exit. Entry/exit span Phase 2 through Phase 7 on that device.
- Batches targeting **different** GPUs run in parallel (different semaphore keys). The service can saturate a multi-GPU pod.
- Batches targeting the **same** GPU serialize at the semaphore; FIFO by asyncio's `acquire` ordering.

### 5.2 Service-process concurrency

The FastAPI app is asyncio-native. Each `/benchmark` request is one coroutine running `supervisor.run_batch`. The asyncio event loop multiplexes many in-flight coroutines; the per-GPU semaphore is the single serialization point.

Uvicorn is configured with `workers=1` — a single process running a single event loop. Multi-worker uvicorn would mean multiple independent `LeaseManager` instances, which would break the per-GPU serialization guarantee across workers. On a single-pod, single-GPU-set deployment, one worker is sufficient and correct.

**Lease scope over the profile child.** The profile child subprocess (§4.4) executes inside Phase 6, which is inside the worker subprocess, which is inside the `async with env.lease_manager.acquire(target)` block held by the service coroutine. The service has not released the per-GPU semaphore when NCU or the profile child runs — the lease spans Phase 2 through Phase 7 inclusive. Concretely:

- The profile child does **not** acquire its own semaphore. It inherits serialization from its grandparent (service) via the still-held `asyncio.Semaphore`.
- NCU runs as a child of the worker; the profile child runs as a child of NCU. Neither NCU nor the profile child opens pynvml or `LeaseManager` — they trust the service's lease to have serialized GPU access before they started.
- At most one profile child (for a single candidate/profile_shape) runs at any time on the leased device. Parallel profiling across candidates on the same device is explicitly forbidden by this lease topology.
- The lease is released by the service coroutine only after `supervisor._finalize`, which happens after `proc.communicate()` returns, which happens after the worker exits, which happens after all profile children have been reaped by NCU. The chain is strict-linear; no concurrent GPU access is possible under this design.

### 5.3 Threading and the GIL inside the worker

Inside the worker subprocess:
- There is no asyncio. The worker is a plain synchronous Python program from end to end.
- `cuda-python` is a C extension binding to the CUDA driver. Its blocking calls (`cuEventSynchronize`, `cuMemcpy`, `cuLaunchKernel` in some paths) release the GIL. We still run everything on the main thread because the harness is single-threaded by design: interleaved sampling, NVTX ranges, and CUDA events must happen in a well-defined order that a threadpool would only obscure.
- Rationale for no-asyncio-in-worker: asyncio buys us concurrency between *I/O-bound* tasks; the worker is CPU-and-GPU bound. Introducing asyncio inside the worker would add complexity (event loop lifecycle, cancellation semantics colliding with CUDA context) without measurable benefit.

### 5.4 NCU subprocess scheduling

`ncu` is invoked via `subprocess.run` (blocking, synchronous) from the worker, once per `(candidate, profile_shape)`. It runs serially with respect to the worker's main flow: the worker finishes Phase 4, computes Phase 5 decisions, then enters Phase 6 and drives `ncu` sequentially.

`ncu` cannot run alongside the timed sampling: the profiler replay of a kernel is not a "normal" run and would invalidate timing. By construction, Phase 6 starts only after Phase 4 has produced all fast-benchmark samples for all candidates.

Concurrency on the same GPU during profile phase: none. The worker holds the lease; `ncu` runs as a child of the worker on the same physical device; only one kernel is measured at a time.

### 5.5 Service shutdown

Uvicorn receives SIGTERM → begins graceful shutdown:
1. Stops accepting new HTTP connections.
2. Waits for in-flight requests up to `SHUTDOWN_GRACE_S` (default ~30 s, env-configurable).
3. The FastAPI lifespan shutdown hook (registered in `service.create_app`):
   - Iterates any tracked active `supervisor` tasks (weakref set) and attempts to wind down their worker subprocesses: SIGTERM to child → wait up to 5 s → SIGKILL.
   - Calls `telemetry.shutdown()` (NVML deinit in the service process).
4. Uvicorn exits with code 0.

If SIGKILL is sent to the pod outright, the kernel reaps the service process and any worker subprocesses become orphans that the kernel also reaps. No CUDA context survives across pod restart.

---

## §6 External Adapters

Four external systems: `cuda-python` driver API, `pynvml`, `ncu` CLI, `nsys` CLI. Each has a thin Python wrapper module. The wrapper surfaces are small (≤10 public functions) and mechanical — all measurement logic and decision logic lives in `plan.py` / `harness.py` / `scoring.py` / `selection.py` / `profiler.py`, not inside the wrappers.

### 6.1 `cuda_driver.py` (wraps `cuda-python`)

```
cuda_driver.init() -> None
cuda_driver.create_primary_context(device_ordinal: int) -> CudaContext
cuda_driver.destroy_primary_context(ctx: CudaContext) -> None
cuda_driver.load_module(cubin: bytes, options: ModuleLoadOptions | None) -> CudaModule
cuda_driver.get_function(module: CudaModule, entrypoint: str) -> CudaFunction
cuda_driver.set_function_attribute(fn: CudaFunction, attr: FunctionAttribute, value: int) -> int
cuda_driver.get_function_attribute(fn: CudaFunction, attr: FunctionAttribute) -> int
cuda_driver.set_cache_config(fn: CudaFunction, cfg: CacheConfig) -> None
cuda_driver.mem_alloc(bytes_: int) -> DevicePtr
cuda_driver.mem_free(ptr: DevicePtr) -> None
cuda_driver.memcpy_htod(ptr: DevicePtr, host_bytes: bytes) -> None
cuda_driver.memcpy_dtoh(ptr: DevicePtr, nbytes: int) -> bytes
cuda_driver.launch(fn: CudaFunction, grid, block, smem, stream, args) -> None
cuda_driver.event_record(event: CudaEvent, stream: CudaStream) -> None
cuda_driver.event_elapsed_ms(start: CudaEvent, stop: CudaEvent) -> float
cuda_driver.stream_synchronize(stream: CudaStream) -> None
# plus lifecycle helpers: create_event / destroy_event / create_stream / destroy_stream
```

`DevicePtr` is an opaque int-like wrapper around a `cuda-python` device pointer. Adapters carry it in `AdapterBuffers.device_ptrs`; no module outside `adapter.py`, `worker._load_candidates`, and `profile_child.main` constructs or frees a `DevicePtr`. `FunctionAttribute` is a StrEnum mirroring the `CU_FUNC_ATTRIBUTE_*` subset in use: `MAX_DYNAMIC_SHARED_SIZE_BYTES`, `PREFERRED_SHARED_MEMORY_CARVEOUT`, `REQUIRED_CLUSTER_WIDTH`, `REQUIRED_CLUSTER_HEIGHT`, `REQUIRED_CLUSTER_DEPTH`, `PREFERRED_CACHE_CONFIG`.

Contract:
- Inputs are native Python types (`bytes`, `int`, `tuple`); outputs are opaque handle types (`CudaContext`, `CudaModule`, `CudaFunction`, `CudaEvent`, `CudaStream`) that are opaque dataclass wrappers around the `cuda-python` handles. Callers never touch raw ctypes.
- Error behavior: any nonzero CUresult raises `CudaDriverError(code: int, symbol: str, message: str)`. Callers catch this at the phase boundary:
  - Caught in `harness.run_sample` → per-sample `runtime_fault`, classified by `fault.attribute`.
  - Caught in worker top-level → `CandidateRuntimeFault` (if isolated to a candidate's execution) or uncaught (→ `os._exit(2)`).
- No retries inside the wrapper. No logging inside the wrapper.

### 6.2 `telemetry.py` (wraps `pynvml`)

```
telemetry.init() -> None
telemetry.shutdown() -> None
telemetry.info_inventory() -> list[DeviceInventoryEntry]
telemetry.preflight(lease: LeasedDevice, policy: ClockPolicyConfig) -> HygieneReport
telemetry.snapshot(lease: LeasedDevice) -> DeviceTelemetrySnapshot
telemetry.postflight(lease: LeasedDevice, pre: DeviceTelemetrySnapshot) -> tuple[DeviceTelemetrySnapshot, AnchorDriftTelemetry]
telemetry.probe_ready() -> bool
```

Contract:
- All functions are synchronous and blocking. They are called from either the asyncio event loop via `asyncio.to_thread` (in the service process) or directly from the worker (in the subprocess).
- Each opens, uses, and closes pynvml handles within the call. No handles are stored as module globals beyond what `pynvml.nvmlInit` implies.
- Error behavior: pynvml exceptions are translated to `NvmlAdapterError(code: int, message: str)`. The preflight function catches these exceptions itself and maps them to `HygieneReport.reason_on_fail` values per spec §6.2; it never raises to the caller for *expected* hygiene failures (throttle, foreign process). Only NVML-library-level failures (e.g., `NVML_ERROR_DRIVER_NOT_LOADED`) propagate as `NvmlAdapterError`.
- `preflight` internally runs a `ncu --version` once at first call per process to resolve profiler counter permission (cached for the lifetime of the process) — this is the single place where `telemetry` reaches into `profiler` for permission probing. Justification: spec §6.2 bundles the permission check with preflight hygiene.

### 6.3 `profiler.py` (wraps `ncu` CLI and parses output)

```
profiler.run_ncu(target_cmd: list[str], nvtx_range: str, set_name: str, replay_mode: ReplayMode, report_out: Path, timeout_s: float) -> NcuRunResult
profiler.parse_report(report_path: Path) -> list[RawProfileMetric]
profiler.normalize(raw: list[RawProfileMetric], arch: str, profiler_version: str) -> ProfileMetrics
profiler.resolve_unavailable_reason(err: NcuRunResult | None, hygiene: HygieneReport, semantics: AdapterIterationSemantics) -> ProfileUnavailableReason | None
profiler.ncu_ready(cfg: ProfilerConfig) -> bool
profiler.ncu_version(cfg: ProfilerConfig) -> str | None
```

Contract:
- `run_ncu` wraps `subprocess.run`, passing a constructed argv. It returns `NcuRunResult { returncode: int, stdout: str, stderr: str, report_path: Path | None, timed_out: bool }`. It never raises; callers inspect the result.
- `parse_report` wraps `ncu --import <report> --print-metrics-json`; parses the JSON into a list of `RawProfileMetric`; missing metrics are `RawProfileMetric(value=None, ...)`, never skipped (INV-BENCH-009). Fields `profiler_name = "ncu"`, `profiler_version` from `ncu --version`, `architecture` from the report header.
- `normalize` applies the spec §6.6 compact-field mapping. When a source metric is missing or `value == None`, the compact field is `None` and its `NormalizedProfileMetricProvenance` still records the intended `source_metrics` + `architecture` + `profiler_version` + `comparable_across_arch = False`.
- `resolve_unavailable_reason` returns the typed `ProfileUnavailableReason` for a given failure shape (timeout, permission denied, arch mismatch, replay refused, adapter semantics, binary missing).

### 6.4 `nsys` CLI (out-of-scope default; helper in `profiler.py`)

Nsight Systems is **not** a default profiler. Per REQ-BENCH-022 it runs only on trigger. The helper lives alongside `profiler.run_ncu`:

```
profiler.run_nsys(target_cmd: list[str], report_out: Path, timeout_s: float) -> NsysRunResult
profiler.nsys_version(cfg: ProfilerConfig) -> str | None
```

Same error-returning (not -raising) shape as `run_ncu`. Not called anywhere in the default Phase 6 path; a future trigger plugin can invoke it.

---

## §7 Dockerfile and Container Lifecycle

### 7.1 Base image

`nvidia/cuda:12.4.1-devel-ubuntu24.04`.

Constraints that drove this choice:
- `ncu` and `nsys` are included in CUDA **devel** variants but not in **runtime** variants. Spec §6 and SC-BENCH-006/SC-BENCH-010 require `ncu` at runtime, so runtime-only is insufficient and devel is the smallest variant that includes it.
- **Ubuntu 24.04 (Noble), not 22.04 (Jammy).** The Benchmarker requires Python 3.12 (matches the repo's toolchain pin and the strict-typing settings in `kerlever/`). The default Jammy APT repositories ship Python 3.10 and do **not** ship a `python3.12` package; sourcing 3.12 on Jammy requires the `ppa:deadsnakes/ppa` PPA, which adds an external reproducibility dependency and an extra apt layer. Noble (24.04) ships `python3.12` in `universe` directly via `apt-get install python3.12 python3.12-venv`, keeping the image reproducible with only NVIDIA's and Canonical's first-party repos.
- **Fallback if `nvidia/cuda:12.4.1-devel-ubuntu24.04` is unavailable in NVIDIA's published image set.** Use `nvidia/cuda:12.4.1-devel-ubuntu22.04` and add the deadsnakes PPA: `apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.12 python3.12-venv`. Document the PPA dependency in the deploy note if this fallback is taken; it is intentionally second-choice.
- Ubuntu 24.04 (glibc 2.39) is compatible with the NVIDIA driver matrix ≥ 550.x, which is the minimum driver required by CUDA 12.4 anyway, so there is no driver-compat regression from moving off 22.04.
- CUDA 12.4.x matches `cuda-python>=12.3` minor compatibility; moving to 12.5 would complicate the `cuda-python` pin. If the deployed driver is older than 12.4 the container's `cuda-python` bindings may raise `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` — the README (deploy note) documents the minimum driver version.

Licensing note: the `nvidia/cuda:*-devel` image ships `ncu` under the CUDA Toolkit EULA. Deployments must accept the EULA via the NVIDIA Container Registry terms (automatic for `nvcr.io` pulls). No additional licensing action is needed beyond those accepted at registry-pull time.

Image size note: the devel image is ~5 GB vs. ~2 GB for runtime. Size is acceptable because (a) the image is pulled once per pod, (b) `ncu` is irreducibly in the devel variant, and (c) the measurement correctness benefit of real profiling dominates storage.

### 7.2 Layer ordering

Optimized for docker-build cache efficiency (system deps seldom change; python deps change occasionally; source changes often).

```
# Layer 1 — base (external, cached)
FROM nvidia/cuda:12.4.1-devel-ubuntu24.04

# Layer 2 — system deps (rare change)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip \
      ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Layer 3 — non-root user (rare change)
RUN groupadd --gid 1000 bench \
 && useradd  --uid 1000 --gid 1000 --create-home --shell /bin/bash bench

# Layer 4 — python deps (moderate change)
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
 && python3.12 -m pip install --no-cache-dir \
      "fastapi>=0.110" "uvicorn[standard]>=0.27" \
      "cuda-python>=12.3" "pynvml>=11.5" "numpy>=1.26" "pydantic>=2.5"

# Layer 5 — source (frequent change)
COPY kerlever/ /app/kerlever/
COPY docker/benchmarker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
 && chown -R bench:bench /app

# Layer 6 — runtime mount point for the shared cubin + ncu-rep store.
# REQ-BENCH-031 pins cubin_uri to an absolute POSIX path on a volume shared
# with the Compiler Service; declaring VOLUME here documents that requirement
# and guarantees the mount point exists inside the image.
RUN mkdir -p /srv/kerlever/artifacts \
 && chown -R bench:bench /srv/kerlever
VOLUME ["/srv/kerlever/artifacts"]

USER bench
EXPOSE 8080

# Layer 7 — container-level readiness probe. Kubernetes probes supersede this
# in production deployments, but the baked-in HEALTHCHECK keeps plain `docker run`
# and docker-compose healthy-status signals accurate.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD curl -fsS http://localhost:8080/healthz || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
```

### 7.3 Entrypoint script

`docker/benchmarker/entrypoint.sh`:

```
#!/usr/bin/env bash
set -euo pipefail
exec python3.12 -m uvicorn kerlever.benchmarker.main:app \
     --host 0.0.0.0 --port 8080 --workers 1 --lifespan on
```

`exec` replaces the shell process with uvicorn so that signal handling is direct from PID 1 container semantics. `--workers 1` is required for correct per-GPU serialization (§5.2). `--lifespan on` ensures the FastAPI startup/shutdown hooks run.

### 7.4 Runtime expectations

Documented in README deploy note:

| Runtime flag | Purpose | Consequence if missing |
|---|---|---|
| `--gpus all` | NVIDIA Container Toolkit passes the GPUs through | NVML init fails; `/healthz` returns 503 with `reason=no_gpu_visible` |
| `--cap-add=SYS_ADMIN` (or `SYS_PTRACE`/`PERFMON`) | `ncu` can read perf counters | Fast bench still runs; every profile returns `profile_unavailable` with `profiler_permission_denied` (SCN-BENCH-002-07) |
| Matching driver ≥ 550.x on host | cuda-python 12.3 requires ≥ 525; 12.4 image is safer on ≥ 550 | NVML init returns `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`; `/healthz` 503 |
| Volume mount for `ARTIFACT_ROOT` (optional) | Persist ncu-rep artifacts beyond pod lifetime | Artifacts stored in tmpfs; GC on container restart |
| `-v <host_path>:/srv/kerlever/artifacts` (required for cubin transport) | Shared cubin + ncu-rep store between Compiler Service (writer) and Benchmarker (reader). REQ-BENCH-031 requires `cubin_uri` to be an absolute POSIX path readable by the Benchmarker; V1 assumes the Compiler Service writes cubins here and the Benchmarker reads them back from the same path. | `CubinUriRejected` → `status=infra_error`, reason `cubin_uri_not_readable` for every candidate. |

### 7.5 Build-time vs runtime separation

- `docker build` runs on any machine (GPU-less CI included). No GPU call at build time. `pip install cuda-python` downloads wheels; it does **not** require a GPU.
- Container start on a non-GPU machine still runs uvicorn; the FastAPI lifespan startup hook catches `NvmlAdapterError(DRIVER_NOT_LOADED)` and **does not crash** — instead it stores the failure in `DeviceInventory.error` and serves 503 from `/healthz` with a structured NVML failure reason. This is essential for the "is the container healthy" check to give a precise diagnostic rather than a generic crash loop.

### 7.6 Healthcheck

A Docker `HEALTHCHECK` **is** baked into the image (see §7.2, Layer 7): `curl -fsS http://localhost:8080/healthz` on a 30 s interval with 3 s timeout and 20 s start-period grace. This makes `docker ps` and compose-managed deployments accurately reflect readiness.

For Kubernetes deployments, `readinessProbe` and `livenessProbe` on `GET /healthz` supersede the baked-in HEALTHCHECK and are the recommended configuration for production. The baked-in check is fallback/dev-ergonomics; the Kubernetes probes are authoritative in clustered environments.

### 7.7 Shared cubin + ncu-rep volume

REQ-BENCH-031 requires `cubin_uri` to resolve to an absolute POSIX path readable by the Benchmarker. In V1 the transport is a shared filesystem volume:

- Mount point inside the container: `/srv/kerlever/artifacts` (declared via `VOLUME` in §7.2).
- Writer: the Compiler Service writes cubins under `/srv/kerlever/artifacts/cubins/<hash>.cubin` (path shape is operator-chosen; the Benchmarker only requires absolute readability).
- Reader: the Benchmarker opens the cubin via `Path(cubin_uri).read_bytes()` inside `worker._load_candidates` (§3.4) and inside `profile_child.main` (§3.5).
- Output writeback: NCU reports are written under `/srv/kerlever/artifacts/ncu/<artifact_id>.ncu-rep` by the NCU subprocess; the Benchmarker re-reads them via `profiler.parse_report`.

Operators must mount the same host path — or the same PVC in Kubernetes — to both the Compiler Service container and the Benchmarker container. If the mount is missing, every candidate fails Phase 1 with `cubin_uri_not_readable` and the batch status is `INFRA_ERROR`.

Relative paths, URI schemes (`file://`, `s3://`, `http://`), and inline cubin bytes are all rejected at Phase 1 normalization (§10.4) — V1 does not support remote transport or in-request bytes.

---

## §8 Initialization Order

### 8.1 Service process startup (FastAPI lifespan)

1. **Parse config**: `BenchmarkerConfig.from_env()` reads env vars. If required env vars are missing → log structured error and exit non-zero (before uvicorn binds the port, so the container is immediately unhealthy).
2. **Init logging**: configure structured JSON logger (single formatter, stdout sink). Log context: `run_id`, `batch_id`, `request_id` when present.
3. **NVML init**: `telemetry.init()`. On `NVML_ERROR_DRIVER_NOT_LOADED` or `NVML_ERROR_NO_PERMISSION`, swallow and record in `DeviceInventory.error`; the container continues so `/healthz` can report the reason. On any other NVML error, also record but keep the service up.
4. **Device inventory**: `telemetry.info_inventory()` → `list[DeviceInventoryEntry]`. Cached on `ServiceEnv.device_inventory`. This is the single NVML-handle opening in the service process; subsequent handles are opened per-request under the lease.
5. **Toolchain identity**: resolve driver version (from NVML), CUDA runtime version (from `cuda.cuda_runtime` binding if available, else from `cuda-python`), `cuda-python` version (package metadata), `pynvml` version (package metadata), `ncu --version` (via `profiler.ncu_version`). `nvcc` is **not** queried — toolchain identity for the request is carried by `request.toolchain_hash` (REQ-BENCH-001); nvcc presence in the image is only needed for building, which happens elsewhere.
6. **Lease manager**: `LeaseManager(cfg.lease, inventory)`.
7. **Artifact store**: `ArtifactStore(cfg.artifact)` — creates `ARTIFACT_ROOT` if missing, checks writability.
8. **Pod-health store**: `PodHealthStore(initial=PodHealth.HEALTHY)`.
9. **Adapter registry seed**: `import kerlever.benchmarker.adapter` triggers auto-registration of the V1 built-ins (`elementwise_add_fp32_v1`, `matmul_fp16_v1`). Then, for each comma-separated path in `KERLEVER_BENCH_ADAPTER_REGISTRY_MODULES`, call `importlib.import_module(path)` so out-of-tree adapters register via their import-time `register_adapter` side effect. Import failure logs structured `adapter_plugin_import_failed` and exits non-zero (container is mis-configured — fail closed rather than accept requests for unregisterable adapters).
10. **FastAPI app**: `create_app(config)` builds the `FastAPI` instance, registers `POST /benchmark`, `GET /healthz`, `GET /info`, installs the exception handler that maps `RequestValidationError` to `ErrorEnvelope` (HTTP 400) and any uncaught exception to `ErrorEnvelope` (HTTP 500) with no stack trace leakage.
11. **Uvicorn**: serves.

### 8.2 Worker subprocess startup

1. **Parse argv**: `--config`, `--request-file`, `--result-file`, `--device-uuid`, `--device-ordinal`.
2. **Load config**: from the config file written by the supervisor.
3. **Read request**: Pydantic parse of `request.json`.
4. **Resolve adapter**: `adapter = adapter.get_adapter(req.operation_adapter_abi, req.operation_adapter_version)`. This happens **before** NVML init, cuda-python init, and any cubin load. On `AdapterUnregistered` → write `INFRA_ERROR` result with `failure_reason = "adapter_not_registered"` and `os._exit(1)`. On `AdapterVersionMismatch` → `failure_reason = "adapter_version_mismatch"` and `os._exit(1)`. Early resolution is deliberate: we do not open NVML or a CUDA context on behalf of a batch we already know is un-runnable.
5. **NVML init (fresh)**: `telemetry.init()`. This is a new pynvml init inside the child process — it does **not** inherit from the parent. This is the only NVML init in this process; it is shut down just before `os._exit(0)`.
6. **cuda-python Driver API init**: `cuda_driver.init()` → `cuInit(0)`.
7. **Resolve leased device**: match `device-uuid` to an ordinal via NVML, sanity-check against `device-ordinal`.
8. **Create primary CUDA context**: `cuda_driver.create_primary_context(ordinal)`. This is the one-and-only CUDA context for this subprocess lifetime.
9. **Load modules**: iterate `request.candidate_module_artifact_refs`, read cubin bytes, `cuda_driver.load_module` + `cuda_driver.get_function` for each. Then apply `function_attribute_policy` via `cuda_driver.set_function_attribute` / `cuda_driver.set_cache_config` and read back via `cuda_driver.get_function_attribute`. Observed values populate `LoadedCandidate.function_attribute_policy_observed` and flow into `MeasurementEnvelope` for the envelope hash (REQ-BENCH-029). Apply failure raises `FunctionAttributePolicyApplyFailed` → per-candidate `fault_class = infra_fault`; other candidates proceed.
10. **Load incumbent**: same pattern for `incumbent_ref` — load module, resolve entrypoint, apply function-attribute policy, read back. The incumbent's own `MeasurementEnvelope` (computed from its own observed values, its own anchor samples later in Phase 4) is the one carried into `decide_incumbent_comparison` (INV-BENCH-015).
11. **Per-shape adapter buffers**: for every shape in `objective_shape_cases ∪ profile_shape_cases`, call `adapter.allocate` + `adapter.seed_inputs` (§3.3). When `effective_cache_policy == WARM_ROTATING_BUFFERS`, also build a per-shape buffer pool of `iterations_per_sample + 1` equivalent buffers. Deferred teardown via `adapter.free` happens just before NVML shutdown at exit.
12. **Enter Phase 3** via `plan.calibrate`. From here the flow is the worker call graph in §3.2 / §3.3.
13. **Exit**: write `result.json`, free adapter buffers, destroy context, NVML shutdown, `os._exit(returncode)`.

Ordering rationale:
- Adapter resolution is step 4 (before NVML/CUDA) so that a mis-configured adapter terminates early without consuming GPU resources.
- NVML init must precede CUDA driver init so that NVML device enumeration is consistent with CUDA ordinal numbering under `CUDA_VISIBLE_DEVICES`.
- Cubins must be loaded before Phase 3 because calibration needs the `CudaFunction` handle for launch timing.
- Function-attribute policy is applied at module-load time (step 9), not later, so the observed values can be recorded into the envelope before any timing begins.
- Adapter buffer allocation (step 11) happens before calibration because `plan.calibrate` issues launches that need buffers bound.

### 8.3 Profile child subprocess startup

Lives for exactly one `(candidate, profile_shape)` profiling invocation. Invoked by NCU with the argv listed in §4.4. Steps:

1. **Parse argv**: `profile_child._parse_argv(argv)`. Every flag listed in §4.4 must be present; missing flag → `os._exit(2)` immediately.
2. **Register built-in adapters**: `adapter.AdapterRegistry.register_builtin_adapters()`. The profile child runs in a fresh interpreter with a fresh registry — V1 built-ins must be re-seeded explicitly. (Out-of-tree adapters loaded via `ADAPTER_REGISTRY_MODULES` — see §9 — also re-register here for symmetry with the service, since the profile child inherits the env.)
3. **Resolve adapter**: `adapter.get_adapter(args.adapter_abi, args.adapter_version)`. On failure → `os._exit(12)` (`EX_ADAPTER_UNREGISTERED`).
4. **NVML init**: optional; the child does not call pynvml in the V1 design — hygiene already ran in the worker. Skipping NVML init here keeps startup under 100 ms.
5. **cuda-python Driver API init**: `cuda_driver.init()`.
6. **Create primary CUDA context**: `cuda_driver.create_primary_context(0)`. The leased device is selected via `CUDA_VISIBLE_DEVICES` inherited from the parent (NCU preserves env across the `--` boundary); inside the child, the leased GPU is always ordinal 0.
7. **Load cubin**: `cuda_driver.load_module(Path(args.cubin_path).read_bytes(), options=None)` → `cuda_driver.get_function(module, args.entrypoint)`.
8. **Apply function-attribute policy**: `cuda_driver.set_function_attribute` for each of `MAX_DYNAMIC_SHARED_SIZE_BYTES`, `PREFERRED_SHARED_MEMORY_CARVEOUT`, `PREFERRED_CACHE_CONFIG` (via `set_cache_config`). On `CudaDriverError` → `os._exit(11)` (`EX_FUNC_ATTR_APPLY_FAILED`).
9. **Allocate adapter buffers**: `adapter.allocate(shape, dtype, device)` → `adapter.seed_inputs(buffers, shape, args.seed)`. Buffer pool for `WARM_ROTATING_BUFFERS` is **not** constructed in the child — a profile child does `iterations_per_sample` launches inside one NVTX range, and buffer rotation for non-`OVERWRITE_PURE` semantics uses a pool that is reset between iterations per spec §6.11 (future-adapter concern; V1 built-ins are `OVERWRITE_PURE`).
10. **Run warmup**: `warmup_count` untimed launches, followed by `cuda_driver.stream_synchronize`.
11. **Emit NVTX + timed launches**: push NVTX range `args.nvtx_range`, issue `args.iterations_per_sample` launches (honoring the adapter semantics dispatch from spec §6.11 if semantics ≠ `OVERWRITE_PURE`), pop NVTX range, `cuda_driver.stream_synchronize`.
12. **Teardown**: `adapter.free(buffers)` → `cuda_driver.destroy_primary_context(ctx)` → `os._exit(0)`.

Ordering rationale for the child mirrors the worker's: adapter resolution before NVML/CUDA so a bad argv or missing adapter exits early without touching the device. NVML is skipped entirely (not just "not initialized in the service process") because the profile child has no hygiene responsibility; the lease and hygiene gates have already been decided upstream.

---

## §9 Configuration Surface

Rule: anything that changes the **measurement semantics** (and therefore must be part of the envelope hash) flows through the `BenchmarkBatchRequest`. Anything that is a **pod-level deployment choice** comes from the environment. This split ensures the envelope hash stays complete and reproducible across pods while still allowing operational tuning.

### 9.1 Request-carried (semantics-affecting)

| Field | Source | Type |
|---|---|---|
| `metric_mode` | request | `MetricMode` |
| `function_attribute_policy` | request | `FunctionAttributePolicy` |
| `cache_policy` | request | `CachePolicy` (may be auto-promoted per §6.1) |
| `clock_policy` | request | `ClockPolicy` |
| `artifact_execution_model` | request | `ArtifactExecutionModel` (must be `COMMON_HARNESS_CUBIN`) |
| `top_k_profile` | request | int (bounded by service cap) |
| `top_m_profile_shift_potential` | request | int (bounded by service cap) |
| `anchor_every_n_samples` | request (optional override) | int |
| `max_interleave_block_len` | request (optional override) | int |
| `bench_rerun_limit` | request (optional override) | int |
| `toolchain_hash` | request (per candidate) | str |
| `operation_adapter_abi` + `operation_adapter_version` | request | str |

### 9.2 Env-carried (deployment-affecting only)

| Env var | Purpose | Default |
|---|---|---|
| `KERLEVER_BENCH_BIND_HOST` | uvicorn host | `0.0.0.0` |
| `KERLEVER_BENCH_BIND_PORT` | uvicorn port | `8080` |
| `KERLEVER_BENCH_ARTIFACT_ROOT` | pod-local path for cubins, reports, samples | `/var/lib/kerlever/bench` |
| `KERLEVER_BENCH_LOG_LEVEL` | log level | `INFO` |
| `KERLEVER_BENCH_BATCH_TIMEOUT_S` | supervisor timeout per batch | `1800` |
| `KERLEVER_BENCH_PROFILE_TIMEOUT_S` | ncu timeout per profile | `300` |
| `KERLEVER_BENCH_SHUTDOWN_GRACE_S` | drain grace period | `30` |
| `KERLEVER_BENCH_NCU_BIN` | path to `ncu` | `/usr/local/cuda/bin/ncu` |
| `KERLEVER_BENCH_NSYS_BIN` | path to `nsys` | `/usr/local/cuda/bin/nsys` |
| `KERLEVER_BENCH_NCU_PROFILE_SET` | `ncu --set` default | `focused` |
| `KERLEVER_BENCH_AMBIGUOUS_FAILURE_LIMIT` | pod-health quarantine threshold | `3` |
| `KERLEVER_BENCH_KERNEL_TIMEOUT_MS` | in-kernel timeout budget | `10000` |
| `KERLEVER_BENCH_POD_ID` | identifier for logs + `run_envelope.pod_id` | hostname |
| `KERLEVER_BENCH_THRESHOLDS_*` | `NOISE_FLOOR_PCT`, `MEASUREMENT_CV_WARN_PCT`, `MEASUREMENT_CV_FAIL_PCT`, `P95_P50_RATIO_WARN`, `ANCHOR_DRIFT_WARN_PCT`, `ANCHOR_DRIFT_FAIL_PCT`, `THERMAL_STEADY_STATE_LIMIT` | spec §6.8 defaults |
| `KERLEVER_BENCH_WARMUP_MIN_RUNS`, `..._MIN_TIMED_BATCH_MS`, `..._MAX_TIMED_BATCH_MS`, `..._REPETITIONS`, `..._MAX_ITERATIONS_PER_SAMPLE`, `..._MIN_P95_SAMPLES` | calibration parameters | spec §6.8 defaults |
| `KERLEVER_BENCH_CLOCK_LOCK_POLICY` | `disabled` / `enabled_when_privileged` | `disabled` |
| `KERLEVER_BENCH_ADAPTER_REGISTRY_MODULES` | Comma-separated Python import paths loaded at service startup (before uvicorn accepts requests) and re-loaded at the top of `profile_child.main`. Each named module is expected to have an import-time side effect that calls `register_adapter(...)` for one or more out-of-tree adapters. The two V1 built-ins (`elementwise_add_fp32_v1`, `matmul_fp16_v1`) always auto-register on import of `kerlever.benchmarker.adapter` and are not listed in this env var. Example: `my_pkg.adapters.custom_gemm_bf16,my_pkg.adapters.reduction_f32`. Missing module → container startup fails fast with `adapter_plugin_import_failed` logged; missing adapter class inside a listed module → ValueError at `get_adapter` time when a request tries to use it. | empty (V1 built-ins only) |

Thresholds are env-carried rather than request-carried because a remote pod's thermal/noise profile is a pod property, not a measurement parameter. Callers that want pod-specific noise floors should deploy different pod configurations; they cannot pass a new noise floor through the request without also breaking envelope-hash comparability across batches.

---

## §10 Error Propagation

Failures flow from deepest layer upward with explicit typed transitions. There is no generic "catch Exception, log, return 500" — every layer either recovers and transforms, or documents why it re-raises.

### 10.1 Exception hierarchy

- `CudaDriverError` — `cuda_driver.py` raises this for any CUresult != SUCCESS.
- `NvmlAdapterError` — `telemetry.py` raises this for library-level NVML failures (not hygiene gate failures).
- `CandidateRuntimeFault` — raised inside the worker when a single candidate's kernel launch/sync fails; carries `candidate_hash`, classified via `fault.attribute` as `candidate_fault` or `ambiguous_fault`.
- `HygieneFailure` — returned (not raised) from `telemetry.preflight` as a populated `HygieneReport.reason_on_fail`; caller decides whether to short-circuit.
- `ProfilerSubprocessError` — returned (not raised) from `profiler.run_ncu` as an `NcuRunResult` with nonzero `returncode` or `timed_out=True`.
- `WorkerCrash` — supervisor-side classification when child exit is nonzero or by signal; carries `exit_code`, `signal`, `stderr_tail`.
- `RequestValidationError` — Pydantic's built-in, raised before `supervisor.run_batch` is even called.
- `AdapterUnregistered` — raised by `adapter.AdapterRegistry.resolve` when the `(abi_name, abi_version)` key is unknown. Caught at worker startup (step 4 in §8.2) and at service startup when importing `ADAPTER_REGISTRY_MODULES`.
- `AdapterVersionMismatch` — raised by `adapter.AdapterRegistry.resolve` when `abi_name` is registered but at a different `abi_version`. Distinct from `AdapterUnregistered` to give the caller a more actionable `failure_reason`.
- `FunctionAttributePolicyApplyFailed` — raised by `worker._load_candidates` when `cuda_driver.set_function_attribute` (or `set_cache_config`) fails on a specific candidate. Carries `candidate_hash` + the underlying `CudaDriverError`. Per-candidate scope — other candidates in the batch proceed.
- `AdapterBufferAllocationFailed` — raised when `adapter.allocate` or `adapter.seed_inputs` fails (typically `CUDA_ERROR_OUT_OF_MEMORY` or `CUDA_ERROR_INVALID_VALUE`). Per-candidate-per-shape scope.
- `CubinUriRejected` — raised in Phase 1 normalization (`normalize.normalize_request`) when `cubin_uri` fails the REQ-BENCH-031 validation (relative path, URI scheme, inline-bytes attempt, unreadable file). Batch-level scope — the whole batch short-circuits to `INFRA_ERROR`.
- `InternalServerError` — the exception-handler bucket for anything genuinely unexpected (bug, not a modeled failure).

### 10.2 Mapping table

| Origin | Class | Transformation | HTTP response |
|---|---|---|---|
| Request body fails Pydantic schema | `RequestValidationError` | `service` exception handler | 400 + `ErrorEnvelope{code:"bad_request", detail:<field errors>}` |
| Hygiene gate fails hard (arch, MIG, ECC, Xid) | `HygieneFailure` via `HygieneReport.reason_on_fail` | `supervisor._hard_gate_result` builds `BenchmarkBatchResult(status=INFRA_ERROR, failure_reason=<reason>, hygiene=<report>)` | 200 + `BenchmarkBatchResult` (status is the transport channel) |
| Candidate kernel `CUDA_ERROR_ILLEGAL_ADDRESS` on healthy pod | `CandidateRuntimeFault` → `fault.attribute` → `CANDIDATE_FAULT` | Per-candidate `measurement_quality=runtime_fault`, `fault_class=candidate_fault` | 200 + `BenchmarkBatchResult(status=SUCCESS or PARTIAL)` |
| Worker subprocess SIGSEGV or `os._exit(2)` | `WorkerCrash` → `fault.attribute(signal/exit)` → `AMBIGUOUS_FAULT` | `BenchmarkBatchResult(status=INFRA_ERROR, fault_class=ambiguous_fault, failure_reason="worker_crash:<signal>")`; pod-health → suspect | 200 + `BenchmarkBatchResult` |
| Worker timeout (`asyncio.timeout`) | `WorkerCrash(timed_out=True)` | `BenchmarkBatchResult(status=TIMEOUT)`; pod-health → suspect | 200 + `BenchmarkBatchResult` |
| ncu timeout / missing / permission denied | `ProfilerSubprocessError` → `profiler.resolve_unavailable_reason` | Per-candidate `profile_status=profile_unavailable` with typed reason; fast-bench result preserved | 200 + `BenchmarkBatchResult(status=SUCCESS or PARTIAL)` |
| Adapter not registered at worker startup (step 4 §8.2) | `AdapterUnregistered` | Worker writes `BenchmarkBatchResult(status=INFRA_ERROR, failure_reason="adapter_not_registered", fault_class=infra_fault)` and exits 1. Supervisor reads and returns. | 200 + `BenchmarkBatchResult` |
| Adapter registered at wrong version at worker startup | `AdapterVersionMismatch` | Same shape as above, `failure_reason="adapter_version_mismatch"`. | 200 + `BenchmarkBatchResult` |
| `cuFuncSetAttribute` fails applying function-attribute policy for a candidate | `FunctionAttributePolicyApplyFailed` | Per-candidate `measurement_quality.status=infra_fault`, `fault_class=infra_fault`, reason `function_attribute_policy_apply_failed` (driver error code appended); other candidates proceed. | 200 + `BenchmarkBatchResult(status=SUCCESS or PARTIAL)` |
| Adapter `allocate` / `seed_inputs` fails for a `(candidate, shape)` | `AdapterBufferAllocationFailed` | Per-candidate-per-shape `measurement_quality.status=infra_fault`, `fault_class=infra_fault`, reason `adapter_buffer_allocation_failed`; shape excluded from the candidate's shape roll-up. | 200 + `BenchmarkBatchResult(status=SUCCESS or PARTIAL)` |
| `cubin_uri` is relative / has URI scheme / unreadable / inline-bytes attempt | `CubinUriRejected` | Phase 1 normalization raises; worker writes `BenchmarkBatchResult(status=INFRA_ERROR, failure_reason ∈ {"cubin_uri_unsupported_scheme","cubin_uri_not_readable"}, fault_class=infra_fault)` before any cubin load. | 200 + `BenchmarkBatchResult` |
| Profile child exit 11 (`EX_FUNC_ATTR_APPLY_FAILED`) or 12 (`EX_ADAPTER_UNREGISTERED`) | `ProfilerSubprocessError` (via `run_ncu` nonzero returncode) | Per-candidate-per-shape `profile_status=profile_unavailable`, reason `profiler_replay_refused`; fast-bench result preserved. | 200 + `BenchmarkBatchResult(status=SUCCESS or PARTIAL)` |
| Uncaught exception in supervisor | `InternalServerError` via FastAPI exception handler | 500 + `ErrorEnvelope{code:"internal_server_error", detail:"see service logs"}`; stack trace goes to structured log, not to response | 500 |

### 10.3 Error envelope type

Local to `types.py`:

```
ErrorEnvelope
  code: Literal["bad_request", "unsupported", "internal_server_error"]
  detail: str
  field_errors: list[{loc: list[str|int], msg: str, type: str}] | None
  request_id: str | None
```

This is the HTTP-level error shape. `BenchmarkBatchResult` is the *domain*-level error shape (preferred channel). HTTP 4xx/5xx are rare and reserved for transport-level failures.

---

## §11 What Is NOT Designed Here

Explicit exclusions, so that future work knows where the gaps are and Coding does not mistakenly invent:

- **No GPU Pipeline Adapter.** The orchestrator-facing shim that maps `BenchmarkBatchResult` → per-candidate `EvaluationResult` is a separate future task. Benchmarker does not import `kerlever.protocols`.
- **No Compiler Service implementation.** The Benchmarker consumes cubins by reference; it does not compile or sanitize. How cubins arrive (object store, pod-local mount, sidecar fetcher) is out of scope here — the `cubin_uri` in the request is required to resolve to a readable file from the pod, and the operational choice of how that happens is a deployment concern.
- **No cubin artifact store.** The artifact store in this design holds **outputs** (samples, ncu reports, raw metrics). **Input** cubins arrive via the request's `cubin_uri`; this design does not own a shared cubin registry.
- **No cross-pod orchestration.** Pod health, lease management, and artifact store are pod-local. A multi-pod scheduler that rings GPUs across pods is explicitly outside Benchmarker (spec §6.10).
- **No Profile Interpreter.** `BottleneckAssessment` fields are emitted empty (spec §6.10).
- **No persistence layer beyond the artifact store.** No database. No in-memory incumbent history. Pod-health is in-memory for the lifetime of the uvicorn process.
- **No default Nsight Systems.** `profiler.run_nsys` exists but is not called from the default Phase 6 flow. Trigger-based invocation is reserved for a future extension point.
- **No clock locking by default.** `CLOCK_LOCK_POLICY=disabled` is the shipped default. Enabling it requires a pool with `CAP_SYS_ADMIN` and the `enabled_when_privileged` env flag. The lock implementation path (invoking `nvidia-smi --lock-gpu-clocks`) is stubbed behind a config flag.
- **No unit tests.** Per user directive; verified by the Manager at coding-verification time.
- **No authentication/authorization.** The service accepts any client that can reach the port. Access control is a network-policy concern outside this module.
- **No `/metrics` Prometheus endpoint.** Observability is via structured JSON logs only; metrics export is future work.
- **No admin API.** No `/reset-pod-health`, no `/kill-batch`. Operators restart the pod if things are wrong — consistent with the "no durable state" design.
- **No cross-architecture ranking.** The Benchmarker never compares measurements taken on different `sm_arch`. Enforcement is envelope-hash-level: mismatched envelopes produce `IncumbentComparison = NOT_COMPARABLE`.
- **No multi-worker uvicorn.** Single worker by configuration. Multi-worker would require cross-process lease coordination (Redis, file-lock) that is both complex and unnecessary for a single-pod deployment.
- **No dynamic candidate admission.** The request's candidate set is fixed at request time; the Benchmarker does not admit late candidates mid-batch.
- **No partial-cubin streaming.** Each cubin is read fully into memory before `cuModuleLoadDataEx`. Large cubins (>100 MB) are not accommodated by design; that is a Compiler Service concern.
- **No adapter auto-discovery from disk.** Adapters register via one of two channels only: (a) import-time side effect of `kerlever.benchmarker.adapter` (V1 built-ins `elementwise_add_fp32_v1`, `matmul_fp16_v1`), or (b) the `KERLEVER_BENCH_ADAPTER_REGISTRY_MODULES` env var listing explicit Python import paths. Scanning `sys.path`, a filesystem directory, or a package entry-point group is out of scope — the operator's explicit `import path` list is the contract.
- **No inline cubin bytes transport.** The request schema does not declare a `cubin_bytes_b64` field; inline bytes in the body are ignored with a warning and the batch proceeds using `cubin_uri` only (REQ-BENCH-031). V2 may add inline-bytes support behind a new SC; V1 rejects.
- **No remote cubin URI fetching.** `cubin_uri` must be an absolute POSIX path. `s3://`, `gs://`, `http://`, `https://`, `file://`, and any other URI scheme are rejected with `cubin_uri_unsupported_scheme`. Remote fetching, signed URLs, and object-store mirroring are deployment / Compiler-Service concerns.
- **No artifact retention policy.** The Benchmarker writes cubin reads, ncu-rep files, and sample JSONs under `ARTIFACT_ROOT` (`/srv/kerlever/artifacts` in the container) but does not delete them. Retention, rotation, archival, and quota are operator-owned (volume-level tooling or a sidecar sweeper).
- **No adapter zoo beyond the two V1 built-ins.** V1 ships `elementwise_add_fp32_v1` and `matmul_fp16_v1`. Additional adapters (reductions, scans, attention, conv, epilogue fusions) are out of scope for this module and are added via the plugin loader without modifying Benchmarker code.

---

## Appendix A — Data flow one-liner

```
HTTP POST /benchmark
  → service.handle_benchmark(req)
    → lease.acquire(target)                      [service process, asyncio]
      → telemetry.preflight(lease)               [to_thread; pynvml]
      → supervisor._spawn_worker                 [asyncio.create_subprocess_exec]
        [[ subprocess boundary — new PID, fresh NVML, fresh cuda-python ctx ]]
        → worker.main(argv)
          → normalize.normalize_request          [Phase 1]
          → cuda_driver.create_primary_context   [Phase 2b — inside worker]
          → cuda_driver.load_module × N          [Phase 3a]
          → plan.calibrate                       [Phase 3]
          → harness.execute_batch                [Phase 4]
            → harness.generate_block_order
            → harness.run_sample × M
              → cuda_driver.event_record / launch / event_elapsed_ms
          → stats.* + scoring.compute_objective_score
                   + scoring.decide_incumbent_comparison   [Phase 5]
          → selection.build_profile_set          [Phase 6a]
          → profiler.run_ncu × K                 [Phase 6b — subprocess of worker]
          → profiler.parse_report + profiler.normalize
          → worker._assemble_result              [Phase 7]
          → write result.json
          → cuda_driver.destroy_primary_context
          → telemetry.shutdown
          → os._exit(0)
        [[ subprocess exits — CUDA ctx gone with it ]]
      → supervisor._read_worker_result           [parse result.json]
      → telemetry.postflight(lease)              [to_thread; pynvml]
      → supervisor._finalize                     [assemble final BenchmarkBatchResult]
    ← BenchmarkBatchResult
  ← HTTP 200 JSON
```

---

## Appendix B — Spec §5 types referenced (coverage check)

| Spec §5 type | Referenced in this design |
|---|---|
| `ArtifactExecutionModel` | §2.1 `types.py`, §9.1 |
| `MetricMode` | §2.1 `types.py`, §9.1 |
| `AdapterIterationSemantics` | §2.1 `types.py`, §6.3, §6.1 |
| `CachePolicy` | §2.1, §9.1 |
| `ClockPolicyMode` / `ClockPolicy` | §2.1, §6.2, §9.1 |
| `IncumbentComparison` | §2.1 `scoring.py`, §10.2 |
| `MeasurementQualityStatus` | §2.1 `harness.py`/`scoring.py`, §10.2 |
| `ProfileStatus` / `ProfileUnavailableReason` | §2.1 `profiler.py`, §10.2, §6.3 |
| `FaultClass` | §2.1 `fault.py`, §4.2, §10.2 |
| `PodHealth` | §2.1 `fault.py`, §4.1 |
| `BatchStatus` | §2.1 `types.py`, §4.2, §10.2 |
| `CandidateArtifactRef` | §3.2 `_load_candidates`, §6.1 `normalize` |
| `FunctionAttributePolicy` | §2.1 `types.py`, §6.1, §9.1 |
| `MeasurementEnvelope` | §2.1 `normalize.py`, §3.2, §6.1 |
| `DeviceTelemetrySnapshot` | §2.1 `telemetry.py`, §6.2 |
| `HygieneReport` | §2.1 `telemetry.py`, §3.1, §10.2 |
| `ShapeMeasurementArtifact` | §2.1 `harness.py`, Appendix A |
| `RawProfileMetric` | §2.1 `profiler.py`, §6.3 |
| `NormalizedProfileMetricProvenance` | §6.3 `profiler.normalize` |
| `ProfileArtifactRef` | §2.1 `types.py` |
| `IncumbentAnchor` | §3.2 `_assemble_result` |
| `CandidateResult` | §3.2 `_assemble_result` |
| `BenchmarkBatchRequest` | §3.1, §4.3 |
| `BenchmarkBatchResult` | §3.1, §3.6, §10.2 |

All 24 spec §5 types are referenced by at least one design module, call, or table.
