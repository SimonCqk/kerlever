# Compiler Service Module Specification

## §1 Problem Statement

### 1.1 Role in the Kerlever Loop

The Compiler Service is the **deterministic GPU-side gate** that decides whether a generated CUDA kernel is executable and correct enough to be measured. It sits on the remote GPU pod between the Coding Agent (which emits candidate source) and the Benchmarker (which measures performance), and it is the first of the three non-LLM services in the Kerlever architecture.

The service converts a request containing CUDA source, a problem specification, and an explicit kernel execution contract into a **structured, audit-ready result** that classifies the candidate as one of:

- `success` — compiles, passes correctness on all required shapes, passes the sanitizer gate; benchmark-ready artifact id emitted;
- `compile_error` / `interface_contract_error` / `correctness_fail` / `sanitizer_fail` / `timeout` / `infra_error` — structured failure with fault attribution, bounded log excerpts, and artifact references.

It does **not** own search state, does **not** decide whether an optimization direction is exhausted, and does **not** produce latency scores. Every outbound record is framed so the Orchestrator can cleanly separate candidate-owned failures from pod-owned failures when updating its durable search memory.

### 1.2 Why This Service Must Exist Standalone

LLM-generated CUDA makes four operating conditions normal rather than exceptional: compilation failure, hardware-observable memory safety errors, ambiguous GPU timeouts that poison the CUDA context for the next candidate, and stale or partial results from interrupted runs. The Kerlever architecture doc treats "the Compiler Service is the measurement truth" as a first-class principle; a thin wrapper around `subprocess.run("nvcc ...")` cannot satisfy that — correctness, fault attribution, sanitizer evidence, pod health, and artifact identity must all be first-class outputs.

Today the entire GPU pipeline is a stub (`kerlever.stubs.StubGPUPipeline`) that returns random latencies. Nothing in the current repository can compile a candidate, execute a correctness run, or distinguish a syntax error from a GPU watchdog timeout. This spec defines V1 of the real service.

### 1.3 Out-of-Module Context

| Module | Relationship to Compiler Service |
|---|---|
| Coding Agent | Produces source; never calls the Compiler Service directly (the Orchestrator's GPU pipeline adapter does) |
| Orchestrator / GPU Pipeline Adapter | Constructs `CompileRequest`, consumes `CompileResult`, maps fault classes into `CandidateOutcome` |
| Benchmarker | Consumes only artifact ids of results with `status=success`; never re-runs correctness |
| Profile Interpreter | Runs after the Benchmarker; never consumes Compiler Service output directly |
| Strategy Navigator / Cross-Candidate Analyzer | Never read Compiler Service output directly; they see the Orchestrator's derived records |

The present task implements the Compiler Service as a standalone Python package plus FastAPI HTTP wrapper plus Dockerfile. Wiring it into `GPUPipelineProtocol` is deferred to a later task.

---

## §2 Goals and Non-Goals

### 2.1 Goals

- **G1. Executable-correctness gate.** Given CUDA source, a problem spec, and an explicit `KernelExecutionSpec`, decide whether the candidate compiles and produces numerically acceptable output on every required shape, and return a typed result that distinguishes every failure class defined in `docs/compiler-service.md`.
- **G2. Structured measurement truth.** Every field that downstream code depends on (register count, shared memory, spills, occupancy, sanitizer outcome, fault class, oracle kind, tolerance source, artifact key) is typed structured data with explicit provenance. Natural-language stdout scraping is not a contract.
- **G3. Never fabricate a fact.** When a source of a static resource fact is unavailable, the service reports `None` with `resource_sources` marking the source as `null`. It never picks a plausible number.
- **G4. Symbol-collision-free correctness.** Reference and candidate are built as separate executables so that candidates may reuse the reference kernel's entrypoint name without compiler-level collisions or source rewriting.
- **G5. Fault attribution that respects search-memory semantics.** `candidate_fault` / `infra_fault` / `ambiguous_fault` are disjoint; ambiguous faults never contribute positive or negative evidence to optimization direction; infra faults never count as bad candidates; candidate faults carry a sub-kind drawn from the design doc's table.
- **G6. Pod-health awareness.** A poisoned CUDA context (illegal-memory-access, kernel watchdog timeout, driver reset) advances the pod from `healthy` to `suspect`; the next GPU execution runs a known-good probe before the candidate; repeated ambiguous failures reach `quarantined` and all subsequent requests short-circuit with `infra_fault`.
- **G7. Idempotent replay.** Retrying the same `request_id` either returns the same completed result or an explicit `infra_error` / `prior_attempt_lost_before_durability`. A collision on `request_id` with a different `artifact_key` is not silently served.
- **G8. Bounded, GC'd artifacts.** Per-class retention plus disk-high-watermark GC prevents long runs from exhausting pod disk, while pinned roots (baseline, incumbent, active batches) and referenced artifacts are never evicted.
- **G9. Clean deployability.** The service ships as a CUDA devel Docker image; `GET /healthz` verifies `nvcc`, driver, GPU visibility, `compute-sanitizer`, and artifact-root writability; container startup exits non-zero if the toolchain is incomplete.
- **G10. Operation-agnostic core.** The 5-phase pipeline never branches on `op_name`; operation-specific knowledge (shape interpretation, ABI, tolerance default, input distribution, comparison mode) is encapsulated behind an `OperationAdapter` protocol. V1 ships a `matmul` adapter and a minimal `elementwise` adapter; any other op returns `interface_contract_error / reason=unsupported_operation`.

### 2.2 Non-Goals

- Orchestrator or Benchmarker integration. No changes to `GPUPipelineProtocol`, `EvaluationResult`, `CandidateOutcome` in this task.
- Cross-request artifact caching. `ARTIFACT_CACHE=disabled` in V1; only same-`request_id` idempotent reuse is supported.
- Disk-persisted idempotency. V1 is in-process only; a process restart forfeits replay safety and the deployment contract must state this explicitly.
- Multi-GPU scheduling beyond one per-device concurrency semaphore.
- AST-level CUDA parsing. Lexical / regex detection is sufficient for sanitizer-escalation triggers.
- Dynamic harness generation by an LLM. Harnesses come only from registered adapters.
- Support for operations other than `matmul` plus a minimal `elementwise` skeleton.
- Unit tests (explicit user instruction), though the spec must be implementable without them.

### 2.3 Alignment with Architecture Principles

- **First-principles over pattern matching (AGENTS.md).** Every fact returned (registers, smem, spills, occupancy) must name its source. No "usually ptxas-ish" claims.
- **No over-engineering.** Idempotency is in-memory; artifact cache is disabled; only two adapters ship. Abstractions are introduced only where the design doc's failure-mode table requires them.
- **Structured data, not natural language.** Every outbound field is typed; stdout is accepted as a source only for bounded, truncated excerpts attached to failure detail, never as the source of a numeric fact.
- **Deterministic where possible, LLM where necessary.** This service is 100% deterministic; an LLM call anywhere in the compile / correctness / sanitizer path would invalidate the measurement-truth contract.
- **Grounded in facts.** `resource_sources`, `oracle_kind`, `tolerance_source`, `fault_class`, `pod_health`, and `idempotency_state` make every outbound claim auditable.

---

## §3 Domain Model

### 3.1 Core Concepts

| Concept | Meaning inside the service |
|---|---|
| **Kernel Execution Spec** | The candidate-owned, hard-required launch contract: entrypoint, `block_dim`, `dynamic_smem_bytes`, `abi_name`, `abi_version`, `metadata_mode`. It is the only way the service learns how to launch the generated kernel. Under normal traffic, the service never infers it. |
| **Operation Adapter** | The stable behavioral boundary between the service core and operation-specific correctness logic. An adapter owns shape interpretation, tensor allocation, ABI, default block geometry, tolerance defaults, input distribution, and comparison mode. |
| **Run Envelope** | The per-request bundle carried alongside every phase: identity (`run_id`, `round_id`, `request_id`, `candidate_hash`), reproducibility hashes, limits, phase timings, pod identity, pod health snapshot, idempotency state. |
| **Toolchain Info** | A reproducibility-anchored snapshot of the pod's environment: `nvcc` version, driver version, GPU name, GPU UUID, compute-sanitizer version, `toolchain_hash`. |
| **Static Analysis (extended)** | The `StaticAnalysis` from `kerlever.types` plus per-fact `resource_sources`, any `resource_conflicts`, and references to cubin/PTX/SASS artifacts. |
| **Correctness Result (extended)** | The `CorrectnessResult` from `kerlever.types` plus `oracle_kind`, `comparison_mode`, `tolerance_source`, and the accumulated `sanitizer_results`. |
| **Sanitizer Outcome** | One `{tool, shape_id, status, report_artifact_id}` record per sanitizer invocation. |
| **Artifact Store** | The pod-local filesystem surface that holds source, cubin, PTX, SASS, compile log, sanitizer report, and correctness log files, each addressed by `artifact_id`. |
| **Artifact Key** | `hash(source_hash, problem_spec_hash, launch_spec_hash, target_arch, toolchain_hash, compile_flags_hash, adapter_version, legacy_inferred_execution_spec)`. Any change in any of those inputs produces a different key. |
| **Fault Class / Candidate Fault Kind** | Disjoint classification of how the request failed: `candidate_fault` (with sub-kinds), `infra_fault`, `ambiguous_fault`, or null on success. |
| **Pod Health** | Singleton state `healthy → suspect → quarantined` guarded by a known-good probe kernel and an ambiguous-failure counter. |
| **Idempotency State** | Per-`request_id` record: `new`, `reused_completed`, or `prior_attempt_lost`. |
| **Failure Detail** | Structured failure context — phase, command, bounded stdout/stderr excerpt, failing shape id, retryable flag. Never unbounded text. |

### 3.2 Typed Data Contract

The service reuses existing types from `kerlever.types` without modification and introduces service-local richer types in `kerlever.compiler_service.types`. No existing type is redefined.

| Reused from `kerlever.types` | Used as |
|---|---|
| `ProblemSpec` | Input to every request: op_name, shapes, dtype, target GPU, reference kernel source |
| `ShapeCase` | Iteration target during correctness |
| `StaticAnalysis` | Structural base of `StaticAnalysisExt` |
| `CorrectnessResult` | Structural base of `CorrectnessResultExt` |
| `CompileStatus` | Legacy-mapping only; **not** used for the richer service-local status set |

| New in `kerlever.compiler_service.types` | Purpose |
|---|---|
| `KernelExecutionSpec` | Candidate-owned launch contract |
| `CompileRequest` | Inbound request: `request_id`, `run_id`, `round_id`, `candidate_hash`, `role`, `source_code`, `problem_spec`, `reference_source`, `execution_spec`, `target_arch`, `legacy_compatibility`, `limits` |
| `RunEnvelope` | Identity + reproducibility + limits + timings + pod identity + pod health + idempotency state |
| `ToolchainInfo` | `nvcc_version`, `driver_version`, `gpu_name`, `gpu_uuid`, `sanitizer_version`, `toolchain_hash` |
| `StaticAnalysisExt` | Extends `StaticAnalysis` with `resource_sources`, `resource_conflicts`, `cubin_artifact_id`, `ptx_artifact_id`, `sass_artifact_id` |
| `SanitizerOutcome` | `{tool, shape_id, status, report_artifact_id}` |
| `CorrectnessResultExt` | Extends `CorrectnessResult` with `oracle_kind`, `comparison_mode`, `tolerance_source`, `sanitizer_results: list[SanitizerOutcome]` |
| `FaultClass` | Enum: `candidate_fault`, `infra_fault`, `ambiguous_fault`, or absent on success |
| `CandidateFaultKind` | Enum: `syntax_error`, `semantic_compile_error`, `interface_contract_error`, `correctness_mismatch`, `memory_safety_error`, `race_or_sync_error`, `uninitialized_memory_error`, `candidate_runtime_error` |
| `FailureDetail` | `phase`, `command`, `stdout_excerpt`, `stderr_excerpt`, `failing_shape_id`, `retryable` |
| `CompileResult` | Top-level outbound record assembled in Phase 5 |
| `CompileResultStatus` | Service-local enum: `success`, `compile_error`, `interface_contract_error`, `correctness_fail`, `sanitizer_fail`, `timeout`, `infra_error` |

### 3.3 Status Enum Separation from Legacy `CompileStatus`

The existing `kerlever.types.CompileStatus` enumerates `SUCCESS`, `COMPILE_ERROR`, `CORRECTNESS_FAIL`. The design doc's richer status surface additionally requires `interface_contract_error`, `sanitizer_fail`, `timeout`, `infra_error`. Conflating them into one enum would either narrow Compiler Service output (losing fault evidence) or widen the global type (breaking existing Orchestrator consumers).

**Decision:** introduce a separate `CompileResultStatus` enum local to the service. Future adapter code that maps `CompileResult` into `EvaluationResult` is responsible for projection into legacy `CompileStatus`. This decision is listed as a shortcut risk in §5 (SR-CS-004).

### 3.4 Invariance Summary

At a domain level, every `CompileResult`:

- carries a non-null `run_envelope.pod_health` sampled at the moment of result assembly;
- carries an `artifact_key` deterministically derived from the eight inputs listed above;
- when `status != success` carries a non-null `fault_class`;
- when `fault_class = candidate_fault` carries a non-null `candidate_fault_kind`;
- when the correctness phase ran, carries a non-null `correctness.oracle_kind`;
- when `legacy_inferred_execution_spec = true`, carries that flag as an input to `artifact_key` and marks the result visibly.

These domain invariants are formalized as INV-CS-* in §7.

---

## §4 Production Path Trace (PPT)

End-to-end behavioral flow for a healthy matmul candidate under a typical `POST /v1/compile`. The language is externally observable — no internal class or method names are referenced; the architecture details belong in `design.md`.

1. An upstream caller (in the delivered system, the future GPU Pipeline Adapter; during manual verification, `curl`) issues `POST /v1/compile` with a JSON body matching the `CompileRequest` schema: candidate CUDA source, the full `ProblemSpec` including `reference_kernel`, a `KernelExecutionSpec` (entrypoint, block dim, dynamic smem bytes, ABI name, ABI version), identity fields (`request_id`, `run_id`, `round_id`, `candidate_hash`), `target_arch`, `legacy_compatibility=false`, and optional `limits`.
2. The service acknowledges receipt and validates that the request is well-formed JSON conforming to the schema. If validation fails the request is rejected with `400` and no artifact is created.
3. The service samples current `pod_health`. If it is `quarantined`, the request short-circuits to a `CompileResult` carrying `status=infra_error`, `fault_class=infra_fault`, `pod_health=quarantined`, with no compile or GPU work performed.
4. The service consults its idempotency registry by `request_id`. If a completed record exists and its stored `artifact_key` matches the current request's computed `artifact_key`, the stored `CompileResult` is returned with `idempotency_state=reused_completed`. If the stored key mismatches, the stored entry is treated as stale and a new attempt proceeds (logged as an anomaly). If a prior attempt is observed started but not completed, the service returns `status=infra_error / reason=prior_attempt_lost_before_durability` with `previous_attempt_lost=true` and `prior_attempt_observed_phase` set.
5. Request normalization computes the four content hashes (`source_hash`, `problem_spec_hash`, `launch_spec_hash`, `compile_flags_hash`), loads cached `ToolchainInfo` (whose `toolchain_hash` was fixed at service startup), computes `artifact_key`, and constructs the `RunEnvelope`.
6. Interface resolution checks that the `KernelExecutionSpec` is present and complete. If any required field is missing and `legacy_compatibility=false`, the service returns `status=interface_contract_error / fault_class=candidate_fault / candidate_fault_kind=interface_contract_error` without invoking `nvcc`. When `legacy_compatibility=true` the service may infer the spec using the legacy rules, marks `legacy_inferred_execution_spec=true`, and continues.
7. Adapter selection reads `problem_spec.op_name`. For `matmul` or `elementwise` the corresponding registered adapter is selected. Any other op_name yields `status=interface_contract_error / reason=unsupported_operation` without compilation.
8. Harness assembly renders two deterministic harness source files, one for the reference kernel and one for the candidate kernel. Each harness is a separate translation unit so the reference and candidate may both use the same entrypoint name without symbol collisions. Input generation is seeded by `hash(problem_spec, shape_id, dtype, "kerlever_correctness")`; reference and candidate receive bit-identical inputs.
9. Compilation invokes `nvcc` twice (once per harness) under a bounded CPU concurrency guard and a wall-clock timeout. Flags are the stable set `-O3 -std=c++17 -lineinfo -arch=<sm_xx> -Xptxas=-v`. Source size is bounded by `MAX_SOURCE_BYTES`; stdout and stderr are bounded by `MAX_LOG_BYTES`. A non-zero exit of either compilation terminates the request via Phase 5 with `status=compile_error / candidate_fault / candidate_fault_kind ∈ {syntax_error, semantic_compile_error}` and a bounded `failure.stderr_excerpt`. The candidate source, the compiled candidate executable, the cubin, the PTX, the SASS, and the bounded compile log are all registered as artifacts, each with an `artifact_id`.
10. Static resource extraction reads function attributes from the loaded candidate module via the CUDA driver API for `registers_per_thread`, `smem_bytes_per_block`, and related fields; it additionally parses `ptxas -v` output as a fallback / cross-check. Each fact is attached with its source (`cuda_func_attribute`, `ptxas`, or `null`); if two sources disagree, both values are preserved in `resource_sources` and `resource_conflicts` records the disagreement. `spill_loads`, `spill_stores` come only from `ptxas`; if the text is unavailable or unparseable they remain `None`. `occupancy_estimate_pct` is computed from the closed-form formula using `block_dim`, `registers_per_thread`, `smem_bytes_per_block`, target arch limits, and max warps per SM — never from an unstructured string.
11. Correctness validation, executed under a per-GPU concurrency semaphore of 1, iterates the `ProblemSpec.shape_cases`. For each shape the service generates deterministic inputs, runs the reference executable to produce the golden output, runs the candidate executable, and compares outputs. Tolerance is resolved in the order `ShapeCase.correctness_tolerance → adapter dtype default → service default`; `comparison_mode` defaults to `tolerance` for float dtypes and only becomes `exact` when the adapter declares exact semantics. A shape failure accumulates into `failing_shape_ids` / `max_abs_error` / `max_rel_error`; the result carries `oracle_kind=reference_kernel` (V1). If any shape fails, `correctness.passed=false` and the run short-circuits to Phase 5 with `status=correctness_fail`.
12. Sanitizer gate, when correctness passes. `memcheck` runs unconditionally on the smallest shape under `SANITIZER_TIMEOUT`. Additional tools escalate by trigger:
    - `racecheck` when the candidate lexically uses `__shared__`, `__syncwarp`, or `cooperative_groups`, or the recent batch shows race-like nondeterminism;
    - `synccheck` when the candidate uses `__syncthreads`, `__syncwarp`, `cooperative_groups`, or barriers;
    - `initcheck` when correctness output contained NaN/Inf, the adapter marks a shape high-risk, or the incoming semantic-diff flag suggests uninitialized reads.
    Each invocation produces a typed `SanitizerOutcome` that is appended to `correctness.sanitizer_results`. A fail status translates to `status=sanitizer_fail / candidate_fault / candidate_fault_kind ∈ {memory_safety_error, race_or_sync_error, uninitialized_memory_error}` depending on the tool that fired. A `compute-sanitizer` timeout yields `status=timeout`.
13. Pod health transitions are evaluated at every Phase 4 step. A clean pass keeps the pod `healthy` and resets the ambiguous failure counter. A watchdog timeout, illegal-memory-access, or driver reset advances the pod to `suspect`; the next Phase 4 request runs a known-good probe kernel before touching the candidate, and a probe failure advances the pod to `quarantined`. If the ambiguous failure counter exceeds `AMBIGUOUS_FAILURE_LIMIT`, the pod is `quarantined` and all subsequent requests short-circuit with `infra_fault`.
14. Result assembly constructs the single `CompileResult` for the request: `status`, `candidate_hash`, `run_envelope` (including current `pod_health` and final `idempotency_state`), `legacy_inferred_execution_spec`, `toolchain`, `static_analysis`, `correctness` (if the phase ran), `sanitizer_results`, `artifacts` map, `fault_class`, `candidate_fault_kind`, `failure` — with unset fields null rather than defaulted. The result is recorded in the idempotency registry keyed by `request_id` with its `artifact_key` before return.
15. The service returns the `CompileResult` JSON to the caller. On `status=success` the `artifacts.cubin_artifact_id` and related ids are consumable by a future Benchmarker running on the same pod. On failure the caller records the outcome and may regenerate a candidate; the service itself never issues retries.

Between requests the artifact store performs cheap GC on unpinned expired entries; when disk usage crosses `ARTIFACT_DISK_HIGH_WATERMARK` an eager pass drops additional entries by class-based TTL. Pinned roots (baseline, current incumbent, active benchmark batch, active profile batch) and any artifact currently referenced by an idempotency registry entry are never evicted.

---

## §5 Shortcut Risks

Documented risks the implementation will be tempted to take, each paired with its structural prevention. Each is phrased against the AGENTS.md "grounded in facts" and "no over-engineering that hides correctness" principles.

| ID | Shortcut | Why it is tempting | Why it is wrong | Structural prevention |
|---|---|---|---|---|
| **SR-CS-001** | Scrape `ptxas -v` text to populate `registers_per_thread` / `smem_bytes_per_block` and silently default to zero when the regex fails | `ptxas -v` always prints a recognizable block on success; "works on the happy path" | Zero registers is a valid-looking number that feeds occupancy formulas. Silent zero means the service becomes a high-confidence liar. Violates G3 (never fabricate). | `StaticAnalysisExt.resource_sources` is a required per-fact provenance field; absence is modeled as `None` with source `null`; occupancy formula short-circuits to `None` when any input is `None`. Enforced by INV-CS-003. |
| **SR-CS-002** | Compile the reference kernel and the candidate kernel into one translation unit for efficiency | One `nvcc` invocation is faster than two; code reuse of the harness | Reference and candidate frequently share the entrypoint name (e.g. both `matmul`). Merging yields a link-time symbol collision or — worse — a deterministic mis-link where one kernel shadows the other and correctness becomes a lie. | Design doc explicitly forbids this; enforced by INV-CS-001. Harness assembly emits two disjoint source trees and two `nvcc` invocations are made regardless of the operation. |
| **SR-CS-003** | Infer `KernelExecutionSpec` whenever it is missing ("there is only one `__global__`, use it") | Avoids rewriting examples, unblocks testing | Silent inference means the candidate identity no longer fully describes what ran. Two requests with the same source but different inferred blocks would share an `artifact_key`. | `legacy_compatibility` is an explicit request flag; absence-of-spec without the flag is `interface_contract_error`; every legacy-inferred result and artifact key carries `legacy_inferred_execution_spec=true`. Enforced by INV-CS-002. |
| **SR-CS-004** | Reuse `kerlever.types.CompileStatus` for the richer service-local status set by widening the enum | "One enum to rule them all" | Widening `CompileStatus` would break every existing orchestrator consumer that switches on the three legacy values and propagate service concepts into the shared type space. | A separate `CompileResultStatus` is declared in `kerlever.compiler_service.types`. The future GPU pipeline adapter is responsible for projecting it into the legacy `CompileStatus` / `CandidateOutcome`. See §3.3. |
| **SR-CS-005** | Treat `request_id` alone as the idempotency key and serve any stored result back | Simplest possible implementation | A collision on `request_id` with different source or different launch spec would silently return a wrong result — the `artifact_key` would have changed but the service would not notice. | Idempotent reuse asserts `stored_artifact_key == current_artifact_key`; mismatch treats the entry as stale and logs an anomaly. Enforced by INV-CS-009. |
| **SR-CS-006** | Build the sanitizer call as a fire-and-forget `subprocess.run` and log a pass/fail message to stdout | Sanitizer output is verbose; structured capture is fiddly | The structured `sanitizer_results` list is the only downstream-visible evidence; forgetting to populate it produces a "passed correctness, sanitizer forgotten" illusion. | Every sanitizer invocation returns a typed `SanitizerOutcome`; Phase 4 accumulates them into `correctness.sanitizer_results` before Phase 5 is entered. Enforced by INV-CS-004. |
| **SR-CS-007** | Default float comparison to bit-exact equality | Bit-exact is simpler, faster, and "more correct" | GPU reduction order, tensor-core paths, and FMA fusion produce numerically valid differences; bit-exact float comparison yields a false `correctness_fail` and teaches the search memory the wrong lesson. | Adapter declares `comparison_mode`; default is `tolerance` for float dtypes, `exact` only when the adapter explicitly opts in. Tolerance is resolved via a single documented order and recorded in `tolerance_source`. Enforced by INV-CS-011. |
| **SR-CS-008** | Let the HTTP server come up even if `nvcc` / driver / sanitizer are absent, and fail individual requests | Keeps the container up and "lets you see the error" | A running-but-broken container looks healthy to the scheduler; the Orchestrator would be handed `infra_error` for every request while the pod appears to be doing its job. | FastAPI startup probe runs the toolchain check and exits non-zero on any missing tool; `GET /healthz` runs the same probe. Enforced by INV-CS-012. |
| **SR-CS-009** | Keep everything in `/var/lib/kerlever/artifacts` until manual cleanup | "We can GC later once we measure how bad it is" | Long-running self-evolution runs generate many large cubin/SASS files; disk exhaustion manifests as `infra_fault` for every subsequent request, with no clean recovery. | Artifact store implements the design doc's per-class retention table and `ARTIFACT_DISK_HIGH_WATERMARK`-triggered GC from day one; pinned roots and currently-referenced artifacts are protected. Enforced by INV-CS-007. |
| **SR-CS-010** | Run multiple correctness executions in parallel on the same GPU | Shorter wall-clock per request | Two CUDA contexts on one device can corrupt each other's memory without the driver noticing; a correctness mismatch then reflects cross-contamination, not the candidate. | Per-device `asyncio.Semaphore(1)` around every Phase 4 GPU invocation. Enforced by INV-CS-010. |
| **SR-CS-011** | Count an `ambiguous_fault` as a `candidate_fault` so the orchestrator at least sees a signal | Any signal > no signal | A kernel watchdog timeout caused by a GPU driver issue, attributed as a candidate correctness miss, permanently poisons the search memory: the direction looks exhausted when it was infrastructure. | `ambiguous_fault` is a first-class `FaultClass` value; the routing table and the attribution helper map poisoned-context evidence to `ambiguous` rather than `candidate`. Enforced by INV-CS-005. |
| **SR-CS-012** | Let the Compiler Service invent a harness via a light LLM call when an adapter is missing | Unblocks adding ops without writing adapters | An LLM-drafted harness would make correctness itself untrusted. The whole point of the service is measurement truth. | Unknown `op_name` yields `status=interface_contract_error / reason=unsupported_operation` without invoking any LLM. Enforced by INV-CS-013. |

---

## §6 Behavioral Specification

This section is the heart of the spec. Each subsystem stands on its own, written in behavioral language the Coding Agent can implement from without guessing. Internal types and functions are not named; structural names already fixed by the design doc (`RunEnvelope`, `CompileResult`, `OperationAdapter`, etc.) are used because they are part of the outbound contract.

### 6.1 Request Normalization and Interface Resolution

Request normalization is the first phase and the only one that runs before any content-based decisions. It takes the inbound `CompileRequest` and produces the `RunEnvelope` plus a resolved `KernelExecutionSpec` that downstream phases will consume. It never touches the GPU.

**Request validation.** Every inbound request must match the `CompileRequest` schema: identity fields (`request_id`, `run_id`, `round_id`, `candidate_hash`), `role` (one of `reference`, `candidate`, `probe`), `source_code`, `problem_spec`, `reference_source`, `execution_spec`, `target_arch`, `legacy_compatibility`, optional `limits`. A body that fails Pydantic validation is rejected with HTTP 400 and does not produce a `CompileResult`; no artifact work is performed.

**Hash computation.** Normalization computes four content-derived hashes in a stable order:

- `source_hash` — SHA-256 of the candidate `source_code` bytes, truncated to a fixed width for display;
- `problem_spec_hash` — SHA-256 of the Pydantic-canonical-JSON serialization of `ProblemSpec` (including `op_name`, `op_semantics`, `dtype`, `target_gpu`, `shape_cases`, `objective`, `target_metric_value`, `max_rounds`, `reference_kernel`);
- `launch_spec_hash` — SHA-256 of the canonical-JSON serialization of the resolved `KernelExecutionSpec` (after legacy inference if applied);
- `compile_flags_hash` — SHA-256 of the canonical-JSON serialization of the resolved `compile_flags` list (including `target_arch` derived flags).

**Toolchain identity.** `ToolchainInfo` is snapshotted once at service startup from the results of the startup toolchain probe, not per-request. Its `toolchain_hash = hash(nvcc_version, driver_version, gpu_uuid, sanitizer_version)`. Requests refer to the cached value; a pod whose toolchain changes after startup is outside V1.

**Artifact key construction.** `artifact_key = hash(source_hash, problem_spec_hash, launch_spec_hash, target_arch, toolchain_hash, compile_flags_hash, adapter_version, legacy_inferred_execution_spec)`. Every change in any of those inputs produces a different key; no other input contributes. `legacy_inferred_execution_spec` is included as a boolean input, which means a legacy-inferred and an explicit candidate for otherwise-identical inputs still have disjoint keys.

**Idempotency registry lookup.** The service maintains an in-memory registry keyed by `request_id`. On each inbound request:

- If the registry has a completed entry for the id and `stored_artifact_key == current_artifact_key`, the stored `CompileResult` is returned with `idempotency_state=reused_completed` after re-sampling the current `pod_health` into the returned envelope. No compile or GPU work is performed.
- If the registry has a completed entry but the keys disagree, the entry is treated as stale, the anomaly is recorded in the request log, and a fresh attempt proceeds with `idempotency_state=new`.
- If the registry has a started-but-not-completed entry (e.g. a prior request began and did not record a completion before a process event), the service returns `status=infra_error / fault_class=infra_fault / reason=prior_attempt_lost_before_durability`, with `previous_attempt_lost=true` and `prior_attempt_observed_phase` set to the last phase name the registry recorded. Since V1 idempotency is in-process, a process restart empties the registry — this limitation is documented explicitly in the deployment contract and is not a bug.
- Otherwise a new entry is created with `idempotency_state=new` and phase updates are recorded as phases complete.

**Interface resolution.** The service inspects the inbound `execution_spec` and `legacy_compatibility` flag:

- If `legacy_compatibility=false` (the normal path) and the `execution_spec` is missing any of `entrypoint`, `block_dim`, `dynamic_smem_bytes`, `abi_name`, or `abi_version`, the request terminates with `status=interface_contract_error / fault_class=candidate_fault / candidate_fault_kind=interface_contract_error / failure.phase="request_normalization"`. No compile is performed. `legacy_inferred_execution_spec=false` is set in the result.
- If `legacy_compatibility=false` and all fields are present, `metadata_mode` must be `explicit` and the spec is used as-is. `legacy_inferred_execution_spec=false`.
- If `legacy_compatibility=true` and fields are missing, the legacy inference rules are applied: the single `__global__` function's name becomes `entrypoint`; `block_dim` becomes the adapter's default block geometry; `dynamic_smem_bytes` becomes `0`; `abi_name`/`abi_version` come from the adapter; `metadata_mode="legacy_inferred"`. `legacy_inferred_execution_spec=true` is set in the result and contributes to the artifact key. If the source does not contain exactly one `__global__` function, inference fails with `interface_contract_error`.

**Run envelope construction.** The envelope carries everything downstream phases will need without re-computing:

- identity — `run_id`, `round_id`, `request_id`, `candidate_hash`;
- reproducibility — `source_hash`, `problem_spec_hash`, `launch_spec_hash`, `toolchain_hash`, `compile_flags_hash`, `adapter_version`;
- limits — `compile_timeout`, `correctness_timeout`, `sanitizer_timeout`, `max_source_bytes`, `max_log_bytes` (from `ServiceConfig` unless overridden by `CompileRequest.limits`);
- observability — empty phase-timings map to be populated by later phases, `pod_id`, `gpu_uuid` from `ToolchainInfo`;
- pod health — current `pod_health` snapshot;
- idempotency — `idempotency_state`, `previous_attempt_lost`, `prior_attempt_observed_phase`.

### 6.2 Operation Adapter Model

Operation adapters are the stable behavioral boundary that keeps the service core free of `op_name` branches. Exactly one adapter is selected per request; it owns everything operation-specific.

**Adapter surface (Protocol).** Every adapter exposes at least the following behaviors:

- `allocate_inputs(problem_spec, shape, seed)` — returns the set of typed input buffers for one shape case, fully deterministic given `seed`;
- `build_harness_source(execution_spec, problem_spec, role, kernel_source)` — returns the CUDA / C++ harness source for one translation unit, including the kernel source (reference or candidate), the `main` that loads inputs, launches the kernel with the `execution_spec` launch parameters and the adapter-derived grid, writes outputs to a deterministic path, and exits with a deterministic status code;
- `compare_outputs(problem_spec, shape, reference_output, candidate_output, tolerance)` — returns per-shape `max_abs_error`, `max_rel_error`, and pass/fail given the resolved tolerance;
- `default_block_dim(problem_spec)` — the adapter's conservative default when legacy inference is active;
- `default_tolerance(dtype)` — the per-dtype tolerance default used when `ShapeCase.correctness_tolerance` is absent;
- `comparison_mode(dtype)` — `tolerance` for float dtypes unless the adapter explicitly opts in to `exact`;
- `abi_contract()` — `abi_name`, `abi_version`;
- `adapter_version()` — a string baked into `artifact_key`;
- `high_risk_shape_ids(problem_spec)` — optional set of shape ids the adapter considers high-risk for sanitizer escalation.

Adapters are registered by `op_name`. The core phases never switch on `op_name`; they call the resolved adapter.

**MatmulAdapter semantics.**

- `ShapeCase.dims = [M, N, K]`;
- dtype maps to a CUDA scalar type (`half` for `fp16`, `float` for `fp32`);
- kernel ABI is `(const T* A, const T* B, T* C, int M, int N, int K)`;
- `default_block_dim` is `(16, 16, 1)`, chosen for conservative launch safety on legacy-inferred candidates;
- grid is derived as `((N + block_x - 1) / block_x, (M + block_y - 1) / block_y, 1)`;
- `default_tolerance`: `1e-2` for `fp16`, `1e-4` for `fp32`;
- `comparison_mode` is always `tolerance`;
- inputs are filled with pseudo-random values from a stable uniform distribution parameterized by `seed`;
- `abi_name="matmul_v1"`, `abi_version="1.0"`.

**ElementwiseAdapter semantics (skeleton).**

- `ShapeCase.dims = [N]`;
- kernel ABI is `(const T* A, const T* B, T* C, int N)`;
- `default_block_dim` is `(256, 1, 1)`;
- grid is `((N + 255) / 256, 1, 1)`;
- `default_tolerance`: `1e-2` for `fp16`, `1e-5` for `fp32`, `0` for integer dtypes;
- `comparison_mode` is `exact` for integer dtypes, `tolerance` for float dtypes;
- `high_risk_shape_ids` returns an empty set by default;
- `abi_name="elementwise_v1"`, `abi_version="1.0"`.

**V1 non-goal (ElementwiseAdapter arity).** The V1 elementwise ABI is restricted to **two-operand arithmetic ops** of the form `C = f(A, B)` — the four-argument signature `(A, B, C, N)` is hard-coded in the rendered harness. Unary ops (`C = f(A)`), n-ary ops, reductions, and ops with extra scalar parameters are out of scope and are planned as a separate adapter task. Submitting a kernel whose arity differs from `(A, B, C, N)` fails at Phase 4 launch time with a candidate runtime error.

The elementwise adapter's role in V1 is not to support real optimization of elementwise ops; it is to keep the adapter interface honest by forcing at least two concrete adapters to coexist without matmul-specific code leaking into the core.

**Unknown operation rejection.** Any `op_name` outside the registry yields `status=interface_contract_error / reason=unsupported_operation` during Phase 2. The service does not attempt an LLM-drafted harness, does not compile, and does not touch the GPU. This is non-negotiable and is enforced by INV-CS-013.

### 6.3 Host Harness Assembly

Phase 2 takes the resolved execution spec and the selected adapter, and produces two disjoint harness source trees, one for the reference kernel and one for the candidate kernel. It does not invoke `nvcc`.

**Separate translation units are mandatory.** Generated CUDA candidates regularly reuse the reference kernel's entrypoint name (e.g. both call their `__global__` function `matmul`). The only structurally safe response is to build two separate executables. Phase 2 writes:

- `workspace/reference.cu` — reference kernel source inlined, adapter harness `main`, launch using `KernelExecutionSpec.entrypoint` as the ABI-compliant symbol;
- `workspace/candidate.cu` — candidate kernel source inlined, adapter harness `main`, launch using the same `entrypoint`.

The two files are compiled into two separate executables in Phase 3, each linked independently. Under no circumstances are both kernels included in a single translation unit. Violation of this rule is a correctness correctness-level bug because symbol collisions can yield deterministic mis-links that "pass" with wrong data.

**Deterministic input seeding.** For each shape case, the harness generates inputs seeded by `hash(problem_spec, shape_id, dtype, "kerlever_correctness")`. The same seed is used by both the reference executable and the candidate executable so that any observed output difference is attributable to kernel logic, not input. Inputs are written to typed binary files in the per-shape input directory before either kernel runs.

**Output capture.** Each harness writes output tensors to a deterministic path under the per-shape output directory; the correctness phase reads both and invokes the adapter's `compare_outputs` on them. Stdout from the harness is captured but not trusted as numeric source; only the binary output files are authoritative.

**Launch parameters.** Block dim, dynamic shared memory, and grid dim come from the `KernelExecutionSpec` plus the adapter (grid comes from the adapter because it is shape-derived). The harness calls `cudaFuncSetAttribute` to grant the requested `dynamic_smem_bytes` before launch; a failure here is attributed to `candidate_runtime_error`.

**Bounded source.** The rendered harness source for each of the reference and candidate is bounded by `MAX_SOURCE_BYTES`; a request whose `source_code` alone exceeds the bound is rejected early with `interface_contract_error / reason=source_too_large`.

### 6.4 nvcc Compile and Artifact Production

Phase 3 compiles the two harness sources and extracts static resource facts. Both compilations must succeed for the request to proceed to Phase 4.

**Stable compile flags.** The fixed flag set is `-O3 -std=c++17 -lineinfo -arch=<sm_xx> -Xptxas=-v`. `sm_xx` is derived from `CompileRequest.target_arch` (e.g. `sm_80` for A100, `sm_90` for H100). The flag set is identical between reference and candidate so that performance- or correctness-differentiation between them is due to the kernel, not the toolchain. Any change in the default flag set is an `adapter_version`-visible event.

**Timeout and byte caps.** Each `nvcc` invocation runs under `COMPILE_TIMEOUT` and is captured as `(returncode, bounded_stdout, bounded_stderr)`. A wall-clock timeout yields `status=timeout / fault_class=infra_fault` (timeouts during compile are attributed to infrastructure in V1; if the root cause is candidate-triggered the Coding Agent will see the stderr excerpt and adjust). Stdout and stderr are truncated to `MAX_LOG_BYTES` with an explicit truncation marker; the bounded excerpts become `failure.stdout_excerpt` and `failure.stderr_excerpt` if the phase fails.

**Failure attribution.**

- If `nvcc` returns non-zero and the stderr excerpt matches lexical patterns associated with parse-level errors (`expected`, `error:`, unclosed braces, missing semicolons), `candidate_fault_kind=syntax_error`.
- Other non-zero exits map to `semantic_compile_error`.
- `nvcc` unable to find an `sm_xx` target, or similar toolchain-level errors map to `infra_fault`.

Phase 5 then constructs the final `CompileResult` with `status=compile_error`, the bounded `failure` record, and no cubin reference. The artifact store still records the candidate source under a compile-failure retention class so the Coding Agent can audit the exact source that failed.

**Artifact persistence.** On success, the per-request artifacts are registered in the artifact store with typed `artifact_id`s:

- `source_artifact_id` — the rendered candidate harness source;
- `executable_artifact_id` — the compiled candidate executable;
- `cubin_artifact_id` — the cubin extracted from the executable;
- `ptx_artifact_id` — the `ptx` file, emitted with `nvcc --ptx`;
- `sass_artifact_id` — SASS output from `cuobjdump`;
- `compile_log_artifact_id` — the bounded combined stdout/stderr of the successful `nvcc` run.

Analogous artifacts are recorded for the reference executable.

### 6.5 Static Resource Extraction

Static resource extraction reads compile-time facts with explicit provenance. The contract is "every fact carries its source, and missing sources become `None`, never a fabricated number".

**Source preference table (from design doc).**

| Fact | Preferred source | Fallback | On unavailable or conflict |
|---|---|---|---|
| `registers_per_thread` | `cuFuncGetAttribute` after module load | `ptxas -v` parse | `None` if both unavailable; both values recorded in `resource_sources` with `resource_conflict` flag if they disagree |
| `smem_bytes_per_block` (static) | `cuFuncGetAttribute` | `ptxas -v` parse | `None`; never inferred from unrelated source text |
| `dynamic_smem_bytes` | `KernelExecutionSpec.dynamic_smem_bytes` | none | required field |
| `spill_loads` / `spill_stores` | `ptxas -v` | SASS inspection | `None`; never extrapolated |
| `max_threads_per_block` | `cuFuncGetAttribute` | device limits × launch spec (validation only) | `None` |

**Provenance.** `StaticAnalysisExt.resource_sources` is a per-fact dict whose values are one of `cuda_func_attribute`, `ptxas`, `sass`, or `null`. `resource_conflicts` is a list of `{fact, sources: [{source, value}], preferred_value}` entries for any fact where two sources disagree.

**Occupancy calculation.** Occupancy is not a scraped field. It is derived from the closed-form formula:

```
threads_per_block = block_x * block_y * block_z
warps_per_block   = ceil(threads_per_block / 32)
active_blocks     = min(
    blocks_by_warps,         # max_warps_per_sm / warps_per_block
    blocks_by_registers,     # max_regs_per_sm / (registers_per_thread * threads_per_block)
    blocks_by_shared_memory, # max_smem_per_sm / (smem_bytes_per_block + dynamic_smem_bytes)
    blocks_by_threads,       # max_threads_per_sm / threads_per_block
    max_blocks_per_sm,
)
occupancy = active_blocks * warps_per_block / max_warps_per_sm
```

If any input to the formula is `None` (unknown register count, unknown shared memory), `occupancy_estimate_pct` is `None`. The formula is never run on a guessed input. Hardware limits (`max_warps_per_sm`, `max_regs_per_sm`, `max_smem_per_sm`, `max_threads_per_sm`, `max_blocks_per_sm`) come from a service-internal per-arch table; unknown arches produce `None`.

**Never-fabricate rule.** INV-CS-003 bans implementations that default missing facts to `0`, `-1`, or a "reasonable" value. The service reports what it knows, not what would look complete.

**V1 non-goal (driver-API attribute reads).** `DriverApiAttributes.read_registers_per_thread`, `.read_static_smem_bytes`, and `.read_max_threads_per_block` all return `None` in V1 and the caller transparently falls back to the `ptxas -v` path. A V2 task will implement `cuModuleLoadData` + `cuFuncGetAttribute` now that the cubin artifact is routinely extracted (Phase 3 extracts cubin / PTX / SASS on every successful compile as of this revision); the V2 driver-API reads will become the preferred source per the table above, with `ptxas` remaining as the fallback. The contract in this §6.5 table is unchanged — only the implementation will move from "always `None`" to "real lookup with fallback".

### 6.6 Correctness Validation

Correctness validation runs after both compilations succeed. It iterates `ProblemSpec.shape_cases` under the per-GPU concurrency semaphore of 1.

**Tolerance resolution order.** For each shape case the tolerance is resolved in the single documented order:

1. `ShapeCase.correctness_tolerance` if present — `tolerance_source="shape_case"`;
2. otherwise the adapter's `default_tolerance(dtype)` — `tolerance_source="adapter_dtype_default"`;
3. otherwise the service default (`1e-4` for float, `0` for integer) — `tolerance_source="service_default"`.

The resolved tolerance and its source are recorded on the `CorrectnessResultExt` so downstream consumers know which layer produced it.

**Comparison mode.** `comparison_mode` comes from the adapter's `comparison_mode(dtype)`. Float kernels default to `tolerance`. Integer kernels and adapters that explicitly declare exact semantics use `exact`. The comparison itself is performed by the adapter's `compare_outputs`, which reports `max_abs_error` and `max_rel_error` even in `exact` mode (an `exact` mode failure has nonzero absolute error by definition).

**Oracle kind.** V1 always uses `oracle_kind="reference_kernel"`. The correctness contract is "candidate matches the problem spec's reference kernel within tolerance on every required shape". Future V1.x may add `adapter_independent` (e.g. cuBLAS for matmul) and `hybrid` (independent on small/supported shapes, reference elsewhere); the spec field exists now so no field migration is required later.

**Per-shape execution.** For each shape case:

1. Inputs are produced from `hash(problem_spec, shape_id, dtype, "kerlever_correctness")`.
2. The reference executable is run with a `CORRECTNESS_TIMEOUT` wall clock; its outputs are read.
3. The candidate executable is run with the same timeout; its outputs are read.
4. The adapter's `compare_outputs` reports `(passed, max_abs_error, max_rel_error)` for this shape.
5. If `passed=false`, the shape id is added to `failing_shape_ids` and the running `max_abs_error` / `max_rel_error` are updated to the maxima across failing shapes.

**Short-circuit vs. all-shapes.** V1 collects all shape results before emitting the `CorrectnessResultExt`; that means all shapes run even after a failure so the `failing_shape_ids` list is informative for the Coding Agent's repair prompt. A timeout on any single shape aborts and the result carries `status=timeout / fault_class=infra_fault` (candidate timeouts in correctness are attributed to infrastructure because we cannot cleanly distinguish them from GPU watchdog events in V1).

**Runtime failures.** A candidate executable exiting with a nonzero status that is not a correctness-mismatch (e.g. an invalid launch configuration reported via `cudaGetLastError`) yields `candidate_fault_kind=candidate_runtime_error`. An illegal-memory-access that also kills the CUDA context advances `pod_health` and may be reclassified as `ambiguous_fault` per §6.8.

**Oracle kind preservation.** INV-CS-006: every correctness result carries a non-null `oracle_kind`. The orchestrator's baseline bootstrap depends on this field to avoid treating a reference-only comparison as an independent proof of correctness.

### 6.7 Sanitizer Gate

The sanitizer gate runs only after `correctness.passed=true`. Its purpose is to catch memory-safety, synchronization, and initialization bugs that slip past value correctness.

**Default behavior.** `memcheck` always runs on the smallest shape (chosen by the `SANITIZER_SHAPE_POLICY`, default "smallest by product of dims"). The invocation is wrapped in `SANITIZER_TIMEOUT`. The result is a typed `SanitizerOutcome` appended to `correctness.sanitizer_results` and regardless of pass/fail always recorded.

**Escalation triggers.** Additional tools run only when specific conditions are met:

- `racecheck` — candidate source lexically contains `__shared__`, `__syncwarp`, or `cooperative_groups`, OR the recent batch (last N requests via pod health tracker) has exhibited race-like nondeterminism flagged via semantic diff metadata;
- `synccheck` — candidate source lexically contains `__syncthreads`, `__syncwarp`, `cooperative_groups`, or explicit barriers;
- `initcheck` — any shape's candidate output contained NaN or Inf while correctness still reported pass (possible when the adapter treats NaN-preserving kernels as acceptable), OR the adapter flagged a shape as high-risk via `high_risk_shape_ids`, OR the request carried a semantic-diff flag suggesting uninitialized reads.

Triggers are evaluated lexically (regex) over the candidate source; AST parsing is out of scope. Each fired tool contributes a `SanitizerOutcome`.

**Escalation ordering.** `memcheck` always runs first. `racecheck`, `synccheck`, `initcheck` run after, in that order, only if their trigger fires. A tool never replaces `memcheck`; it augments.

**Failure to candidate-fault-kind mapping.**

| Sanitizer tool | On fail | `candidate_fault_kind` |
|---|---|---|
| `memcheck` | out-of-bounds, misaligned, illegal address | `memory_safety_error` |
| `racecheck` | shared-memory race | `race_or_sync_error` |
| `synccheck` | barrier misuse | `race_or_sync_error` |
| `initcheck` | uninitialized device memory read | `uninitialized_memory_error` |

Any sanitizer `fail` triggers `status=sanitizer_fail`. A sanitizer `timeout` yields `status=timeout / fault_class=infra_fault`. An `unsupported` status (tool cannot run on this target) is preserved in `sanitizer_results` but does not cause failure.

**Reports and artifacts.** A sanitizer failure produces a `report_artifact_id` pointing at the bounded sanitizer report in the artifact store. A pass produces `report_artifact_id=null`; the pod is not required to keep pass-only sanitizer reports.

**Per-result preservation.** Every sanitizer invocation's outcome ends up in `correctness.sanitizer_results`; the field is a list, never collapsed into a single aggregate pass/fail. This is required by INV-CS-004; downstream consumers need to see which tool caught what on which shape.

### 6.8 Pod Health

The pod health tracker is a service-wide singleton that records the pod's CUDA context hygiene across requests. Its states are `healthy`, `suspect`, `quarantined`.

**Transitions driven by Phase 4 outcomes.**

- Every clean Phase 4 pass (correctness ok, sanitizer ok) keeps the pod in its current state and resets the `ambiguous_failure_count` to 0.
- A Phase 4 event classified as ambiguous (see §6.9) advances `healthy → suspect` and increments `ambiguous_failure_count`.
- The next Phase 4 request after entering `suspect` runs the known-good probe kernel (e.g. `vec_add.cu` from `reference_kernels/`) before accepting the candidate. A probe pass returns the pod to `healthy`. A probe fail advances `suspect → quarantined`.
- `ambiguous_failure_count >= AMBIGUOUS_FAILURE_LIMIT` advances the pod to `quarantined` regardless of probe availability.
- `quarantined` is terminal for the life of the process; no request consults the GPU; every request short-circuits to `status=infra_error / fault_class=infra_fault / pod_health=quarantined`.

**Probe semantics.** The probe kernel is a tiny, known-safe vector-add-style kernel whose correctness is trivially verifiable. It is compiled once at service startup (or lazily on first use) and reused; its binary is stored in a pinned artifact. Probe failure means "the pod cannot even run a trivial kernel" and is a strong signal that the GPU, driver, or CUDA context is broken.

**Envelope preservation.** Every outbound `CompileResult`'s `run_envelope.pod_health` reflects the pod's state at result assembly time, not at request intake. A candidate that was accepted while `healthy` but caused a transition to `suspect` emits a result with `pod_health=suspect` and, if its failure was ambiguous, `fault_class=ambiguous_fault`. This is required by INV-CS-008.

**Ambiguity in V1.** Detecting a poisoned CUDA context in V1 is bounded: we treat `cudaErrorIllegalAddress`, `cudaErrorLaunchTimeout`, `cudaErrorMisalignedAddress` during Phase 4 as ambiguity-triggering. A full CUDA context reset is not attempted; once a request triggers such an error, the pod enters `suspect` and the next request runs the probe first.

### 6.9 Fault Attribution

Every non-success `CompileResult` carries a `fault_class`. The three classes are disjoint and their search-memory meaning is distinct:

- **`candidate_fault`** — a deterministic failure caused by the candidate on a healthy pod. Always carries a non-null `candidate_fault_kind`. Contributes to the orchestrator's direction- and candidate-specific learning signal.
- **`infra_fault`** — a pod- or tool-level failure independent of the candidate (nvcc missing, artifact write failure, disk full, probe failure, pod quarantined, toolchain not ready). Never contributes to optimization signal.
- **`ambiguous_fault`** — a pod-observable failure where candidate cause cannot be cleanly separated from pod cause (kernel watchdog timeout, CUDA context poisoning without a clear memcheck fingerprint, process kill during GPU run). Never contributes positive or negative optimization signal; retry policy is orchestrator's choice.

**Attribution rules.**

| Observed event | Fault class | Candidate fault kind |
|---|---|---|
| `nvcc` returns non-zero, stderr matches parse-level patterns | `candidate_fault` | `syntax_error` |
| `nvcc` returns non-zero, other pattern | `candidate_fault` | `semantic_compile_error` |
| Missing or malformed `KernelExecutionSpec`, unknown `op_name` | `candidate_fault` | `interface_contract_error` |
| Shape correctness diff on healthy pod | `candidate_fault` | `correctness_mismatch` |
| `memcheck` fail on healthy pod | `candidate_fault` | `memory_safety_error` |
| `racecheck` or `synccheck` fail | `candidate_fault` | `race_or_sync_error` |
| `initcheck` fail | `candidate_fault` | `uninitialized_memory_error` |
| Candidate runtime failure (non-zero exit, bad launch config) on healthy pod | `candidate_fault` | `candidate_runtime_error` |
| `nvcc` missing, `compute-sanitizer` missing, GPU gone, driver reset, artifact disk full, pod quarantined | `infra_fault` | none |
| `compile_timeout` / `correctness_timeout` / `sanitizer_timeout` in V1 | `infra_fault` | none (V1 simplification) |
| Kernel watchdog timeout, CUDA context poisoning, process kill during GPU run | `ambiguous_fault` | none |
| Pod is `suspect` and probe fails | `infra_fault` | none |
| Pod becomes `suspect` during the request | result is downgraded from the candidate-side attribution to `ambiguous_fault` | none |

**Conservative attribution.** When the pod health transitions during the request, the request's final attribution is `ambiguous_fault` — never `candidate_fault` — even if the proximate symptom was candidate-like. The search memory semantics in the design doc explicitly permit this: an ambiguous fault is rerun at the orchestrator's discretion and does not pollute direction exhaustion.

**Separation from search meaning.** `ambiguous_fault` never contributes to optimization signal semantics in the returned record. The orchestrator consults the same `fault_class` / `candidate_fault_kind` table (design doc §Fault Attribution) when mapping `CompileResult` into `AttemptRecord` / `CandidateOutcome`; the service's job is only to label correctly, not to decide retry policy.

### 6.10 Idempotency

Idempotency provides "retrying the same `request_id` is safe" guarantees bounded by "V1 is in-process only".

**Key.** Every idempotency registry entry is keyed by `request_id`. An entry stores `{started_at, phase_observed, completed_at (optional), artifact_key, compile_result (optional)}`.

**Life cycle.**

1. On request intake, the registry is consulted. See §6.1 for the four cases.
2. After each phase completes, `phase_observed` is updated to the phase name (`request_normalization`, `harness_assembly`, `compile`, `correctness`, `sanitizer`, `output`). Any exception path must still leave the registry in a consistent state; an in-flight entry with no completion is what produces `prior_attempt_lost` on replay.
3. On result assembly in Phase 5, the registry entry is finalized with `completed_at` and the assembled `CompileResult`. Only then may subsequent replays return `idempotency_state=reused_completed`.

**Reuse check.** On reuse, `stored_artifact_key == current_artifact_key` is asserted. Mismatch means the `request_id` was sent with different source/launch/spec and the stored result cannot be trusted; the entry is treated as stale, an anomaly is logged, and a fresh attempt proceeds with `idempotency_state=new`. This is required by INV-CS-009.

**TTL.** Entries older than `IDEMPOTENCY_TTL` are eligible for purge. A purged entry means a subsequent replay is indistinguishable from a fresh `new` request; callers should not rely on idempotent replay past the TTL.

**Process-restart bound.** V1 does not persist the registry. A process restart empties it; a retry of an in-flight `request_id` after restart is served as `new`. This limitation is documented in the deployment contract so operators cannot infer stronger durability than the service actually has.

**Replay of in-flight.** A replay whose original entry is still `started` (no `completed_at`, no `compile_result`) produces `status=infra_error / fault_class=infra_fault / reason=prior_attempt_lost_before_durability` with `previous_attempt_lost=true` and `prior_attempt_observed_phase` set. The original attempt's progress is not resumed; the orchestrator decides whether to retry with the same id or a new one.

### 6.11 Artifact Store

The artifact store is the pod-local filesystem surface that holds every durable byte the service produces. It is owned by the service and visible through `GET /v1/artifacts/{id}`; artifact ids are opaque and uniquely identify one file.

**Workspace layout.** The configured root is `KERLEVER_ARTIFACT_ROOT` (default `/var/lib/kerlever/artifacts`). Per-request artifacts live under `root/<run_id>/<candidate_hash>/<artifact_kind>/...`. Pinned reference artifacts (probe kernels, baseline cubin) live under `root/pinned/<role>/...`.

**Artifact identity.** Every artifact has a stable `artifact_id` whose form is `sha256-truncated-hash-of-path-and-bytes`; a client who receives an id can always retrieve the bytes as long as GC has not evicted them.

**Per-class retention (from design doc table).** The retention policy is implemented as a typed `RetentionPolicy` mapping artifact class to keep/drop rules:

| Artifact class | Keep | Drop / GC |
|---|---|---|
| baseline, current incumbent | source, cubin, PTX, SASS, compile log, static metadata, correctness summary | retain through run completion + configured retention TTL |
| successful non-incumbent top-K / profiled | cubin, PTX/SASS refs, compile log, static metadata, profile inputs | drop after analysis TTL unless referenced by durable record |
| successful non-profiled non-incumbent | compact metadata, score refs | drop bulky cubin/PTX/SASS after short TTL or when Benchmarker no longer needs them |
| compile failure | source hash, bounded source excerpt/ref, bounded compile log, failure classification | no cubin; trim oversized logs |
| correctness failure | source hash, cubin (short TTL), failing shape ids, max errors, bounded logs | drop cubin/PTX/SASS after repair/debug TTL unless pinned |
| sanitizer failure | sanitizer report, tool name, shape id, bounded logs, cubin (short TTL) | drop cubin/PTX/SASS after repair/debug TTL unless pinned |
| infra or ambiguous failure | pod logs, health state, bounded stderr, request envelope | no optimization artifacts unless needed for retry diagnosis |

**Pinned roots.** The service exposes `pin(role, artifact_id)` / `unpin(role, artifact_id)` for `baseline`, `incumbent`, `active_benchmark_batch`, `active_profile_batch`, and `probe_kernel`. Pinned artifacts are never subject to GC regardless of class TTL. Pinning is in-memory in V1.

**GC passes.**

- A cheap pass runs after every request completion: scan unpinned entries whose TTL has elapsed and delete them.
- An eager pass runs when disk utilization crosses `ARTIFACT_DISK_HIGH_WATERMARK`; it drops additional entries in order of class TTL priority until utilization falls below the watermark.
- Neither pass may delete a pinned artifact, and neither may delete an artifact currently referenced by an idempotency registry entry whose `completed_at` is within `IDEMPOTENCY_TTL`. INV-CS-007.

**Artifact API.** `GET /v1/artifacts/{artifact_id}` streams bytes with a content-type keyed by class (`application/octet-stream` for cubin, `text/plain` for logs, etc.). Unknown or GC'd ids return `404`.

### 6.12 FastAPI Surface

The service exposes a narrow HTTP surface. Each route is typed by Pydantic schemas that wrap the service-local types.

| Route | Method | Behavior |
|---|---|---|
| `/healthz` | GET | Runs the toolchain probe (nvcc version, driver version, GPU visibility, compute-sanitizer version, artifact root writability) and the current `pod_health`. On all-green returns `200` with `{status: "ready", toolchain: {...}, gpu: {...}, pod_health: "..."}`; on any missing dependency returns `503` with a structured error. Never touches the GPU for a kernel launch. |
| `/v1/compile` | POST | Body: `CompileRequest` JSON. Returns `CompileResult` JSON. Honors idempotency by `request_id`. Returns HTTP `200` with status-in-body for all service-level outcomes (including failures); returns HTTP `400` only on schema-malformed bodies and `503` when `pod_health=quarantined`. |
| `/v1/artifacts/{artifact_id}` | GET | Streams the artifact by id. `404` for unknown or GC'd ids. |
| `/v1/pod-status` | GET | Returns `{pod_health, ambiguous_failure_count, toolchain: {...}, disk_watermark: {...}}`. Never touches the GPU. |

**Startup readiness.** At FastAPI startup the service runs the same toolchain probe as `/healthz`. If any required tool is missing (`nvcc`, driver, sanitizer, artifact root unwritable), the startup hook exits the process with a non-zero code so the container becomes visibly unhealthy immediately. A container whose HTTP server is up but whose probe would fail is a bug — INV-CS-012 forbids it.

**Concurrency.** Every `/v1/compile` handler is `async`. Inside, the service holds a global asyncio lock around the idempotency registry (brief, for entry creation and update), a bounded `asyncio.Semaphore(CPU_COMPILE_CONCURRENCY)` around `nvcc` invocations, and a per-GPU `asyncio.Semaphore(1)` around Phase 4 GPU work. Requests are served out of order if their phase patterns differ.

### 6.13 Configuration Parameters

All tunables are declared in `ServiceConfig`, materialized at service startup from environment variables or defaults, and carried into the `RunEnvelope.limits` per request (so replays see the same limits).

| Parameter | Default direction | Behavior |
|---|---|---|
| `COMPILE_TIMEOUT` | service default (e.g. 60 s) | Wall-clock cap per `nvcc` invocation |
| `CORRECTNESS_TIMEOUT` | service default (e.g. 120 s) | Wall-clock cap covering all shape correctness runs |
| `SANITIZER_TIMEOUT` | service default (e.g. 300 s) | Wall-clock cap per `compute-sanitizer` invocation |
| `MAX_SOURCE_BYTES` | small bounded limit (e.g. 256 KiB) | Rejects pathological generated source before compile |
| `MAX_LOG_BYTES` | service default (e.g. 64 KiB) | Truncation bound for compile stdout/stderr, sanitizer reports, correctness logs |
| `CPU_COMPILE_CONCURRENCY` | bounded by pod CPU (e.g. 4) | Parallel `nvcc` capacity |
| `GPU_RUN_CONCURRENCY` | 1 per visible GPU | Phase 4 per-device semaphore |
| `ARTIFACT_RETENTION` | class-based TTL + disk watermark | `RetentionPolicy` table from §6.11 |
| `ARTIFACT_DISK_HIGH_WATERMARK` | service default (e.g. 85 %) | Trigger of eager GC pass |
| `ARTIFACT_PIN_ROOTS` | `{baseline, incumbent, active_benchmark_batch, active_profile_batch, probe_kernel}` | Roots GC must not evict |
| `SANITIZER_SHAPE_POLICY` | `smallest_by_dim_product` | Choice of the default memcheck shape |
| `SANITIZER_DEFAULT_TOOL` | `memcheck` | The unconditional sanitizer tool |
| `SANITIZER_ESCALATION_TOOLS` | `[racecheck, synccheck, initcheck]` | Extra tools available under triggers |
| `SANITIZER_ESCALATION_POLICY` | `trigger_based` | Only these triggers, only these tools, only specified shapes |
| `DEFAULT_COMPILE_FLAGS` | `[-O3, -std=c++17, -lineinfo, -arch=<sm_xx>, -Xptxas=-v]` | Applied identically to reference and candidate |
| `POD_HEALTH_PROBE` | `reference_kernels/vec_add.cu` | The known-good probe used from `suspect` transitions |
| `AMBIGUOUS_FAILURE_LIMIT` | small fixed count (e.g. 3) | Threshold for `quarantined` transition |
| `IDEMPOTENCY_TTL` | run-scoped (e.g. 24 h) | Registry purge age |
| `ARTIFACT_CACHE` | `disabled` | Cross-request cubin reuse is off in V1 |

All configuration parameters are declared in one place; the rest of the codebase does not hard-code their values. Missing env var falls back to the service default.

---

## §7 Safety Invariants

Each invariant is phrased as an always-true property with the mechanism that enforces it. Violation is a correctness bug, not a warning.

**INV-CS-001: Reference and candidate never share a translation unit.** Phase 2 emits two disjoint harness source trees; Phase 3 compiles two separate executables. There is no code path in the service that concatenates reference kernel source and candidate kernel source into a single file.
*Enforcement:* the harness assembly subsystem has no "merge" mode; the compile subsystem is invoked twice with distinct source paths; code review enforces this because the failure mode is silently produced wrong correctness.

**INV-CS-002: `KernelExecutionSpec` is never inferred silently for normal traffic.** A request with `legacy_compatibility=false` and a missing or incomplete `execution_spec` always short-circuits to `interface_contract_error`. When `legacy_compatibility=true`, the result's `legacy_inferred_execution_spec=true` flag is set and feeds `artifact_key`.
*Enforcement:* interface resolution only enters the legacy inference branch when the request flag is true; the result assembler refuses to emit a result with `metadata_mode="legacy_inferred"` and `legacy_inferred_execution_spec=false`.

**INV-CS-003: Static resource facts are never fabricated.** `registers_per_thread`, `smem_bytes_per_block`, `spill_loads`, `spill_stores`, `occupancy_estimate_pct` are `None` when unavailable. `resource_sources` records the source per fact. `resource_conflicts` records disagreement.
*Enforcement:* extraction code returns `Optional[int]`; there is no code path that writes a default numeric value on extraction failure; the occupancy formula short-circuits to `None` when any input is `None`.

**INV-CS-004: Every sanitizer invocation produces a preserved `SanitizerOutcome`.** The service never calls `compute-sanitizer` without capturing its structured outcome into `correctness.sanitizer_results`. A pass, fail, timeout, or unsupported status are all recorded.
*Enforcement:* sanitizer execution returns a typed result whose construction is the only way to signal completion; Phase 4 appends to the list before advancing; Phase 5 refuses to emit a `CompileResult` whose `correctness.sanitizer_results` is shorter than the number of sanitizer invocations attempted.

**INV-CS-005: Every `CompileResult` carries a `fault_class` consistent with `status`.** `status=success` → `fault_class=null`; `status ∈ {compile_error, interface_contract_error, correctness_fail, sanitizer_fail}` on a healthy pod → `fault_class=candidate_fault`; `status ∈ {timeout, infra_error}` → `fault_class=infra_fault`; ambiguous events → `fault_class=ambiguous_fault`.
*Enforcement:* the result assembler takes status and pod health as inputs and derives fault class by the §6.9 table; every code path that sets status passes through the assembler; no other place in the codebase sets `fault_class`.

**INV-CS-006: Every correctness result carries an `oracle_kind`.** Whenever `correctness` is non-null on the `CompileResult`, `oracle_kind` is one of `reference_kernel`, `adapter_independent`, `hybrid`. V1 always uses `reference_kernel`.
*Enforcement:* `CorrectnessResultExt` has `oracle_kind` as a required field; the Pydantic model rejects construction without it.

**INV-CS-007: Artifact GC never deletes a pinned or currently-referenced artifact.** Pinned roots and artifacts referenced by an in-TTL idempotency registry entry are skipped by every GC pass.
*Enforcement:* GC walks a candidate set that excludes pinned ids and ids referenced by `artifact_refs` on in-TTL registry entries; the delete API refuses to act on a pinned id.

**INV-CS-008: `POST /v1/compile` output envelope always carries the current `pod_health`.** The field is sampled at result assembly, after Phase 4 transitions are applied.
*Enforcement:* the envelope construction path reads `pod_health` from the tracker singleton at assembly time, not at intake; the Pydantic model for `RunEnvelope` requires the field.

**INV-CS-009: Idempotent reuse asserts `artifact_key` match.** A stored completed entry is only reused when its stored `artifact_key` equals the current request's computed key.
*Enforcement:* the reuse check compares keys explicitly; a mismatch logs an anomaly and falls through to a new attempt with `idempotency_state=new`.

**INV-CS-010: Concurrent Phase 4 executions on the same GPU are impossible.** The per-device semaphore is acquired before any kernel launch and released only after all shape executions and sanitizer invocations for the request complete.
*Enforcement:* the semaphore is acquired around the entire correctness phase, not per-shape; tests of the correctness scheduler assert acquisition before kernel calls.

**INV-CS-011: Float comparison never defaults to `exact`.** Float dtypes default to `comparison_mode="tolerance"` and only switch to `exact` when the adapter explicitly opts in. `tolerance_source` always reflects which layer produced the tolerance.
*Enforcement:* the adapter's `comparison_mode(dtype)` is the only source; there is no code path that constructs a correctness result with `comparison_mode="exact"` on a float dtype without the adapter having declared it.

**INV-CS-012: Container never reports ready while toolchain is incomplete.** Startup probe + `/healthz` share the same check; startup exits non-zero on failure; `/healthz` returns `503`.
*Enforcement:* the startup hook and the health handler call the same function; startup hook uses `sys.exit(1)` on any failure.

**INV-CS-013: The core phases never branch on `op_name`.** Operation-specific behavior is reachable only via the adapter protocol. Unknown `op_name` yields `interface_contract_error / reason=unsupported_operation` during adapter selection.
*Enforcement:* code review and `ruff`/`mypy`; no `if op_name == "matmul"` in phase modules; adapter registry is the only place `op_name` is read outside Phase 2.

**INV-CS-014: `ambiguous_fault` never contributes optimization signal semantics in the returned record.** The record is structurally separable by `fault_class`; the service never emits a `candidate_fault` label for an event that advanced pod health to suspect.
*Enforcement:* attribution rules in §6.9 downgrade to `ambiguous_fault` when pod health transitions during the request; the result assembler refuses to emit `candidate_fault` with `pod_health=suspect` sampled during the same request.

**INV-CS-015: Every `CompileResult` is assembled in exactly one place.** Phase 5 is the single constructor of `CompileResult`; every failure path in earlier phases short-circuits to Phase 5 with its failure context; no code outside Phase 5 returns a `CompileResult`.
*Enforcement:* the `CompileResult` constructor is called from one file; static analysis (grep) confirms the rule; code review enforces.

---

## §8 Requirements

Every requirement is numbered `REQ-CS-NNN`; every scenario is numbered `SCN-CS-NNN-NN` and follows the `GIVEN / WHEN / THEN` pattern (AND clauses allowed). Each REQ traces to at least one SC-CS-*.

### 8.1 Successful Compile Path

**REQ-CS-001: Successful matmul compile returns populated static analysis with provenance.** [traces SC-CS-001]
When a request delivers well-formed matmul source whose entrypoint and ABI match `MatmulAdapter` and whose target arch is supported, the service must compile both the reference and candidate harnesses with `nvcc`, extract `registers_per_thread`, `smem_bytes_per_block`, `spill_loads`, `spill_stores`, and `occupancy_estimate_pct`, attach per-fact `resource_sources`, and return `status=success` with a non-null `static_analysis`.

**SCN-CS-001-01: Reference matmul produces success with cuda_func_attribute provenance.**
- GIVEN: a `CompileRequest` with `problem_spec.op_name="matmul"`, a reference matmul source that uses `entrypoint="matmul"`, `KernelExecutionSpec.block_dim=(16,16,1)`, `dynamic_smem_bytes=0`, `abi_name="matmul_v1"`, `abi_version="1.0"`
- AND: `target_arch` corresponds to a supported GPU (e.g. `sm_80`)
- AND: the pod is `healthy` and toolchain is complete
- WHEN: `POST /v1/compile` is invoked
- THEN: the response `status=success`
- AND: `static_analysis.registers_per_thread` is a non-null integer
- AND: `static_analysis.resource_sources.registers_per_thread ∈ {cuda_func_attribute, ptxas}`
- AND: `correctness.passed=true`, `correctness.oracle_kind="reference_kernel"`
- AND: `sanitizer_results` contains at least one outcome with `tool="memcheck"` and `status="pass"`
- AND: `artifacts.cubin_artifact_id` is non-null
- AND: `run_envelope.pod_health="healthy"`, `run_envelope.idempotency_state="new"`

**SCN-CS-001-02: cuFuncGetAttribute unavailable falls back to ptxas.**
- GIVEN: the driver API attribute read for `registers_per_thread` fails (e.g. `cuda-python` not importable)
- AND: `ptxas -v` output is parseable
- WHEN: static resource extraction runs
- THEN: `static_analysis.registers_per_thread` is populated from `ptxas`
- AND: `resource_sources.registers_per_thread="ptxas"`
- AND: no fabricated value is written

**SCN-CS-001-03: Both sources disagree; both recorded.**
- GIVEN: `cuFuncGetAttribute` reports 64 registers and `ptxas` reports 72
- WHEN: static resource extraction runs
- THEN: `resource_sources.registers_per_thread` records both sources with both values
- AND: `resource_conflicts` contains an entry for `registers_per_thread`
- AND: the reported `registers_per_thread` field equals the preferred value (from `cuFuncGetAttribute`)

### 8.2 Compile Error Paths

**REQ-CS-002: Syntax errors produce `compile_error` with bounded log and `syntax_error` kind.** [traces SC-CS-002]
When the candidate source fails to compile because of a parse-level error, the service must return `status=compile_error`, `fault_class=candidate_fault`, `candidate_fault_kind=syntax_error`, a bounded `failure.stderr_excerpt`, no cubin reference, and a non-null `candidate_fault_kind`.

**SCN-CS-002-01: Missing close brace yields syntax_error.**
- GIVEN: a candidate source with a `__global__` function body that omits its closing `}`
- WHEN: `POST /v1/compile` is invoked
- THEN: `status="compile_error"`, `fault_class="candidate_fault"`, `candidate_fault_kind="syntax_error"`
- AND: `failure.phase="compile"`, `failure.stderr_excerpt` is non-empty and bounded by `MAX_LOG_BYTES`
- AND: `artifacts.cubin_artifact_id` is null
- AND: the bounded excerpt contains the `nvcc` error token (e.g. `expected`)

**SCN-CS-002-02: Type mismatch yields semantic_compile_error.**
- GIVEN: a candidate source that passes parse but fails semantic analysis (e.g. assigning `half*` to `float*` without cast)
- WHEN: the request is processed
- THEN: `status="compile_error"`, `candidate_fault_kind="semantic_compile_error"`
- AND: `failure.stderr_excerpt` is bounded
- AND: `artifacts.cubin_artifact_id` is null

**SCN-CS-002-03: Log excerpt is bounded.**
- GIVEN: `nvcc` emits 512 KiB of stderr
- AND: `MAX_LOG_BYTES=65536`
- WHEN: failure record is assembled
- THEN: `failure.stderr_excerpt` is at most 65536 bytes
- AND: the excerpt ends with a truncation marker

### 8.3 Correctness Paths

**REQ-CS-003: Correctness mismatch returns `correctness_fail` with failing shapes and oracle kind.** [traces SC-CS-003, SC-CS-009]
When the candidate compiles but produces outputs outside tolerance on one or more shapes, the service must return `status=correctness_fail`, `fault_class=candidate_fault`, `candidate_fault_kind=correctness_mismatch`, a populated `correctness.failing_shape_ids`, `max_abs_error`, `max_rel_error`, and `correctness.oracle_kind="reference_kernel"`. No benchmark-ready artifact is emitted.

**SCN-CS-003-01: One failing shape among three.**
- GIVEN: `problem_spec.shape_cases` has three shapes
- AND: candidate output differs from reference on shape `small_1` beyond tolerance
- WHEN: the request is processed
- THEN: `status="correctness_fail"`, `candidate_fault_kind="correctness_mismatch"`
- AND: `correctness.failing_shape_ids=["small_1"]`
- AND: `correctness.max_abs_error` is non-null
- AND: `correctness.oracle_kind="reference_kernel"`
- AND: `artifacts.benchmark_ready_cubin_artifact_id` is absent

**SCN-CS-003-02: Zero-producing matmul candidate.**
- GIVEN: a candidate kernel that writes all zeros to output `C`
- WHEN: the request is processed
- THEN: `status="correctness_fail"`
- AND: every shape id in `problem_spec.shape_cases` appears in `correctness.failing_shape_ids`

**SCN-CS-003-03: Tolerance order resolution.**
- GIVEN: `ShapeCase.correctness_tolerance=1e-5` for shape `small_1`
- AND: adapter `default_tolerance(float)=1e-4`
- AND: service default `1e-3`
- WHEN: correctness runs on shape `small_1`
- THEN: the applied tolerance is `1e-5`
- AND: `correctness.tolerance_source="shape_case"`

**SCN-CS-003-04: Float comparison defaults to tolerance.**
- GIVEN: `problem_spec.dtype="fp16"`, adapter does not declare `exact`
- WHEN: correctness runs
- THEN: `correctness.comparison_mode="tolerance"`

### 8.4 Sanitizer Paths

**REQ-CS-004: Sanitizer gate runs memcheck by default and escalates on triggers.** [traces SC-CS-004]
Given a candidate that passes correctness, the service must run `memcheck` on the smallest shape, and escalate to `racecheck` / `synccheck` / `initcheck` when the documented triggers fire. Each invocation produces a typed `SanitizerOutcome` appended to `correctness.sanitizer_results`.

**SCN-CS-004-01: Default memcheck on smallest shape.**
- GIVEN: a candidate passes correctness on three shapes `[small, medium, large]`
- AND: the candidate does not use `__shared__`, `__syncthreads`, or cooperative groups
- WHEN: the sanitizer gate runs
- THEN: `correctness.sanitizer_results` contains exactly one entry with `tool="memcheck"` and `shape_id="small"`
- AND: `racecheck`/`synccheck`/`initcheck` are not invoked

**SCN-CS-004-02: Shared-memory kernel triggers racecheck and synccheck.**
- GIVEN: the candidate source lexically contains `__shared__` and `__syncthreads`
- WHEN: the sanitizer gate runs
- THEN: `correctness.sanitizer_results` includes entries with `tool="memcheck"`, `tool="racecheck"`, `tool="synccheck"`
- AND: each entry has a `shape_id` and a `status`

**SCN-CS-004-03: Memcheck failure yields sanitizer_fail / memory_safety_error.**
- GIVEN: memcheck reports an out-of-bounds write
- WHEN: the result is assembled
- THEN: `status="sanitizer_fail"`, `fault_class="candidate_fault"`, `candidate_fault_kind="memory_safety_error"`
- AND: the failing `SanitizerOutcome` has a non-null `report_artifact_id`

**SCN-CS-004-04: Racecheck failure yields race_or_sync_error.**
- GIVEN: racecheck reports a shared-memory race
- WHEN: the result is assembled
- THEN: `status="sanitizer_fail"`, `candidate_fault_kind="race_or_sync_error"`

**SCN-CS-004-05: Sanitizer timeout is infra_fault.**
- GIVEN: a candidate passes correctness
- AND: `compute-sanitizer memcheck` runs longer than `SANITIZER_TIMEOUT`
- WHEN: the invocation times out
- THEN: `status="timeout"`, `fault_class="infra_fault"`
- AND: the `SanitizerOutcome` has `status="timeout"`

### 8.5 Interface Contract Paths

**REQ-CS-005: Missing `KernelExecutionSpec` fields yield `interface_contract_error`.** [traces SC-CS-005]
When the inbound request does not include `entrypoint`, `block_dim`, `dynamic_smem_bytes`, `abi_name`, or `abi_version`, and `legacy_compatibility=false`, the service must return `status=interface_contract_error`, `fault_class=candidate_fault`, `candidate_fault_kind=interface_contract_error`, without invoking `nvcc` or touching the GPU.

**SCN-CS-005-01: Missing entrypoint on normal traffic.**
- GIVEN: a `CompileRequest` with `execution_spec.entrypoint=None` and `legacy_compatibility=false`
- WHEN: the request is processed
- THEN: `status="interface_contract_error"`, `candidate_fault_kind="interface_contract_error"`
- AND: `legacy_inferred_execution_spec=false`
- AND: no cubin artifact is produced
- AND: `failure.phase="request_normalization"`

**SCN-CS-005-02: Legacy compatibility infers when flag set.**
- GIVEN: a `CompileRequest` with missing `entrypoint` but `legacy_compatibility=true`
- AND: the source contains exactly one `__global__` function named `matmul`
- WHEN: the request is processed
- THEN: the inferred `entrypoint="matmul"`, `block_dim=MatmulAdapter.default_block_dim()`, `dynamic_smem_bytes=0`
- AND: `legacy_inferred_execution_spec=true`
- AND: `artifact_key` includes `legacy_inferred_execution_spec=true` as an input

**SCN-CS-005-03: Legacy compatibility without unique __global__ still fails.**
- GIVEN: `legacy_compatibility=true` and the source contains two `__global__` functions
- WHEN: legacy inference attempts
- THEN: `status="interface_contract_error"` because the inference rule's precondition is not satisfied

**REQ-CS-011: Unknown op_name yields unsupported_operation.** [traces SC-CS-011]
A request whose `problem_spec.op_name` is not in the adapter registry must return `status=interface_contract_error`, `reason="unsupported_operation"`, without compiling and without invoking an LLM.

**SCN-CS-011-01: Softmax request is rejected.**
- GIVEN: `problem_spec.op_name="softmax"`
- WHEN: the request is processed
- THEN: `status="interface_contract_error"`
- AND: `reason="unsupported_operation"`
- AND: no compile log is produced
- AND: no artifact is created

### 8.6 Idempotency

**REQ-CS-006: Idempotent replay returns the same result or reports lost prior attempt.** [traces SC-CS-006]
Two requests sharing the same `request_id` within `IDEMPOTENCY_TTL` must produce the same completed `CompileResult` (with `idempotency_state=reused_completed`) when the stored `artifact_key` matches, or `status=infra_error / reason=prior_attempt_lost_before_durability` when the prior attempt did not durably complete.

**SCN-CS-006-01: Second request returns reused_completed.**
- GIVEN: a first `POST /v1/compile` with `request_id=req_A` succeeded and the response was stored
- WHEN: a second `POST /v1/compile` with the same `request_id=req_A` and identical body is sent
- THEN: the response is byte-identical to the first except for `run_envelope.idempotency_state`
- AND: `run_envelope.idempotency_state="reused_completed"`
- AND: no re-compilation is performed

**SCN-CS-006-02: Replay with mismatched artifact_key falls through.**
- GIVEN: a stored completed entry for `request_id=req_B`
- AND: a second request with the same `request_id` but a different `source_code` (different `artifact_key`)
- WHEN: the service processes the replay
- THEN: an anomaly is logged
- AND: the stored entry is treated as stale
- AND: a fresh attempt runs with `idempotency_state="new"`

**SCN-CS-006-03: Replay of in-flight yields prior_attempt_lost.**
- GIVEN: a registry entry for `request_id=req_C` exists with `started_at` but no `completed_at`
- WHEN: a second request with `request_id=req_C` arrives
- THEN: `status="infra_error"`, `fault_class="infra_fault"`, `reason="prior_attempt_lost_before_durability"`
- AND: `run_envelope.previous_attempt_lost=true`
- AND: `run_envelope.prior_attempt_observed_phase` is the last recorded phase

### 8.7 Pod Health

**REQ-CS-007: Pod health is observable and advances on ambiguous faults.** [traces SC-CS-007]
Every result envelope carries the current `pod_health`. An ambiguous Phase 4 event transitions `healthy → suspect`. The next request runs a known-good probe before any candidate work. A probe failure or an ambiguous-failure counter above `AMBIGUOUS_FAILURE_LIMIT` transitions to `quarantined`, after which every subsequent result short-circuits with `infra_fault / pod_health=quarantined`.

**SCN-CS-007-01: Ambiguous fault transitions healthy to suspect.**
- GIVEN: the pod is `healthy`
- AND: a request triggers a `cudaErrorIllegalAddress` during Phase 4
- WHEN: the result is assembled
- THEN: `pod_health="suspect"` in the returned envelope
- AND: `fault_class="ambiguous_fault"`

**SCN-CS-007-02: Next request runs probe before candidate.**
- GIVEN: the pod is `suspect` after a prior ambiguous fault
- WHEN: a new `POST /v1/compile` arrives
- THEN: the service runs the known-good probe before the candidate's Phase 4
- AND: if the probe passes, the pod returns to `healthy` and the candidate proceeds

**SCN-CS-007-03: Probe failure quarantines the pod.**
- GIVEN: the pod is `suspect` and the probe kernel fails
- WHEN: the service assembles the result
- THEN: `pod_health="quarantined"`
- AND: the candidate did not run
- AND: `status="infra_error"`, `fault_class="infra_fault"`

**SCN-CS-007-04: Quarantined pod short-circuits subsequent requests.**
- GIVEN: the pod is `quarantined`
- WHEN: any `POST /v1/compile` arrives
- THEN: the response is `status="infra_error"`, `fault_class="infra_fault"`, `run_envelope.pod_health="quarantined"`
- AND: no compile or GPU work occurs

**SCN-CS-007-05: Ambiguous-failure counter over limit quarantines.**
- GIVEN: `AMBIGUOUS_FAILURE_LIMIT=3` and three consecutive ambiguous events occurred
- WHEN: the counter threshold is crossed
- THEN: `pod_health="quarantined"` before the fourth request is processed

### 8.8 Artifact Key

**REQ-CS-008: `artifact_key` reflects every executable-behavior input.** [traces SC-CS-008]
Every `CompileResult` includes an `artifact_key` computed from `hash(source_hash, problem_spec_hash, launch_spec_hash, target_arch, toolchain_hash, compile_flags_hash, adapter_version, legacy_inferred_execution_spec)`. Any change in any of those inputs produces a different key.

**SCN-CS-008-01: Source change produces a different artifact_key.**
- GIVEN: two requests identical except for one comment line in `source_code`
- WHEN: both are processed
- THEN: their `artifact_key` values differ

**SCN-CS-008-02: Launch spec change produces a different artifact_key.**
- GIVEN: two requests identical except for `execution_spec.block_dim`
- WHEN: both are processed
- THEN: their `artifact_key` values differ

**SCN-CS-008-03: Legacy-inferred flag produces a different artifact_key.**
- GIVEN: two requests with identical source, problem spec, and effective launch geometry
- AND: one has `legacy_inferred_execution_spec=false`, the other `=true`
- WHEN: both are processed
- THEN: their `artifact_key` values differ

### 8.9 Fault Attribution

**REQ-CS-009: Faults are attributed per the design doc table.** [traces SC-CS-009]
Each failure event maps to exactly one `fault_class` and (for `candidate_fault`) one `candidate_fault_kind`. Infra-class events never carry `candidate_fault_kind`; ambiguous events never carry `candidate_fault_kind` and never contribute optimization signal semantics.

**SCN-CS-009-01: nvcc missing yields infra_fault.**
- GIVEN: the pod starts with `nvcc` absent
- WHEN: the startup probe runs
- THEN: the process exits non-zero
- AND: had a request made it through, it would be `fault_class="infra_fault"` with no `candidate_fault_kind`

**SCN-CS-009-02: Kernel watchdog timeout yields ambiguous_fault.**
- GIVEN: a candidate triggers a kernel launch timeout
- WHEN: the result is assembled
- THEN: `fault_class="ambiguous_fault"`, `candidate_fault_kind` is null

**SCN-CS-009-03: Deterministic out-of-bounds on healthy pod yields candidate_fault.**
- GIVEN: the pod is `healthy`
- AND: memcheck reports a reproducible out-of-bounds write
- WHEN: the result is assembled
- THEN: `fault_class="candidate_fault"`, `candidate_fault_kind="memory_safety_error"`

**SCN-CS-009-04: Pod transition during request downgrades attribution.**
- GIVEN: a correctness-mismatch-looking event coincides with pod health advancing to `suspect`
- WHEN: the result is assembled
- THEN: `fault_class="ambiguous_fault"`, not `candidate_fault`

### 8.10 FastAPI Surface and Deployment

**REQ-CS-010: `POST /v1/compile` accepts `CompileRequest` JSON and returns `CompileResult` JSON; startup exits non-zero when toolchain incomplete.** [traces SC-CS-010, SC-CS-013]
The service exposes the four routes in §6.12. On container start, the toolchain probe runs; missing `nvcc`, driver, `compute-sanitizer`, or unwritable artifact root triggers a non-zero process exit.

**SCN-CS-010-01: Well-formed request returns typed response.**
- GIVEN: a running container on a GPU host with complete toolchain
- WHEN: `POST /v1/compile` is invoked with a valid body
- THEN: the HTTP response has status 200
- AND: the body parses as `CompileResult`
- AND: `status` is one of the seven `CompileResultStatus` values

**SCN-CS-010-02: Missing nvcc exits container.**
- GIVEN: a container whose `nvcc` is absent from the image
- WHEN: the container starts
- THEN: the process exits non-zero before `uvicorn` listens

**SCN-CS-010-03: /healthz returns 503 when driver missing.**
- GIVEN: `nvcc` is present but the NVIDIA driver is not visible
- WHEN: `GET /healthz` is invoked
- THEN: the response status is 503
- AND: the body identifies the missing dependency

**SCN-CS-010-04: Quarantined pod returns 503 on POST /v1/compile.**
- GIVEN: the pod is `quarantined`
- WHEN: `POST /v1/compile` is invoked
- THEN: the HTTP status is 503
- AND: the body is still a `CompileResult` with `status="infra_error"` and `pod_health="quarantined"`

**REQ-CS-013: Docker image critical path works on GPU host.** [traces SC-CS-013]
Building the delivered `Dockerfile` yields an image that, when run on a GPU host, passes `/healthz` and serves a successful `POST /v1/compile` for a reference matmul request.

**SCN-CS-013-01: Image builds and runs.**
- GIVEN: `docker build -f docker/compiler-service/Dockerfile` is invoked on an x86_64 host
- WHEN: the build completes
- THEN: `docker run` of the image on a GPU host answers `GET /healthz` with 200
- AND: `POST /v1/compile` with the reference matmul kernel from `examples/matmul_spec.yaml` returns `status="success"`

### 8.11 Artifact Store

**REQ-CS-012: Artifact store retains pinned roots and GC's by class TTL and disk watermark.** [traces SC-CS-012]
The store must never evict a pinned artifact or an artifact referenced by an in-TTL idempotency record. Non-pinned artifacts are evicted by class TTL; when disk utilization crosses `ARTIFACT_DISK_HIGH_WATERMARK`, an eager pass drops additional entries in class-TTL priority order.

**SCN-CS-012-01: Pinned baseline survives GC.**
- GIVEN: an artifact pinned under `role=baseline`
- WHEN: a GC pass runs over expired non-pinned entries
- THEN: the pinned artifact remains retrievable via `GET /v1/artifacts/{id}`

**SCN-CS-012-02: Expired non-pinned artifact is evicted.**
- GIVEN: a non-pinned compile-failure source artifact whose TTL has elapsed
- WHEN: GC runs
- THEN: `GET /v1/artifacts/{id}` returns 404

**SCN-CS-012-03: High watermark triggers eager GC.**
- GIVEN: disk utilization exceeds `ARTIFACT_DISK_HIGH_WATERMARK`
- WHEN: the next request completes
- THEN: eager GC runs
- AND: at least one non-pinned entry is evicted (if one exists)
- AND: no pinned or idempotency-referenced artifact is evicted

**SCN-CS-012-04: GC never deletes currently-referenced artifact.**
- GIVEN: an artifact referenced by an in-TTL idempotency entry
- WHEN: GC runs
- THEN: the artifact remains retrievable

---

## §9 Traceability Matrix

Every SC-CS-* from `.planning/plan.md` traces to at least one REQ-CS-*, one SCN-CS-*, and (where relevant) one INV-CS-*.

| SC-CS-* | REQ-CS-* | SCN-CS-* | INV-CS-* |
|---|---|---|---|
| SC-CS-001 (successful matmul with provenance) | REQ-CS-001 | SCN-CS-001-01, 01-02, 01-03 | INV-CS-003 |
| SC-CS-002 (compile error with bounded log) | REQ-CS-002 | SCN-CS-002-01, 02-02, 02-03 | INV-CS-005, INV-CS-015 |
| SC-CS-003 (correctness fail with oracle kind) | REQ-CS-003 | SCN-CS-003-01, 03-02, 03-03, 03-04 | INV-CS-006, INV-CS-011 |
| SC-CS-004 (sanitizer default memcheck + escalation) | REQ-CS-004 | SCN-CS-004-01, 04-02, 04-03, 04-04, 04-05 | INV-CS-004 |
| SC-CS-005 (missing execution spec → interface_contract_error) | REQ-CS-005 | SCN-CS-005-01, 05-02, 05-03 | INV-CS-002 |
| SC-CS-006 (idempotent replay or prior_attempt_lost) | REQ-CS-006 | SCN-CS-006-01, 06-02, 06-03 | INV-CS-009 |
| SC-CS-007 (pod health observable, probe, quarantine) | REQ-CS-007 | SCN-CS-007-01, 07-02, 07-03, 07-04, 07-05 | INV-CS-008, INV-CS-014 |
| SC-CS-008 (artifact_key covers all executable inputs) | REQ-CS-008 | SCN-CS-008-01, 08-02, 08-03 | INV-CS-002 |
| SC-CS-009 (fault attribution table) | REQ-CS-009, REQ-CS-003, REQ-CS-004 | SCN-CS-009-01, 09-02, 09-03, 09-04 | INV-CS-005, INV-CS-014 |
| SC-CS-010 (FastAPI POST /v1/compile + startup exit) | REQ-CS-010 | SCN-CS-010-01, 10-02, 10-03, 10-04 | INV-CS-012 |
| SC-CS-011 (unknown op_name rejected) | REQ-CS-011 | SCN-CS-011-01 | INV-CS-013 |
| SC-CS-012 (artifact store retention + GC + pinning) | REQ-CS-012 | SCN-CS-012-01, 12-02, 12-03, 12-04 | INV-CS-007 |
| SC-CS-013 (Dockerfile critical path on GPU host) | REQ-CS-013, REQ-CS-010 | SCN-CS-013-01, 10-01, 10-02 | INV-CS-012 |

**Additional structural invariants (not tied to a single SC but enforcing cross-cutting properties):**

- INV-CS-001 (ref + candidate never share TU) — prevents FM-CS-002; implied by SC-CS-001, SC-CS-003.
- INV-CS-010 (per-device GPU concurrency 1) — prevents FM-CS-010; implied by SC-CS-003, SC-CS-004.
- INV-CS-015 (single CompileResult constructor) — structural guarantee that every status/fault pair in SC-CS-002..SC-CS-009 is produced consistently.

Every SC-CS-* in the plan has at least one REQ-CS-* and at least one SCN-CS-*. Every REQ-CS-* traces back to one or more SC-CS-*. There are no dangling SCN-CS-* scenarios — each belongs to a REQ-CS-* that itself traces to an SC-CS-*.
