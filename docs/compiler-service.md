# Compiler Service

The Compiler Service is the deterministic GPU-side gate for CUDA source code. It decides **whether a kernel is executable and correct enough to measure**, not whether it is fast. It runs on the remote GPU pod, compiles reference and candidate kernels with the target CUDA toolchain, extracts static resource facts, validates correctness across the workload shapes, and hands only passing artifacts to the Benchmarker.

It is a **measurement-producing service**, not a search-policy module. It does not own persistence, candidate generation, benchmarking, profiling, or optimization decisions.

---

## Inputs

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│ Kernel Source         │  │ Problem Spec /       │  │ Baseline / Reference │  │ Remote GPU Pod     │
│ candidate or          │  │ Oracle Context       │  │ Kernel               │  │ Toolchain          │
│ reference source      │  │ op, shapes, dtype,   │  │ source used for      │  │ nvcc, driver, GPU, │
│ code_hash + intent    │  │ target GPU, tolerance│  │ correctness compare  │  │ artifact root      │
└─────────┬────────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └─────────┬──────────┘
          │                          │                          │                        │
          └─────────────┬────────────┴──────────────┬───────────┴────────────────────────┘
                        │                           │
                        ▼                           ▼
               ┌───────────────────────────────────────────────────────────────────────────┐
               │             GPU Pipeline Adapter / Orchestrator Request                  │
               │  request id, run id, role, candidate hash, execution spec                 │
               └──────────────────────────────────────┬────────────────────────────────────┘
                                                      │
                                                      ▼
      ═════════════════════════════════════════════════════════════════════
       Phase 1 - Request Normalization / Kernel Interface Resolution
      ═════════════════════════════════════════════════════════════════════
```

The service consumes a structured request from the GPU pipeline adapter. The Orchestrator may keep its current `GPUPipelineProtocol.evaluate()` abstraction while the adapter splits real work into Compiler Service, Benchmarker, and Profile Interpreter calls.

Persistence of global optimization state remains with the Orchestrator. Pod-local artifacts are implementation artifacts, not canonical search memory.

---

## Phase 1 - Request Normalization / Kernel Interface Resolution (Deterministic)

Validate that the source and problem context are concrete enough to compile and run. This phase does not touch the GPU.

```
    ┌─────────────────────────────────────────────────────────────┐
    │              Normalize Compile Request                      │
    │                                                             │
    │  1. Compute source hash and problem-spec hash               │
    │  2. Resolve target GPU to CUDA arch, for example sm_80      │
    │  3. Pin toolchain identity: nvcc, CUDA, driver, GPU name    │
    │  4. Select operation adapter from ProblemSpec.op_name       │
    │  5. Resolve candidate-owned execution spec                  │
    │  6. Create isolated per-request workspace                   │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Resolve Kernel Interface                       │
    │                                                             │
    │  entrypoint:        candidate execution spec (required)     │
    │  launch geometry:   candidate execution spec plus adapter   │
    │                     shape-derived grid                      │
    │  dynamic smem:      candidate execution spec (required)     │
    │  kernel ABI:        operation adapter contract              │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                         [ enter Phase 2 ]
```

### Kernel Execution Spec

Launch metadata belongs with the candidate, not with the Compiler Service. It is part of what makes the generated source executable and reproducible, so it should be attached to `KernelCandidate` and copied into the compile request by the GPU pipeline adapter.

```
KernelExecutionSpec
  entrypoint: string
  block_dim: [x, y, z]
  dynamic_smem_bytes: int
  abi_name: string
  abi_version: string
  metadata_mode: "explicit"
```

`grid_dim` should not be a required V1 field. For adapter-owned operations such as matmul, the grid is derived from the shape and `block_dim`. If a future kernel family needs custom grid policy or per-shape overrides, that should be added as structured execution metadata rather than inferred from source comments.

`KernelExecutionSpec` is a hard-required V1 field for new candidates. The Compiler Service must not infer entrypoint, block geometry, dynamic shared memory, or ABI for normal search traffic. A missing or incomplete execution spec is an `interface_contract_error`, not a convenience path.

Legacy compatibility inference is allowed only for migrating old candidates or old fixtures under an explicit compatibility flag:

- if exactly one `__global__` function exists, use it as the entrypoint;
- use the operation adapter's default block geometry;
- set dynamic shared memory to zero;
- set `metadata_mode = "legacy_inferred"`;
- set `legacy_inferred_execution_spec = true` in the run envelope and result.

Legacy-inferred candidates must be visibly marked in every result and artifact key. They should not be generated by the Coding Agent in normal operation. The point is migration support, not a second production contract.

### Run Envelope

Request normalization also creates the run envelope that follows the job through compile, correctness, artifact storage, and result mapping. This is part of the core service contract because LLM-generated CUDA makes bad source, timeouts, and poisoned GPU contexts normal operating conditions.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                         Run Envelope                         │
    │                                                             │
    │  identity:                                                   │
    │    run_id, round_id, candidate_hash, request_id              │
    │                                                             │
    │  reproducibility:                                            │
    │    source_hash, problem_spec_hash, launch_spec_hash,         │
    │    toolchain_hash, compile_flags_hash, adapter_version       │
    │                                                             │
    │  limits:                                                     │
    │    compile_timeout, run_timeout, sanitizer_timeout,           │
    │    max_source_bytes, max_log_bytes                            │
    │                                                             │
    │  observability:                                              │
    │    phase timings, pod id, gpu uuid, health state              │
    │                                                             │
    │  idempotency:                                                │
    │    idempotency_state, previous_attempt_lost,                  │
    │    prior_attempt_observed_phase                               │
    └─────────────────────────────────────────────────────────────┘
```

The service should treat `request_id` as an idempotency key. Retrying the same request should either return the same completed result or clearly state that the previous attempt was lost before artifact durability.

V1 should support idempotent result reuse for the same `request_id`, but should not add cross-request compile artifact caching. Artifact caching is easy to get wrong before identity is stable; reusing the wrong cubin is worse than compiling again.

If a prior attempt is known to have started but no durable result exists, return an explicit infrastructure result rather than silently recompiling:

```
status: "infra_error"
reason: "prior_attempt_lost_before_durability"
previous_attempt_lost: true
prior_attempt_observed_phase: "compile" | "correctness" | "sanitizer" | null
```

If V1 cannot persist idempotency state across process restarts, state that limitation explicitly in the deployment contract: in-process idempotency only, no cross-process reuse guarantee. Do not imply stronger durability than the service actually has.

Artifact identity must still be defined up front and include every input that can change executable behavior:

```
artifact_key = hash(
  source_hash,
  problem_spec_hash,
  launch_spec_hash,
  target_arch,
  toolchain_hash,
  compile_flags_hash,
  adapter_version,
  legacy_inferred_execution_spec
)
```

This prevents reusing a cubin compiled under a different CUDA version, architecture, launch contract, or adapter harness.

### Why Interface Resolution Is A First-Class Phase

CUDA source alone is not a runnable contract. To execute a generated kernel, the service must know:

- which `__global__` function to launch;
- block dimensions;
- dynamic shared memory bytes;
- expected kernel argument ABI;
- shape dimension mapping.

The design decision is direct: **launch metadata is candidate-owned structured data**, not a source comment, LLM convention, or Compiler Service guess.

---

## Phase 2 - Operation Harness Assembly (Deterministic)

The Compiler Service does not interpret `ProblemSpec.op_semantics` prose. It uses operation adapters keyed by `ProblemSpec.op_name`.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                 Select Operation Adapter                    │
    │                                                             │
    │  op_name = "matmul"  ──► MatmulAdapter                      │
    │  unknown op          ──► unsupported operation failure      │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 Render Host Harness                         │
    │                                                             │
    │  - allocate inputs and outputs for each ShapeCase           │
    │  - generate deterministic inputs from a stable seed         │
    │  - launch reference executable and candidate executable     │
    │  - compare outputs with shape/dtype tolerance               │
    │  - report per-shape error and runtime failure context       │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                         [ enter Phase 3 ]
```

### Why Operation Adapters

`op_semantics` helps agents reason, but it is not executable. A deterministic correctness harness needs hard facts:

| Concern | Owned By |
|---------|----------|
| Shape interpretation, for example `[M, N, K]` | Operation adapter |
| Tensor allocation and layout | Operation adapter |
| Kernel ABI | Operation adapter |
| Default launch geometry | Operation adapter |
| Tolerance defaults | Operation adapter |
| Input distribution | Operation adapter |

### V1 Adapter Boundary

V1 should start with `matmul`, because the repo already has `examples/matmul_spec.yaml`.

The matmul adapter can assume:

- `ShapeCase.dims = [M, N, K]`;
- dtype maps to a CUDA scalar type such as `half` or `float`;
- kernel ABI is `A, B, C, M, N, K`;
- default block geometry is conservative, for example `16x16x1`;
- grid geometry derives from `M` and `N`.

This keeps the first version honest and small. More operations are added by adding adapters, not by adding a general semantic parser.

Unsupported operations return an explicit unsupported-operation result. They must not fall back to an LLM-generated harness at runtime. The Compiler Service is part of the measurement truth; letting an LLM invent the harness would make correctness itself untrusted. LLMs can help draft new adapters offline, but an adapter only enters this service after it has deterministic tests and a registered ABI contract.

---

## Phase 3 - Compile / Static Resource Extraction (Deterministic)

Compile the harnessed CUDA source and extract resource facts. This is the first point where CUDA tooling becomes the source of truth.

```
    ┌─────────────────────────────────────────────────────────────┐
    │                       nvcc Compile                          │
    │                                                             │
    │  stable flags:                                              │
    │    -O3 -std=c++17 -lineinfo -arch=<sm_xx> -Xptxas=-v        │
    │                                                             │
    │  produce:                                                   │
    │    executable, cubin, ptx, sass, compile log                │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                 ┌──────────────┴───────────────┐
                 │                              │
          COMPILE FAIL                         PASS
                 │                              │
                 ▼                              ▼
    structured failure context       ┌────────────────────────────┐
    returned to Orchestrator          │ Static Resource Extraction │
                                      └──────────────┬─────────────┘
                                                     │
                                                     ▼
                                              [ enter Phase 4 ]
```

### Static Signals Summary

| Signal | Source | Why It Matters |
|--------|--------|----------------|
| `registers_per_thread` | CUDA function attributes, cross-checked with ptxas if available | register pressure, occupancy binding resource |
| `smem_bytes_per_block` | CUDA function attributes + launch metadata, cross-checked with ptxas if available | shared-memory pressure, block residency |
| `spill_loads`, `spill_stores` | ptxas output / SASS inspection if available | local-memory spill evidence |
| `occupancy_estimate_pct` | resource model | first-order latency-hiding bound |
| cubin / PTX / SASS references | build artifacts | downstream static inspection and audit |
| toolchain identity | pod runtime | comparability across runs and pods |

### Static Resource Extraction Sources

`ptxas -v` is useful but it is text output, not the only contract. The service should cross-check static resource facts through structured CUDA APIs after loading the compiled module:

- runtime API: `cudaFuncGetAttributes`, when the harness uses runtime symbols;
- driver API: `cuFuncGetAttribute`, when loading cubin/PTX modules;
- library/kernel API: `cuKernelGetAttribute`, when the CUDA version and artifact type expose kernels through that path.

Extraction policy:

| Fact | Preferred Source | Fallback | Missing/Disagreement Policy |
|------|------------------|----------|-----------------------------|
| registers per thread | CUDA function attribute | ptxas verbose output | if unavailable, `None`; if values disagree, keep both in `resource_sources` and mark `resource_conflict` |
| static shared memory | CUDA function attribute | ptxas verbose output | if unavailable, `None`; never infer from unrelated text |
| dynamic shared memory | `KernelExecutionSpec.dynamic_smem_bytes` | none | required by execution spec |
| spills | ptxas/SASS evidence | none | if unavailable, `None`; do not extrapolate |
| max threads per block | CUDA function attribute | device limits + launch spec for validation only | if unavailable, `None` |

The static analysis result should preserve provenance:

```
resource_sources:
  registers_per_thread: "cuda_func_attribute" | "ptxas" | null
  smem_bytes_per_block: "cuda_func_attribute" | "ptxas" | null
  spill_loads: "ptxas" | "sass" | null
  resource_conflicts: [ ... ]
```

### Occupancy Is A Derived Fact, Not A Guess

Occupancy should be computed from compile facts and target hardware constraints:

```
threads_per_block = block.x * block.y * block.z
warps_per_block   = ceil(threads_per_block / 32)

active_blocks = min(
  blocks_by_warps,
  blocks_by_registers,
  blocks_by_shared_memory,
  blocks_by_threads,
  max_blocks_per_sm
)

occupancy = active_blocks * warps_per_block / max_warps_per_sm
```

If a required input is unavailable, return the missing field as `None`. Do not fabricate numbers to make downstream policy look complete.

---

## Phase 4 - Correctness Validation (Deterministic GPU Execution)

Correctness is the gate between "compiles" and "worth benchmarking." The Benchmarker should never time a kernel that has not passed correctness for the workload shapes.

```
    ┌─────────────────────────────────────────────────────────────┐
    │              Build Reference And Candidate Separately       │
    │                                                             │
    │  reference executable: reference kernel + same harness      │
    │  candidate executable: candidate kernel + same harness      │
    │                                                             │
    │  reason: avoid symbol collisions and fragile source rewrite │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Run All Required Shape Cases                   │
    │                                                             │
    │  for each ShapeCase:                                        │
    │    - generate deterministic inputs                          │
    │    - run reference                                          │
    │    - run candidate                                          │
    │    - compare outputs                                        │
    │    - record max abs/rel error and failing shape ids         │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  │                            │
        CORRECTNESS FAIL                      PASS
                  │                            │
                  ▼                            ▼
       return failure context        optional sanitizer check
```

### Correctness Inputs

Inputs must be deterministic:

```
seed = hash(problem_spec, shape_id, dtype, "kerlever_correctness")
```

Reference and candidate executions use identical inputs. Candidate correctness must not depend on wall-clock randomness or request ordering.

### Tolerance Order

Tolerance should be resolved in this order:

1. `ShapeCase.correctness_tolerance`, if provided.
2. Operation adapter dtype default.
3. Service default for exact integer comparison or conservative floating comparison.

### Floating-Point Determinism

Deterministic inputs do not imply bit-exact floating-point outputs. Reference and candidate executions usually both run on GPU, and different reduction orders, fused instructions, tensor core paths, or warp-level scheduling can produce numerically valid differences. For floating dtypes, the correctness contract is tolerance-based unless an operation adapter explicitly declares bit-exact semantics.

The Compiler Service should record:

```
comparison_mode: "exact" | "tolerance"
tolerance_source: "shape_case" | "adapter_dtype_default" | "service_default"
max_abs_error
max_rel_error
failing_shape_ids
```

Agents should not be rewarded for forcing bitwise equality when the mathematical contract only requires tolerance. Bit-exact comparison belongs to integer or explicitly exact operations, not generic floating-point kernels.

### Oracle Model

V1 treats the reference kernel as the candidate comparison oracle. That is enough to answer "does this candidate match the baseline reference on the workload shapes?", but it is not an independent proof that the reference implements the mathematical operation correctly.

The correctness result should therefore record `oracle_kind`:

| Oracle Kind | Meaning |
|-------------|---------|
| `reference_kernel` | Candidate output is compared against `ProblemSpec.reference_kernel` output |
| `adapter_independent` | Reference and candidate are checked against an adapter-provided independent oracle |
| `hybrid` | Independent oracle is used on small or supported shapes, reference kernel is used elsewhere |

For matmul, the preferred independent oracle path after V1 is:

1. cuBLAS or cuBLASLt when dtype and layout are supported.
2. CPU float32 accumulation for small correctness and sanitizer shapes.
3. Reference-kernel comparison as fallback for shapes or dtypes not covered above.

Baseline bootstrap must preserve the chosen `oracle_kind` in metadata. Reports and search memory should not flatten all of these into an unqualified "correct" claim.

### Sanitizer Gate

After value correctness passes, run Compute Sanitizer `memcheck` on one small deterministic shape by default. A sanitizer failure is treated as a correctness failure because the kernel is not safe to benchmark, even if the sampled output happened to match.

Do not run sanitizer on every shape in the first version. It adds high latency and little extra signal for the control loop. Instead, use trigger-based escalation when the candidate or recent run history makes memory safety more suspicious:

- the candidate uses dynamic shared memory;
- the candidate has custom execution metadata rather than adapter defaults;
- correctness passes but output contains NaN or Inf anomalies;
- a previous candidate from the same generation batch failed sanitizer;
- the pod is healthy but a recent run produced an ambiguous GPU fault;
- the operation adapter marks a shape as high risk for indexing or layout.

Escalation means running sanitizer on additional selected shapes, not automatically every shape.

Sanitizer escalation should name the sub-tool. "Run sanitizer" is too vague for LLM-generated CUDA:

| Tool | Default / Trigger | What It Catches |
|------|-------------------|-----------------|
| `memcheck` | default small-shape gate after value correctness | out-of-bounds, misaligned accesses, device memory errors, hardware exceptions |
| `racecheck` | candidate uses shared memory tiling, cooperative groups, warp/shared communication, or recent race-like nondeterminism | shared-memory data access hazards |
| `synccheck` | candidate uses `__syncthreads`, `__syncwarp`, cooperative groups, barriers, or warp-specialized paths | invalid synchronization usage |
| `initcheck` | NaN/Inf anomalies, uninitialized accumulator suspicion, missing initialization in semantic diff, or adapter high-risk marker | uninitialized device memory reads |

Recommended order is `memcheck` first, then trigger-specific tools. `racecheck`, `synccheck`, and `initcheck` do not replace `memcheck`.

The result should preserve the exact tool and shape:

```
sanitizer_results:
  - tool: "memcheck" | "racecheck" | "synccheck" | "initcheck"
    shape_id: string
    status: "pass" | "fail" | "timeout" | "unsupported"
    report_artifact_id: string | null
```

### Pod Health During GPU Execution

GPU execution failures can poison the process or CUDA context for later candidates. Correctness and sanitizer runs should update a simple pod health state:

```
healthy -> suspect -> quarantined
```

- `healthy`: accept normal compile and correctness work.
- `suspect`: one ambiguous GPU/runtime failure occurred; finish cleanup, then run a known-good probe before accepting more GPU execution.
- `quarantined`: the known-good probe failed, the driver or GPU disappeared, artifact storage is broken, or repeated ambiguous failures occurred; stop accepting work and let the pod be recycled.

This health state is infra evidence, not optimization evidence. If a pod becomes suspect or quarantined, the result should preserve that attribution so the Orchestrator does not treat the candidate as a clean failed optimization attempt.

---

## Phase 5 - Output To GPU Pipeline Adapter

The service returns structured facts and a routing outcome. It does not directly update search memory or choose the next optimization direction.

```
    ┌───────────────────────────────────────────────────────────┐
    │                 Compiler Service Result                    │
    │                                                           │
    │  {                                                        │
    │    "status": "success" | "compile_error" |               │
    │              "interface_contract_error" |                 │
    │              "correctness_fail" | "sanitizer_fail" |      │
    │              "timeout" | "infra_error",                  │
    │    "candidate_hash": "...",                              │
    │    "run_envelope": { ... },                              │
    │    "idempotency": {                                      │
    │       "idempotency_state": "new" | "reused_completed" |  │
    │                            "prior_attempt_lost",         │
    │       "previous_attempt_lost": false                     │
    │    },                                                    │
    │    "legacy_inferred_execution_spec": false,              │
    │    "toolchain": { ... },                                 │
    │    "static_analysis": {                                  │
    │       "registers_per_thread": 64,                        │
    │       "smem_bytes_per_block": 49152,                     │
    │       "spill_loads": 0,                                  │
    │       "spill_stores": 0,                                 │
    │       "occupancy_estimate_pct": 50.0,                    │
    │       "resource_sources": { ... }                        │
    │    },                                                    │
    │    "correctness": { "oracle_kind": "...", ... },          │
    │    "sanitizer_results": [ ... ],                         │
    │    "artifacts": { "artifact_id": "...", ... },           │
    │    "fault_class": null | "candidate_fault" |             │
    │                   "infra_fault" | "ambiguous_fault",     │
    │    "candidate_fault_kind": null | "...",                 │
    │    "failure": { ... } | null                             │
    │  }                                                        │
    └──────────────────────────┬────────────────────────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 │                            │
                 ▼                            ▼
    ┌──────────────────────┐    ┌────────────────────────────┐
    │    Benchmarker       │    │       Orchestrator         │
    │    only on success   │    │ record failure outcome     │
    └──────────────────────┘    └────────────────────────────┘
```

### Routing Semantics

| Compiler Result | Next Step |
|-----------------|-----------|
| `success` | send artifact id to Benchmarker |
| `compile_error` | record compile failure; return context to Coding Agent in next prompt |
| `interface_contract_error` | record candidate contract failure; do not compile or benchmark |
| `correctness_fail` | record correctness failure; do not benchmark |
| `sanitizer_fail` | record correctness failure; do not benchmark |
| `timeout` | record infrastructure/error outcome unless clearly candidate-caused |
| `infra_error` | record error outcome; candidate should not be treated as a bad optimization idea |
| `infra_error` with `prior_attempt_lost_before_durability` | retry with same request id or fresh request according to Orchestrator policy; no optimization signal |

Failure context should identify the phase, command, bounded stdout/stderr excerpt, failing shape id when relevant, and whether the failure is retryable. It must be structured enough for the Coding Agent to repair code without relying on prose scraping.

### Fault Attribution

The result must distinguish candidate failures from infra failures because the Orchestrator's search memory treats failures as learning signal. It must also subclass candidate failures because not every candidate fault has the same search meaning. A missing semicolon should not exhaust a memory-tiling strategy, and a sanitizer race should usually get a repair attempt before the direction is treated as bad.

| Fault Class | Examples | Search-Memory Meaning |
|-------------|----------|-----------------------|
| `candidate_fault` | CUDA syntax error, unresolved symbol, deterministic correctness mismatch, sanitizer memory error | candidate-owned failure; inspect `candidate_fault_kind` before updating tabu or exhaustion |
| `infra_fault` | pod disk full, artifact write failure, nvcc missing, GPU lost, driver reset, service timeout before process start | no optimization signal; retry elsewhere or mark service unhealthy |
| `ambiguous_fault` | kernel watchdog timeout, CUDA illegal access that poisons the context, process killed during GPU run | do not benchmark; retry policy depends on pod health and reproducibility |

The service should prefer conservative attribution. If pod health is degraded after a request, mark the result `ambiguous_fault` or `infra_fault`, not a clean candidate failure.

Candidate fault kinds:

| Candidate Fault Kind | Examples | Search-Memory Treatment | Retry Policy |
|----------------------|----------|-------------------------|--------------|
| `syntax_error` | CUDA/C++ parse error, missing semicolon, malformed template | generation-quality signal only; should not contribute to direction tabu | no service retry; Coding Agent repair prompt |
| `semantic_compile_error` | type mismatch, invalid CUDA construct, unsupported intrinsic for target arch | weak negative signal for implementation pattern, not necessarily optimization direction | repair prompt, may retry same direction |
| `interface_contract_error` | missing `KernelExecutionSpec`, entrypoint absent, ABI mismatch, wrong argument contract | candidate contract failure; blocks evaluation but says little about optimization idea | repair execution metadata or adapter contract |
| `correctness_mismatch` | deterministic output outside tolerance, wrong shape/layout math | strong negative signal for the candidate idea in this context | no automatic retry unless failure is localized and repairable |
| `memory_safety_error` | memcheck out-of-bounds, misaligned access, illegal address with healthy pod | implementation may be repairable; weaker than correctness-mismatch for direction exhaustion | one repair retry may be valuable |
| `race_or_sync_error` | racecheck hazard, synccheck barrier/sync violation | implementation repair signal, especially for shared-memory/cooperative designs | repair retry before direction exhaustion |
| `uninitialized_memory_error` | initcheck failure, uninitialized accumulator/input read | implementation repair signal | repair retry before direction exhaustion |
| `candidate_runtime_error` | deterministic launch failure on healthy pod, invalid launch config | contract or implementation failure depending on detail | classify further when possible |

The GPU pipeline adapter should preserve fault attribution when mapping Compiler Service results into Orchestrator-visible outcomes:

| Fault Class / Kind | Attempt Record | Tabu / Direction Exhaustion | Retry Policy |
|--------------------|----------------|-----------------------------|--------------|
| `candidate_fault.syntax_error` | record as generation failure | never exhausts optimization direction by itself | Coding Agent repair |
| `candidate_fault.semantic_compile_error` | record as candidate attempt | low weight unless repeated in same pattern | repair or regenerate |
| `candidate_fault.interface_contract_error` | record as contract failure | no direction exhaustion | fix metadata/ABI |
| `candidate_fault.correctness_mismatch` | record as candidate attempt | can contribute strongly in comparable context | usually no service retry |
| `candidate_fault.memory_safety_error` | record as candidate attempt | contributes only after repair retry or repetition | repair retry useful |
| `candidate_fault.race_or_sync_error` | record as candidate attempt | contributes only after repair retry or repetition | repair retry useful |
| `candidate_fault.uninitialized_memory_error` | record as candidate attempt | contributes only after repair retry or repetition | repair retry useful |
| `infra_fault` | record as infra evaluation failure, not candidate evidence | never contributes | retry on another healthy pod or after recovery |
| `ambiguous_fault` | record as ambiguous evaluation failure | does not contribute until reproduced as candidate fault | one controlled retry with same run envelope, preferably fresh pod |

This keeps pod instability from polluting search memory. A failed GPU service run is not evidence that an optimization direction is bad.

---

## Full Phase Flow Summary

```
    INPUTS ──► Phase 1: Request Normalization / Interface Resolution
                (deterministic, no GPU execution)
                     │
                     ▼
               Phase 2: Operation Harness Assembly
                (adapter-driven, no natural-language parsing)
                     │
                     ▼
               Phase 3: Compile / Static Resource Extraction
                (nvcc, ptxas, cubin/PTX/SASS, resource facts)
                     │
          ┌──────────┴───────────┐
          │                      │
      compile fail              pass
          │                      │
          ▼                      ▼
  structured failure      Phase 4: Correctness Validation
  to Orchestrator          (all shapes + sanitizer gate)
                                 │
                    ┌────────────┴────────────┐
                    │                         │
             correctness fail                pass
                    │                         │
                    ▼                         ▼
          structured failure          Phase 5: Output Artifact
          to Orchestrator             to Benchmarker
```

---

## Boundary Decisions

### Compiler Service vs. Benchmarker

The Compiler Service may execute kernels to prove correctness, but it must not produce optimization scores. Correctness execution can be noisy, unsynchronized for timing, or sanitizer-instrumented. Benchmark results require a separate controlled measurement path.

### Compiler Service vs. Profile Interpreter

Static resource facts are compile-time facts: registers, shared memory, spills, occupancy estimate, artifact references. Bottleneck labels such as `memory_bandwidth` or `tensor_core_underutilized` belong to profiling and interpretation after benchmark selection.

### Compiler Service vs. Coding Agent

The Coding Agent generates source and intent. The Compiler Service returns structured failure evidence. It does not patch code, infer optimization strategy, or decide whether a failed idea is exhausted.

### Compiler Service vs. Orchestrator

The Orchestrator owns durable state. The Compiler Service owns pod-local artifacts and returns references. Global attempt records, incumbent updates, and round summaries are outside this module.

---

## Design Rationale

### Why Build Reference And Candidate Separately?

Generated kernels often reuse names from the reference kernel. Combining both into one translation unit creates symbol collisions and invites brittle source rewriting. Separate harness executables avoid that and make the comparison model simple: same inputs, same harness behavior, different kernel binary.

### Why Require Operation Adapters?

A CUDA correctness harness needs more structure than `op_semantics` can safely provide. Adapters keep deterministic behavior in code and make unsupported operations fail clearly rather than silently measuring nonsense.

### Why Keep Artifacts On The Pod?

PTX, cubin, SASS, logs, and executables can be large. The Benchmarker runs on the same remote GPU pod, so it can consume artifact ids locally. The Orchestrator needs metadata and references for audit, not large binary blobs.

Artifacts still need garbage collection from day one. "Keep everything while immature" will eventually crash long self-evolution runs. The service should enforce a per-run artifact budget and a high-watermark cleanup policy:

| Artifact Class | Keep | Drop / GC |
|----------------|------|-----------|
| baseline and current incumbent | source, cubin, PTX, SASS, compile log, static metadata, correctness summary | retain until run completion plus configured retention TTL |
| successful non-incumbent top-K/profiled candidate | cubin, PTX/SASS refs, compile log, static metadata, profile inputs needed by downstream analysis | drop after analysis TTL unless referenced by Orchestrator search memory |
| successful non-profiled non-incumbent | compact metadata and score references | drop bulky cubin/PTX/SASS after short TTL or when not needed by Benchmarker |
| compile failure | source hash, bounded source excerpt/ref, bounded compile log, failure classification | no cubin/SASS to keep; trim oversized logs |
| correctness failure | source hash, cubin short TTL, failing shape ids, max errors, bounded logs | drop cubin/PTX/SASS after repair/debug TTL unless explicitly pinned |
| sanitizer failure | sanitizer report artifact, tool name, shape id, bounded logs, cubin short TTL | drop cubin/PTX/SASS after repair/debug TTL unless explicitly pinned |
| infra or ambiguous failure | pod logs, health state, bounded stderr, request envelope | no optimization artifacts unless needed for retry diagnosis |

The artifact store should expose pinned roots (`baseline`, `incumbent`, active benchmark batch, active profile batch) and collect everything else by TTL and disk watermark. GC must never delete an artifact referenced by a durable Orchestrator record.

### Why Candidate-Owned Execution Metadata?

Execution metadata is part of the generated kernel artifact. If it lives only in the Compiler Service, then candidate identity no longer fully describes what was measured. Keeping entrypoint, block geometry, dynamic shared memory, and ABI version with the candidate makes compile artifacts reproducible and makes benchmark results auditable.

---

## Configuration Parameters

| Parameter | Default Direction | Description |
|-----------|-------------------|-------------|
| `COMPILE_TIMEOUT` | fixed service default | Maximum wall time for nvcc and artifact extraction |
| `CORRECTNESS_TIMEOUT` | fixed service default | Maximum wall time for all shape correctness runs |
| `SANITIZER_TIMEOUT` | fixed service default | Maximum wall time for the sanitizer shape |
| `MAX_SOURCE_BYTES` | small bounded limit | Protects the pod from pathological generated source |
| `CPU_COMPILE_CONCURRENCY` | bounded by pod CPU | Parallel compile capacity |
| `GPU_RUN_CONCURRENCY` | 1 per visible GPU | Prevents correctness jobs from corrupting each other |
| `ARTIFACT_RETENTION` | class-based TTL + disk watermark | Retain baseline/incumbent and active analysis artifacts; GC bulky unreferenced artifacts |
| `ARTIFACT_DISK_HIGH_WATERMARK` | service default | Trigger GC before pod disk exhaustion |
| `ARTIFACT_PIN_ROOTS` | baseline/incumbent/active batches | Durable references that GC must not delete |
| `SANITIZER_SHAPE_POLICY` | smallest shape | Choose a cheap deterministic shape for default memcheck |
| `SANITIZER_DEFAULT_TOOL` | memcheck | Default safety gate after value correctness |
| `SANITIZER_ESCALATION_TOOLS` | racecheck, synccheck, initcheck | Extra tools available when candidate structure or failures justify them |
| `SANITIZER_ESCALATION_POLICY` | trigger-based | Run additional tools or shapes only for high-risk candidates or recent suspicious failures |
| `DEFAULT_COMPILE_FLAGS` | stable per run | Keep reference and candidate comparable |
| `POD_HEALTH_PROBE` | known-good kernel | Probe used before returning from suspect to healthy |
| `AMBIGUOUS_FAILURE_LIMIT` | low fixed count | Quarantine pod after repeated ambiguous GPU/runtime failures |
| `IDEMPOTENCY_TTL` | run-scoped | How long completed request ids are retained for safe retry |
| `ARTIFACT_CACHE` | disabled in V1 | Cross-request cubin reuse waits until artifact identity and toolchain hashing are stable |
