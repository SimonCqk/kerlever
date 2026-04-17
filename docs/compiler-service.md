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
    │  entrypoint:        candidate execution spec, or strict     │
    │                     one-__global__ compatibility inference  │
    │  launch geometry:   candidate execution spec, or adapter    │
    │                     default                                 │
    │  dynamic smem:      candidate execution spec, or zero       │
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
```

`grid_dim` should not be a required V1 field. For adapter-owned operations such as matmul, the grid is derived from the shape and `block_dim`. If a future kernel family needs custom grid policy or per-shape overrides, that should be added as structured execution metadata rather than inferred from source comments.

Compatibility fallback remains allowed only for early simple candidates:

- if exactly one `__global__` function exists, use it as the entrypoint;
- use the operation adapter's default block geometry;
- set dynamic shared memory to zero;
- mark the run envelope as using inferred execution metadata.

The fallback is not the canonical contract for production GPU search.

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
    └─────────────────────────────────────────────────────────────┘
```

The service should treat `request_id` as an idempotency key. Retrying the same request should either return the same completed result or clearly state that the previous attempt was lost before artifact durability.

V1 should support idempotent result reuse for the same `request_id`, but should not add cross-request compile artifact caching. Artifact caching is easy to get wrong before identity is stable; reusing the wrong cubin is worse than compiling again.

Artifact identity must still be defined up front and include every input that can change executable behavior:

```
artifact_key = hash(
  source_hash,
  problem_spec_hash,
  launch_spec_hash,
  target_arch,
  toolchain_hash,
  compile_flags_hash,
  adapter_version
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
| `registers_per_thread` | ptxas output / CUDA metadata | register pressure, occupancy binding resource |
| `smem_bytes_per_block` | ptxas + launch metadata | shared-memory pressure, block residency |
| `spill_loads`, `spill_stores` | ptxas output | local-memory spill evidence |
| `occupancy_estimate_pct` | resource model | first-order latency-hiding bound |
| cubin / PTX / SASS references | build artifacts | downstream static inspection and audit |
| toolchain identity | pod runtime | comparability across runs and pods |

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

After value correctness passes, run Compute Sanitizer on one small deterministic shape by default. A sanitizer failure is treated as a correctness failure because the kernel is not safe to benchmark, even if the sampled output happened to match.

Do not run sanitizer on every shape in the first version. It adds high latency and little extra signal for the control loop. Instead, use trigger-based escalation when the candidate or recent run history makes memory safety more suspicious:

- the candidate uses dynamic shared memory;
- the candidate has custom execution metadata rather than adapter defaults;
- correctness passes but output contains NaN or Inf anomalies;
- a previous candidate from the same generation batch failed sanitizer;
- the pod is healthy but a recent run produced an ambiguous GPU fault;
- the operation adapter marks a shape as high risk for indexing or layout.

Escalation means running sanitizer on additional selected shapes, not automatically every shape.

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
    │              "correctness_fail" | "sanitizer_fail" |      │
    │              "timeout" | "infra_error",                  │
    │    "candidate_hash": "...",                              │
    │    "run_envelope": { ... },                              │
    │    "toolchain": { ... },                                 │
    │    "static_analysis": {                                  │
    │       "registers_per_thread": 64,                        │
    │       "smem_bytes_per_block": 49152,                     │
    │       "spill_loads": 0,                                  │
    │       "spill_stores": 0,                                 │
    │       "occupancy_estimate_pct": 50.0                     │
    │    },                                                    │
    │    "correctness": { "oracle_kind": "...", ... },          │
    │    "artifacts": { "artifact_id": "...", ... },           │
    │    "fault_class": null | "candidate_fault" |             │
    │                   "infra_fault" | "ambiguous_fault",     │
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
| `correctness_fail` | record correctness failure; do not benchmark |
| `sanitizer_fail` | record correctness failure; do not benchmark |
| `timeout` | record infrastructure/error outcome unless clearly candidate-caused |
| `infra_error` | record error outcome; candidate should not be treated as a bad optimization idea |

Failure context should identify the phase, command, bounded stdout/stderr excerpt, failing shape id when relevant, and whether the failure is retryable. It must be structured enough for the Coding Agent to repair code without relying on prose scraping.

### Fault Attribution

The result must distinguish candidate failures from infra failures because the Orchestrator's search memory treats failures as learning signal. A CUDA syntax error should inform the Coding Agent; a pod disk failure or lost GPU should not poison the optimization direction.

| Fault Class | Examples | Search-Memory Meaning |
|-------------|----------|-----------------------|
| `candidate_fault` | CUDA syntax error, unresolved symbol, deterministic correctness mismatch, sanitizer memory error | valid negative signal for this candidate and possibly this generation pattern |
| `infra_fault` | pod disk full, artifact write failure, nvcc missing, GPU lost, driver reset, service timeout before process start | no optimization signal; retry elsewhere or mark service unhealthy |
| `ambiguous_fault` | kernel watchdog timeout, CUDA illegal access that poisons the context, process killed during GPU run | do not benchmark; retry policy depends on pod health and reproducibility |

The service should prefer conservative attribution. If pod health is degraded after a request, mark the result `ambiguous_fault` or `infra_fault`, not a clean candidate failure.

The GPU pipeline adapter should preserve fault attribution when mapping Compiler Service results into Orchestrator-visible outcomes:

| Fault Class | Attempt Record | Tabu / Direction Exhaustion | Retry Policy |
|-------------|----------------|-----------------------------|--------------|
| `candidate_fault` | record as candidate attempt | allowed to contribute to tabu and exhaustion logic | usually no retry unless failure is nondeterministic |
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
| `ARTIFACT_RETENTION` | keep all early | Retain enough evidence while the system is immature |
| `SANITIZER_SHAPE_POLICY` | smallest shape | Choose a cheap deterministic shape for memcheck |
| `SANITIZER_ESCALATION_POLICY` | trigger-based | Run sanitizer on additional shapes only for high-risk candidates or recent suspicious failures |
| `DEFAULT_COMPILE_FLAGS` | stable per run | Keep reference and candidate comparable |
| `POD_HEALTH_PROBE` | known-good kernel | Probe used before returning from suspect to healthy |
| `AMBIGUOUS_FAILURE_LIMIT` | low fixed count | Quarantine pod after repeated ambiguous GPU/runtime failures |
| `IDEMPOTENCY_TTL` | run-scoped | How long completed request ids are retained for safe retry |
| `ARTIFACT_CACHE` | disabled in V1 | Cross-request cubin reuse waits until artifact identity and toolchain hashing are stable |
