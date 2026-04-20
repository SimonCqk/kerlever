# Compiler Service Module Design

> **Scope.** This document is the **implementation wiring** companion to
> `docs/compiler-service/spec.md`. The spec owns *what* the service does
> (behavior, invariants, external contract). This doc owns *how that behavior is
> composed in Python code* — module tree, class surface, dependency wiring,
> initialization order, call graph, concurrency primitives, external-tool
> wrappers, and container contract.
>
> Behavioral rules are NOT re-stated here. Where the behavior is defined in the
> spec, this doc links: "see spec §6.N". Duplication is a design smell.

---

## §1 Purpose and relationship to spec.md

| Concern | Lives in |
|---|---|
| Behavioral what/why of each subsystem (REQ-*, SCN-*, INV-*, PPT, Shortcut Risks) | `spec.md` §1–§9 |
| Configuration knobs and their meaning | `spec.md` §6.13 |
| External contract surface (routes, fields, status semantics) | `spec.md` §6.12, §3 |
| Class / function-level call graph, exact signatures | this doc §4, §6 |
| Module layout, DI, init order | this doc §3, §5 |
| Concurrency primitives, external-tool wrappers | this doc §8, §16 |
| FastAPI app factory, CLI entry, Dockerfile, `pyproject.toml` | this doc §12–§15 |

The Coding Agent reads both. If something is unspecified in *either* document,
the Architect owes a revision — the Coding Agent must ESCALATE rather than
silently invent. If the two documents disagree, spec.md wins on behavior and
design.md wins on wiring.

The design adheres to the plan's layout (`.planning/plan.md` §"Package
Layout") without deviation. The few name refinements below are explicitly
marked.

---

## §2 Architecture diagram

```
                        ┌──────────────────────────────────────────┐
                        │              HTTP / CLI Boundary          │
                        │  (FastAPI app or `python -m …` one-shot)  │
                        └──────────────────┬───────────────────────┘
                                           │  CompileRequest
                                           ▼
         ┌────────────────────────────────────────────────────────────┐
         │                      CompilerService                        │
         │   (stateless orchestrator — runs the 5 phases in order)     │
         └──────┬────────────┬────────────┬────────────┬─────────────┘
                │            │            │            │
                ▼            ▼            ▼            ▼
        ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐
        │ Phase1   │→│ Phase2    │→│ Phase3     │→│ Phase4     │→│ Phase5   │
        │ Request  │  │ Harness   │  │ Compile    │  │ Correctness│  │ Output   │
        │ Normalize│  │ Assemble  │  │ (nvcc,     │  │ (run ref + │  │ Assemble │
        │          │  │           │  │  ptxas,    │  │  cand,     │  │          │
        │          │  │           │  │  resources)│  │  sanitizer)│  │          │
        └────┬─────┘  └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘
             │              │              │               │              │
             │    ┌─────────┴──┐     ┌─────┴──────┐   ┌────┴──────┐       │
             │    │ Adapter    │     │ Toolchain  │   │ GPU       │       │
             │    │ Registry   │     │ wrappers   │   │ Semaphore │       │
             │    │            │     │ NvccRunner │   │ (per-dev) │       │
             │    │ Matmul     │     │ PtxasParser│   │           │       │
             │    │ Elementwise│     │ Cuobjdump  │   │ Compute-  │       │
             │    └────────────┘     │ Runner     │   │ Sanitizer │       │
             │                       │ Driver API │   │ Runner    │       │
             │                       └──────┬─────┘   └────┬──────┘       │
             │                              │              │              │
             ▼                              ▼              ▼              ▼
     ┌──────────────┐              ┌──────────────┐   ┌──────────────┐   ▼
     │ Idempotency  │              │ ToolchainInfo│   │ PodHealth    │ CompileResult
     │ Registry     │              │ (snapshot)   │   │ Tracker      │
     │ (asyncio.Lock)│             │              │   │ (singleton)  │
     └──────┬───────┘              └──────────────┘   └──────┬───────┘
            │                                                 │
            │             ┌──────────────────┐                │
            └────────────►│  ArtifactStore   │◄───────────────┘
                          │  (pinned roots + │
                          │   class TTL GC)  │
                          └─────────┬────────┘
                                    │
                                    ▼
                      filesystem: /var/lib/kerlever/artifacts/…

   External dependencies (bold boundaries):
      nvcc, ptxas, cuobjdump   — `/usr/local/cuda/bin/*` subprocesses
      compute-sanitizer        — `/usr/local/cuda/bin/compute-sanitizer`
      CUDA Driver API          — `cuda-python` bindings (cuFuncGetAttribute…)
      NVIDIA driver + GPU      — visible to the container via nvidia-runtime
```

**Protocol / ABC boundaries** (stable — the core never branches on `op_name`):

```
  OperationAdapter (Protocol)       ───►  MatmulAdapter, ElementwiseAdapter
  ResourceExtractor (Protocol)      ───►  DriverApiResourceExtractor,
                                          PtxasResourceExtractor
  SanitizerTool (StrEnum)           ───►  memcheck | racecheck | synccheck | initcheck
  FaultClassifier (pure function)   ───►  consumed by Phase5ResultAssembler only
```

Every arrow in the diagram is one-way. Phases never loop back; every failure
short-circuits forward to `Phase5ResultAssembler` (the sole constructor of
`CompileResult` per INV-CS-015).

---

## §3 Module layout

The package tree below is the plan's layout with only cosmetic refinements
(one new `errors.py`, one renamed `reference_kernels/` kept as-is). No module
is merged or split.

```
kerlever/compiler_service/
  __init__.py                  # Public re-exports: CompilerService,
                               # CompileRequest, CompileResult, create_app
  __main__.py                  # `python -m kerlever.compiler_service` → cli.main()

  types.py                     # All service-local Pydantic models + enums.
  config.py                    # ServiceConfig (env-var driven singleton).
  errors.py                    # Internal exception hierarchy
                               # (never surfaces across HTTP — converted by Phase5).

  identity.py                  # Hashing helpers: source_hash, problem_spec_hash,
                               # launch_spec_hash, compile_flags_hash,
                               # toolchain_hash, artifact_key.
  envelope.py                  # RunEnvelopeBuilder + PhaseTimer.
  faults.py                    # FaultClass, CandidateFaultKind enums,
                               # attribute_fault(status, pod_health, …) function.

  idempotency.py               # IdempotencyRegistry (in-memory, asyncio.Lock).
  pod_health.py                # PodHealthTracker singleton + known-good probe.
  artifact_store.py            # ArtifactStore + PinnedRoots + RetentionPolicy
                               # + GC passes.

  toolchain.py                 # ToolchainProbe (startup + /healthz)
                               # + ToolchainInfo snapshot + NvccRunner
                               # + PtxasParser + CuobjdumpRunner + driver-API
                               # bindings facade.
  sanitizer.py                 # ComputeSanitizerRunner + SanitizerPolicy
                               # (escalation triggers, lexical matching).
  resource_extraction.py       # StaticResourceExtractor (driver API preferred,
                               # ptxas fallback, provenance tagged).
  static_resource_model.py     # Occupancy formula + per-arch limits table.

  service.py                   # CompilerService orchestrator class
                               # (owns the phase instances, runs them).

  adapters/
    __init__.py                # AdapterRegistry + OperationAdapter Protocol.
    base.py                    # OperationAdapter Protocol definition.
    matmul.py                  # MatmulAdapter.
    elementwise.py             # ElementwiseAdapter (skeleton).

  phases/
    __init__.py
    phase1_request.py          # Phase1RequestNormalizer.
    phase2_harness.py          # Phase2HarnessAssembler.
    phase3_compile.py          # Phase3Compiler.
    phase4_correctness.py      # Phase4CorrectnessValidator.
    phase5_output.py           # Phase5ResultAssembler (sole CompileResult ctor).

  api/
    __init__.py
    app.py                     # create_app(config: ServiceConfig) -> FastAPI.
    schemas.py                 # HTTP-edge schemas (wrap service types).
    dependencies.py            # FastAPI Depends(...) providers.
    handlers.py                # Route handlers, bound via router.

  cli.py                       # One-shot compile CLI (stdin/file → stdout JSON).

  reference_kernels/
    vec_add.cu                 # Known-good probe kernel. Shipped with the package.

docker/compiler-service/
  Dockerfile
  entrypoint.sh
  .dockerignore
```

**Responsibility one-liners.**

| Module | Single responsibility |
|---|---|
| `types.py` | Pydantic v2 models + enums for requests, results, envelopes, faults (see §4.1). |
| `config.py` | Parse environment → `ServiceConfig`; cache the singleton. |
| `errors.py` | `CompilerServiceError` base + one subclass per internal short-circuit reason (never leaves the service). |
| `identity.py` | Deterministic hashes; no I/O. |
| `envelope.py` | Build `RunEnvelope`; record phase durations. |
| `faults.py` | Pure-function fault attribution per spec §6.9 table. |
| `idempotency.py` | `request_id` → attempt record; `stored_artifact_key` match enforcement (INV-CS-009). |
| `pod_health.py` | `healthy / suspect / quarantined` FSM + probe coordination (§6.8). |
| `artifact_store.py` | Write/read/pin/unpin/gc artifacts (§6.11). |
| `toolchain.py` | Detect & wrap `nvcc`, driver, sanitizer; snapshot `ToolchainInfo`. |
| `sanitizer.py` | Run `compute-sanitizer` sub-tools + decide escalation (§6.7). |
| `resource_extraction.py` | `StaticAnalysisExt` with provenance (§6.5, INV-CS-003). |
| `static_resource_model.py` | Closed-form occupancy formula + per-arch hardware table. |
| `service.py` | `CompilerService.compile(request) -> CompileResult` — no other entry point. |
| `adapters/*` | `OperationAdapter` Protocol + per-op implementations. |
| `phases/*` | One class per phase; immutable input → immutable output. |
| `api/app.py` | FastAPI app factory; startup/shutdown hooks. |
| `api/schemas.py` | HTTP edge adaptation (thin pass-through in V1). |
| `api/dependencies.py` | `Depends(...)` → singleton accessors. |
| `api/handlers.py` | Route handlers binding URL shapes to `CompilerService`. |
| `cli.py` | One-shot `python -m kerlever.compiler_service`. |

---

## §4 Core classes and function signatures

This section documents the *Python surface* that the Coding Agent must
implement. Every class carrying non-trivial state is listed with its
constructor and its key methods. Field semantics belong to spec.md — they are
not re-stated here.

### 4.1 `types.py` exports

Pydantic v2 models, all `frozen=True` except the mutable envelope fields.

```python
# Enums
class CompileResultStatus(StrEnum):
    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    INTERFACE_CONTRACT_ERROR = "interface_contract_error"
    CORRECTNESS_FAIL = "correctness_fail"
    SANITIZER_FAIL = "sanitizer_fail"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"

class CandidateRole(StrEnum):
    REFERENCE = "reference"
    CANDIDATE = "candidate"
    PROBE = "probe"

class MetadataMode(StrEnum):
    EXPLICIT = "explicit"
    LEGACY_INFERRED = "legacy_inferred"

class PodHealth(StrEnum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    QUARANTINED = "quarantined"

class IdempotencyState(StrEnum):
    NEW = "new"
    REUSED_COMPLETED = "reused_completed"
    PRIOR_ATTEMPT_LOST = "prior_attempt_lost"

class FaultClass(StrEnum):
    CANDIDATE_FAULT = "candidate_fault"
    INFRA_FAULT = "infra_fault"
    AMBIGUOUS_FAULT = "ambiguous_fault"

class CandidateFaultKind(StrEnum):
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_COMPILE_ERROR = "semantic_compile_error"
    INTERFACE_CONTRACT_ERROR = "interface_contract_error"
    CORRECTNESS_MISMATCH = "correctness_mismatch"
    MEMORY_SAFETY_ERROR = "memory_safety_error"
    RACE_OR_SYNC_ERROR = "race_or_sync_error"
    UNINITIALIZED_MEMORY_ERROR = "uninitialized_memory_error"
    CANDIDATE_RUNTIME_ERROR = "candidate_runtime_error"

class SanitizerTool(StrEnum):
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"
    INITCHECK = "initcheck"

class SanitizerStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"

class ResourceSource(StrEnum):
    CUDA_FUNC_ATTRIBUTE = "cuda_func_attribute"
    PTXAS = "ptxas"
    SASS = "sass"
    NULL = "null"

class OracleKind(StrEnum):
    REFERENCE_KERNEL = "reference_kernel"
    ADAPTER_INDEPENDENT = "adapter_independent"
    HYBRID = "hybrid"

class ComparisonMode(StrEnum):
    TOLERANCE = "tolerance"
    EXACT = "exact"

class ToleranceSource(StrEnum):
    SHAPE_CASE = "shape_case"
    ADAPTER_DTYPE_DEFAULT = "adapter_dtype_default"
    SERVICE_DEFAULT = "service_default"

class PhaseName(StrEnum):
    REQUEST_NORMALIZATION = "request_normalization"
    HARNESS_ASSEMBLY = "harness_assembly"
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    SANITIZER = "sanitizer"
    OUTPUT = "output"

# Requests / contracts
class KernelExecutionSpec(BaseModel):
    entrypoint: str | None = None
    block_dim: tuple[int, int, int] | None = None
    dynamic_smem_bytes: int | None = None
    abi_name: str | None = None
    abi_version: str | None = None
    metadata_mode: MetadataMode = MetadataMode.EXPLICIT

class RequestLimits(BaseModel):
    compile_timeout_s: float | None = None
    correctness_timeout_s: float | None = None
    sanitizer_timeout_s: float | None = None
    max_source_bytes: int | None = None
    max_log_bytes: int | None = None

class CompileRequest(BaseModel):
    request_id: str
    run_id: str
    round_id: str
    candidate_hash: str
    role: CandidateRole
    source_code: str
    problem_spec: ProblemSpec                 # reused from kerlever.types
    reference_source: str
    execution_spec: KernelExecutionSpec
    target_arch: str                          # e.g. "sm_80"
    legacy_compatibility: bool = False
    limits: RequestLimits | None = None

class ToolchainInfo(BaseModel, frozen=True):
    nvcc_version: str
    driver_version: str
    gpu_name: str
    gpu_uuid: str
    sanitizer_version: str
    toolchain_hash: str

class RunEnvelope(BaseModel):
    # identity
    run_id: str
    round_id: str
    request_id: str
    candidate_hash: str
    # reproducibility
    source_hash: str
    problem_spec_hash: str
    launch_spec_hash: str
    toolchain_hash: str
    compile_flags_hash: str
    adapter_version: str
    artifact_key: str
    # limits (resolved per-request)
    limits: RequestLimits
    # observability
    pod_id: str
    gpu_uuid: str
    phase_timings_ms: dict[PhaseName, float] = Field(default_factory=dict)
    # pod health
    pod_health: PodHealth
    # idempotency
    idempotency_state: IdempotencyState
    previous_attempt_lost: bool = False
    prior_attempt_observed_phase: PhaseName | None = None

class ResourceConflict(BaseModel, frozen=True):
    fact: str
    sources: list[tuple[ResourceSource, int | None]]
    preferred_value: int | None

class StaticAnalysisExt(BaseModel, frozen=True):
    # extends kerlever.types.StaticAnalysis
    base: StaticAnalysis
    resource_sources: dict[str, ResourceSource]
    resource_conflicts: list[ResourceConflict] = Field(default_factory=list)
    cubin_artifact_id: str | None = None
    ptx_artifact_id: str | None = None
    sass_artifact_id: str | None = None

class SanitizerOutcome(BaseModel, frozen=True):
    tool: SanitizerTool
    shape_id: str
    status: SanitizerStatus
    report_artifact_id: str | None = None

class CorrectnessResultExt(BaseModel, frozen=True):
    base: CorrectnessResult                   # reused from kerlever.types
    oracle_kind: OracleKind
    comparison_mode: ComparisonMode
    tolerance_source: ToleranceSource
    tolerance_value: float
    sanitizer_results: list[SanitizerOutcome] = Field(default_factory=list)

class FailureDetail(BaseModel, frozen=True):
    phase: PhaseName
    command: str | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    failing_shape_id: str | None = None
    retryable: bool = False
    reason: str | None = None                 # e.g. "unsupported_operation"

class ArtifactRefs(BaseModel, frozen=True):
    source_artifact_id: str | None = None
    executable_artifact_id: str | None = None
    reference_executable_artifact_id: str | None = None
    cubin_artifact_id: str | None = None
    ptx_artifact_id: str | None = None
    sass_artifact_id: str | None = None
    compile_log_artifact_id: str | None = None
    sanitizer_report_artifact_ids: list[str] = Field(default_factory=list)
    correctness_log_artifact_id: str | None = None

class CompileResult(BaseModel, frozen=True):
    status: CompileResultStatus
    candidate_hash: str
    run_envelope: RunEnvelope
    legacy_inferred_execution_spec: bool = False
    toolchain: ToolchainInfo
    static_analysis: StaticAnalysisExt | None = None
    correctness: CorrectnessResultExt | None = None
    artifacts: ArtifactRefs = Field(default_factory=ArtifactRefs)
    fault_class: FaultClass | None = None
    candidate_fault_kind: CandidateFaultKind | None = None
    failure: FailureDetail | None = None
```

### 4.2 `service.py` — the orchestrator

```python
@dataclass(frozen=True)
class CompilerServiceDeps:
    config: ServiceConfig
    toolchain: ToolchainInfo
    artifact_store: ArtifactStore
    pod_health: PodHealthTracker
    idempotency: IdempotencyRegistry
    adapter_registry: AdapterRegistry
    # per-device Phase4 semaphore, keyed by GPU index
    gpu_semaphores: Mapping[int, asyncio.Semaphore]
    # Phase3 global compile parallelism
    compile_semaphore: asyncio.Semaphore
    # External-tool wrappers (stateless objects; can be shared)
    nvcc: NvccRunner
    cuobjdump: CuobjdumpRunner
    sanitizer: ComputeSanitizerRunner
    resource_extractor: StaticResourceExtractor
    # Pod identity used in envelopes
    pod_id: str


class CompilerService:
    def __init__(self, deps: CompilerServiceDeps) -> None: ...

    async def compile(self, request: CompileRequest) -> CompileResult:
        """Run Phase 1→5. Sole coroutine a request handler calls."""
```

`CompilerService.compile` owns the happy-path phase sequencing. It does NOT
construct `CompileResult` on any path — every phase hands control to
`Phase5ResultAssembler` on both success and failure (INV-CS-015). The private
sequencer is implemented as:

```python
async def compile(self, request: CompileRequest) -> CompileResult:
    timer = PhaseTimer()
    phase5 = Phase5ResultAssembler(self._deps, timer)

    try:
        phase1_out = await self._phase1.run(request, timer)
        if phase1_out.short_circuit is not None:
            return phase5.from_short_circuit(request, phase1_out)

        phase2_out = await self._phase2.run(phase1_out, timer)
        if phase2_out.short_circuit is not None:
            return phase5.from_short_circuit(request, phase2_out)

        phase3_out = await self._phase3.run(phase2_out, timer)
        if phase3_out.short_circuit is not None:
            return phase5.from_short_circuit(request, phase3_out)

        phase4_out = await self._phase4.run(phase3_out, timer)
        return phase5.assemble(request, phase1_out, phase3_out, phase4_out)
    finally:
        self._deps.idempotency.finalize_if_pending(request.request_id)
        await self._deps.artifact_store.gc_cheap_pass()
```

Every `PhaseNOutput` dataclass carries both the happy-path payload and an
optional `short_circuit: PhaseShortCircuit | None`. The assembler is the only
object that reads `short_circuit`.

### 4.3 Phase classes

Each phase class is a small, stateless object whose constructor receives the
`CompilerServiceDeps` it needs (narrow, not the full bag). Each has **one**
public async method, `run(...)`, returning its own frozen output dataclass.

```python
# phase1_request.py
@dataclass(frozen=True)
class Phase1Output:
    request: CompileRequest
    envelope_seed: RunEnvelopeSeed              # computed hashes + pod_health + idemp state
    resolved_execution_spec: KernelExecutionSpec
    legacy_inferred_execution_spec: bool
    short_circuit: PhaseShortCircuit | None = None

class Phase1RequestNormalizer:
    def __init__(
        self,
        config: ServiceConfig,
        toolchain: ToolchainInfo,
        pod_health: PodHealthTracker,
        idempotency: IdempotencyRegistry,
        adapter_registry: AdapterRegistry,
    ) -> None: ...

    async def run(self, request: CompileRequest, timer: PhaseTimer) -> Phase1Output: ...


# phase2_harness.py
@dataclass(frozen=True)
class HarnessArtifacts:
    reference_source_path: Path
    candidate_source_path: Path
    adapter: OperationAdapter

@dataclass(frozen=True)
class Phase2Output:
    phase1: Phase1Output
    harness: HarnessArtifacts
    short_circuit: PhaseShortCircuit | None = None

class Phase2HarnessAssembler:
    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
        adapter_registry: AdapterRegistry,
    ) -> None: ...

    async def run(self, phase1: Phase1Output, timer: PhaseTimer) -> Phase2Output: ...


# phase3_compile.py
@dataclass(frozen=True)
class CompileArtifacts:
    reference_executable: Path
    candidate_executable: Path
    cubin_artifact_id: str
    ptx_artifact_id: str
    sass_artifact_id: str
    compile_log_artifact_id: str
    source_artifact_id: str

@dataclass(frozen=True)
class Phase3Output:
    phase2: Phase2Output
    compile: CompileArtifacts
    static_analysis: StaticAnalysisExt
    short_circuit: PhaseShortCircuit | None = None

class Phase3Compiler:
    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
        nvcc: NvccRunner,
        cuobjdump: CuobjdumpRunner,
        resource_extractor: StaticResourceExtractor,
        compile_semaphore: asyncio.Semaphore,
    ) -> None: ...

    async def run(self, phase2: Phase2Output, timer: PhaseTimer) -> Phase3Output: ...


# phase4_correctness.py
@dataclass(frozen=True)
class CorrectnessOutcome:
    correctness: CorrectnessResultExt            # holds sanitizer_results
    pod_health_transition: PodHealthTransition | None

@dataclass(frozen=True)
class Phase4Output:
    phase3: Phase3Output
    correctness_outcome: CorrectnessOutcome
    short_circuit: PhaseShortCircuit | None = None

class Phase4CorrectnessValidator:
    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
        sanitizer_runner: ComputeSanitizerRunner,
        sanitizer_policy: SanitizerPolicy,
        pod_health: PodHealthTracker,
        gpu_semaphores: Mapping[int, asyncio.Semaphore],
    ) -> None: ...

    async def run(self, phase3: Phase3Output, timer: PhaseTimer) -> Phase4Output: ...


# phase5_output.py
class Phase5ResultAssembler:
    def __init__(
        self,
        deps: CompilerServiceDeps,
        timer: PhaseTimer,
    ) -> None: ...

    def assemble(
        self,
        request: CompileRequest,
        phase1: Phase1Output,
        phase3: Phase3Output,
        phase4: Phase4Output,
    ) -> CompileResult: ...

    def from_short_circuit(
        self,
        request: CompileRequest,
        phase_output: PhaseShortCircuitCarrier,
    ) -> CompileResult: ...
```

`Phase5ResultAssembler` is the **only** place `CompileResult(...)` appears in
the codebase (INV-CS-015). Grep enforcement is part of review.

### 4.4 Adapters

```python
# adapters/base.py
class OperationAdapter(Protocol):
    op_name: ClassVar[str]

    def adapter_version(self) -> str: ...
    def abi_contract(self) -> tuple[str, str]:  # (abi_name, abi_version)
        ...
    def default_block_dim(self, problem_spec: ProblemSpec) -> tuple[int, int, int]: ...
    def default_tolerance(self, dtype: str) -> float: ...
    def comparison_mode(self, dtype: str) -> ComparisonMode: ...
    def high_risk_shape_ids(self, problem_spec: ProblemSpec) -> set[str]: ...

    def allocate_inputs(
        self, problem_spec: ProblemSpec, shape: ShapeCase, seed: int
    ) -> InputBundle: ...

    def build_harness_source(
        self,
        execution_spec: KernelExecutionSpec,
        problem_spec: ProblemSpec,
        role: CandidateRole,
        kernel_source: str,
    ) -> str: ...

    def compare_outputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        reference_output: Path,
        candidate_output: Path,
        tolerance: float,
        comparison_mode: ComparisonMode,
    ) -> ShapeComparisonResult: ...


# adapters/__init__.py
class AdapterRegistry:
    def __init__(self, adapters: Sequence[OperationAdapter]) -> None: ...
    def get(self, op_name: str) -> OperationAdapter | None: ...
    def names(self) -> frozenset[str]: ...


# adapters/matmul.py
class MatmulAdapter:
    op_name: ClassVar[str] = "matmul"
    # …implements OperationAdapter per spec §6.2


# adapters/elementwise.py
class ElementwiseAdapter:
    op_name: ClassVar[str] = "elementwise"
    # …implements OperationAdapter per spec §6.2
```

The registry is constructed once at startup with the V1 adapter list; it is
frozen (`frozenset` keys, immutable list of instances).

### 4.5 Singletons and registries

```python
# idempotency.py
@dataclass
class IdempotencyEntry:
    request_id: str
    started_at: datetime
    phase_observed: PhaseName
    artifact_key: str | None = None
    artifact_refs: list[str] = field(default_factory=list)
    completed_at: datetime | None = None
    compile_result: CompileResult | None = None

class IdempotencyRegistry:
    def __init__(self, ttl: timedelta) -> None: ...
    async def observe_intake(self, request_id: str) -> IdempotencyIntake: ...
    async def record_phase(self, request_id: str, phase: PhaseName) -> None: ...
    async def finalize(
        self, request_id: str, artifact_key: str,
        refs: Sequence[str], result: CompileResult
    ) -> None: ...
    def finalize_if_pending(self, request_id: str) -> None: ...
    def referenced_artifact_ids(self) -> frozenset[str]: ...  # for GC skip-set
    def purge_expired(self) -> int: ...


# pod_health.py
@dataclass(frozen=True)
class PodHealthTransition:
    previous: PodHealth
    current: PodHealth
    reason: Literal["clean_pass", "ambiguous_event", "probe_pass",
                    "probe_fail", "ambiguous_limit_exceeded"]

class PodHealthTracker:
    def __init__(
        self,
        ambiguous_limit: int,
        probe_source_path: Path,
        probe_binary_artifact_id_ref: Callable[[], str | None],
    ) -> None: ...

    def snapshot(self) -> PodHealth: ...
    def needs_probe(self) -> bool: ...
    async def run_probe_if_needed(
        self, runner: ProbeRunner
    ) -> PodHealthTransition | None: ...
    def record_phase4_outcome(
        self, classification: Phase4Classification
    ) -> PodHealthTransition | None: ...


# artifact_store.py
class ArtifactStore:
    def __init__(
        self,
        root: Path,
        retention: RetentionPolicy,
        high_watermark_pct: float,
        pinned_roots: PinnedRoots,
    ) -> None: ...

    async def write(
        self, kind: ArtifactKind, bytes_: bytes, run_id: str, candidate_hash: str
    ) -> str: ...
    async def write_stream(
        self, kind: ArtifactKind, source: Path, run_id: str, candidate_hash: str
    ) -> str: ...
    async def read(self, artifact_id: str) -> AsyncIterator[bytes]: ...
    async def path_of(self, artifact_id: str) -> Path | None: ...

    def pin(self, role: PinRole, artifact_id: str) -> None: ...
    def unpin(self, role: PinRole, artifact_id: str) -> None: ...

    async def gc_cheap_pass(self) -> int: ...
    async def gc_eager_if_over_watermark(
        self, referenced_ids: frozenset[str]
    ) -> int: ...


# toolchain.py
class ToolchainProbe:
    def __init__(self, config: ServiceConfig) -> None: ...
    def run(self) -> ToolchainProbeResult: ...
    def snapshot(self) -> ToolchainInfo: ...                 # requires run() succeeded

class NvccRunner:
    def __init__(self, nvcc_path: Path, config: ServiceConfig) -> None: ...
    async def compile(
        self,
        source: Path,
        output: Path,
        target_arch: str,
        extra_flags: Sequence[str] = (),
        timeout_s: float | None = None,
        max_log_bytes: int | None = None,
    ) -> NvccResult: ...

class PtxasParser:
    @staticmethod
    def parse(stdout: str, stderr: str) -> PtxasReport: ...

class CuobjdumpRunner:
    def __init__(self, cuobjdump_path: Path) -> None: ...
    async def extract_sass(
        self, executable: Path, output: Path, timeout_s: float
    ) -> CuobjdumpResult: ...

class DriverApiAttributes:
    """Thin facade over cuda-python cuFuncGetAttribute.
    Importable at startup; if the import fails, all methods raise
    DriverApiUnavailable and callers fall back to ptxas."""
    def __init__(self) -> None: ...
    def read_registers_per_thread(self, binary: Path, entrypoint: str) -> int | None: ...
    def read_static_smem_bytes(self, binary: Path, entrypoint: str) -> int | None: ...
    def read_max_threads_per_block(self, binary: Path, entrypoint: str) -> int | None: ...


# sanitizer.py
class ComputeSanitizerRunner:
    def __init__(self, sanitizer_path: Path, config: ServiceConfig) -> None: ...
    async def run(
        self,
        tool: SanitizerTool,
        executable: Path,
        shape: ShapeCase,
        input_dir: Path,
        timeout_s: float,
    ) -> SanitizerOutcome: ...

class SanitizerPolicy:
    def decide(
        self,
        candidate_source: str,
        correctness_outputs: Sequence[Path],
        problem_spec: ProblemSpec,
        adapter: OperationAdapter,
        recent_history: PodHealthTracker,
    ) -> list[SanitizerTool]: ...
    @staticmethod
    def smallest_shape(shapes: Sequence[ShapeCase]) -> ShapeCase: ...


# resource_extraction.py
class StaticResourceExtractor:
    def __init__(
        self,
        driver_api: DriverApiAttributes | None,        # None if unavailable
        ptxas_parser: PtxasParser,
        arch_model: StaticResourceModel,
    ) -> None: ...
    def extract(
        self,
        binary: Path,
        entrypoint: str,
        block_dim: tuple[int, int, int],
        dynamic_smem_bytes: int,
        target_arch: str,
        ptxas_stdout: str,
    ) -> StaticAnalysisExt: ...


# static_resource_model.py
class StaticResourceModel:
    def limits_for(self, target_arch: str) -> ArchLimits | None: ...
    def compute_occupancy(
        self,
        block_dim: tuple[int, int, int],
        registers_per_thread: int | None,
        smem_bytes_per_block: int | None,
        dynamic_smem_bytes: int,
        target_arch: str,
    ) -> float | None: ...
```

### 4.6 `identity.py` — hash helpers

Pure functions, no I/O, no state. Each returns a hex-encoded SHA-256 string
(truncated to a display width where noted in the spec).

```python
def source_hash(source_code: str) -> str: ...
def problem_spec_hash(ps: ProblemSpec) -> str: ...
def launch_spec_hash(es: KernelExecutionSpec) -> str: ...
def compile_flags_hash(flags: Sequence[str]) -> str: ...
def toolchain_hash(info: Mapping[str, str]) -> str: ...

def artifact_key(
    *,
    source_hash: str,
    problem_spec_hash: str,
    launch_spec_hash: str,
    target_arch: str,
    toolchain_hash: str,
    compile_flags_hash: str,
    adapter_version: str,
    legacy_inferred_execution_spec: bool,
) -> str: ...
```

### 4.7 `faults.py` — attribution

```python
def attribute_fault(
    *,
    status: CompileResultStatus,
    pod_health_during_request: PodHealth,          # sampled at assembly time
    pod_health_transitioned: bool,                 # did it move during request?
    last_sanitizer_tool: SanitizerTool | None,
    cuda_error: CudaErrorKind | None,              # illegal_address, watchdog…
    compile_stderr_pattern: SyntaxPatternHit | None,
) -> tuple[FaultClass | None, CandidateFaultKind | None]:
    """Pure function implementing the spec §6.9 table."""
```

`attribute_fault` is consumed only by `Phase5ResultAssembler`.

---

## §5 Dependency injection and init order

### 5.1 `create_app` lifecycle

```python
# api/app.py
def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """
    Build the FastAPI app plus every singleton needed by the request path.
    Construction order matters because some singletons depend on others.
    """
    config = config or ServiceConfig.from_env()
    app = FastAPI(title="kerlever-compiler-service", lifespan=_lifespan(config))
    app.include_router(build_router())
    return app


@asynccontextmanager
async def _lifespan(config: ServiceConfig) -> AsyncIterator[None]:
    # --- 1. Toolchain probe MUST succeed before any other singleton ---
    probe = ToolchainProbe(config)
    probe_result = probe.run()
    if not probe_result.ok:
        # INV-CS-012: startup exits non-zero on toolchain missing.
        sys.stderr.write(probe_result.as_error_json() + "\n")
        raise SystemExit(1)
    toolchain_info = probe.snapshot()

    # --- 2. Pod identity + CUDA device enumeration ---
    pod_id = resolve_pod_id(config)
    visible_gpus = enumerate_visible_gpus(config)     # e.g. {0: gpu_uuid_0}

    # --- 3. Artifact store (needs its root to be writable; probe already checked). ---
    retention = RetentionPolicy.from_config(config)
    pinned_roots = PinnedRoots(roots=config.ARTIFACT_PIN_ROOTS)
    artifact_store = ArtifactStore(
        root=config.KERLEVER_ARTIFACT_ROOT,
        retention=retention,
        high_watermark_pct=config.ARTIFACT_DISK_HIGH_WATERMARK,
        pinned_roots=pinned_roots,
    )

    # --- 4. Idempotency registry (in-memory, TTL-bounded). ---
    idempotency = IdempotencyRegistry(ttl=config.IDEMPOTENCY_TTL)

    # --- 5. Pod health + probe kernel compile (lazy, but kernel source known). ---
    pod_health = PodHealthTracker(
        ambiguous_limit=config.AMBIGUOUS_FAILURE_LIMIT,
        probe_source_path=config.POD_HEALTH_PROBE,
        probe_binary_artifact_id_ref=lambda: _probe_binary_id(artifact_store),
    )

    # --- 6. Adapter registry (frozen after construction). ---
    adapter_registry = AdapterRegistry([MatmulAdapter(), ElementwiseAdapter()])

    # --- 7. Concurrency primitives. ---
    compile_semaphore = asyncio.Semaphore(config.CPU_COMPILE_CONCURRENCY)
    gpu_semaphores: dict[int, asyncio.Semaphore] = {
        gpu_index: asyncio.Semaphore(config.GPU_RUN_CONCURRENCY)
        for gpu_index in visible_gpus
    }

    # --- 8. External-tool wrappers (stateless). ---
    nvcc = NvccRunner(probe_result.nvcc_path, config)
    cuobjdump = CuobjdumpRunner(probe_result.cuobjdump_path)
    sanitizer_runner = ComputeSanitizerRunner(probe_result.sanitizer_path, config)
    driver_api = DriverApiAttributes.try_load()     # None if cuda-python missing
    resource_extractor = StaticResourceExtractor(
        driver_api=driver_api,
        ptxas_parser=PtxasParser(),
        arch_model=StaticResourceModel.default(),
    )

    # --- 9. Build deps bag and bind to app state. ---
    deps = CompilerServiceDeps(
        config=config,
        toolchain=toolchain_info,
        artifact_store=artifact_store,
        pod_health=pod_health,
        idempotency=idempotency,
        adapter_registry=adapter_registry,
        gpu_semaphores=gpu_semaphores,
        compile_semaphore=compile_semaphore,
        nvcc=nvcc,
        cuobjdump=cuobjdump,
        sanitizer=sanitizer_runner,
        resource_extractor=resource_extractor,
        pod_id=pod_id,
    )
    service = CompilerService(deps)
    app.state.compiler_service = service
    app.state.deps = deps

    yield

    # --- Shutdown: no persistent state to flush; nothing to do in V1. ---
```

### 5.2 Singleton vs per-request matrix

| Object | Lifetime | Why |
|---|---|---|
| `ServiceConfig` | Singleton (at `create_app`) | env-var snapshot; immutable for process lifetime. |
| `ToolchainInfo` | Singleton | toolchain fixed at startup (spec §6.1). |
| `ArtifactStore` | Singleton | filesystem handle + GC bookkeeping. |
| `IdempotencyRegistry` | Singleton | in-memory map shared across requests. |
| `PodHealthTracker` | Singleton | pod-wide state machine (spec §6.8). |
| `AdapterRegistry` | Singleton | immutable adapter list. |
| `asyncio.Semaphore` (GPU, per-device) | Singleton | enforces INV-CS-010. |
| `asyncio.Semaphore` (CPU compile) | Singleton | enforces CPU_COMPILE_CONCURRENCY bound. |
| `NvccRunner`, `CuobjdumpRunner`, `ComputeSanitizerRunner`, `DriverApiAttributes`, `StaticResourceExtractor` | Singleton | stateless wrappers around tool paths. |
| `CompilerService` | Singleton | thin orchestrator (depends only on singletons). |
| `CompilerServiceDeps` | Singleton | frozen dataclass bag. |
| `Phase1…Phase5` instances | Singleton (bound in `CompilerService.__init__`) | stateless; share singletons. |
| `CompileRequest` | Per-request | inbound parsed body. |
| `RunEnvelope` | Per-request | built by `Phase1RequestNormalizer`. |
| `PhaseTimer` | Per-request | records phase durations into envelope. |
| `Phase1Output…Phase4Output` | Per-request | immutable phase handoff dataclasses. |
| `CompileResult` | Per-request | built exactly once by `Phase5ResultAssembler`. |

No mutable state lives on per-request objects besides `PhaseTimer` and the
`phase_timings_ms` dict of the envelope; both are fully populated by the time
`Phase5ResultAssembler.assemble` runs.

### 5.3 FastAPI `Depends(...)` wiring

```python
# api/dependencies.py
def get_service(request: Request) -> CompilerService:
    return request.app.state.compiler_service

def get_deps(request: Request) -> CompilerServiceDeps:
    return request.app.state.deps

def get_pod_health(deps: CompilerServiceDeps = Depends(get_deps)) -> PodHealthTracker:
    return deps.pod_health

def get_toolchain(deps: CompilerServiceDeps = Depends(get_deps)) -> ToolchainInfo:
    return deps.toolchain

def get_artifact_store(deps: CompilerServiceDeps = Depends(get_deps)) -> ArtifactStore:
    return deps.artifact_store
```

Route handlers always resolve singletons through `Depends(...)`; nothing is
imported from a module-level global. This makes the app trivial to construct
in tests or the CLI with a different `ServiceConfig`.

---

## §6 Request lifecycle (sequence diagram)

Happy-path matmul request under `POST /v1/compile`. Failure handoffs are
shown in §7.

```
Caller           FastAPI       CompilerService   Phase1..5 instances               Singletons
  │ POST /v1/compile              │                    │
  ├────────────────────────────►  │                    │
  │                               │ 1. schemas.parse   │
  │                               ├──CompileRequest──► │                    CompilerServiceDeps
  │                               │                    │ 2. await compile() │ (frozen bag)
  │                               │                    │                    │
  │                               │                    │ 3. Phase1.run()    │
  │                               │                    ├──────────────────► │
  │                               │                    │                    ├─ idempotency.observe_intake
  │                               │                    │                    ├─ pod_health.snapshot()
  │                               │                    │                    ├─ identity.* (hashes)
  │                               │                    │                    ├─ adapter_registry.get(op_name)
  │                               │                    │                    └─► Phase1Output (envelope seed,
  │                               │                    │                        resolved execution_spec,
  │                               │                    │                        short_circuit=None|…)
  │                               │                    │                    │
  │                               │                    │ 4. Phase2.run()    │
  │                               │                    ├──────────────────► │
  │                               │                    │                    ├─ adapter.build_harness_source(REFERENCE)
  │                               │                    │                    ├─ adapter.build_harness_source(CANDIDATE)
  │                               │                    │                    ├─ artifact_store.write(…)
  │                               │                    │                    └─► HarnessArtifacts
  │                               │                    │                    │
  │                               │                    │ 5. Phase3.run()    │
  │                               │                    ├──────────────────► │
  │                               │                    │                    ├─ async with compile_semaphore:
  │                               │                    │                    │     nvcc.compile(reference)
  │                               │                    │                    │     nvcc.compile(candidate)
  │                               │                    │                    ├─ cuobjdump.extract_sass(candidate)
  │                               │                    │                    ├─ resource_extractor.extract(…)
  │                               │                    │                    ├─ artifact_store.write(cubin/ptx/sass/log)
  │                               │                    │                    └─► CompileArtifacts + StaticAnalysisExt
  │                               │                    │                    │
  │                               │                    │ 6. Phase4.run()    │
  │                               │                    ├──────────────────► │
  │                               │                    │                    ├─ if pod_health.needs_probe(): run probe
  │                               │                    │                    ├─ async with gpu_semaphores[gpu_index]:
  │                               │                    │                    │     for each shape:
  │                               │                    │                    │       run reference executable
  │                               │                    │                    │       run candidate executable
  │                               │                    │                    │       adapter.compare_outputs(…)
  │                               │                    │                    │     sanitizer_policy.decide(…)
  │                               │                    │                    │     for each tool: sanitizer.run(…)
  │                               │                    │                    ├─ pod_health.record_phase4_outcome(…)
  │                               │                    │                    └─► CorrectnessOutcome + transitions
  │                               │                    │                    │
  │                               │                    │ 7. Phase5.assemble(request, p1, p3, p4)
  │                               │                    ├──────────────────► │
  │                               │                    │                    ├─ attribute_fault(…)
  │                               │                    │                    ├─ RunEnvelopeBuilder.build(pod_health.snapshot())
  │                               │                    │                    ├─ idempotency.finalize(request_id,
  │                               │                    │                    │                        artifact_key, refs, result)
  │                               │                    │                    └─► CompileResult
  │                               │ ◄─CompileResult────┤                    │
  │                               │                    │ finally: gc_cheap_pass()
  │ ◄─CompileResult───────────────┤                    │
```

**Data handoff shape** across the phases:

```
  CompileRequest
        │
        ▼  Phase1Output { request, envelope_seed, resolved_execution_spec, legacy_inferred_execution_spec, adapter }
        ▼  Phase2Output { phase1, harness: HarnessArtifacts }
        ▼  Phase3Output { phase2, compile: CompileArtifacts, static_analysis: StaticAnalysisExt }
        ▼  Phase4Output { phase3, correctness_outcome: CorrectnessOutcome, pod_health_transition }
        ▼  CompileResult  (Phase5 sole constructor)
```

Every `PhaseNOutput` references its predecessor by composition, so Phase 5 can
reconstruct the full request trail without additional bookkeeping.

---

## §7 Error and short-circuit flow

Every non-success status is a Phase-5-constructed `CompileResult`. No earlier
phase returns a `CompileResult` directly; they return a `PhaseShortCircuit`
packet and let the assembler do the construction (INV-CS-015).

`PhaseShortCircuit` packet:

```python
@dataclass(frozen=True)
class PhaseShortCircuit:
    phase: PhaseName
    status: CompileResultStatus
    candidate_fault_kind: CandidateFaultKind | None
    cuda_error: CudaErrorKind | None
    failure: FailureDetail
```

### 7.1 Branch matrix (originating phase → status → fault class/kind)

| Originating phase | Trigger | `status` | `FaultClass` | `CandidateFaultKind` |
|---|---|---|---|---|
| Phase 1 | Pod `quarantined` at intake | `infra_error` | `infra_fault` | none |
| Phase 1 | Prior idempotency entry started, not completed | `infra_error` | `infra_fault` | none (`reason=prior_attempt_lost_before_durability`) |
| Phase 1 | `execution_spec` fields missing, `legacy_compatibility=false` | `interface_contract_error` | `candidate_fault` | `interface_contract_error` |
| Phase 1 | Legacy inference preconditions fail | `interface_contract_error` | `candidate_fault` | `interface_contract_error` |
| Phase 2 | `problem_spec.op_name` unknown to registry | `interface_contract_error` | `candidate_fault` | `interface_contract_error` (`reason=unsupported_operation`) |
| Phase 2 | Rendered source exceeds `MAX_SOURCE_BYTES` | `interface_contract_error` | `candidate_fault` | `interface_contract_error` (`reason=source_too_large`) |
| Phase 3 | `nvcc` non-zero, stderr matches parse patterns | `compile_error` | `candidate_fault` | `syntax_error` |
| Phase 3 | `nvcc` non-zero, other pattern | `compile_error` | `candidate_fault` | `semantic_compile_error` |
| Phase 3 | `nvcc` missing / unknown target `sm_xx` | `compile_error` | `infra_fault` | none |
| Phase 3 | `nvcc` wall-clock timeout | `timeout` | `infra_fault` | none |
| Phase 4 | Any shape correctness mismatch on healthy pod | `correctness_fail` | `candidate_fault` | `correctness_mismatch` |
| Phase 4 | Candidate exits nonzero / bad launch config | `correctness_fail` | `candidate_fault` | `candidate_runtime_error` |
| Phase 4 | `cudaErrorIllegalAddress` / `LaunchTimeout` / `Misaligned` | varies (`correctness_fail` or `sanitizer_fail`) | `ambiguous_fault` | none |
| Phase 4 | `correctness` wall-clock timeout | `timeout` | `infra_fault` | none |
| Phase 4 | `memcheck` fail (healthy pod) | `sanitizer_fail` | `candidate_fault` | `memory_safety_error` |
| Phase 4 | `racecheck` / `synccheck` fail | `sanitizer_fail` | `candidate_fault` | `race_or_sync_error` |
| Phase 4 | `initcheck` fail | `sanitizer_fail` | `candidate_fault` | `uninitialized_memory_error` |
| Phase 4 | Sanitizer wall-clock timeout | `timeout` | `infra_fault` | none |
| Phase 4 | Probe fails before candidate runs | `infra_error` | `infra_fault` | none |

Phase 5's `attribute_fault(...)` implements this table exactly; no other
function in the codebase sets `fault_class`. Attribution downgrading to
`ambiguous_fault` when pod health transitions during the request is the job of
the assembler (per spec §6.9 "Conservative attribution").

### 7.2 Short-circuit flow diagram

```
   Phase 1 ──short_circuit?──► Phase5.from_short_circuit
      │
      ▼ ok
   Phase 2 ──short_circuit?──► Phase5.from_short_circuit
      │
      ▼ ok
   Phase 3 ──short_circuit?──► Phase5.from_short_circuit
      │
      ▼ ok
   Phase 4 ──short_circuit?──► Phase5.from_short_circuit
      │
      ▼ ok
   Phase5.assemble (success)

   Internal exceptions (CompilerServiceError subclasses) raised from any phase
   are caught in CompilerService.compile and routed to Phase5.from_exception
   (which maps to `infra_error` + infra_fault by default, unless the exception
   carries an explicit mapping).
```

No phase throws a bare `Exception`; phases either return a short_circuit
packet (expected, domain failure) or raise `CompilerServiceError` (guarded,
internal). The FastAPI handler never catches these; the service converts them
to results before they surface.

---

## §8 External tool wrappers

All external tools are invoked via `asyncio.create_subprocess_exec` and never
via `subprocess.run` or threads. Each wrapper is a stateless class whose
public surface is narrow.

### 8.1 `NvccRunner`

```python
@dataclass(frozen=True)
class NvccResult:
    returncode: int
    stdout_excerpt: str              # bounded by max_log_bytes
    stderr_excerpt: str              # bounded by max_log_bytes
    truncated: bool                  # true if either stream was cut
    command: str                     # joined argv for logging
    timed_out: bool

class NvccRunner:
    async def compile(
        self,
        source: Path,
        output: Path,
        target_arch: str,
        extra_flags: Sequence[str] = (),
        timeout_s: float | None = None,
        max_log_bytes: int | None = None,
    ) -> NvccResult:
        # flags = DEFAULT_COMPILE_FLAGS + [f"-arch={target_arch}",
        #         "-Xptxas=-v", "-lineinfo", *extra_flags, "-o", str(output), str(source)]
        # process = await asyncio.create_subprocess_exec(
        #     self._nvcc_path, *flags,
        #     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Read up to max_log_bytes from each stream; kill on timeout.
        ...
```

Byte caps: implemented via `asyncio.wait_for` on bounded `StreamReader.read(n)`
calls, then `process.kill()` + drain on overflow.
Timeouts: `asyncio.wait_for(process.wait(), timeout=timeout_s)`; on
`TimeoutError`, the runner kills the process and returns with `timed_out=True`.
Flag set is the single `DEFAULT_COMPILE_FLAGS` constant from `ServiceConfig`,
never assembled per-call.

### 8.2 `PtxasParser`

Pure parsing; no subprocess. Consumes `NvccResult.stdout_excerpt` +
`stderr_excerpt` (since `-Xptxas=-v` prints to stderr on Linux CUDA) and
returns a `PtxasReport` dataclass with `registers_per_thread`,
`smem_bytes_per_block`, `spill_loads`, `spill_stores`, each `int | None`.
Never writes a zero on parse failure; only `None`.

### 8.3 `CuobjdumpRunner`

```python
@dataclass(frozen=True)
class CuobjdumpResult:
    returncode: int
    sass_path: Path | None
    stderr_excerpt: str
    timed_out: bool

class CuobjdumpRunner:
    async def extract_sass(
        self, executable: Path, output: Path, timeout_s: float
    ) -> CuobjdumpResult: ...
```

Runs `cuobjdump --dump-sass <exe>` and writes stdout to `output`; size capped
so a runaway SASS dump never exhausts disk.

### 8.4 `ComputeSanitizerRunner`

```python
@dataclass(frozen=True)
class SanitizerRawResult:
    returncode: int
    report_path: Path | None          # --save path, if the tool wrote one
    stdout_excerpt: str
    stderr_excerpt: str
    timed_out: bool

class ComputeSanitizerRunner:
    async def run(
        self,
        tool: SanitizerTool,
        executable: Path,
        shape: ShapeCase,
        input_dir: Path,
        timeout_s: float,
    ) -> SanitizerOutcome:
        # 1. choose --tool (memcheck|racecheck|synccheck|initcheck)
        # 2. --save <tmp.report> --error-exitcode=1
        # 3. run bounded; classify returncode into SanitizerStatus
        # 4. if report produced and non-empty, store in ArtifactStore
        # 5. return SanitizerOutcome
        ...
```

Every invocation produces a `SanitizerOutcome` even on timeout or unsupported
(INV-CS-004). The caller (Phase 4) accumulates them into
`correctness.sanitizer_results`.

### 8.5 `DriverApiAttributes` (cuda-python facade)

```python
class DriverApiAttributes:
    @classmethod
    def try_load(cls) -> DriverApiAttributes | None:
        """Returns None if cuda-python import fails or the driver is absent.
        StaticResourceExtractor then falls back to ptxas transparently."""

    def read_registers_per_thread(self, cubin: Path, entrypoint: str) -> int | None: ...
    def read_static_smem_bytes(self, cubin: Path, entrypoint: str) -> int | None: ...
    def read_max_threads_per_block(self, cubin: Path, entrypoint: str) -> int | None: ...
```

Uses `cuda.cuModuleLoadData` + `cuda.cuModuleGetFunction` +
`cuda.cuFuncGetAttribute` under the hood. All calls are synchronous and very
fast; we run them on the event loop thread without a thread pool.

### 8.6 Summary: where timeouts and byte caps are enforced

| Tool | Timeout source | Byte cap source |
|---|---|---|
| `nvcc` | `request.limits.compile_timeout_s` or `COMPILE_TIMEOUT` | `MAX_LOG_BYTES` on stdout + stderr |
| `cuobjdump` | hard-coded short cap (e.g. 30 s) | `MAX_ARTIFACT_BYTES` on SASS file |
| `compute-sanitizer` | `SANITIZER_TIMEOUT` (per tool invocation) | `MAX_LOG_BYTES` on report excerpt |
| driver API | none (in-process, fast) | not applicable |
| Phase 4 correctness run (reference & candidate executables) | `CORRECTNESS_TIMEOUT` per shape | `MAX_LOG_BYTES` on captured stdout/stderr |

---

## §9 Artifact store lifecycle

```
     ┌─────────────────────────────────────────────────────────┐
     │                       ArtifactStore                      │
     │   ┌────────────────┐   ┌────────────────┐   ┌─────────┐  │
     │   │ PinnedRoots    │   │ RetentionPolicy│   │  GC     │  │
     │   │  role → {ids}  │   │  class → TTL   │   │ passes  │  │
     │   └────────────────┘   └────────────────┘   └─────────┘  │
     │                                                          │
     │    write()  ─► filesystem path, return artifact_id       │
     │    read()   ─► stream bytes of id (or 404 if GC'd)       │
     │    pin()    ─► add id to PinnedRoots[role]               │
     │    unpin()  ─► remove id from PinnedRoots[role]          │
     │    gc_cheap_pass()   ─► delete unpinned, expired, unreferenced
     │    gc_eager_if_over_watermark() ─► drop by class TTL priority
     └─────────────────────────────────────────────────────────┘
```

### 9.1 Data shape

```python
class ArtifactKind(StrEnum):
    SOURCE_CANDIDATE = "source_candidate"
    SOURCE_REFERENCE = "source_reference"
    EXECUTABLE = "executable"
    CUBIN = "cubin"
    PTX = "ptx"
    SASS = "sass"
    COMPILE_LOG = "compile_log"
    SANITIZER_REPORT = "sanitizer_report"
    CORRECTNESS_LOG = "correctness_log"
    PROBE_BINARY = "probe_binary"

class ArtifactClass(StrEnum):
    # Retention classes from spec §6.11
    BASELINE_INCUMBENT = "baseline_incumbent"
    SUCCESS_TOPK = "success_topk"
    SUCCESS_NON_PROFILED = "success_non_profiled"
    COMPILE_FAILURE = "compile_failure"
    CORRECTNESS_FAILURE = "correctness_failure"
    SANITIZER_FAILURE = "sanitizer_failure"
    INFRA_OR_AMBIGUOUS_FAILURE = "infra_or_ambiguous_failure"

class PinRole(StrEnum):
    BASELINE = "baseline"
    INCUMBENT = "incumbent"
    ACTIVE_BENCHMARK_BATCH = "active_benchmark_batch"
    ACTIVE_PROFILE_BATCH = "active_profile_batch"
    PROBE_KERNEL = "probe_kernel"

@dataclass(frozen=True)
class RetentionEntry:
    class_: ArtifactClass
    ttl: timedelta

class RetentionPolicy:
    def entry_for(self, class_: ArtifactClass) -> RetentionEntry: ...

class PinnedRoots:
    def __init__(self, roots: frozenset[PinRole]) -> None: ...
    def pin(self, role: PinRole, artifact_id: str) -> None: ...
    def unpin(self, role: PinRole, artifact_id: str) -> None: ...
    def pinned_ids(self) -> frozenset[str]: ...
```

### 9.1.1 Per-request scratch workspace

Phase 2 creates a `tempfile.mkdtemp` workspace (named
`kerlever-<request_id>-*`) to hold:

- Rendered harness sources (`reference.cu`, `candidate.cu`).
- Compiled executables produced by Phase 3 (`reference.out`,
  `candidate.out`).
- Objdump outputs (`candidate.cubin`, `candidate.ptx`, `candidate.sass`).
- Per-shape input/output buffers (`shapes/<shape_id>/A.bin`, `B.bin`,
  `ref.bin`, `cand.bin`) written by Phase 4.

Every one of these files is already copied into the `ArtifactStore`
(via `ArtifactStore.write`) before Phase 5 returns. The workspace is
intermediate scratch and is removed by `CompilerService.compile`'s
`finally` block via `shutil.rmtree(..., ignore_errors=True)` —
unconditionally, **after** `gc_cheap_pass` and `purge_expired` so the
skip-set computations see a consistent artifact-store view. Cleanup is
best-effort: failures are logged under the `workspace_cleanup_failed`
event and never raised.

This keeps the pod's disk usage bounded to the artifact store root
alone; without this cleanup each request would leak O(MB) of
intermediate files forever.

### 9.2 GC trigger points

| Trigger | Method | Where invoked |
|---|---|---|
| After each request completion | `gc_cheap_pass()` | `CompilerService.compile` `finally` block |
| When disk usage ≥ `ARTIFACT_DISK_HIGH_WATERMARK` | `gc_eager_if_over_watermark()` | Checked inside `gc_cheap_pass()` tail; also callable from `/v1/pod-status` for observability |

**No background timer.** V1 does not run a periodic `asyncio.Task`; GC is
request-triggered only (matches plan §"Artifact Store"). This avoids the
common hazard of a background task silently failing and letting disk grow.

### 9.3 References GC must skip

The skip-set for any GC pass is the union of:

1. `PinnedRoots.pinned_ids()` — all roles currently pinned.
2. `IdempotencyRegistry.referenced_artifact_ids()` — every `artifact_refs`
   entry across all idempotency entries whose `completed_at` is within
   `IDEMPOTENCY_TTL`.
3. The probe kernel binary id (pinned under `PinRole.PROBE_KERNEL`).

`gc_cheap_pass()` takes the skip set at call time; a later `pin()` cannot
un-delete an already-deleted artifact, but a concurrent `pin()` is safe
because pinning is an in-memory set operation that happens before the delete
scan reads the skip set (serialized by an internal `asyncio.Lock` on the
store).

### 9.4 `pin()` / `unpin()` call sites

| Call site | Role | Purpose |
|---|---|---|
| `_lifespan` after probe compile | `PROBE_KERNEL` | Keep `vec_add.cu` binary alive for the life of the process. |
| Future Benchmarker integration (out of scope V1) | `ACTIVE_BENCHMARK_BATCH` | Hold cubin/PTX/SASS across a benchmark cycle. |
| Future Orchestrator integration (out of scope V1) | `BASELINE`, `INCUMBENT` | Cross-run protection. |

V1 only wires `PROBE_KERNEL`; the other roles are declared but unused until
the GPU pipeline adapter lands.

---

## §10 Pod health state machine

```
                     ┌──────────┐
                     │ healthy  │◄─── (clean Phase 4 pass)
       ┌─── reset ──►│          │◄─── (probe pass from suspect)
       │             └────┬─────┘
       │                  │ ambiguous_event
       │                  ▼
       │             ┌──────────┐
       │             │ suspect  │─── probe_fail ───────────┐
       │             │          │                          │
       └────probe_pass────┘─────┘                          │
                                                            ▼
                                                      ┌────────────┐
                      ambiguous_limit_exceeded ──────►│ quarantined│
                      (cumulative across requests)    │  (terminal)│
                                                      └────────────┘
```

### 10.1 Transition hooks (where each edge is fired)

| Transition | Fired from |
|---|---|
| `healthy → suspect` (ambiguous event) | `Phase4CorrectnessValidator.run(...)` — after observing `cudaErrorIllegalAddress`, `cudaErrorLaunchTimeout`, `cudaErrorMisalignedAddress`, or a driver reset. |
| `suspect → healthy` (probe pass) | `Phase4CorrectnessValidator.run(...)` — before running the candidate, if `pod_health.needs_probe()` returns True. |
| `suspect → quarantined` (probe fail) | Same; probe failure short-circuits to Phase 5 with `infra_error` + `infra_fault`. |
| `*      → quarantined` (limit exceeded) | Computed inside `PodHealthTracker.record_phase4_outcome(...)`; set before the method returns. |
| `quarantined` short-circuit | `Phase1RequestNormalizer.run(...)` — if `pod_health.snapshot() == QUARANTINED` at intake, the request is short-circuited immediately. |

`PodHealthTracker` runs the probe lazily on the event loop thread; probe
concurrency is serialized by the per-device GPU semaphore (same semaphore
Phase 4 uses). That means a probe and a candidate cannot run on the same GPU
at the same time — the tracker acquires the semaphore before the probe and
releases it after, then Phase 4 acquires it for the candidate.

### 10.2 Cross-request coordination

```python
class PodHealthTracker:
    _state: PodHealth
    _ambiguous_count: int
    _last_transition: PodHealthTransition | None
    _lock: asyncio.Lock                         # protects _state + _ambiguous_count

    async def run_probe_if_needed(self, runner: ProbeRunner) -> …:
        async with self._lock:
            if self._state is not PodHealth.SUSPECT:
                return None
        # run probe OUTSIDE the lock (holds the GPU semaphore instead)
        probe_outcome = await runner.run(...)
        async with self._lock:
            # apply transition based on probe_outcome; return transition record
            ...
```

The tracker holds its mutation lock for the smallest possible critical
section; probe execution is outside the lock so concurrent `snapshot()` calls
by other in-flight requests are not blocked by a slow GPU probe.

All `RunEnvelope.pod_health` samples read `snapshot()` at Phase 5 time (spec
§6.8 "Envelope preservation"), not at intake, so a request that transitioned
the pod sees the post-transition value.

### 10.3 Probe kernel artifact

The probe source lives at
`kerlever/compiler_service/reference_kernels/vec_add.cu` and is a
**self-contained program** (kernel + `main` that allocates small buffers,
launches, synchronises, verifies the output element-wise, and exits 0 on
success or 1 on any CUDA error / value mismatch). Being a single-file
program means the probe compile is a one-shot `NvccRunner.compile` call
with no harness assembly required.

The probe binary is produced **at startup**, inside `build_deps`, before
the `PodHealthTracker` is constructed:

1. `build_deps` invokes `NvccRunner.compile(probe_source, probe_output,
   target_arch=config.probe_target_arch)` into
   `<artifact_root>/probe/probe.out`. A failed compile prints a
   structured JSON payload to `stderr` and `sys.exit(1)` — **INV-CS-012
   extension**: a pod that cannot compile its own probe is not ready.
2. The compiled binary bytes are persisted via
   `ArtifactStore.write(kind=ArtifactKind.PROBE_BINARY, run_id="pod",
   candidate_hash="probe")`.
3. The returned `artifact_id` is pinned under
   `PinRole.PROBE_KERNEL` so class-TTL GC never evicts it.
4. The `Path` to `probe.out` is passed into `PodHealthTracker`'s
   constructor as `probe_executable_path` (replacing the older
   `probe_binary_artifact_id_ref` placeholder).

At Phase 4 time, `Phase4CorrectnessValidator.run` calls
`pod_health.needs_probe()`; if True, it constructs a **local** probe
runner closure that:

- acquires the per-device GPU semaphore (the same one `_run_all_shapes`
  uses — only one GPU consumer at a time, per INV-CS-010),
- invokes the pre-compiled binary via
  `asyncio.create_subprocess_exec(str(probe_executable_path))`,
- waits with `config.probe_timeout_s` (default 10 s),
- decodes `ProbeOutcome(passed=returncode == 0, detail=stderr_excerpt)`.

The closure is then passed into
`PodHealthTracker.run_probe_if_needed(runner)`. A `probe_fail`
transition short-circuits Phase 4 to `status=INFRA_ERROR /
fault_class=INFRA_FAULT / reason="probe_failed_pod_quarantined"` —
no candidate work occurs. A `probe_pass` transition proceeds with the
candidate on a now-healthy pod. The returned transition is merged into
`Phase4Output.pod_health_transition` so `Phase5ResultAssembler`
observes it for envelope + attribution.

`ServiceConfig` grows two knobs for this path: `probe_target_arch`
(env `KERLEVER_PROBE_TARGET_ARCH`, default `sm_80`) and
`probe_timeout_s` (env `KERLEVER_PROBE_TIMEOUT_S`, default `10.0`).

---

## §11 Idempotency registry internals

### 11.1 In-memory layout

```python
class IdempotencyRegistry:
    _entries: dict[str, IdempotencyEntry]       # request_id → entry
    _per_id_locks: WeakValueDictionary[str, asyncio.Lock]
    _registry_lock: asyncio.Lock                # protects _entries + _per_id_locks
    _ttl: timedelta
```

- `_entries` is the single source of truth.
- `_per_id_locks` maps a `request_id` to an `asyncio.Lock`; the `WeakValueDictionary` allows locks to be reclaimed once no request is in flight.
- `_registry_lock` guards the two-step read-then-create-lock pattern below. It is held only while resolving a per-id lock, not during the phase work itself.

### 11.2 Per-id single-flight

The design uses a **per-`request_id` lock** (not the plan-optional global one)
because the cost is negligible and it prevents a genuine race where two
replays arrive simultaneously before either has recorded a completion. The
registry still uses `_registry_lock` for its internal map mutations.

```python
async def _acquire_id_lock(self, request_id: str) -> asyncio.Lock:
    async with self._registry_lock:
        lock = self._per_id_locks.get(request_id)
        if lock is None:
            lock = asyncio.Lock()
            self._per_id_locks[request_id] = lock
    return lock
```

`Phase1RequestNormalizer.run` does:

```python
id_lock = await registry._acquire_id_lock(request.request_id)
async with id_lock:
    intake = await registry.observe_intake(request.request_id)
    # intake returns one of: NEW | REUSED_COMPLETED | PRIOR_ATTEMPT_LOST
    # with the stored CompileResult if REUSED_COMPLETED.
```

The lock is released after the full request finishes (via
`CompilerService.compile`'s `finally` block), so a concurrent replay waits
for the original to complete and then reads its finalized entry.

### 11.3 Entry life cycle fields

| Field | Set by | Set when |
|---|---|---|
| `started_at` | `observe_intake` | On `NEW` intake |
| `phase_observed` | `record_phase` | After each `PhaseN.run` returns |
| `artifact_key` | `finalize` | Once in Phase 5 |
| `artifact_refs` | `finalize` | Once in Phase 5 (list of artifact ids held by the result) |
| `completed_at` | `finalize` | Once in Phase 5 |
| `compile_result` | `finalize` | Once in Phase 5 |

`finalize` is atomic under `_registry_lock` — after it returns, any replay
will see `REUSED_COMPLETED`. Before it returns, replays see either `NEW` (if
they raced through `_acquire_id_lock` first, which is blocked by
single-flight) or `PRIOR_ATTEMPT_LOST` (only possible if the process crashed
mid-flight; V1 does not persist the registry, so this is the post-restart
case).

### 11.4 `stored_artifact_key == current_artifact_key` check

Implemented in `observe_intake` exactly as spec §6.10 requires:

```python
if entry.compile_result is not None:
    if entry.artifact_key == current_artifact_key:
        return IdempotencyIntake(state=REUSED_COMPLETED, result=entry.compile_result)
    else:
        # INV-CS-009 mismatch: log anomaly, treat as stale, fall through to NEW.
        self._log_stale_anomaly(request_id, entry.artifact_key, current_artifact_key)
        entry = IdempotencyEntry.new(request_id)
        self._entries[request_id] = entry
        return IdempotencyIntake(state=NEW)
```

### 11.5 GC interaction

`IdempotencyRegistry.referenced_artifact_ids()` returns the union of
`artifact_refs` across all entries with `completed_at` within `IDEMPOTENCY_TTL`.
`ArtifactStore.gc_cheap_pass()` is called with this frozenset so it never
drops an artifact that a live replay could still return.

---

## §12 FastAPI app factory

### 12.1 `create_app` signature and route table

```python
def create_app(config: ServiceConfig | None = None) -> FastAPI: ...
```

Router mounted by `create_app` (defined in `api/handlers.py`):

| Method | Path | Request model | Response model | Depends (FastAPI) |
|---|---|---|---|---|
| `GET` | `/healthz` | — | `HealthzResponse` | `get_deps` |
| `POST` | `/v1/compile` | `CompileRequest` | `CompileResult` | `get_service` |
| `GET` | `/v1/artifacts/{artifact_id}` | path param | `StreamingResponse` | `get_artifact_store` |
| `GET` | `/v1/pod-status` | — | `PodStatusResponse` | `get_deps` |

HTTP semantics:

| Condition | HTTP status |
|---|---|
| `/healthz` all-green | 200 |
| `/healthz` any missing dependency | 503 |
| `POST /v1/compile` body fails Pydantic validation | 400 (FastAPI default) |
| `POST /v1/compile` pod quarantined | 503 (body is still a typed `CompileResult`) |
| `POST /v1/compile` any other outcome | 200 (status-in-body) |
| `GET /v1/artifacts/{id}` not found or GC'd | 404 |
| `GET /v1/pod-status` | 200 |

### 12.2 Startup / shutdown events

Startup runs inside `_lifespan` (§5.1). Shutdown is a no-op in V1: no
persisted state, no background task. The `SystemExit(1)` path on toolchain
failure is the single non-trivial startup behavior — it must use
`sys.stderr` + `raise SystemExit(1)` so uvicorn exits non-zero (INV-CS-012).

### 12.3 Error handlers

A single `exception_handler(CompilerServiceError)` catches anything the
service did not convert to a `CompileResult`, logs it, and returns a 500 —
but this should be unreachable in V1 because every phase either returns a
short-circuit or an exception that Phase 5 converts to `infra_error`. The
handler is a safety net, not a documented path.

`RequestValidationError` is left to FastAPI's default 400 handler; the body
shape is FastAPI's standard JSON error envelope.

### 12.4 OpenAPI schema source

FastAPI derives the schema from the Pydantic models in `api/schemas.py`.
V1's schemas are thin aliases: the `POST /v1/compile` request schema IS
`CompileRequest` from `types.py`, and the response schema IS `CompileResult`.
No DTO translation layer — Pydantic v2 serializes the service-local models
directly.

### 12.5 Concurrency on the handler

Each `POST /v1/compile` handler is `async def` and simply awaits
`service.compile(request)`. Uvicorn's default event loop handles request
interleaving; FastAPI does not start a thread per request. The semaphores
and locks described in §16 are the only coordination primitives.

---

## §13 CLI entry point

`python -m kerlever.compiler_service` invokes `cli.main()`:

```python
# cli.py
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kerlever.compiler_service")
    parser.add_argument("--request-json", type=Path, required=True,
                        help="Path to a CompileRequest JSON file.")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="If set, write the CompileResult JSON here.")
    parser.add_argument("--probe-only", action="store_true",
                        help="Run ToolchainProbe and exit.")
    args = parser.parse_args(argv)

    config = ServiceConfig.from_env()
    if args.probe_only:
        result = ToolchainProbe(config).run()
        print(result.model_dump_json(indent=2))
        return 0 if result.ok else 1

    # Build the same deps bag as create_app, without FastAPI.
    deps, teardown = asyncio.run(_build_standalone_deps(config))
    try:
        service = CompilerService(deps)
        request = CompileRequest.model_validate_json(args.request_json.read_text())
        result = asyncio.run(service.compile(request))
    finally:
        asyncio.run(teardown())

    payload = result.model_dump_json(indent=2)
    if args.output_json is not None:
        args.output_json.write_text(payload)
    else:
        print(payload)
    return 0
```

Purpose: smoke-test the service end-to-end without the HTTP layer, e.g. from
a CI job or a local debug session. It reuses `_build_standalone_deps(config)`
which is the same body as `_lifespan` minus FastAPI wiring, so the CLI path
shares the exact construction order with the HTTP path.

---

## §14 Dockerfile and container contract

### 14.1 `docker/compiler-service/Dockerfile`

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps: Python 3.12 + build essentials for cuda-python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3.12 python3.12-venv python3-pip ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# compute-sanitizer ships at /usr/local/cuda/bin/compute-sanitizer
ENV PATH="/usr/local/cuda/bin:${PATH}"

# App install.
COPY . /opt/kerlever
WORKDIR /opt/kerlever
RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
    && python3.12 -m pip install --no-cache-dir ".[service]"

# Artifact root (matches ServiceConfig default).
ENV KERLEVER_ARTIFACT_ROOT=/var/lib/kerlever/artifacts
RUN mkdir -p "${KERLEVER_ARTIFACT_ROOT}" \
    && chmod 0750 "${KERLEVER_ARTIFACT_ROOT}"

EXPOSE 8080
ENTRYPOINT ["/opt/kerlever/docker/compiler-service/entrypoint.sh"]
CMD ["uvicorn", "kerlever.compiler_service.api.app:create_app",
     "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

### 14.2 `entrypoint.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run the toolchain probe ONCE before uvicorn. On failure, exit non-zero so
# the container is marked unhealthy (INV-CS-012 + SC-CS-010).
python3.12 -m kerlever.compiler_service --probe-only

exec "$@"
```

### 14.3 Startup probe contract

`ToolchainProbe.run()` checks, in order:

1. `nvcc --version` succeeds → parse version.
2. `nvidia-smi` is reachable → parse driver version + `gpu_uuid` + `gpu_name`.
3. `compute-sanitizer --version` succeeds → parse version.
4. `${KERLEVER_ARTIFACT_ROOT}` exists and a test-write under it succeeds.
5. Optional: `cuda-python` importable (not fatal; logs a warning and falls
   back to ptxas per INV-CS-003).

The probe result is the input to both `_lifespan`'s `SystemExit(1)` decision
and the `/healthz` endpoint. One function, two call sites — satisfies
INV-CS-012.

After the toolchain probe passes, `build_deps` also performs a
**known-good probe compile** (INV-CS-012 extension). `NvccRunner.compile`
is invoked on `config.pod_health_probe_path` with
`config.probe_target_arch`, output to `<artifact_root>/probe/probe.out`.
A failed probe compile writes a structured JSON error payload to
`stderr` and exits the process with code 1 — a pod that cannot build
its own known-good probe is not ready. The resulting binary is pinned
under `PinRole.PROBE_KERNEL` so GC never drops it (spec §10.3).

### 14.4 Directories created at startup

| Path | Owner | Purpose |
|---|---|---|
| `/var/lib/kerlever/artifacts/` | Dockerfile `RUN mkdir` | Artifact root |
| `/var/lib/kerlever/artifacts/pinned/probe_kernel/` | `ArtifactStore` on first pin | Probe binary storage |
| `/tmp/kerlever/<request_id>/` | `Phase2HarnessAssembler` per request | Rendered harness + inputs + outputs; cleaned up at phase exit unless a failure artifact needed persistence |

---

## §15 `pyproject.toml` changes

Diff-like bullets (adds only; no removals):

- Add a new optional-dependency extra:

  ```toml
  [project.optional-dependencies]
  service = [
      "fastapi>=0.110",
      "uvicorn[standard]>=0.29",
      "cuda-python>=12.0",
  ]
  ```

- Update `[tool.mypy]`:

  ```toml
  packages = ["kerlever", "kerlever.compiler_service"]
  ```

  (Explicit sub-package is documentation — mypy already picks it up via
  `packages = ["kerlever"]`, but listing it makes the intent obvious.)

- No change to `[tool.ruff.lint]` selects. The new code passes
  `E, F, W, I, N, UP, B, A, SIM` without exceptions.

- No change to `[project.dependencies]`. The core `kerlever` package must not
  pull FastAPI/uvicorn/cuda-python by default — only the `service` extra
  installs them. Consumers of the library face zero new runtime deps.

- No change to `[project.scripts]`. The CLI uses `python -m`, which is
  preferred by the plan for containers.

---

## §16 Concurrency model

Every synchronization primitive the service uses, its scope, and why.

| Primitive | Granularity | Scope / acquired where | Prevents |
|---|---|---|---|
| `asyncio.Lock` on `IdempotencyRegistry._registry_lock` | Service-global | `observe_intake`, `record_phase`, `finalize`, `_acquire_id_lock` — held only during map mutations, never during phase work. | Torn read/write of the `_entries` dict. |
| `asyncio.Lock` per `request_id` (`_per_id_locks`) | Per-`request_id` | Acquired at the top of `Phase1RequestNormalizer.run` and released in `CompilerService.compile`'s `finally`. | Concurrent replays of the same id racing each other and producing inconsistent `idempotency_state`. |
| `asyncio.Semaphore(CPU_COMPILE_CONCURRENCY)` | Service-global | Acquired inside `Phase3Compiler.run` around both nvcc invocations (acquired twice — once per invocation — so parallel compiles of reference and candidate count as two). | Unbounded concurrent `nvcc` processes starving the pod CPU. |
| `asyncio.Semaphore(GPU_RUN_CONCURRENCY)` per visible GPU | Per-device | Acquired inside `Phase4CorrectnessValidator.run` around the entire correctness phase (all shapes + all sanitizer invocations for one request) AND inside `PodHealthTracker.run_probe_if_needed`. | Two CUDA contexts on one device poisoning each other (INV-CS-010 / FM-CS-010). |
| `asyncio.Lock` on `PodHealthTracker._lock` | Service-global | Around state mutations only; NOT held during probe kernel execution. | Torn reads of `_state`, `_ambiguous_count`. |
| `asyncio.Lock` inside `ArtifactStore` | Service-global | Around the critical GC section that reads pinned + referenced sets and deletes files. | Double-free during concurrent GC passes. |

### 16.1 What is deliberately NOT used

| Primitive | Why not |
|---|---|
| `threading.Lock` / `threading.Thread` | Plan forbids threads. All I/O is `asyncio`. |
| `multiprocessing` | Plan forbids. Containers already scale horizontally. |
| Periodic `asyncio.create_task` for GC | GC is request-triggered per plan §"Artifact Store" and §9.2. Background tasks silently fail; request-triggered is observable. |
| Global asyncio `Lock` for a whole request | Would serialize everything. Per-id lock is sufficient for idempotency correctness; per-device semaphore is sufficient for GPU correctness. |

### 16.2 Acquisition order (to avoid deadlocks)

If a phase acquires more than one primitive, it does so in this fixed order:

```
per-request-id lock  ─►  registry_lock (briefly)  ─►  compile_semaphore
                                                 ─►  gpu_semaphore[device]
                                                 ─►  pod_health._lock (briefly)
                                                 ─►  artifact_store._lock (briefly)
```

No code path acquires these in a different order. This is a property the
Coding Agent must preserve during implementation; the design sketch in §5.1
already does.

### 16.3 Cancellation

`CompilerService.compile` does NOT install custom cancellation handlers.
FastAPI's default behavior — cancel the handler task on client disconnect —
is acceptable because:

- `NvccRunner`, `CuobjdumpRunner`, `ComputeSanitizerRunner` use
  `asyncio.create_subprocess_exec` and kill their child processes in
  `finally` blocks when their `await` is cancelled;
- `IdempotencyRegistry.finalize_if_pending` in `CompilerService.compile`'s
  `finally` records the cancellation as an in-flight entry, so a retry sees
  `PRIOR_ATTEMPT_LOST` rather than data loss;
- per-device GPU semaphores are released via `async with`, so a cancelled
  request cannot leak the GPU lock.

---

## Appendix A: Cross-reference to spec IDs

This is a small map from design components back into the spec so reviewers
can verify coverage without re-reading the whole spec. If a spec ID is not
listed, the wiring for it is subsumed by a component listed here.

| Design component | Enforces / implements |
|---|---|
| `Phase1RequestNormalizer` | REQ-CS-005, REQ-CS-006, REQ-CS-011 (partial), INV-CS-002, INV-CS-009 |
| `Phase2HarnessAssembler` | REQ-CS-011, INV-CS-001, INV-CS-013 |
| `Phase3Compiler` + `StaticResourceExtractor` | REQ-CS-001, REQ-CS-002, INV-CS-001, INV-CS-003 |
| `Phase4CorrectnessValidator` + `ComputeSanitizerRunner` + `SanitizerPolicy` | REQ-CS-003, REQ-CS-004, INV-CS-004, INV-CS-006, INV-CS-010, INV-CS-011 |
| `Phase5ResultAssembler` + `attribute_fault` | REQ-CS-009, INV-CS-005, INV-CS-014, INV-CS-015 |
| `IdempotencyRegistry` + per-id lock | REQ-CS-006, INV-CS-009 |
| `PodHealthTracker` | REQ-CS-007, INV-CS-008, INV-CS-014 |
| `ArtifactStore` + `PinnedRoots` + `RetentionPolicy` | REQ-CS-012, INV-CS-007 |
| `identity.artifact_key` | REQ-CS-008 |
| `create_app` + `ToolchainProbe` | REQ-CS-010, REQ-CS-013, INV-CS-012 |
| `AdapterRegistry` + `OperationAdapter` Protocol | REQ-CS-011, INV-CS-013 |

---

*End of design.md. Behavioral rules remain sole property of spec.md; wiring
remains sole property of this document.*
