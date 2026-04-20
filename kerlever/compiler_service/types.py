"""Compiler Service types — Pydantic v2 models and enums.

All service-local data contracts live here. The module reuses
``ProblemSpec``, ``ShapeCase``, ``StaticAnalysis``, ``CorrectnessResult``
from ``kerlever.types`` unchanged; every other structure visible across
phases is defined here.

Spec: docs/compiler-service/spec.md §3, §6
Design: docs/compiler-service/design.md §4.1
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from kerlever.types import CorrectnessResult, ProblemSpec, StaticAnalysis

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CompileResultStatus(StrEnum):
    """Top-level service-local compile result status.

    Spec: §3.3. Intentionally distinct from ``kerlever.types.CompileStatus``
    — see SR-CS-004.
    """

    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    INTERFACE_CONTRACT_ERROR = "interface_contract_error"
    CORRECTNESS_FAIL = "correctness_fail"
    SANITIZER_FAIL = "sanitizer_fail"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"


class CandidateRole(StrEnum):
    """Role of a source under compilation."""

    REFERENCE = "reference"
    CANDIDATE = "candidate"
    PROBE = "probe"


class MetadataMode(StrEnum):
    """Whether the execution spec was explicit or inferred from source."""

    EXPLICIT = "explicit"
    LEGACY_INFERRED = "legacy_inferred"


class PodHealth(StrEnum):
    """Pod-wide CUDA context hygiene state (spec §6.8)."""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    QUARANTINED = "quarantined"


class IdempotencyState(StrEnum):
    """Idempotency registry state for a given request (spec §6.10)."""

    NEW = "new"
    REUSED_COMPLETED = "reused_completed"
    PRIOR_ATTEMPT_LOST = "prior_attempt_lost"


class FaultClass(StrEnum):
    """Disjoint failure attribution (spec §6.9)."""

    CANDIDATE_FAULT = "candidate_fault"
    INFRA_FAULT = "infra_fault"
    AMBIGUOUS_FAULT = "ambiguous_fault"


class CandidateFaultKind(StrEnum):
    """Sub-kind of ``CANDIDATE_FAULT`` (spec §6.9 attribution table)."""

    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_COMPILE_ERROR = "semantic_compile_error"
    INTERFACE_CONTRACT_ERROR = "interface_contract_error"
    CORRECTNESS_MISMATCH = "correctness_mismatch"
    MEMORY_SAFETY_ERROR = "memory_safety_error"
    RACE_OR_SYNC_ERROR = "race_or_sync_error"
    UNINITIALIZED_MEMORY_ERROR = "uninitialized_memory_error"
    CANDIDATE_RUNTIME_ERROR = "candidate_runtime_error"


class SanitizerTool(StrEnum):
    """Compute Sanitizer sub-tool names (spec §6.7)."""

    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"
    INITCHECK = "initcheck"


class SanitizerStatus(StrEnum):
    """Outcome of a single sanitizer invocation."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"


class ResourceSource(StrEnum):
    """Provenance of a static resource fact (spec §6.5, INV-CS-003)."""

    CUDA_FUNC_ATTRIBUTE = "cuda_func_attribute"
    PTXAS = "ptxas"
    SASS = "sass"
    NULL = "null"


class OracleKind(StrEnum):
    """Oracle used to judge correctness (spec §6.6, INV-CS-006)."""

    REFERENCE_KERNEL = "reference_kernel"
    ADAPTER_INDEPENDENT = "adapter_independent"
    HYBRID = "hybrid"


class ComparisonMode(StrEnum):
    """Comparison mode for output correctness (spec §6.6, INV-CS-011)."""

    TOLERANCE = "tolerance"
    EXACT = "exact"


class ToleranceSource(StrEnum):
    """Which layer resolved the tolerance value (spec §6.6)."""

    SHAPE_CASE = "shape_case"
    ADAPTER_DTYPE_DEFAULT = "adapter_dtype_default"
    SERVICE_DEFAULT = "service_default"


class PhaseName(StrEnum):
    """Pipeline phase names (spec §6)."""

    REQUEST_NORMALIZATION = "request_normalization"
    HARNESS_ASSEMBLY = "harness_assembly"
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    SANITIZER = "sanitizer"
    OUTPUT = "output"


class ArtifactKind(StrEnum):
    """Artifact store content class (design §9.1)."""

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
    """Retention class for artifact-store GC (spec §6.11)."""

    BASELINE_INCUMBENT = "baseline_incumbent"
    SUCCESS_TOPK = "success_topk"
    SUCCESS_NON_PROFILED = "success_non_profiled"
    COMPILE_FAILURE = "compile_failure"
    CORRECTNESS_FAILURE = "correctness_failure"
    SANITIZER_FAILURE = "sanitizer_failure"
    INFRA_OR_AMBIGUOUS_FAILURE = "infra_or_ambiguous_failure"


class PinRole(StrEnum):
    """Artifact store pinning role (spec §6.11)."""

    BASELINE = "baseline"
    INCUMBENT = "incumbent"
    ACTIVE_BENCHMARK_BATCH = "active_benchmark_batch"
    ACTIVE_PROFILE_BATCH = "active_profile_batch"
    PROBE_KERNEL = "probe_kernel"


class CudaErrorKind(StrEnum):
    """Classified CUDA runtime errors observed during Phase 4."""

    ILLEGAL_ADDRESS = "illegal_address"
    LAUNCH_TIMEOUT = "launch_timeout"
    MISALIGNED_ADDRESS = "misaligned_address"
    DRIVER_RESET = "driver_reset"


class SyntaxPatternHit(StrEnum):
    """Lexical classification of nvcc stderr for fault attribution."""

    PARSE_LEVEL = "parse_level"
    SEMANTIC = "semantic"
    TOOLCHAIN = "toolchain"


# ---------------------------------------------------------------------------
# Request and launch-time contracts
# ---------------------------------------------------------------------------


class KernelExecutionSpec(BaseModel):
    """Candidate-owned launch contract (spec §3, §6.1).

    For ``legacy_compatibility=False`` requests every field must be set;
    interface resolution rejects a partial spec as
    ``interface_contract_error`` (INV-CS-002).
    """

    entrypoint: str | None = None
    block_dim: tuple[int, int, int] | None = None
    dynamic_smem_bytes: int | None = None
    abi_name: str | None = None
    abi_version: str | None = None
    metadata_mode: MetadataMode = MetadataMode.EXPLICIT


class RequestLimits(BaseModel):
    """Per-request override of service-default limits."""

    compile_timeout_s: float | None = None
    correctness_timeout_s: float | None = None
    sanitizer_timeout_s: float | None = None
    max_source_bytes: int | None = None
    max_log_bytes: int | None = None


class CompileRequest(BaseModel):
    """Inbound request body for ``POST /v1/compile`` (spec §6.1).

    Implements: REQ-CS-005, REQ-CS-010
    """

    request_id: str
    run_id: str
    round_id: str
    candidate_hash: str
    role: CandidateRole
    source_code: str
    problem_spec: ProblemSpec
    reference_source: str
    execution_spec: KernelExecutionSpec
    target_arch: str
    legacy_compatibility: bool = False
    limits: RequestLimits | None = None


# ---------------------------------------------------------------------------
# Toolchain and envelope
# ---------------------------------------------------------------------------


class ToolchainInfo(BaseModel):
    """Reproducibility-anchored snapshot of pod environment (spec §6.1)."""

    model_config = ConfigDict(frozen=True)

    nvcc_version: str
    driver_version: str
    gpu_name: str
    gpu_uuid: str
    sanitizer_version: str
    toolchain_hash: str


class RunEnvelope(BaseModel):
    """Per-request bundle carried alongside every phase (spec §6.1).

    Implements: REQ-CS-006, REQ-CS-007, REQ-CS-008
    Invariant: INV-CS-008 (pod_health sampled at assembly time)
    """

    # identity
    run_id: str
    round_id: str
    request_id: str
    candidate_hash: str
    # reproducibility hashes
    source_hash: str
    problem_spec_hash: str
    launch_spec_hash: str
    toolchain_hash: str
    compile_flags_hash: str
    adapter_version: str
    artifact_key: str
    # resolved limits
    limits: RequestLimits
    # observability
    pod_id: str
    gpu_uuid: str
    phase_timings_ms: dict[PhaseName, float] = Field(default_factory=dict)
    # pod health (sampled at assembly time — INV-CS-008)
    pod_health: PodHealth
    # idempotency
    idempotency_state: IdempotencyState
    previous_attempt_lost: bool = False
    prior_attempt_observed_phase: PhaseName | None = None


# ---------------------------------------------------------------------------
# Static analysis extensions
# ---------------------------------------------------------------------------


class ResourceConflict(BaseModel):
    """Two sources disagree on one static resource fact (INV-CS-003)."""

    model_config = ConfigDict(frozen=True)

    fact: str
    sources: list[tuple[ResourceSource, int | None]]
    preferred_value: int | None


class StaticAnalysisExt(BaseModel):
    """Extends ``StaticAnalysis`` with provenance per fact.

    Implements: REQ-CS-001
    Invariant: INV-CS-003 (never fabricate a missing fact)
    """

    model_config = ConfigDict(frozen=True)

    base: StaticAnalysis
    resource_sources: dict[str, ResourceSource]
    resource_conflicts: list[ResourceConflict] = Field(default_factory=list)
    cubin_artifact_id: str | None = None
    ptx_artifact_id: str | None = None
    sass_artifact_id: str | None = None


# ---------------------------------------------------------------------------
# Correctness + sanitizer results
# ---------------------------------------------------------------------------


class SanitizerOutcome(BaseModel):
    """One ``compute-sanitizer`` invocation's structured outcome.

    Invariant: INV-CS-004 (every invocation preserved in
    ``correctness.sanitizer_results``)
    """

    model_config = ConfigDict(frozen=True)

    tool: SanitizerTool
    shape_id: str
    status: SanitizerStatus
    report_artifact_id: str | None = None


class CorrectnessResultExt(BaseModel):
    """Extends ``CorrectnessResult`` with oracle/tolerance provenance.

    Implements: REQ-CS-003, REQ-CS-004
    Invariant: INV-CS-006 (``oracle_kind`` non-null), INV-CS-011
        (``comparison_mode`` follows adapter; float defaults to tolerance)
    """

    model_config = ConfigDict(frozen=True)

    base: CorrectnessResult
    oracle_kind: OracleKind
    comparison_mode: ComparisonMode
    tolerance_source: ToleranceSource
    tolerance_value: float
    sanitizer_results: list[SanitizerOutcome] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure + artifacts
# ---------------------------------------------------------------------------


class FailureDetail(BaseModel):
    """Structured failure context (spec §3)."""

    model_config = ConfigDict(frozen=True)

    phase: PhaseName
    command: str | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    failing_shape_id: str | None = None
    retryable: bool = False
    reason: str | None = None


class ArtifactRefs(BaseModel):
    """Artifact id map returned on a compiled result (spec §6.4, §6.11)."""

    model_config = ConfigDict(frozen=True)

    source_artifact_id: str | None = None
    executable_artifact_id: str | None = None
    reference_executable_artifact_id: str | None = None
    cubin_artifact_id: str | None = None
    ptx_artifact_id: str | None = None
    sass_artifact_id: str | None = None
    compile_log_artifact_id: str | None = None
    sanitizer_report_artifact_ids: list[str] = Field(default_factory=list)
    correctness_log_artifact_id: str | None = None


class CompileResult(BaseModel):
    """Top-level outbound record (spec §3, §6).

    Constructed in exactly one place: ``Phase5ResultAssembler`` (INV-CS-015).
    Every other code path short-circuits by returning a ``PhaseShortCircuit``
    packet and letting the assembler construct this value.
    """

    model_config = ConfigDict(frozen=True)

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


# ---------------------------------------------------------------------------
# Re-export reused types for convenience
# ---------------------------------------------------------------------------

__all__ = [
    "ArtifactClass",
    "ArtifactKind",
    "ArtifactRefs",
    "CandidateFaultKind",
    "CandidateRole",
    "ComparisonMode",
    "CompileRequest",
    "CompileResult",
    "CompileResultStatus",
    "CorrectnessResult",
    "CorrectnessResultExt",
    "CudaErrorKind",
    "FailureDetail",
    "FaultClass",
    "IdempotencyState",
    "KernelExecutionSpec",
    "MetadataMode",
    "OracleKind",
    "PhaseName",
    "PinRole",
    "PodHealth",
    "ProblemSpec",
    "RequestLimits",
    "ResourceConflict",
    "ResourceSource",
    "RunEnvelope",
    "SanitizerOutcome",
    "SanitizerStatus",
    "SanitizerTool",
    "StaticAnalysis",
    "StaticAnalysisExt",
    "SyntaxPatternHit",
    "ToleranceSource",
    "ToolchainInfo",
]
