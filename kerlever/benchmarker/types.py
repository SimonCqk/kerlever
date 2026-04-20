"""Benchmarker — request/response/envelope Pydantic models and StrEnums.

All types declared in spec §5 (shape definitions) live here. Shared types
from ``kerlever.types`` are re-used verbatim via import; we never copy their
definitions.

Spec: docs/benchmarker/spec.md
Design: docs/benchmarker/design.md
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kerlever.types import (
    BenchmarkBundle,
    BottleneckAssessment,
    ObjectiveScore,
    PerformanceObjective,
    ProblemSpec,
    ProfileBundle,
    ProfileMetrics,
    ShapeBenchResult,
    ShapeCase,
    StaticAnalysis,
)

# ---------------------------------------------------------------------------
# StrEnums (spec §5)
# ---------------------------------------------------------------------------


class ArtifactExecutionModel(StrEnum):
    """Artifact execution model. V1 supports only COMMON_HARNESS_CUBIN.

    Implements: REQ-BENCH-024
    """

    COMMON_HARNESS_CUBIN = "common_harness_cubin"


class MetricMode(StrEnum):
    """Timing metric mode; default is DEVICE_KERNEL_US."""

    DEVICE_KERNEL_US = "device_kernel_us"
    HOST_LAUNCH_US = "host_launch_us"
    OPERATOR_END_TO_END_US = "operator_end_to_end_us"
    CUDA_GRAPH_REPLAY_US = "cuda_graph_replay_us"


class AdapterIterationSemantics(StrEnum):
    """Adapter-declared per-iteration semantics.

    Controls repeated-launch timing and profile replay policy.
    """

    OVERWRITE_PURE = "overwrite_pure"
    REQUIRES_OUTPUT_RESET = "requires_output_reset"
    REQUIRES_FULL_INPUT_RESET = "requires_full_input_reset"
    NOT_REPEATABLE = "not_repeatable"


class CachePolicy(StrEnum):
    """Cache policy for repeated-launch timing."""

    WARM_SAME_BUFFERS = "warm_same_buffers"
    WARM_ROTATING_BUFFERS = "warm_rotating_buffers"
    COLD_FLUSH_BUFFER = "cold_flush_buffer"
    RESET_PERSISTING_L2 = "reset_persisting_l2"


class ClockPolicyMode(StrEnum):
    """Resolved clock policy mode."""

    OBSERVED_ONLY = "observed_only"
    LOCKED = "locked"
    LOCK_REQUESTED_UNAVAILABLE = "lock_requested_unavailable"


class IncumbentComparison(StrEnum):
    """Per-candidate comparison outcome versus the in-episode anchor."""

    IMPROVED = "improved"
    STATISTICAL_TIE = "statistical_tie"
    REGRESSED = "regressed"
    UNSTABLE = "unstable"
    NOT_COMPARABLE = "not_comparable"


class MeasurementQualityStatus(StrEnum):
    """Per-shape measurement quality classification."""

    VALID = "valid"
    VALID_WITH_WARNING = "valid_with_warning"
    UNSTABLE = "unstable"
    RUNTIME_FAULT = "runtime_fault"
    INFRA_FAULT = "infra_fault"


class ProfileStatus(StrEnum):
    """Profile collection status per candidate."""

    PRESENT = "present"
    PROFILE_UNAVAILABLE = "profile_unavailable"


class ProfileUnavailableReason(StrEnum):
    """Closed enum of profile-unavailable reasons."""

    PROFILER_PERMISSION_DENIED = "profiler_permission_denied"
    ADAPTER_NOT_REPEATABLE = "adapter_not_repeatable"
    ARCH_MISMATCH = "arch_mismatch"
    PROFILER_TIMEOUT = "profiler_timeout"
    PROFILER_BINARY_MISSING = "profiler_binary_missing"
    PROFILER_REPLAY_REFUSED = "profiler_replay_refused"
    MIG_PROFILE_MISMATCH = "mig_profile_mismatch"


class FaultClass(StrEnum):
    """Per-candidate fault attribution."""

    CANDIDATE_FAULT = "candidate_fault"
    INFRA_FAULT = "infra_fault"
    AMBIGUOUS_FAULT = "ambiguous_fault"


class PodHealth(StrEnum):
    """Pod-health state machine states."""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    QUARANTINED = "quarantined"


class BatchStatus(StrEnum):
    """Top-level batch status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    UNSTABLE = "unstable"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"


class CacheConfig(StrEnum):
    """cuFuncSetCacheConfig policies (driver API)."""

    PREFER_NONE = "prefer_none"
    PREFER_SHARED = "prefer_shared"
    PREFER_L1 = "prefer_l1"
    PREFER_EQUAL = "prefer_equal"


class ReplayMode(StrEnum):
    """NCU replay mode."""

    APPLICATION = "application"
    KERNEL = "kernel"


class ProfilerName(StrEnum):
    """Profiler tool name for provenance."""

    NCU = "ncu"
    NSYS = "nsys"


class FunctionAttribute(IntEnum):
    """CUDA function attribute enumeration used by ``cuFuncSetAttribute``.

    Values mirror the ``CUfunction_attribute`` enum in the cuda-python
    driver bindings; we only expose the subset the Benchmarker uses.

    Implements: REQ-BENCH-029
    """

    CACHE_MODE_CA = 7
    MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    PREFERRED_SHARED_MEMORY_CARVEOUT = 9
    REQUIRED_CLUSTER_WIDTH = 12
    REQUIRED_CLUSTER_HEIGHT = 13
    REQUIRED_CLUSTER_DEPTH = 14
    NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 15


# ---------------------------------------------------------------------------
# Pydantic models — identity, envelope, hygiene
# ---------------------------------------------------------------------------


_BASE_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False)


class LaunchSpec(BaseModel):
    """Per-candidate launch specification used to resolve the CUDA function.

    Implements: REQ-BENCH-001
    """

    model_config = _BASE_MODEL_CONFIG

    entrypoint: str
    block_dim: tuple[int, int, int]
    grid_dim: tuple[int, int, int] | None = None
    dynamic_smem_bytes: int
    abi_name: str
    abi_version: str
    metadata_mode: str | None = None


class CorrectnessForward(BaseModel):
    """Correctness forwarded from Compiler Service.

    The Benchmarker never reruns correctness; it only checks the passed flag.
    """

    model_config = _BASE_MODEL_CONFIG

    passed: bool
    failing_shape_ids: list[str] = Field(default_factory=list)
    max_abs_error: float | None = None
    max_rel_error: float | None = None


class CandidateArtifactRef(BaseModel):
    """Reference to one candidate's cubin artifact + launch metadata.

    Implements: REQ-BENCH-001
    """

    model_config = _BASE_MODEL_CONFIG

    candidate_hash: str
    artifact_id: str
    cubin_uri: str
    launch_spec: LaunchSpec
    launch_spec_hash: str
    source_hash: str
    toolchain_hash: str
    static_analysis: StaticAnalysis | None = None
    correctness: CorrectnessForward | None = None
    adapter_iteration_semantics: AdapterIterationSemantics


class FunctionAttributePolicy(BaseModel):
    """Requested or observed function-attribute policy."""

    model_config = _BASE_MODEL_CONFIG

    max_dynamic_shared_memory_size: int | None = None
    preferred_shared_memory_carveout_pct: int | None = None
    cache_config: CacheConfig | None = None
    cluster_dims: tuple[int, int, int] | None = None
    non_portable_cluster_size_allowed: bool | None = None


class ClockPolicy(BaseModel):
    """Clock policy resolution record."""

    model_config = _BASE_MODEL_CONFIG

    mode: ClockPolicyMode
    requested_sm_clock_mhz: int | None = None
    requested_mem_clock_mhz: int | None = None


class WarmupPolicy(BaseModel):
    """Warmup policy block in the envelope."""

    model_config = _BASE_MODEL_CONFIG

    min_runs: int
    cache_state: Literal["untouched", "touched"]


class RepeatPolicy(BaseModel):
    """Repeat / calibration policy block in the envelope."""

    model_config = _BASE_MODEL_CONFIG

    repetitions: int
    iterations_per_sample: int
    min_timed_batch_ms: float
    max_timed_batch_ms: float


class CachePolicyBlock(BaseModel):
    """Requested + effective cache policy with reason on promotion.

    Implements: REQ-BENCH-009
    """

    model_config = _BASE_MODEL_CONFIG

    requested: CachePolicy
    effective: CachePolicy
    reason: str | None = None


class MeasurementEnvelope(BaseModel):
    """Immutable per-candidate measurement envelope; dictates comparability.

    Implements: REQ-BENCH-001, REQ-BENCH-008
    Invariant: INV-BENCH-011 (envelope mismatch short-circuits to not_comparable)
    """

    model_config = _BASE_MODEL_CONFIG

    # identity
    run_id: str
    round_id: str | None = None
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
    mig_profile: str | None = None
    sm_arch: str
    driver_version: str
    cuda_runtime_version: str
    # timing policy
    metric_mode: MetricMode
    function_attribute_policy_requested: FunctionAttributePolicy
    function_attribute_policy_observed: FunctionAttributePolicy
    warmup_policy: WarmupPolicy
    repeat_policy: RepeatPolicy
    cache_policy: CachePolicyBlock
    clock_policy: ClockPolicy
    interleave_seed: int | None = None


class DeviceTelemetrySnapshot(BaseModel):
    """Snapshot of device telemetry sampled via pynvml."""

    model_config = _BASE_MODEL_CONFIG

    taken_at_ms: int
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    gpu_temp_c: float | None = None
    power_w: float | None = None
    throttle_reasons: list[str] = Field(default_factory=list)
    ecc_sbe_total: int | None = None
    ecc_dbe_total: int | None = None
    xid_events_since_last: int = 0


class HygieneReport(BaseModel):
    """Preflight hygiene report.

    Implements: REQ-BENCH-003
    """

    model_config = _BASE_MODEL_CONFIG

    gpu_uuid: str
    sm_arch: str
    mig_profile: str | None = None
    compute_mode: str
    foreign_processes: list[str] = Field(default_factory=list)
    clocks_event_reasons: list[str] = Field(default_factory=list)
    gpu_temp_c: float | None = None
    power_w: float | None = None
    ecc_ok: bool = True
    xid_ok: bool = True
    probe_ok: bool | None = None
    profiler_counter_permission: bool = False
    telemetry: DeviceTelemetrySnapshot | None = None
    reason_on_fail: str | None = None


class AnchorDriftTelemetry(BaseModel):
    """Telemetry delta annotation collected at postflight."""

    model_config = _BASE_MODEL_CONFIG

    sm_clock_drift_mhz: int | None = None
    mem_clock_drift_mhz: int | None = None
    temp_drift_c: float | None = None
    power_drift_w: float | None = None


# ---------------------------------------------------------------------------
# Shape-level measurement artifact and profile artifact types
# ---------------------------------------------------------------------------


class MeasurementQuality(BaseModel):
    """Measurement quality classification with reason and warnings."""

    model_config = _BASE_MODEL_CONFIG

    status: MeasurementQualityStatus
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ShapeMeasurementArtifact(BaseModel):
    """Rich per-shape measurement artifact.

    Implements: REQ-BENCH-011, REQ-BENCH-021
    """

    model_config = _BASE_MODEL_CONFIG

    shape_id: str
    samples_us: list[float] = Field(default_factory=list)
    warmup_count: int
    iterations_per_sample: int
    min_samples_required: int
    p50_us: float
    p95_us: float | None = None
    mean_us: float
    stdev_us: float
    cv_pct: float | None = None
    min_us: float
    max_us: float
    cache_policy: CachePolicy
    requested_cache_policy: CachePolicy
    effective_cache_policy: CachePolicy
    interleave_block_len: int | None = None
    anchor_every_n_samples: int | None = None
    anchor_pre_us: float | None = None
    anchor_post_us: float | None = None
    anchor_drift_pct: float | None = None
    interleave_block_order: list[str] = Field(default_factory=list)
    artifact_execution_model: ArtifactExecutionModel
    adapter_iteration_semantics: AdapterIterationSemantics
    metric_mode: MetricMode
    max_timed_batch_ms: float
    function_attribute_policy: FunctionAttributePolicy
    useful_bytes: int | None = None
    actual_bytes: int | None = None
    algorithmic_flops: int | None = None
    effective_bandwidth_gbps: float | None = None
    achieved_flops: float | None = None
    arithmetic_intensity_flop_per_byte: float | None = None
    measurement_quality: MeasurementQuality
    telemetry_before: DeviceTelemetrySnapshot
    telemetry_after: DeviceTelemetrySnapshot


class RawProfileMetric(BaseModel):
    """Raw NCU metric entry with mandatory provenance.

    Implements: REQ-BENCH-018
    Invariant: INV-BENCH-009 (missing metric → value is null, never fabricated)
    """

    model_config = _BASE_MODEL_CONFIG

    metric_name: str
    value: float | int | None
    unit: str | None = None
    architecture: str
    profiler_name: ProfilerName
    profiler_version: str
    collection_section: str | None = None


class NormalizedProfileMetricProvenance(BaseModel):
    """Provenance for a normalized (compact) profile metric."""

    model_config = _BASE_MODEL_CONFIG

    source_metrics: list[str]
    architecture: str
    profiler_version: str
    comparable_across_arch: Literal[False] = False


class ProfileArtifactRef(BaseModel):
    """Pointer into the pod-local artifact store."""

    model_config = _BASE_MODEL_CONFIG

    artifact_id: str
    kind: Literal["ncu_report", "nsys_report", "raw_metrics_json", "samples_json"]
    uri: str
    size_bytes: int
    created_at_ms: int


# ---------------------------------------------------------------------------
# Batch assembly types
# ---------------------------------------------------------------------------


class IncumbentAnchor(BaseModel):
    """In-episode incumbent anchor block.

    Implements: REQ-BENCH-013
    """

    model_config = _BASE_MODEL_CONFIG

    incumbent_artifact_id: str
    shape_results: list[ShapeBenchResult]
    objective_score: ObjectiveScore
    anchor_drift_pct_per_shape: dict[str, float] = Field(default_factory=dict)
    measurement_quality_per_shape: dict[str, MeasurementQualityStatus] = Field(
        default_factory=dict,
    )


class CandidateResult(BaseModel):
    """Per-candidate result block.

    Implements: REQ-BENCH-008, REQ-BENCH-030
    """

    model_config = _BASE_MODEL_CONFIG

    candidate_hash: str
    envelope: MeasurementEnvelope
    benchmark: BenchmarkBundle | None = None
    incumbent_comparison: IncumbentComparison
    measurement_quality: MeasurementQualityStatus
    measurement_quality_reason: str | None = None
    shape_measurement_artifact_refs: dict[str, str] = Field(default_factory=dict)
    profile_status: ProfileStatus
    profile_unavailable_reason: ProfileUnavailableReason | None = None
    # Spec §6.6 + REQ-BENCH-030: one ProfileBundle per (candidate, profile_shape).
    # The first shape keeps the single-bundle semantic for compatibility via the
    # ``profile_bundle`` property below.
    profile_bundles: list[ProfileBundle] = Field(default_factory=list)
    raw_profile_metrics_ref: str | None = None
    profile_artifact_refs: list[ProfileArtifactRef] = Field(default_factory=list)
    fault_class: FaultClass | None = None
    failure_reason: str | None = None

    @property
    def profile_bundle(self) -> ProfileBundle | None:
        """Compatibility alias returning the first profile bundle or ``None``.

        Old call sites continue to read ``result.profile_bundle``; new code
        iterates ``profile_bundles`` for the full cartesian (candidate, shape)
        set.
        """
        return self.profile_bundles[0] if self.profile_bundles else None


class BaselineRef(BaseModel):
    """Baseline reference carried on the request."""

    model_config = _BASE_MODEL_CONFIG

    artifact_id: str
    objective_score: ObjectiveScore


class IncumbentRef(BaseModel):
    """Incumbent reference carried on the request."""

    model_config = _BASE_MODEL_CONFIG

    artifact_id: str
    objective_score: ObjectiveScore
    cubin_uri: str | None = None
    launch_spec: LaunchSpec | None = None
    launch_spec_hash: str | None = None
    source_hash: str | None = None
    toolchain_hash: str | None = None


class BenchmarkBatchRequest(BaseModel):
    """Full HTTP request body for ``POST /benchmark``.

    Implements: REQ-BENCH-001, REQ-BENCH-023, REQ-BENCH-024, REQ-BENCH-026
    Invariant: INV-BENCH-001 (no import of kerlever.protocols)
    """

    model_config = _BASE_MODEL_CONFIG

    request_id: str
    run_id: str
    round_id: str | None = None
    batch_id: str
    problem_spec: ProblemSpec
    objective_shape_cases: list[ShapeCase]
    profile_shape_cases: list[ShapeCase] = Field(default_factory=list)
    baseline_ref: BaselineRef
    incumbent_ref: IncumbentRef
    candidate_module_artifact_refs: list[CandidateArtifactRef]
    operation_adapter_abi: str
    operation_adapter_version: str
    artifact_execution_model: ArtifactExecutionModel
    metric_mode: MetricMode = MetricMode.DEVICE_KERNEL_US
    function_attribute_policy: FunctionAttributePolicy = Field(
        default_factory=FunctionAttributePolicy,
    )
    cache_policy: CachePolicy = CachePolicy.WARM_SAME_BUFFERS
    clock_policy: ClockPolicy = Field(
        default_factory=lambda: ClockPolicy(mode=ClockPolicyMode.OBSERVED_ONLY),
    )
    top_k_profile: int = 2
    top_m_profile_shift_potential: int = 1
    anchor_every_n_samples: int | None = None
    max_interleave_block_len: int | None = None
    bench_rerun_limit: int | None = None


class ToolchainIdentity(BaseModel):
    """Toolchain identity block in run envelope."""

    model_config = _BASE_MODEL_CONFIG

    driver_version: str
    cuda_runtime_version: str
    cuda_python_version: str
    pynvml_version: str
    ncu_version: str | None = None


class VisibleGpu(BaseModel):
    """Visible GPU identity block in run envelope."""

    model_config = _BASE_MODEL_CONFIG

    gpu_uuid: str
    pci_bus_id: str
    sm_arch: str
    mig_profile: str | None = None


class RunEnvelope(BaseModel):
    """Batch-level run envelope emitted in the response."""

    model_config = _BASE_MODEL_CONFIG

    run_id: str
    round_id: str | None = None
    batch_id: str
    request_id: str
    pod_id: str
    pod_health: PodHealth
    ambiguous_failure_count: int = 0
    toolchain: ToolchainIdentity
    visible_gpu: VisibleGpu


class MeasurementContext(BaseModel):
    """Batch-wide measurement context emitted in the response."""

    model_config = _BASE_MODEL_CONFIG

    artifact_execution_model: ArtifactExecutionModel
    metric_mode: MetricMode
    cache_policy_requested: CachePolicy
    cache_policy_effective: CachePolicy
    clock_policy: ClockPolicy
    interleave_enabled: bool
    anchor_every_n_samples: int | None = None
    max_interleave_block_len: int | None = None
    noise_floor_pct: float
    guard_pct: float


class BenchmarkBatchResult(BaseModel):
    """Full HTTP response body for ``POST /benchmark``.

    Implements: REQ-BENCH-008, REQ-BENCH-023
    """

    model_config = _BASE_MODEL_CONFIG

    status: BatchStatus
    run_envelope: RunEnvelope
    measurement_context: MeasurementContext
    hygiene: HygieneReport
    incumbent_anchor: IncumbentAnchor
    candidate_results: list[CandidateResult] = Field(default_factory=list)
    top_k_profiled: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# /healthz and /info response types
# ---------------------------------------------------------------------------


class DeviceInventoryEntry(BaseModel):
    """Cached device inventory entry populated at service startup."""

    model_config = _BASE_MODEL_CONFIG

    ordinal: int
    gpu_uuid: str
    pci_bus_id: str
    sm_arch: str
    mig_profile: str | None = None
    name: str | None = None


class HealthReport(BaseModel):
    """Readiness report for ``GET /healthz``."""

    model_config = _BASE_MODEL_CONFIG

    status: Literal["ready", "not_ready"]
    toolchain: ToolchainIdentity | None = None
    gpus: list[DeviceInventoryEntry] = Field(default_factory=list)
    pod_health: PodHealth = PodHealth.HEALTHY
    missing: list[str] = Field(default_factory=list)
    reason: str | None = None


class InfoResponse(BaseModel):
    """Response body for ``GET /info``."""

    model_config = _BASE_MODEL_CONFIG

    service_version: str
    build_hash: str | None = None
    toolchain: ToolchainIdentity
    gpus: list[DeviceInventoryEntry] = Field(default_factory=list)
    default_metric_mode: MetricMode
    artifact_execution_model: ArtifactExecutionModel
    supported_adapter_abis: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error envelope (HTTP transport-level only; §10.3)
# ---------------------------------------------------------------------------


class ErrorEnvelopeFieldError(BaseModel):
    """Single field validation error."""

    model_config = _BASE_MODEL_CONFIG

    loc: list[str | int]
    msg: str
    type: str


class ErrorEnvelope(BaseModel):
    """HTTP-level error shape (4xx/5xx only; domain errors use BenchmarkBatchResult)."""

    model_config = _BASE_MODEL_CONFIG

    code: Literal["bad_request", "unsupported", "internal_server_error"]
    detail: str
    field_errors: list[ErrorEnvelopeFieldError] | None = None
    request_id: str | None = None


__all__ = [
    "AdapterIterationSemantics",
    "AnchorDriftTelemetry",
    "ArtifactExecutionModel",
    "BaselineRef",
    "BatchStatus",
    "BenchmarkBatchRequest",
    "BenchmarkBatchResult",
    "BenchmarkBundle",
    "BottleneckAssessment",
    "CacheConfig",
    "CachePolicy",
    "CachePolicyBlock",
    "CandidateArtifactRef",
    "CandidateResult",
    "ClockPolicy",
    "ClockPolicyMode",
    "CorrectnessForward",
    "DeviceInventoryEntry",
    "DeviceTelemetrySnapshot",
    "ErrorEnvelope",
    "ErrorEnvelopeFieldError",
    "FaultClass",
    "FunctionAttribute",
    "FunctionAttributePolicy",
    "HealthReport",
    "HygieneReport",
    "IncumbentAnchor",
    "IncumbentComparison",
    "IncumbentRef",
    "InfoResponse",
    "LaunchSpec",
    "MeasurementContext",
    "MeasurementEnvelope",
    "MeasurementQuality",
    "MeasurementQualityStatus",
    "MetricMode",
    "NormalizedProfileMetricProvenance",
    "ObjectiveScore",
    "PerformanceObjective",
    "PodHealth",
    "ProblemSpec",
    "ProfileArtifactRef",
    "ProfileBundle",
    "ProfileMetrics",
    "ProfileStatus",
    "ProfileUnavailableReason",
    "ProfilerName",
    "RawProfileMetric",
    "RepeatPolicy",
    "ReplayMode",
    "RunEnvelope",
    "ShapeBenchResult",
    "ShapeCase",
    "ShapeMeasurementArtifact",
    "StaticAnalysis",
    "ToolchainIdentity",
    "VisibleGpu",
    "WarmupPolicy",
]
