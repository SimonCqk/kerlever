"""Benchmarker — disposable worker subprocess entrypoint.

Runs Phases 1b..7 inside a fresh process with a single CUDA context.
Writes a :class:`BenchmarkBatchResult` JSON to the configured result file
before calling :func:`os._exit` — this is the INV-BENCH-012 seam that
contains GPU faults to one batch.

Exit codes (design §3.2):

* ``0`` — success or controlled completion.
* ``1`` — candidate fault flushed before exit.
* ``2`` — uncaught exception; best-effort infra_error result flushed.

Spec: docs/benchmarker/spec.md §6 phases
Design: docs/benchmarker/design.md §3.2, §4.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from kerlever.benchmarker.adapter import (
    AdapterBuffers,
    OperationAdapter,
    get_adapter,
)
from kerlever.benchmarker.config import BenchmarkerConfig, HygieneThresholds
from kerlever.benchmarker.fault import attribute
from kerlever.benchmarker.harness import (
    BatchMeasurement,
    HarnessConfig,
    execute_batch,
)
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.normalize import (
    NormalizationError,
    NormalizedBatch,
    normalize_request,
)
from kerlever.benchmarker.plan import CalibratedPlan, LoadedCandidate, calibrate
from kerlever.benchmarker.profiler import (
    build_nvtx_range,
    ncu_version,
    parse_report,
    resolve_unavailable_reason,
    run_ncu,
)
from kerlever.benchmarker.profiler import (
    normalize as profiler_normalize,
)
from kerlever.benchmarker.scoring import (
    compute_objective_score,
    decide_incumbent_comparison,
)
from kerlever.benchmarker.selection import (
    ScoredCandidate,
    build_profile_set,
)
from kerlever.benchmarker.stats import (
    anchor_drift_pct,
    cv_pct,
    p50,
    p95,
    stdev,
)
from kerlever.benchmarker.stats import (
    mean as stats_mean,
)
from kerlever.benchmarker.telemetry import (
    cuda_python_version,
    cuda_runtime_version,
    driver_version,
    pynvml_version,
)
from kerlever.benchmarker.telemetry import init as telemetry_init
from kerlever.benchmarker.telemetry import shutdown as telemetry_shutdown
from kerlever.benchmarker.telemetry import snapshot as telemetry_snapshot
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    ArtifactExecutionModel,
    BatchStatus,
    BenchmarkBatchRequest,
    BenchmarkBatchResult,
    BenchmarkBundle,
    BottleneckAssessment,
    CachePolicy,
    CandidateArtifactRef,
    CandidateResult,
    DeviceTelemetrySnapshot,
    FaultClass,
    FunctionAttributePolicy,
    HygieneReport,
    IncumbentAnchor,
    IncumbentComparison,
    MeasurementContext,
    MeasurementEnvelope,
    MeasurementQuality,
    MeasurementQualityStatus,
    MetricMode,
    PodHealth,
    ProfileArtifactRef,
    ProfileBundle,
    ProfilerName,
    ProfileStatus,
    ProfileUnavailableReason,
    RawProfileMetric,
    RepeatPolicy,
    ReplayMode,
    RunEnvelope,
    ShapeBenchResult,
    ShapeMeasurementArtifact,
    ToolchainIdentity,
    VisibleGpu,
    WarmupPolicy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerArgs:
    """Parsed worker argv."""

    config_path: Path
    request_path: Path
    result_path: Path
    device_uuid: str
    device_ordinal: int


def _parse_argv(argv: list[str]) -> WorkerArgs:
    """Parse the worker argv into a :class:`WorkerArgs` record."""
    parser = argparse.ArgumentParser(prog="kerlever.benchmarker.worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--device-ordinal", type=int, required=True)
    ns = parser.parse_args(argv)
    return WorkerArgs(
        config_path=Path(ns.config),
        request_path=Path(ns.request_file),
        result_path=Path(ns.result_file),
        device_uuid=ns.device_uuid,
        device_ordinal=ns.device_ordinal,
    )


def _load_config(path: Path) -> BenchmarkerConfig:
    """Load BenchmarkerConfig from the JSON file written by the supervisor."""
    return BenchmarkerConfig.from_dict(json.loads(path.read_text()))


def _read_request(path: Path) -> BenchmarkBatchRequest:
    """Pydantic-parse the request JSON written by the supervisor."""
    return BenchmarkBatchRequest.model_validate_json(path.read_text())


def _write_result(path: Path, result: BenchmarkBatchResult) -> None:
    """Write result.json atomically (tmp + rename) per design §4.3."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(result.model_dump_json())
    os.replace(tmp, path)


_ARTIFACT_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_artifact_token(value: str) -> str:
    """Return a filesystem-safe token for pod-local artifact paths."""
    safe = _ARTIFACT_TOKEN_RE.sub("_", value).strip("._")
    return safe or "empty"


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically write a text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _artifact_id(kind: str, batch_id: str, filename: str) -> str:
    """Build the opaque artifact id used in response references."""
    return f"{kind}/{_safe_artifact_token(batch_id)}/{filename}"


def _artifact_path(root: Path, kind: str, batch_id: str, filename: str) -> Path:
    """Resolve a durable artifact path under the pod-local artifact root."""
    return root / kind / _safe_artifact_token(batch_id) / filename


def _write_shape_artifact(
    root: Path,
    batch_id: str,
    candidate_hash: str,
    shape_id: str,
    artifact: ShapeMeasurementArtifact,
) -> str:
    """Persist a ShapeMeasurementArtifact and return its artifact id."""
    filename = (
        f"{_safe_artifact_token(candidate_hash)}_"
        f"{_safe_artifact_token(shape_id)}.json"
    )
    path = _artifact_path(root, "samples", batch_id, filename)
    _write_text_atomic(path, artifact.model_dump_json())
    return _artifact_id("samples", batch_id, filename)


def _write_raw_metrics_artifact(
    root: Path,
    batch_id: str,
    candidate_hash: str,
    shape_id: str,
    raw: list[RawProfileMetric],
) -> ProfileArtifactRef:
    """Persist raw profiler metrics and return a typed artifact reference."""
    filename = (
        f"{_safe_artifact_token(candidate_hash)}_"
        f"{_safe_artifact_token(shape_id)}.json"
    )
    path = _artifact_path(root, "raw_metrics", batch_id, filename)
    payload = json.dumps([m.model_dump() for m in raw], separators=(",", ":"))
    _write_text_atomic(path, payload)
    return ProfileArtifactRef(
        artifact_id=_artifact_id("raw_metrics", batch_id, filename),
        kind="raw_metrics_json",
        uri=str(path),
        size_bytes=path.stat().st_size,
        created_at_ms=int(time.time() * 1000),
    )


def _ncu_artifact_ref(path: Path, batch_id: str) -> ProfileArtifactRef:
    """Build a typed artifact reference for a durable NCU report."""
    return ProfileArtifactRef(
        artifact_id=_artifact_id("ncu", batch_id, path.name),
        kind="ncu_report",
        uri=str(path),
        size_bytes=path.stat().st_size,
        created_at_ms=int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# Result assembly helpers (Phase 7)
# ---------------------------------------------------------------------------


def _build_run_envelope(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
    pod_health: PodHealth,
    ambiguous_count: int,
) -> RunEnvelope:
    """Construct the response's ``run_envelope`` block."""
    return RunEnvelope(
        run_id=req.run_id,
        round_id=req.round_id,
        batch_id=req.batch_id,
        request_id=req.request_id,
        pod_id=cfg.pod_id,
        pod_health=pod_health,
        ambiguous_failure_count=ambiguous_count,
        toolchain=ToolchainIdentity(
            driver_version=driver_version(),
            cuda_runtime_version=cuda_runtime_version(),
            cuda_python_version=cuda_python_version(),
            pynvml_version=pynvml_version(),
            ncu_version=ncu_version(cfg.profiler),
        ),
        visible_gpu=VisibleGpu(
            gpu_uuid=device.gpu_uuid,
            pci_bus_id=device.pci_bus_id,
            sm_arch=device.sm_arch,
            mig_profile=device.mig_profile,
        ),
    )


def _build_measurement_context(
    req: BenchmarkBatchRequest,
    normalized: NormalizedBatch,
    cfg: BenchmarkerConfig,
) -> MeasurementContext:
    """Construct the response's ``measurement_context`` block."""
    return MeasurementContext(
        artifact_execution_model=req.artifact_execution_model,
        metric_mode=req.metric_mode,
        cache_policy_requested=normalized.requested_cache_policy,
        cache_policy_effective=normalized.effective_cache_policy,
        clock_policy=normalized.clock_policy,
        interleave_enabled=normalized.interleave_enabled,
        anchor_every_n_samples=req.anchor_every_n_samples
        or cfg.calibration.anchor_every_n_samples,
        max_interleave_block_len=req.max_interleave_block_len
        or cfg.calibration.max_interleave_block_len,
        noise_floor_pct=cfg.thresholds.noise_floor_pct,
        guard_pct=req.problem_spec.objective.regression_guard_pct,
    )


def _empty_hygiene(device: LeasedDevice, reason: str | None) -> HygieneReport:
    """Worker-local hygiene stub; the authoritative report is built service-side."""
    return HygieneReport(
        gpu_uuid=device.gpu_uuid,
        sm_arch=device.sm_arch,
        mig_profile=device.mig_profile,
        compute_mode="UNKNOWN",
        reason_on_fail=reason,
    )


def _build_empty_incumbent_anchor(
    req: BenchmarkBatchRequest,
) -> IncumbentAnchor:
    """Produce an incumbent anchor block when no anchors were collected."""
    return IncumbentAnchor(
        incumbent_artifact_id=req.incumbent_ref.artifact_id,
        shape_results=[],
        objective_score=req.incumbent_ref.objective_score,
    )


# ---------------------------------------------------------------------------
# Candidate loading (Phase 3 prelude)
# ---------------------------------------------------------------------------


def _dry_launcher(
    _candidate: LoadedCandidate, _shape: object, iterations: int
) -> float:
    """Calibration launcher used when CUDA is unavailable on the dev host.

    Returns a fixed ``iterations * base_ms`` value so calibration converges
    without touching the GPU; the real worker path replaces this with an
    actual cuEventElapsedTime call.
    """
    return iterations * 0.2


def _cuda_calibration_launcher(
    stream_handle: object,
    build_args: object,
    resolve_grid_dim: Callable[[LoadedCandidate, object], tuple[int, int, int]],
) -> object:
    """Return a launcher that runs one calibration sample on the GPU.

    The returned closure issues ``iterations`` launches, records events,
    and returns elapsed_ms. Kept out of :mod:`plan` so the plan module
    stays free of CUDA imports.
    """
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    def _launch_one(
        cand: LoadedCandidate,
        shape: object,
        iterations: int,
    ) -> float:
        start = cd.create_event()
        stop = cd.create_event()
        try:
            cd.event_record(start, stream_handle)  # type: ignore[arg-type]
            for _ in range(iterations):
                args = build_args(cand, shape)  # type: ignore[operator]
                cd.launch(
                    cand.function,
                    resolve_grid_dim(cand, shape),
                    cand.block_dim,
                    cand.dynamic_smem_bytes,
                    stream_handle,  # type: ignore[arg-type]
                    args,
                )
            cd.event_record(stop, stream_handle)  # type: ignore[arg-type]
            elapsed_ms = cd.event_elapsed_ms(start, stop)
        finally:
            cd.destroy_event(start)
            cd.destroy_event(stop)
        return elapsed_ms

    return _launch_one


def _load_candidates(
    admit: list[CandidateArtifactRef],
    adapter: OperationAdapter,
) -> tuple[
    list[LoadedCandidate],
    dict[str, FunctionAttributePolicy],
    dict[str, str],
]:
    """Load cubin bytes, resolve functions, and apply function-attribute policy.

    For each admitted candidate we:

    1. Read cubin bytes (spec §8.2 step 9).
    2. ``cuModuleLoadDataEx`` → ``cuModuleGetFunction``.
    3. Apply every non-null field of the candidate's requested
       ``FunctionAttributePolicy`` via ``cuFuncSetAttribute``.
    4. Read observed values back so the envelope captures the clamped
       value (REQ-BENCH-008, REQ-BENCH-029).

    Apply-failures are classified as per-candidate infra faults and the
    candidate is returned in the ``rejected`` map so other candidates can
    still run (spec §10.2).

    Returns:
        ``(loaded, observed_policies, rejected_with_reason)`` where
        ``rejected_with_reason`` maps ``candidate_hash`` → reason token
        (e.g., ``"function_attribute_policy_apply_failed"``).

    Implements: REQ-BENCH-029 (V1 scope: ``max_dynamic_shared_memory_size``
        only; other FunctionAttributePolicy fields deferred per spec §9)
    """
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415
    from kerlever.benchmarker.types import FunctionAttribute  # noqa: PLC0415

    loaded: list[LoadedCandidate] = []
    observed_policies: dict[str, FunctionAttributePolicy] = {}
    rejected: dict[str, str] = {}
    for ref in admit:
        try:
            cubin_path = Path(ref.cubin_uri)
            cubin_bytes = cubin_path.read_bytes()
            module = cd.load_module(cubin_bytes, None)
            fn = cd.get_function(module, ref.launch_spec.entrypoint)
        except Exception as exc:  # noqa: BLE001 — classified below
            logger.warning(
                "worker.load_candidate.failed",
                extra={
                    "candidate_hash": ref.candidate_hash,
                    "error": str(exc),
                },
            )
            rejected[ref.candidate_hash] = "cubin_load_failed"
            continue

        # REQ-BENCH-029: apply function-attribute policy and read back.
        observed = FunctionAttributePolicy()
        try:
            if ref.launch_spec.dynamic_smem_bytes > 0:
                observed_smem = cd.set_function_attribute(
                    fn,
                    FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    int(ref.launch_spec.dynamic_smem_bytes),
                )
                observed = observed.model_copy(
                    update={"max_dynamic_shared_memory_size": observed_smem}
                )
        except cd.CudaDriverError as exc:
            logger.warning(
                "worker.function_attribute_apply_failed",
                extra={
                    "candidate_hash": ref.candidate_hash,
                    "error": str(exc),
                },
            )
            rejected[ref.candidate_hash] = (
                "function_attribute_policy_apply_failed"
            )
            continue

        observed_policies[ref.candidate_hash] = observed
        loaded.append(
            LoadedCandidate(
                candidate_hash=ref.candidate_hash,
                function=fn,
                launch_args_factory=None,
                adapter_iteration_semantics=adapter.iteration_semantics,
                function_attribute_policy_observed=observed,
                block_dim=ref.launch_spec.block_dim,
                grid_dim=ref.launch_spec.grid_dim,
                dynamic_smem_bytes=ref.launch_spec.dynamic_smem_bytes,
            )
        )
    return loaded, observed_policies, rejected


# ---------------------------------------------------------------------------
# Adapter-driven helpers (buffer allocation, launch-arg building, reset hooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfraFaultRecord:
    """Compact reject record used when load-time failures add to the batch.

    Deliberately identical in shape to :class:`normalize.InfraFault` so
    ``_reject_candidate_result`` can consume either.
    """

    candidate_hash: str
    reason: str
    fault_class: FaultClass


def _build_args_for(
    adapter: OperationAdapter,
    buffers: dict[tuple[str, str], AdapterBuffers],
    cand: LoadedCandidate,
    shape: object,
) -> tuple[object, ...]:
    """Dispatch to the adapter for the cand-specific buffer entry.

    Anchor launches use the incumbent's buffer pool, so we accept whichever
    key is present in ``buffers``. A missing buffer is a spec-level bug —
    every (candidate, shape) must be allocated up front in Phase 3 prep.
    """
    from kerlever.benchmarker.types import ShapeCase as _ShapeCase  # noqa: PLC0415

    if not isinstance(shape, _ShapeCase):
        raise TypeError(f"expected ShapeCase, got {type(shape)!r}")
    key = (cand.candidate_hash, shape.shape_id)
    bufs = buffers.get(key)
    if bufs is None:
        raise RuntimeError(
            f"no adapter buffer for {key}; Phase 3 prep missed this pair"
        )
    return adapter.build_launch_args(bufs, shape)


def _make_reset_hook(
    adapter: OperationAdapter,
    buffers: dict[tuple[str, str], AdapterBuffers],
    cand: LoadedCandidate,
) -> Callable[[], None]:
    """Return a reset-hook closure for ``cand`` that resets every shape's buffer.

    Implements: INV-BENCH-013
    """

    def _reset() -> None:
        for (cand_hash, _shape_id), bufs in buffers.items():
            if cand_hash != cand.candidate_hash:
                continue
            adapter.reset_between_iterations(
                bufs, cand.adapter_iteration_semantics
            )

    return _reset


def _adapter_seed(
    run_id: str, batch_id: str, shape_id: str, candidate_hash: str
) -> int:
    """Deterministic seed aligned with :func:`profile_child._profile_seed`.

    Spec §6.11 requires the profile child to re-seed with the *same* value
    so the profiled launch reads the same inputs the measurement saw.
    """
    key = (
        f"{run_id}|{batch_id}|{shape_id}|{candidate_hash}|"
        "kerlever_profile_seed"
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _allocate_shape_buffers(
    *,
    adapter: OperationAdapter,
    candidates: list[LoadedCandidate],
    incumbent: LoadedCandidate,
    shapes: Sequence[object],
    device: LeasedDevice,
    run_id: str,
    batch_id: str,
    dtype: str,
    out: dict[tuple[str, str], AdapterBuffers],
    pools: dict[tuple[str, str], list[AdapterBuffers]],
    allocated: list[AdapterBuffers],
    pool_size: int,
) -> None:
    """Allocate + seed one :class:`AdapterBuffers` per (candidate, shape) pair.

    Incumbent is included so anchor launches have a matching buffer set
    (spec §6.11 / design §3.3). Out-parameter pattern keeps the caller's
    ``finally`` block simple: free whatever is in ``out``. ``dtype`` is
    the element dtype from ``problem_spec.dtype`` — adapters store it on
    :class:`AdapterBuffers` for debugging.
    """
    from kerlever.benchmarker.types import ShapeCase as _ShapeCase  # noqa: PLC0415

    all_cands = list(candidates) + [incumbent]
    for cand in all_cands:
        for shape in shapes:
            if not isinstance(shape, _ShapeCase):
                raise TypeError(
                    f"_allocate_shape_buffers expects ShapeCase; got {type(shape)!r}"
                )
            seed = _adapter_seed(
                run_id, batch_id, shape.shape_id, cand.candidate_hash
            )
            key = (cand.candidate_hash, shape.shape_id)
            pool: list[AdapterBuffers] = []
            for _ in range(max(1, pool_size)):
                bufs = adapter.allocate(shape, dtype, device)
                adapter.seed_inputs(bufs, shape, seed)
                pool.append(bufs)
                allocated.append(bufs)
            pools[key] = pool
            out[key] = pool[0]


# ---------------------------------------------------------------------------
# Scoring assembly (Phase 5 → Phase 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeArtifactContext:
    """Per-(candidate, shape) context threaded into ``_candidate_shape_result``.

    Every field here is taken from real runtime state so the rich artifact
    matches the spec §6.11 / REQ-BENCH-032 fidelity contract. Static defaults
    are a spec violation (INV-BENCH-014).
    """

    warmup_count: int
    iterations_per_sample: int
    max_timed_batch_ms: float
    interleave_block_len: int | None
    interleave_block_order: list[str]
    anchor_pre_us: float | None
    anchor_post_us: float | None
    anchor_drift_pct: float | None
    requested_cache: CachePolicy
    effective_cache: CachePolicy
    metric_mode: MetricMode
    artifact_execution_model: ArtifactExecutionModel
    adapter_iteration_semantics: AdapterIterationSemantics
    function_attribute_policy: FunctionAttributePolicy
    telemetry_before: DeviceTelemetrySnapshot
    telemetry_after: DeviceTelemetrySnapshot
    anchor_every_n_samples: int | None


def _candidate_shape_result(
    shape_id: str,
    samples: list[float],
    min_p95_samples: int,
    ctx: ShapeArtifactContext,
    *,
    thresholds: HygieneThresholds,
) -> tuple[ShapeBenchResult, ShapeMeasurementArtifact]:
    """Build the compact ``ShapeBenchResult`` + rich ``ShapeMeasurementArtifact``.

    Every field of :class:`ShapeMeasurementArtifact` is derived from real
    runtime state threaded through ``ctx``; no static defaults leak into
    the artifact (REQ-BENCH-032, INV-BENCH-014). ``thresholds`` feeds the
    warn/fail CV classification into :func:`_classify_artifact_quality`.

    Implements: REQ-BENCH-032
    Invariant: INV-BENCH-014
    """
    # Degraded-but-still-valid case: zero samples surfaced from the worker
    # get zeroed numerics; the artifact records the empty sample list.
    if not samples:
        compact = ShapeBenchResult(
            shape_id=shape_id,
            # REQ-BENCH-033 sentinel — no p95 available.
            latency_p50_us=0.0,
            latency_p95_us=-1.0,
            stdev_us=0.0,
            run_count=0,
        )
        return compact, _empty_artifact(shape_id, min_p95_samples, ctx)
    p50_val = p50(samples)
    p95_val = p95(samples, min_p95_samples)
    mean_val = stats_mean(samples)
    stdev_val = stdev(samples)
    cv_val = cv_pct(mean_val, stdev_val)
    # REQ-BENCH-033: ShapeBenchResult.latency_p95_us uses -1.0 as the
    # "p95 unavailable" sentinel when samples < min_p95_samples. Consumers
    # needing p95 must read ShapeMeasurementArtifact.p95_us (which is None).
    latency_p95 = p95_val if p95_val is not None else -1.0
    compact = ShapeBenchResult(
        shape_id=shape_id,
        latency_p50_us=p50_val,
        latency_p95_us=latency_p95,
        stdev_us=stdev_val,
        run_count=len(samples),
    )
    quality_status = _classify_artifact_quality(
        samples=samples,
        p95_val=p95_val,
        cv_val=cv_val,
        thresholds=thresholds,
    )
    artifact = ShapeMeasurementArtifact(
        shape_id=shape_id,
        samples_us=list(samples),
        warmup_count=ctx.warmup_count,
        iterations_per_sample=ctx.iterations_per_sample,
        min_samples_required=min_p95_samples,
        p50_us=p50_val,
        p95_us=p95_val,
        mean_us=mean_val,
        stdev_us=stdev_val,
        cv_pct=cv_val,
        min_us=min(samples),
        max_us=max(samples),
        cache_policy=ctx.effective_cache,
        requested_cache_policy=ctx.requested_cache,
        effective_cache_policy=ctx.effective_cache,
        interleave_block_len=ctx.interleave_block_len,
        anchor_every_n_samples=ctx.anchor_every_n_samples,
        anchor_pre_us=ctx.anchor_pre_us,
        anchor_post_us=ctx.anchor_post_us,
        anchor_drift_pct=ctx.anchor_drift_pct,
        interleave_block_order=list(ctx.interleave_block_order),
        artifact_execution_model=ctx.artifact_execution_model,
        adapter_iteration_semantics=ctx.adapter_iteration_semantics,
        metric_mode=ctx.metric_mode,
        max_timed_batch_ms=ctx.max_timed_batch_ms,
        function_attribute_policy=ctx.function_attribute_policy,
        measurement_quality=MeasurementQuality(status=quality_status),
        telemetry_before=ctx.telemetry_before,
        telemetry_after=ctx.telemetry_after,
    )
    return compact, artifact


def _classify_artifact_quality(
    samples: list[float],
    p95_val: float | None,
    cv_val: float | None,
    thresholds: HygieneThresholds,
) -> MeasurementQualityStatus:
    """Classify the artifact's measurement quality from real sample stats.

    Uses the configured fail/warn CV thresholds (spec §6.4):

    * no samples → ``INFRA_FAULT``.
    * ``cv_pct > thresholds.measurement_cv_fail_pct`` → ``UNSTABLE``.
    * ``cv_pct > thresholds.measurement_cv_warn_pct`` → ``VALID_WITH_WARNING``.
    * else → ``VALID``.

    The caller aggregates this artifact-level status with other signals
    (p95/p50 ratio, calibration warnings) in :func:`_shape_quality_status`.
    """
    _ = p95_val
    if not samples:
        return MeasurementQualityStatus.INFRA_FAULT
    if cv_val is not None and cv_val > thresholds.measurement_cv_fail_pct:
        return MeasurementQualityStatus.UNSTABLE
    if cv_val is not None and cv_val > thresholds.measurement_cv_warn_pct:
        return MeasurementQualityStatus.VALID_WITH_WARNING
    return MeasurementQualityStatus.VALID


def _empty_artifact(
    shape_id: str,
    min_required: int,
    ctx: ShapeArtifactContext,
) -> ShapeMeasurementArtifact:
    """Construct an artifact populated from real context when no samples survived.

    Required by REQ-BENCH-032: even in the no-samples path we surface the
    realized warmup count, calibration plan, cache policy, and telemetry so
    consumers can diagnose why the shape produced nothing.
    """
    return ShapeMeasurementArtifact(
        shape_id=shape_id,
        samples_us=[],
        warmup_count=ctx.warmup_count,
        iterations_per_sample=ctx.iterations_per_sample,
        min_samples_required=min_required,
        p50_us=0.0,
        p95_us=None,
        mean_us=0.0,
        stdev_us=0.0,
        cv_pct=None,
        min_us=0.0,
        max_us=0.0,
        cache_policy=ctx.effective_cache,
        requested_cache_policy=ctx.requested_cache,
        effective_cache_policy=ctx.effective_cache,
        interleave_block_len=ctx.interleave_block_len,
        anchor_every_n_samples=ctx.anchor_every_n_samples,
        anchor_pre_us=ctx.anchor_pre_us,
        anchor_post_us=ctx.anchor_post_us,
        anchor_drift_pct=ctx.anchor_drift_pct,
        interleave_block_order=list(ctx.interleave_block_order),
        artifact_execution_model=ctx.artifact_execution_model,
        adapter_iteration_semantics=ctx.adapter_iteration_semantics,
        metric_mode=ctx.metric_mode,
        max_timed_batch_ms=ctx.max_timed_batch_ms,
        function_attribute_policy=ctx.function_attribute_policy,
        measurement_quality=MeasurementQuality(
            status=MeasurementQualityStatus.INFRA_FAULT,
            reason="no_samples_collected",
        ),
        telemetry_before=ctx.telemetry_before,
        telemetry_after=ctx.telemetry_after,
    )


def _empty_telemetry_snapshot() -> DeviceTelemetrySnapshot:
    """Return an empty :class:`DeviceTelemetrySnapshot` for degraded paths."""
    return DeviceTelemetrySnapshot(taken_at_ms=int(time.time() * 1000))


# ---------------------------------------------------------------------------
# Main phase driver
# ---------------------------------------------------------------------------


def _run_phases(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
    config_file_path: Path,
    request_file_path: Path,
) -> BenchmarkBatchResult:
    """Run Phases 1b..7 end to end and return a :class:`BenchmarkBatchResult`.

    Any caught :class:`NormalizationError` short-circuits to
    ``status=infra_error``. All other exceptions propagate to
    :func:`main` which writes an ``INFRA_ERROR`` result and exits 2.

    ``config_file_path`` and ``request_file_path`` are threaded through so
    Phase 6 can hand them to the profile-child subprocess (spec §6.12 /
    design §3.5).

    Implements: REQ-BENCH-001, REQ-BENCH-006, REQ-BENCH-011,
        REQ-BENCH-012, REQ-BENCH-014, REQ-BENCH-015, REQ-BENCH-016,
        REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-019, REQ-BENCH-024,
        REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030, REQ-BENCH-032,
        REQ-BENCH-033, REQ-BENCH-034
    """
    # Phase 1 — normalize.
    try:
        normalized = normalize_request(req, cfg, device)
    except NormalizationError as exc:
        return _build_infra_error_result(req, cfg, device, exc.reason)

    # REQ-BENCH-028: resolve adapter up front so a mis-registered adapter
    # exits before we open NVML / CUDA. Normalize already verified the
    # registration; we resolve here for the instance.
    adapter = get_adapter(
        req.operation_adapter_abi, req.operation_adapter_version
    )
    if adapter is None:
        return _build_infra_error_result(
            req, cfg, device, "adapter_not_registered"
        )

    # Phase 2 preflight telemetry — REQ-BENCH-032 binds this snapshot into
    # every ShapeMeasurementArtifact.telemetry_before. NVML may be absent on
    # dev hosts; we log and substitute an empty snapshot without aborting
    # (spec SCN-BENCH-015-06 tolerates NVML-unavailable with a warning).
    try:
        preflight_telemetry = telemetry_snapshot(device)
    except Exception as exc:  # noqa: BLE001 — degraded path allowed
        logger.warning(
            "worker.telemetry.nvml_unavailable",
            extra={"phase": "preflight", "error": str(exc)},
        )
        preflight_telemetry = _empty_telemetry_snapshot()

    # Phase 2b/3: CUDA context, load candidates.
    # Import lazily so pure-import of this module does not require cuda-python.
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    cd.init()
    ctx = cd.create_primary_context(device.ordinal)
    stream = cd.create_stream()
    # Per-(candidate_hash, shape_id) AdapterBuffers for teardown in the
    # ``finally`` block below. Populated below once the plan is known.
    per_shape_buffers: dict[tuple[str, str], AdapterBuffers] = {}
    try:
        loaded, observed_policies, load_rejects = _load_candidates(
            normalized.admit_candidates,
            adapter,
        )
        # Accumulate load-time rejections into normalized.reject_candidates so
        # the final candidate_results surface them. Build a mutable dict
        # because NormalizedBatch is frozen.
        load_reject_map: dict[str, InfraFaultRecord] = {
            h: InfraFaultRecord(
                candidate_hash=h,
                reason=reason,
                fault_class=FaultClass.INFRA_FAULT,
            )
            for h, reason in load_rejects.items()
        }
        # Incumbent load (when the request carries a cubin).
        incumbent_loaded = _load_incumbent(req, adapter)

        def _resolve_grid_dim(
            cand: LoadedCandidate, shape: object
        ) -> tuple[int, int, int]:
            from kerlever.benchmarker.types import (
                ShapeCase as _ShapeCase,  # noqa: PLC0415
            )

            if not isinstance(shape, _ShapeCase):
                raise TypeError(f"expected ShapeCase, got {type(shape)!r}")
            return cand.grid_dim or adapter.grid_dim(shape, cand.block_dim)

        # Phase 3: calibration.
        calibration_launcher = _cuda_calibration_launcher(
            stream,
            lambda cand, shape: _build_args_for(
                adapter, per_shape_buffers, cand, shape
            ),
            _resolve_grid_dim,
        )
        # Allocate + seed buffers for every shape (objective + profile).
        all_shapes = list(req.objective_shape_cases) + list(
            req.profile_shape_cases
        )
        buffer_pools: dict[tuple[str, str], list[AdapterBuffers]] = {}
        allocated_buffers: list[AdapterBuffers] = []
        pool_size = (
            max(2, cfg.calibration.max_interleave_block_len)
            if normalized.effective_cache_policy == CachePolicy.WARM_ROTATING_BUFFERS
            else 1
        )
        _allocate_shape_buffers(
            adapter=adapter,
            candidates=loaded,
            incumbent=incumbent_loaded,
            shapes=all_shapes,
            device=device,
            run_id=req.run_id,
            batch_id=req.batch_id,
            dtype=req.problem_spec.dtype,
            out=per_shape_buffers,
            pools=buffer_pools,
            allocated=allocated_buffers,
            pool_size=pool_size,
        )

        def _before_sample(cand: LoadedCandidate, shape: object) -> None:
            from kerlever.benchmarker.types import (
                ShapeCase as _ShapeCase,  # noqa: PLC0415
            )

            if normalized.effective_cache_policy != CachePolicy.WARM_ROTATING_BUFFERS:
                return
            if not isinstance(shape, _ShapeCase):
                raise TypeError(f"expected ShapeCase, got {type(shape)!r}")
            key = (cand.candidate_hash, shape.shape_id)
            pool = buffer_pools.get(key)
            if not pool:
                raise RuntimeError(f"missing buffer pool for {key}")
            per_shape_buffers[key] = adapter.rotate_buffers(pool)

        plan_out = calibrate(
            candidates=loaded,
            shapes=req.objective_shape_cases,
            cfg=cfg.calibration,
            metric_mode=req.metric_mode,
            requested_cache_policy=normalized.requested_cache_policy,
            effective_cache_policy=normalized.effective_cache_policy,
            cache_policy_reason=normalized.cache_policy_reason,
            launcher=calibration_launcher,
        )

        # Phase 4: harness execute_batch, with adapter-driven args + reset.
        harness_cfg = HarnessConfig(
            repetitions=cfg.calibration.repetitions,
            anchor_every_n_samples=(
                req.anchor_every_n_samples or cfg.calibration.anchor_every_n_samples
            ),
            max_interleave_block_len=(
                req.max_interleave_block_len
                or cfg.calibration.max_interleave_block_len
            ),
            kernel_timeout_ms=cfg.supervisor.kernel_timeout_ms,
        )
        reset_hooks: dict[str, Callable[[], None]] = {}
        for cand in list(loaded) + [incumbent_loaded]:
            if (
                cand.adapter_iteration_semantics
                != AdapterIterationSemantics.OVERWRITE_PURE
            ):
                reset_hooks[cand.candidate_hash] = _make_reset_hook(
                    adapter, per_shape_buffers, cand
                )
        batch_meas = execute_batch(
            plan=plan_out,
            candidates=loaded,
            incumbent=incumbent_loaded,
            shapes=req.objective_shape_cases,
            seeds=normalized.interleave_seed_per_shape,
            cfg=harness_cfg,
            stream=stream,
            build_args=lambda cand, shape: _build_args_for(
                adapter, per_shape_buffers, cand, shape
            ),
            resolve_grid_dim=_resolve_grid_dim,
            reset_hook_per_candidate=reset_hooks,
            before_sample=_before_sample,
        )

        # Phase 4 postflight telemetry — REQ-BENCH-032 binds this snapshot
        # into every ShapeMeasurementArtifact.telemetry_after. Same degraded
        # path as preflight: log + substitute empty on NVML errors.
        try:
            postflight_telemetry = telemetry_snapshot(device)
        except Exception as exc:  # noqa: BLE001 — degraded path allowed
            logger.warning(
                "worker.telemetry.nvml_unavailable",
                extra={"phase": "postflight", "error": str(exc)},
            )
            postflight_telemetry = _empty_telemetry_snapshot()

        # Phase 5: stats, objective scores, incumbent comparisons.
        shape_weights = {
            s.shape_id: s.weight for s in req.objective_shape_cases
        }
        incumbent_shape_results = _incumbent_shape_results(
            batch_meas,
            cfg,
            plan_out,
            normalized,
            req,
            incumbent_loaded,
            preflight_telemetry,
            postflight_telemetry,
        )
        incumbent_score = compute_objective_score(
            incumbent_shape_results,
            req.problem_spec.objective,
            shape_weights,
            baseline_value=req.baseline_ref.objective_score.value,
            incumbent_anchor_value=req.incumbent_ref.objective_score.value,
        )
        # REQ-BENCH-034: incumbent CV comes from the anchor samples we just
        # measured; we take the max across shapes as conservative.
        incumbent_cv_pct = _compute_incumbent_cv_pct(batch_meas)
        # INV-BENCH-015: build the incumbent's own envelope from real state
        # rather than cloning a candidate's.
        incumbent_envelope = _build_incumbent_envelope(
            req=req,
            normalized=normalized,
            device=device,
            incumbent=incumbent_loaded,
        )
        scored_cands: list[ScoredCandidate] = []
        candidate_results: list[CandidateResult] = []
        for cand in loaded:
            envelope = normalized.envelope_per_candidate[cand.candidate_hash]
            (
                shape_results,
                artifact_refs,
                quality_statuses,
                quality_reasons,
                cand_cv_pct,
            ) = (
                _assemble_candidate_shapes(
                    cand,
                    batch_meas,
                    plan_out,
                    cfg,
                    normalized=normalized,
                    req=req,
                    preflight_telemetry=preflight_telemetry,
                    postflight_telemetry=postflight_telemetry,
                )
            )
            cand_score = compute_objective_score(
                shape_results,
                req.problem_spec.objective,
                shape_weights,
                baseline_value=req.baseline_ref.objective_score.value,
                incumbent_anchor_value=max(incumbent_score.value, 1e-9),
            )
            anchor_drift = _aggregate_anchor_drift(batch_meas)
            if not math.isfinite(cand_score.value):
                comparison = IncumbentComparison.NOT_COMPARABLE
            elif incumbent_envelope is None:
                # INV-BENCH-015: no real incumbent envelope available →
                # refuse to fabricate one; mark not_comparable.
                comparison = IncumbentComparison.NOT_COMPARABLE
            else:
                comparison = decide_incumbent_comparison(
                    candidate_envelope=envelope,
                    candidate_score=cand_score.value,
                    candidate_quality=list(quality_statuses.values()),
                    candidate_cv_pct=cand_cv_pct,
                    incumbent_envelope=incumbent_envelope,
                    incumbent_score=incumbent_score.value,
                    incumbent_cv_pct=incumbent_cv_pct,
                    anchor_drift_fraction=anchor_drift,
                    guard_pct=req.problem_spec.objective.regression_guard_pct,
                    noise_floor_pct=cfg.thresholds.noise_floor_pct,
                )
            scored_cands.append(
                ScoredCandidate(
                    candidate_hash=cand.candidate_hash,
                    incumbent_comparison=comparison,
                    objective_score=cand_score,
                    candidate_cv_pct=cand_cv_pct,
                )
            )
            worst_quality = _worst_quality(list(quality_statuses.values()))
            quality_reason = _first_quality_reason(quality_reasons, worst_quality)
            fault_cls = _fault_for_candidate(cand, batch_meas, worst_quality)
            bundle = BenchmarkBundle(
                shape_results=shape_results,
                objective_score=cand_score,
                regressed_vs_incumbent=(comparison == IncumbentComparison.REGRESSED),
            )
            # REQ-BENCH-008: envelope records observed attribute readback so
            # downstream consumers see the clamped value for this candidate.
            observed = observed_policies.get(cand.candidate_hash)
            if observed is not None:
                envelope = envelope.model_copy(
                    update={"function_attribute_policy_observed": observed}
                )
            candidate_results.append(
                CandidateResult(
                    candidate_hash=cand.candidate_hash,
                    envelope=envelope,
                    benchmark=bundle,
                    incumbent_comparison=comparison,
                    measurement_quality=worst_quality,
                    measurement_quality_reason=quality_reason,
                    shape_measurement_artifact_refs=artifact_refs,
                    profile_status=ProfileStatus.PROFILE_UNAVAILABLE,
                    profile_unavailable_reason=None,
                    profile_bundles=[],
                    fault_class=fault_cls,
                )
            )

        # Phase 6: profile target selection + NCU.
        incumbent_scored = ScoredCandidate(
            candidate_hash="__incumbent__",
            incumbent_comparison=IncumbentComparison.IMPROVED,
            objective_score=incumbent_score,
        )
        profile_set = build_profile_set(
            scoreable=scored_cands,
            k=req.top_k_profile,
            m=req.top_m_profile_shift_potential,
            incumbent=incumbent_scored,
            include_incumbent=cfg.profiler.include_incumbent,
            hints_per_candidate={},
        )
        profile_set_hashes = {c.candidate_hash for c in profile_set}
        _run_profile_phase(
            req=req,
            cfg=cfg,
            device=device,
            candidate_results=candidate_results,
            loaded=loaded,
            profile_set_hashes=profile_set_hashes,
            plan_out=plan_out,
            config_file_path=config_file_path,
            request_file_path=request_file_path,
        )

        # Phase 7: assembly.
        top_k_profiled = [
            c.candidate_hash
            for c in candidate_results
            if c.profile_status == ProfileStatus.PRESENT
        ]
        # Rejected candidates from Phase 1 + load-time surface as
        # CandidateResult entries with fault_class=INFRA_FAULT so the caller
        # sees the full batch.
        for norm_reject in normalized.reject_candidates.values():
            candidate_results.append(
                _reject_candidate_result(norm_reject, req, device)
            )
        for load_reject in load_reject_map.values():
            candidate_results.append(
                _reject_candidate_result(load_reject, req, device)
            )

        incumbent_anchor = IncumbentAnchor(
            incumbent_artifact_id=req.incumbent_ref.artifact_id,
            shape_results=incumbent_shape_results,
            objective_score=incumbent_score,
        )

        status = _resolve_status(candidate_results)
        return BenchmarkBatchResult(
            status=status,
            run_envelope=_build_run_envelope(req, cfg, device, PodHealth.HEALTHY, 0),
            measurement_context=_build_measurement_context(req, normalized, cfg),
            hygiene=_empty_hygiene(device, None),
            incumbent_anchor=incumbent_anchor,
            candidate_results=candidate_results,
            top_k_profiled=top_k_profiled,
            failure_reason=None,
        )
    finally:
        # Free adapter buffers before destroying the context.
        buffers_to_free = (
            allocated_buffers
            if "allocated_buffers" in locals()
            else list(per_shape_buffers.values())
        )
        seen_buffer_ids: set[int] = set()
        for bufs in buffers_to_free:
            buf_id = id(bufs)
            if buf_id in seen_buffer_ids:
                continue
            seen_buffer_ids.add(buf_id)
            try:
                adapter.free(bufs)
            except Exception as exc:  # noqa: BLE001 — teardown best-effort
                logger.warning(
                    "worker.adapter.free_failed",
                    extra={"error": str(exc)},
                )
        cd.destroy_stream(stream)
        cd.destroy_primary_context(ctx)


def _load_incumbent(
    req: BenchmarkBatchRequest, adapter: OperationAdapter
) -> LoadedCandidate:
    """Return a LoadedCandidate representing the incumbent anchor.

    Phase 1 normalization requires incumbent cubin + launch metadata, so this
    function never fabricates an anchor from a candidate.
    """
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415
    from kerlever.benchmarker.types import FunctionAttribute  # noqa: PLC0415

    if req.incumbent_ref.cubin_uri is None or req.incumbent_ref.launch_spec is None:
        raise RuntimeError("incumbent artifact required")
    cubin = Path(req.incumbent_ref.cubin_uri).read_bytes()
    module = cd.load_module(cubin, None)
    fn = cd.get_function(module, req.incumbent_ref.launch_spec.entrypoint)
    observed = FunctionAttributePolicy()
    if req.incumbent_ref.launch_spec.dynamic_smem_bytes > 0:
        observed_smem = cd.set_function_attribute(
            fn,
            FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
            int(req.incumbent_ref.launch_spec.dynamic_smem_bytes),
        )
        observed = observed.model_copy(
            update={"max_dynamic_shared_memory_size": observed_smem}
        )
    return LoadedCandidate(
        candidate_hash="__incumbent__",
        function=fn,
        launch_args_factory=None,
        adapter_iteration_semantics=adapter.iteration_semantics,
        function_attribute_policy_observed=observed,
        block_dim=req.incumbent_ref.launch_spec.block_dim,
        grid_dim=req.incumbent_ref.launch_spec.grid_dim,
        dynamic_smem_bytes=req.incumbent_ref.launch_spec.dynamic_smem_bytes,
    )


def _incumbent_shape_results(
    batch_meas: BatchMeasurement,
    cfg: BenchmarkerConfig,
    plan_out: CalibratedPlan,
    normalized: NormalizedBatch,
    req: BenchmarkBatchRequest,
    incumbent: LoadedCandidate,
    preflight_telemetry: DeviceTelemetrySnapshot,
    postflight_telemetry: DeviceTelemetrySnapshot,
) -> list[ShapeBenchResult]:
    """Aggregate incumbent anchor samples per shape into :class:`ShapeBenchResult`.

    Implements: REQ-BENCH-032
    Invariant: INV-BENCH-014
    """
    out: list[ShapeBenchResult] = []
    for shape_id, per_shape in batch_meas.per_shape.items():
        samples = list(per_shape.anchor_samples)
        ctx = _shape_artifact_context(
            shape_id=shape_id,
            per_shape=per_shape,
            plan_out=plan_out,
            cfg=cfg,
            normalized=normalized,
            req=req,
            cand=incumbent,
            interleaved=normalized.interleave_enabled,
            telemetry_before=preflight_telemetry,
            telemetry_after=postflight_telemetry,
        )
        compact, _art = _candidate_shape_result(
            shape_id,
            samples,
            cfg.calibration.min_p95_samples,
            ctx,
            thresholds=cfg.thresholds,
        )
        out.append(compact)
    return out


def _shape_artifact_context(
    *,
    shape_id: str,
    per_shape: object,
    plan_out: CalibratedPlan,
    cfg: BenchmarkerConfig,
    normalized: NormalizedBatch,
    req: BenchmarkBatchRequest,
    cand: LoadedCandidate,
    interleaved: bool,
    telemetry_before: DeviceTelemetrySnapshot,
    telemetry_after: DeviceTelemetrySnapshot,
) -> ShapeArtifactContext:
    """Derive a :class:`ShapeArtifactContext` from real runtime state.

    ``telemetry_before`` / ``telemetry_after`` MUST be real snapshots
    captured around the Phase 4 ``execute_batch`` call — Phase 2 preflight
    binds ``telemetry_before``, Phase 4 postflight binds ``telemetry_after``
    (INV-BENCH-014, REQ-BENCH-032). When NVML is unavailable the caller
    logs a warning and passes an ``_empty_telemetry_snapshot()``.

    Implements: REQ-BENCH-032, INV-BENCH-014
    Invariant: INV-BENCH-014
    """
    from kerlever.benchmarker.harness import PerShapeMeasurement  # noqa: PLC0415

    assert isinstance(per_shape, PerShapeMeasurement)
    sp = plan_out.sample_plans.get((cand.candidate_hash, shape_id))
    warmup_count = sp.warmup_count if sp is not None else 0
    iterations_per_sample = sp.iterations_per_sample if sp is not None else 0

    # Anchor drift — only populated when the batch actually interleaved.
    anchor_pre: float | None = None
    anchor_post: float | None = None
    anchor_drift: float | None = None
    if interleaved:
        if per_shape.anchor_pre_samples:
            anchor_pre = stats_mean(per_shape.anchor_pre_samples)
        if per_shape.anchor_post_samples:
            anchor_post = stats_mean(per_shape.anchor_post_samples)
        if anchor_pre is not None and anchor_post is not None:
            anchor_drift = anchor_drift_pct(anchor_pre, anchor_post)

    return ShapeArtifactContext(
        warmup_count=warmup_count,
        iterations_per_sample=iterations_per_sample,
        max_timed_batch_ms=cfg.calibration.max_timed_batch_ms,
        interleave_block_len=per_shape.interleave_block_len,
        interleave_block_order=list(per_shape.block_order),
        anchor_pre_us=anchor_pre,
        anchor_post_us=anchor_post,
        anchor_drift_pct=anchor_drift,
        requested_cache=normalized.requested_cache_policy,
        effective_cache=normalized.effective_cache_policy,
        metric_mode=req.metric_mode,
        artifact_execution_model=req.artifact_execution_model,
        adapter_iteration_semantics=cand.adapter_iteration_semantics,
        function_attribute_policy=cand.function_attribute_policy_observed,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        anchor_every_n_samples=per_shape.anchor_every_n_samples,
    )


def _assemble_candidate_shapes(
    cand: LoadedCandidate,
    batch_meas: BatchMeasurement,
    plan_out: CalibratedPlan,
    cfg: BenchmarkerConfig,
    *,
    normalized: NormalizedBatch,
    req: BenchmarkBatchRequest,
    preflight_telemetry: DeviceTelemetrySnapshot,
    postflight_telemetry: DeviceTelemetrySnapshot,
) -> tuple[
    list[ShapeBenchResult],
    dict[str, str],
    dict[str, MeasurementQualityStatus],
    dict[str, str],
    float | None,
]:
    """Build compact shape results + artifact refs + quality statuses.

    Implements: REQ-BENCH-032, REQ-BENCH-033
    Invariant: INV-BENCH-014
    """
    shape_results: list[ShapeBenchResult] = []
    artifact_refs: dict[str, str] = {}
    quality: dict[str, MeasurementQualityStatus] = {}
    quality_reasons: dict[str, str] = {}
    all_cvs: list[float] = []
    for shape_id, per_shape in batch_meas.per_shape.items():
        samples = per_shape.candidate_samples.get(cand.candidate_hash, [])
        runtime_fault = batch_meas.runtime_faults.get(cand.candidate_hash, {}).get(
            shape_id
        )
        ctx = _shape_artifact_context(
            shape_id=shape_id,
            per_shape=per_shape,
            plan_out=plan_out,
            cfg=cfg,
            normalized=normalized,
            req=req,
            cand=cand,
            interleaved=normalized.interleave_enabled,
            telemetry_before=preflight_telemetry,
            telemetry_after=postflight_telemetry,
        )
        compact, artifact = _candidate_shape_result(
            shape_id,
            samples,
            cfg.calibration.min_p95_samples,
            ctx,
            thresholds=cfg.thresholds,
        )
        status = _shape_quality_status(
            compact,
            artifact,
            plan_out,
            cand.candidate_hash,
            shape_id,
            cfg,
            runtime_fault=runtime_fault,
        )
        if status != artifact.measurement_quality.status or runtime_fault is not None:
            artifact = artifact.model_copy(
                update={
                    "measurement_quality": MeasurementQuality(
                        status=status,
                        reason=runtime_fault,
                    )
                }
            )
        shape_results.append(compact)
        artifact_refs[shape_id] = _write_shape_artifact(
            cfg.artifact.root,
            req.batch_id,
            cand.candidate_hash,
            shape_id,
            artifact,
        )
        quality[shape_id] = status
        if runtime_fault is not None:
            quality_reasons[shape_id] = runtime_fault
        if artifact.cv_pct is not None:
            all_cvs.append(artifact.cv_pct)
    candidate_cv = max(all_cvs) if all_cvs else None
    return shape_results, artifact_refs, quality, quality_reasons, candidate_cv


def _compute_incumbent_cv_pct(batch_meas: BatchMeasurement) -> float | None:
    """Compute the incumbent CV (percent) from anchor samples.

    Spec §6.5 / REQ-BENCH-034: feed a real incumbent CV into
    :func:`scoring.decide_incumbent_comparison` instead of ``None``. We take
    the max shape CV as the conservative aggregate.

    Implements: REQ-BENCH-034
    """
    cvs: list[float] = []
    for per_shape in batch_meas.per_shape.values():
        samples = per_shape.anchor_samples
        if len(samples) < 2:
            continue
        mu = stats_mean(samples)
        sd = stdev(samples)
        cv = cv_pct(mu, sd)
        if cv is not None:
            cvs.append(cv)
    return max(cvs) if cvs else None


def _build_incumbent_envelope(
    *,
    req: BenchmarkBatchRequest,
    normalized: NormalizedBatch,
    device: LeasedDevice,
    incumbent: LoadedCandidate,
) -> MeasurementEnvelope | None:
    """Construct the incumbent's own :class:`MeasurementEnvelope`.

    INV-BENCH-015 forbids copying a candidate's envelope; the incumbent's
    envelope must reflect its own artifact identity and observed attributes.
    If the request omits the incumbent's launch metadata (V1 fallback),
    we return ``None`` and the caller marks candidates ``not_comparable``
    rather than synthesizing an envelope.

    Implements: REQ-BENCH-015
    Invariant: INV-BENCH-015
    """
    if (
        req.incumbent_ref.cubin_uri is None
        or req.incumbent_ref.launch_spec is None
        or req.incumbent_ref.launch_spec_hash is None
        or req.incumbent_ref.source_hash is None
        or req.incumbent_ref.toolchain_hash is None
    ):
        return None
    from kerlever.benchmarker.types import CachePolicyBlock  # noqa: PLC0415

    shape_ids = [s.shape_id for s in req.objective_shape_cases]
    return MeasurementEnvelope(
        run_id=req.run_id,
        round_id=req.round_id,
        batch_id=req.batch_id,
        request_id=req.request_id,
        candidate_hash="__incumbent__",
        artifact_id=req.incumbent_ref.artifact_id,
        source_hash=req.incumbent_ref.source_hash,
        launch_spec_hash=req.incumbent_ref.launch_spec_hash,
        toolchain_hash=req.incumbent_ref.toolchain_hash,
        module_artifact_hash=req.incumbent_ref.artifact_id,
        artifact_execution_model=req.artifact_execution_model,
        problem_spec_hash=next(
            iter(normalized.envelope_per_candidate.values())
        ).problem_spec_hash
        if normalized.envelope_per_candidate
        else "__incumbent__",
        objective_hash=next(
            iter(normalized.envelope_per_candidate.values())
        ).objective_hash
        if normalized.envelope_per_candidate
        else "__incumbent__",
        shape_ids=shape_ids,
        operation_adapter_abi=req.operation_adapter_abi,
        operation_adapter_version=req.operation_adapter_version,
        target_gpu=req.problem_spec.target_gpu,
        gpu_uuid=device.gpu_uuid,
        pci_bus_id=device.pci_bus_id,
        mig_profile=device.mig_profile,
        sm_arch=device.sm_arch,
        driver_version=driver_version(),
        cuda_runtime_version=cuda_runtime_version(),
        metric_mode=req.metric_mode,
        function_attribute_policy_requested=req.function_attribute_policy,
        function_attribute_policy_observed=(
            incumbent.function_attribute_policy_observed
        ),
        warmup_policy=_stub_warmup(),
        repeat_policy=_stub_repeat(),
        cache_policy=CachePolicyBlock(
            requested=normalized.requested_cache_policy,
            effective=normalized.effective_cache_policy,
            reason=normalized.cache_policy_reason,
        ),
        clock_policy=normalized.clock_policy,
        interleave_seed=None,
    )


def _shape_quality_status(
    compact: ShapeBenchResult,
    artifact: ShapeMeasurementArtifact,
    plan_out: CalibratedPlan,
    candidate_hash: str,
    shape_id: str,
    cfg: BenchmarkerConfig,
    runtime_fault: str | None = None,
) -> MeasurementQualityStatus:
    """Classify a shape's quality per spec §6.4 decision table."""
    if runtime_fault is not None:
        return MeasurementQualityStatus.RUNTIME_FAULT
    # No samples → infra_fault.
    if compact.run_count == 0:
        return MeasurementQualityStatus.INFRA_FAULT
    # CV fail.
    if (
        artifact.cv_pct is not None
        and artifact.cv_pct > cfg.thresholds.measurement_cv_fail_pct
    ):
        return MeasurementQualityStatus.UNSTABLE
    # p95/p50 ratio warn.
    warn = False
    if artifact.p95_us is not None and artifact.p50_us > 0:
        ratio = artifact.p95_us / artifact.p50_us
        if ratio > cfg.thresholds.p95_p50_ratio_warn:
            warn = True
    if (
        artifact.cv_pct is not None
        and artifact.cv_pct > cfg.thresholds.measurement_cv_warn_pct
    ):
        warn = True
    if plan_out.calibration_warnings.get((candidate_hash, shape_id)):
        warn = True
    return (
        MeasurementQualityStatus.VALID_WITH_WARNING
        if warn
        else MeasurementQualityStatus.VALID
    )


def _aggregate_anchor_drift(batch_meas: BatchMeasurement) -> float:
    """Aggregate anchor drift across shapes using max-drift semantics."""
    drifts: list[float] = []
    for per_shape in batch_meas.per_shape.values():
        if per_shape.anchor_pre_samples and per_shape.anchor_post_samples:
            drifts.append(
                anchor_drift_pct(
                    stats_mean(per_shape.anchor_pre_samples),
                    stats_mean(per_shape.anchor_post_samples),
                )
            )
    return max(drifts) if drifts else 0.0


def _worst_quality(
    statuses: list[MeasurementQualityStatus],
) -> MeasurementQualityStatus:
    """Return the worst quality status across a candidate's shapes."""
    order = [
        MeasurementQualityStatus.INFRA_FAULT,
        MeasurementQualityStatus.RUNTIME_FAULT,
        MeasurementQualityStatus.UNSTABLE,
        MeasurementQualityStatus.VALID_WITH_WARNING,
        MeasurementQualityStatus.VALID,
    ]
    for candidate_status in order:
        if candidate_status in statuses:
            return candidate_status
    return MeasurementQualityStatus.VALID


def _first_quality_reason(
    reasons: dict[str, str],
    worst: MeasurementQualityStatus,
) -> str | None:
    """Return the first quality reason worth surfacing at candidate level."""
    _ = worst
    if not reasons:
        return None
    shape_id = sorted(reasons.keys())[0]
    return f"{shape_id}:{reasons[shape_id]}"


def _fault_for_candidate(
    cand: LoadedCandidate,
    batch_meas: BatchMeasurement,
    worst: MeasurementQualityStatus,
) -> FaultClass | None:
    """Map a candidate's fault dict to a :class:`FaultClass` (or ``None``)."""
    per_cand = batch_meas.runtime_faults.get(cand.candidate_hash)
    if not per_cand:
        return None
    # Any runtime fault on a healthy pod is candidate_fault.
    representative_message = next(iter(per_cand.values()))
    return attribute(
        exc=RuntimeError(representative_message),
        exit_signal=None,
        exit_code=None,
        pod_health=PodHealth.HEALTHY,
    ) if worst != MeasurementQualityStatus.VALID else None


def _reject_candidate_result(
    reject: object,
    req: BenchmarkBatchRequest,
    device: LeasedDevice,
) -> CandidateResult:
    """Build a CandidateResult entry for a rejected candidate.

    Accepts either :class:`normalize.InfraFault` or the worker-local
    :class:`InfraFaultRecord` — both expose ``candidate_hash``, ``reason``,
    ``fault_class``.
    """
    candidate_hash = getattr(reject, "candidate_hash", "unknown")
    reason = getattr(reject, "reason", "rejected")
    fault_class = getattr(reject, "fault_class", FaultClass.INFRA_FAULT)
    envelope = _stub_envelope_for_reject(req, device, candidate_hash)
    return CandidateResult(
        candidate_hash=candidate_hash,
        envelope=envelope,
        benchmark=None,
        incumbent_comparison=IncumbentComparison.NOT_COMPARABLE,
        measurement_quality=MeasurementQualityStatus.INFRA_FAULT,
        measurement_quality_reason=reason,
        shape_measurement_artifact_refs={},
        profile_status=ProfileStatus.PROFILE_UNAVAILABLE,
        profile_unavailable_reason=None,
        profile_bundles=[],
        raw_profile_metrics_ref=None,
        profile_artifact_refs=[],
        fault_class=fault_class,
        failure_reason=reason,
    )


def _stub_envelope_for_reject(
    req: BenchmarkBatchRequest, device: LeasedDevice, candidate_hash: str
) -> MeasurementEnvelope:
    """Minimal envelope for a rejected candidate so the response is well-formed."""
    from kerlever.benchmarker.types import CachePolicyBlock  # noqa: PLC0415

    return MeasurementEnvelope(
        run_id=req.run_id,
        round_id=req.round_id,
        batch_id=req.batch_id,
        request_id=req.request_id,
        candidate_hash=candidate_hash,
        artifact_id="rejected",
        source_hash="rejected",
        launch_spec_hash="rejected",
        toolchain_hash="rejected",
        module_artifact_hash="rejected",
        artifact_execution_model=req.artifact_execution_model,
        problem_spec_hash="rejected",
        objective_hash="rejected",
        shape_ids=[s.shape_id for s in req.objective_shape_cases],
        operation_adapter_abi=req.operation_adapter_abi,
        operation_adapter_version=req.operation_adapter_version,
        target_gpu=req.problem_spec.target_gpu,
        gpu_uuid=device.gpu_uuid,
        pci_bus_id=device.pci_bus_id,
        mig_profile=device.mig_profile,
        sm_arch=device.sm_arch,
        driver_version=driver_version(),
        cuda_runtime_version=cuda_runtime_version(),
        metric_mode=req.metric_mode,
        function_attribute_policy_requested=req.function_attribute_policy,
        function_attribute_policy_observed=req.function_attribute_policy,
        warmup_policy=_stub_warmup(),
        repeat_policy=_stub_repeat(),
        cache_policy=CachePolicyBlock(
            requested=CachePolicy.WARM_SAME_BUFFERS,
            effective=CachePolicy.WARM_SAME_BUFFERS,
            reason=None,
        ),
        clock_policy=req.clock_policy,
        interleave_seed=None,
    )


def _stub_warmup() -> WarmupPolicy:
    """Return a stub WarmupPolicy for rejected candidates."""
    return WarmupPolicy(min_runs=0, cache_state="untouched")


def _stub_repeat() -> RepeatPolicy:
    """Return a stub RepeatPolicy for rejected candidates."""
    return RepeatPolicy(
        repetitions=0,
        iterations_per_sample=0,
        min_timed_batch_ms=0.0,
        max_timed_batch_ms=0.0,
    )


def _run_profile_phase(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
    candidate_results: list[CandidateResult],
    loaded: list[LoadedCandidate],
    profile_set_hashes: set[str],
    plan_out: CalibratedPlan,
    config_file_path: Path,
    request_file_path: Path,
) -> None:
    """Run NCU for each selected ``(candidate, profile_shape)`` tuple.

    Each selected (candidate, shape) pair spawns its own NCU command line
    whose target is ``python -m kerlever.benchmarker.profile_child ...``
    (spec §6.12, REQ-BENCH-030). Every resulting :class:`ProfileBundle` is
    appended to ``cres.profile_bundles`` — no early break after the first
    shape (spec §6.6 mandates the full cartesian product).

    Implements: REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-030
        (spec §6.6 cartesian product)
    Invariant: INV-BENCH-008, INV-BENCH-009
    """
    if not profile_set_hashes:
        return
    profile_shapes = req.profile_shape_cases or req.objective_shape_cases[:1]
    by_hash = {c.candidate_hash: c for c in loaded}
    for cres in candidate_results:
        if cres.candidate_hash not in profile_set_hashes:
            continue
        cand = by_hash.get(cres.candidate_hash)
        if cand is None:
            continue
        semantics = cand.adapter_iteration_semantics
        replay_mode = (
            ReplayMode.APPLICATION
            if semantics == AdapterIterationSemantics.OVERWRITE_PURE
            else ReplayMode.KERNEL
        )
        first_reason: ProfileUnavailableReason | None = None
        for shape in profile_shapes:
            nvtx_range = build_nvtx_range(
                req.run_id, req.batch_id, cand.candidate_hash, shape.shape_id
            )
            report_out = (
                cfg.artifact.root
                / "ncu"
                / _safe_artifact_token(req.batch_id)
                / (
                    f"{_safe_artifact_token(cand.candidate_hash)}_"
                    f"{_safe_artifact_token(shape.shape_id)}.ncu-rep"
                )
            )
            report_out.parent.mkdir(parents=True, exist_ok=True)
            sp = plan_out.sample_plans.get(
                (cand.candidate_hash, shape.shape_id)
            )
            iterations = sp.iterations_per_sample if sp is not None else 1
            target_cmd = [
                sys.executable,
                "-m",
                "kerlever.benchmarker.profile_child",
                "--config-file",
                str(config_file_path),
                "--request-file",
                str(request_file_path),
                "--candidate-hash",
                cand.candidate_hash,
                "--shape-id",
                shape.shape_id,
                "--nvtx-range",
                nvtx_range,
                "--iterations",
                str(iterations),
                "--device-ordinal",
                str(device.ordinal),
                "--device-uuid",
                device.gpu_uuid,
            ]
            ncu_result = run_ncu(
                cfg=cfg.profiler,
                target_cmd=target_cmd,
                nvtx_range=nvtx_range,
                set_name=cfg.profiler.ncu_profile_set,
                replay_mode=replay_mode,
                report_out=report_out,
                timeout_s=cfg.profiler.profile_timeout_s,
            )
            reason = resolve_unavailable_reason(
                ncu_result,
                HygieneReport(
                    gpu_uuid=device.gpu_uuid,
                    sm_arch=device.sm_arch,
                    compute_mode="UNKNOWN",
                ),
                semantics,
            )
            if ncu_result.report_path is None or reason is not None:
                if first_reason is None:
                    first_reason = reason
                continue
            cres.profile_artifact_refs.append(
                _ncu_artifact_ref(ncu_result.report_path, req.batch_id)
            )
            raw = parse_report(
                cfg.profiler,
                ncu_result.report_path,
                architecture=device.sm_arch,
                profiler_version=ncu_version(cfg.profiler) or "unknown",
            )
            metrics, _provenance = profiler_normalize(
                raw,
                arch=device.sm_arch,
                profiler_version=ncu_version(cfg.profiler) or "unknown",
            )
            raw_ref = _write_raw_metrics_artifact(
                cfg.artifact.root,
                req.batch_id,
                cand.candidate_hash,
                shape.shape_id,
                raw,
            )
            cres.profile_artifact_refs.append(raw_ref)
            cres.profile_bundles.append(
                ProfileBundle(
                    shape_id=shape.shape_id,
                    metrics=metrics,
                    assessment=BottleneckAssessment(
                        tags=[],
                        primary_tag=None,
                        evidence={},
                        rule_trace=[],
                    ),
                )
            )

        if cres.profile_bundles:
            cres.profile_status = ProfileStatus.PRESENT
            cres.profile_unavailable_reason = None
            if cres.raw_profile_metrics_ref is None:
                for ref in cres.profile_artifact_refs:
                    if ref.kind == "raw_metrics_json":
                        cres.raw_profile_metrics_ref = ref.artifact_id
                        break
        else:
            cres.profile_status = ProfileStatus.PROFILE_UNAVAILABLE
            cres.profile_unavailable_reason = first_reason
            cres.raw_profile_metrics_ref = None


def _resolve_status(candidate_results: list[CandidateResult]) -> BatchStatus:
    """Resolve the top-level :class:`BatchStatus` from candidate outcomes."""
    if any(
        c.measurement_quality == MeasurementQualityStatus.INFRA_FAULT
        and c.benchmark is not None
        for c in candidate_results
    ):
        return BatchStatus.PARTIAL
    if any(
        c.measurement_quality == MeasurementQualityStatus.UNSTABLE
        for c in candidate_results
    ):
        return BatchStatus.UNSTABLE
    return BatchStatus.SUCCESS


def _build_infra_error_result(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
    reason: str,
) -> BenchmarkBatchResult:
    """Build a BenchmarkBatchResult for Phase 1 infra-error short-circuit."""
    return BenchmarkBatchResult(
        status=BatchStatus.INFRA_ERROR,
        run_envelope=_build_run_envelope(req, cfg, device, PodHealth.HEALTHY, 0),
        measurement_context=MeasurementContext(
            artifact_execution_model=req.artifact_execution_model,
            metric_mode=req.metric_mode,
            cache_policy_requested=req.cache_policy,
            cache_policy_effective=req.cache_policy,
            clock_policy=req.clock_policy,
            interleave_enabled=False,
            anchor_every_n_samples=None,
            max_interleave_block_len=None,
            noise_floor_pct=cfg.thresholds.noise_floor_pct,
            guard_pct=req.problem_spec.objective.regression_guard_pct,
        ),
        hygiene=_empty_hygiene(device, reason),
        incumbent_anchor=_build_empty_incumbent_anchor(req),
        candidate_results=[],
        top_k_profiled=[],
        failure_reason=reason,
    )


def _best_effort_infra_error(
    args: WorkerArgs,
    device: LeasedDevice,
    cfg: BenchmarkerConfig,
    req: BenchmarkBatchRequest | None,
    reason: str,
) -> None:
    """Flush a best-effort INFRA_ERROR result on uncaught exception."""
    try:
        if req is not None:
            result = _build_infra_error_result(req, cfg, device, reason)
            _write_result(args.result_path, result)
    except Exception as exc:
        logger.warning(
            "worker.best_effort_result.failed",
            extra={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> NoReturn:
    """Worker subprocess entrypoint.

    Contract (design §3.2):

    * ``os._exit(0)`` on success.
    * ``os._exit(1)`` on controlled candidate fault (result flushed).
    * ``os._exit(2)`` on uncaught; best-effort infra_error flushed first.

    The final :func:`os._exit` is the INV-BENCH-012 seam: the Python
    interpreter does not run atexit / CUDA-finalizers, so a poisoned
    context dies with the process rather than leaking to the next batch.

    Implements: REQ-BENCH-025
    Invariant: INV-BENCH-012
    """
    args = _parse_argv(argv)
    cfg = _load_config(args.config_path)
    logging.basicConfig(level=cfg.log_level)
    req: BenchmarkBatchRequest | None = None
    device = LeasedDevice(
        ordinal=args.device_ordinal,
        gpu_uuid=args.device_uuid,
        pci_bus_id="unknown",
        sm_arch="unknown",
        mig_profile=None,
        name=None,
    )
    try:
        req = _read_request(args.request_path)
        try:
            telemetry_init()
        except Exception as exc:
            logger.warning("nvml.init.worker_failed", extra={"error": str(exc)})
        try:
            # Re-fetch device identity from NVML so envelope/hygiene have real sm_arch.
            # Telemetry snapshots for preflight/postflight are taken inside
            # :func:`_run_phases` and threaded into the ShapeMeasurementArtifact
            # per REQ-BENCH-032 / INV-BENCH-014.
            device = _resolve_leased_device(args.device_ordinal, args.device_uuid)
        except Exception as exc:
            logger.warning(
                "worker.device_resolve.failed", extra={"error": str(exc)}
            )
        result = _run_phases(
            req,
            cfg,
            device,
            config_file_path=args.config_path,
            request_file_path=args.request_path,
        )
        _write_result(args.result_path, result)
        telemetry_shutdown()
        os._exit(0)
    except NormalizationError as exc:
        best = _build_infra_error_result(
            req or BenchmarkBatchRequest.model_validate_json(
                args.request_path.read_text()
            ),
            cfg,
            device,
            exc.reason,
        )
        _write_result(args.result_path, best)
        os._exit(1)
    except BaseException as exc:  # noqa: BLE001 — we convert to structured result
        _best_effort_infra_error(
            args, device, cfg, req, f"uncaught:{type(exc).__name__}:{exc}"
        )
        os._exit(2)


def _resolve_leased_device(ordinal: int, uuid_hint: str) -> LeasedDevice:
    """Rebuild the LeasedDevice using NVML from inside the worker."""
    from kerlever.benchmarker.telemetry import info_inventory  # noqa: PLC0415

    for entry in info_inventory():
        if entry.ordinal == ordinal or entry.gpu_uuid == uuid_hint:
            return LeasedDevice(
                ordinal=entry.ordinal,
                gpu_uuid=entry.gpu_uuid,
                pci_bus_id=entry.pci_bus_id,
                sm_arch=entry.sm_arch,
                mig_profile=entry.mig_profile,
                name=entry.name,
            )
    return LeasedDevice(
        ordinal=ordinal,
        gpu_uuid=uuid_hint,
        pci_bus_id="unknown",
        sm_arch="unknown",
    )


def _profiler_name_default() -> ProfilerName:
    """Convenience helper for tests and external callers."""
    return ProfilerName.NCU


def _raw_metric_example(arch: str) -> RawProfileMetric:
    """Shape-of-raw-metric helper for documentation; returns a null-valued sample."""
    return RawProfileMetric(
        metric_name="example",
        value=None,
        unit=None,
        architecture=arch,
        profiler_name=ProfilerName.NCU,
        profiler_version="unknown",
        collection_section=None,
    )


if __name__ == "__main__":  # pragma: no cover - direct subprocess entry
    main(sys.argv[1:])
