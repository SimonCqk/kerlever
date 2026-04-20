"""Benchmarker — FastAPI service factory.

Builds the uvicorn-visible ``app`` via :func:`create_app`. Holds the
:class:`Supervisor`, :class:`LeaseManager`, :class:`DeviceInventory`,
:class:`PodHealthStore`, and cached toolchain identity on ``app.state.env``.

No CUDA / pynvml symbol is imported at module scope — callers that merely
``import kerlever.benchmarker.service`` do not need a GPU driver.

Spec: docs/benchmarker/spec.md §6.10, §SC-BENCH-010
Design: docs/benchmarker/design.md §2.1 service.py, §3.1
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

# Force-register the V1 built-in adapters at service import time so a
# fresh uvicorn worker has them available before lifespan kicks off.
import kerlever.benchmarker.adapter  # noqa: F401
from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.fault import BatchOutcomeSignal, PodHealthStore
from kerlever.benchmarker.lease import (
    DeviceInventory,
    LeaseManager,
    NoCompatibleDeviceError,
    parse_target,
)
from kerlever.benchmarker.profiler import ncu_ready, ncu_version
from kerlever.benchmarker.supervisor import Supervisor, ensure_artifact_root
from kerlever.benchmarker.telemetry import (
    cuda_python_version,
    cuda_runtime_version,
    driver_version,
    info_inventory,
    probe_ready,
    pynvml_version,
)
from kerlever.benchmarker.types import (
    ArtifactExecutionModel,
    BenchmarkBatchRequest,
    BenchmarkBatchResult,
    DeviceInventoryEntry,
    ErrorEnvelope,
    ErrorEnvelopeFieldError,
    FaultClass,
    HealthReport,
    InfoResponse,
    MetricMode,
    PodHealth,
    ToolchainIdentity,
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceEnv:
    """Service-process state attached to ``app.state.env``.

    Mutable only in documented ways: the ``PodHealthStore`` advances state
    after each batch; everything else is immutable after startup.
    """

    config: BenchmarkerConfig
    supervisor: Supervisor
    lease_manager: LeaseManager
    device_inventory: DeviceInventory
    pod_health_store: PodHealthStore
    toolchain: ToolchainIdentity
    inventory_entries: list[DeviceInventoryEntry] = field(default_factory=list)


def _build_toolchain_identity(cfg: BenchmarkerConfig) -> ToolchainIdentity:
    """Snapshot toolchain identity for /info and run envelopes."""
    return ToolchainIdentity(
        driver_version=driver_version(),
        cuda_runtime_version=cuda_runtime_version(),
        cuda_python_version=cuda_python_version(),
        pynvml_version=pynvml_version(),
        ncu_version=ncu_version(cfg.profiler),
    )


def _build_service_env(cfg: BenchmarkerConfig) -> ServiceEnv:
    """Run the lifespan startup sequence (design §8.1).

    Graceful-on-no-GPU: NVML failures are logged but do not crash startup
    so ``/healthz`` can still serve 503 with a structured reason (§7.5).

    Step 9 of design §8.1 — iterate ``cfg.adapter_registry_modules`` and
    import each so its module-level ``register_adapter`` side effect runs
    (REQ-BENCH-028, spec §6.11). V1 built-ins already registered at module
    import time above.
    """
    ensure_artifact_root(cfg)
    for mod_path in cfg.adapter_registry_modules:
        try:
            importlib.import_module(mod_path)
        except Exception as exc:
            logger.error(
                "service.adapter_plugin_import_failed",
                extra={"module": mod_path, "error": str(exc)},
            )
            raise
    entries = info_inventory()
    inventory = DeviceInventory(entries)
    lease_manager = LeaseManager(cfg.lease, inventory)
    pod_health = PodHealthStore()
    toolchain = _build_toolchain_identity(cfg)
    supervisor = Supervisor(cfg)
    return ServiceEnv(
        config=cfg,
        supervisor=supervisor,
        lease_manager=lease_manager,
        device_inventory=inventory,
        pod_health_store=pod_health,
        toolchain=toolchain,
        inventory_entries=entries,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan hook: init env at startup, clean up on shutdown."""
    cfg: BenchmarkerConfig = app.state.config
    env = _build_service_env(cfg)
    app.state.env = env
    logger.info(
        "service.started",
        extra={
            "pod_id": cfg.pod_id,
            "visible_gpus": len(env.inventory_entries),
        },
    )
    try:
        yield
    finally:
        logger.info("service.stopping", extra={"pod_id": cfg.pod_id})


def create_app(config: BenchmarkerConfig | None = None) -> FastAPI:
    """Build the FastAPI app and register handlers.

    Implements: REQ-BENCH-023
    Invariant: INV-BENCH-001 (no import of kerlever.protocols)
    """
    cfg = config or BenchmarkerConfig.from_env()
    app = FastAPI(lifespan=_lifespan)
    app.state.config = cfg

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        envelope = ErrorEnvelope(
            code="bad_request",
            detail="request validation failed",
            field_errors=[
                ErrorEnvelopeFieldError(
                    loc=[str(p) for p in err.get("loc", [])],
                    msg=str(err.get("msg", "")),
                    type=str(err.get("type", "")),
                )
                for err in exc.errors()
            ],
            request_id=request.headers.get("x-request-id"),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=envelope.model_dump(),
        )

    @app.exception_handler(Exception)
    async def _on_internal_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("service.internal_error", exc_info=exc)
        envelope = ErrorEnvelope(
            code="internal_server_error",
            detail="see service logs",
            request_id=request.headers.get("x-request-id"),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=envelope.model_dump(),
        )

    @app.post("/benchmark", response_model=BenchmarkBatchResult)
    async def handle_benchmark(
        req: BenchmarkBatchRequest, request: Request
    ) -> BenchmarkBatchResult:
        """Run one benchmark batch.

        Implements: REQ-BENCH-023
        """
        env: ServiceEnv = request.app.state.env
        if env.pod_health_store.current() is PodHealth.QUARANTINED:
            # Spec §6.2: quarantined pods short-circuit without touching GPU.
            from kerlever.benchmarker.types import (  # noqa: PLC0415
                BatchStatus,
                HygieneReport,
                IncumbentAnchor,
                MeasurementContext,
                RunEnvelope,
                VisibleGpu,
            )

            hygiene = HygieneReport(
                gpu_uuid="",
                sm_arch="",
                compute_mode="UNKNOWN",
                reason_on_fail="pod_quarantined",
            )
            run_envelope = RunEnvelope(
                run_id=req.run_id,
                round_id=req.round_id,
                batch_id=req.batch_id,
                request_id=req.request_id,
                pod_id=env.config.pod_id,
                pod_health=PodHealth.QUARANTINED,
                ambiguous_failure_count=env.pod_health_store.ambiguous_count(),
                toolchain=env.toolchain,
                visible_gpu=VisibleGpu(
                    gpu_uuid="",
                    pci_bus_id="",
                    sm_arch="",
                ),
            )
            return BenchmarkBatchResult(
                status=BatchStatus.INFRA_ERROR,
                run_envelope=run_envelope,
                measurement_context=MeasurementContext(
                    artifact_execution_model=req.artifact_execution_model,
                    metric_mode=req.metric_mode,
                    cache_policy_requested=req.cache_policy,
                    cache_policy_effective=req.cache_policy,
                    clock_policy=req.clock_policy,
                    interleave_enabled=False,
                    noise_floor_pct=env.config.thresholds.noise_floor_pct,
                    guard_pct=req.problem_spec.objective.regression_guard_pct,
                ),
                hygiene=hygiene,
                incumbent_anchor=IncumbentAnchor(
                    incumbent_artifact_id=req.incumbent_ref.artifact_id,
                    shape_results=[],
                    objective_score=req.incumbent_ref.objective_score,
                ),
                candidate_results=[],
                top_k_profiled=[],
                failure_reason="pod_quarantined",
            )

        try:
            target = parse_target(
                req.problem_spec.target_gpu, _resolve_sm_arch(req)
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            async with env.lease_manager.acquire(target) as device:
                finalized = await env.supervisor.run_batch(
                    req,
                    device,
                    env.pod_health_store.current(),
                    env.pod_health_store.ambiguous_count(),
                )
        except NoCompatibleDeviceError as exc:
            logger.warning(
                "service.no_compatible_device",
                extra={"target": target, "reason": str(exc)},
            )
            return _infra_error_response(env, req, "arch_mismatch")

        new_pod_health = env.pod_health_store.update(
            BatchOutcomeSignal(
                had_ambiguous_fault=_has_fault_class(
                    finalized.result, FaultClass.AMBIGUOUS_FAULT
                ),
                had_infra_fault=_has_fault_class(
                    finalized.result, FaultClass.INFRA_FAULT
                ),
                had_candidate_fault=_has_fault_class(
                    finalized.result, FaultClass.CANDIDATE_FAULT
                ),
                worker_timed_out=False,
            ),
            ambiguous_failure_limit=env.config.supervisor.ambiguous_failure_limit,
        )
        # Reflect advanced pod health back into the run envelope.
        result = finalized.result
        if result.run_envelope.pod_health != new_pod_health:
            amb = env.pod_health_store.ambiguous_count()
            result = result.model_copy(
                update={
                    "run_envelope": result.run_envelope.model_copy(
                        update={
                            "pod_health": new_pod_health,
                            "ambiguous_failure_count": amb,
                        }
                    ),
                }
            )
        return result

    @app.get("/healthz")
    async def handle_healthz(request: Request) -> JSONResponse:
        """Return the service readiness report.

        Implements: REQ-BENCH-023
        """
        env: ServiceEnv | None = getattr(request.app.state, "env", None)
        if env is None:
            report = HealthReport(
                status="not_ready",
                reason="service_initializing",
                missing=["env"],
            )
            return JSONResponse(
                status_code=503,
                content=report.model_dump(),
            )
        missing: list[str] = []
        reason: str | None = None
        if not probe_ready():
            missing.append("gpu")
            reason = "no_gpu_visible"
        if not ncu_ready(env.config.profiler):
            missing.append("ncu")
            reason = reason or "ncu_missing"
        if env.pod_health_store.current() is PodHealth.QUARANTINED:
            reason = "pod_quarantined"
        if missing or reason == "pod_quarantined":
            report = HealthReport(
                status="not_ready",
                toolchain=env.toolchain,
                gpus=env.inventory_entries,
                pod_health=env.pod_health_store.current(),
                missing=missing,
                reason=reason,
            )
            return JSONResponse(
                status_code=503, content=report.model_dump()
            )
        report = HealthReport(
            status="ready",
            toolchain=env.toolchain,
            gpus=env.inventory_entries,
            pod_health=env.pod_health_store.current(),
        )
        return JSONResponse(status_code=200, content=report.model_dump())

    @app.get("/info", response_model=InfoResponse)
    async def handle_info(request: Request) -> InfoResponse:
        """Return service identity, toolchain, and supported adapters.

        Implements: REQ-BENCH-023
        """
        env: ServiceEnv = request.app.state.env
        return InfoResponse(
            service_version=env.config.service_version,
            build_hash=env.config.build_hash,
            toolchain=env.toolchain,
            gpus=env.inventory_entries,
            default_metric_mode=MetricMode.DEVICE_KERNEL_US,
            artifact_execution_model=ArtifactExecutionModel.COMMON_HARNESS_CUBIN,
            supported_adapter_abis=list(env.config.supported_adapter_abis),
            thresholds=_threshold_map(env.config),
        )

    return app


def _threshold_map(cfg: BenchmarkerConfig) -> dict[str, float]:
    """Export the numeric thresholds block for ``GET /info``."""
    t = cfg.thresholds
    return {
        "noise_floor_pct": t.noise_floor_pct,
        "measurement_cv_warn_pct": t.measurement_cv_warn_pct,
        "measurement_cv_fail_pct": t.measurement_cv_fail_pct,
        "p95_p50_ratio_warn": t.p95_p50_ratio_warn,
        "anchor_drift_warn_pct": t.anchor_drift_warn_pct,
        "anchor_drift_fail_pct": t.anchor_drift_fail_pct,
        "thermal_steady_state_limit_c": t.thermal_steady_state_limit_c,
    }


def _resolve_sm_arch(req: BenchmarkBatchRequest) -> str:
    """Derive ``sm_arch`` from the request.

    The ``ProblemSpec`` does not carry ``sm_arch`` directly; we accept a
    free-form ``target_gpu`` like ``"H100-SXM5"`` and let the deployment
    map it via a future policy. For V1, if the request includes candidates
    that were compiled against a specific arch, we read the arch from the
    first candidate's :attr:`CandidateArtifactRef.toolchain_hash` is
    insufficient; instead we accept any arch and delegate matching to the
    lease manager via the inventory's reported ``sm_arch``. We return
    the literal ``target_gpu`` string so the inventory match uses it.
    """
    # Deliberately return the target string; lease.find_compatible checks
    # inventory sm_arch against this value. For a real deployment this
    # should be replaced with a configured SKU→sm_arch map.
    return req.problem_spec.target_gpu


def _has_fault_class(
    result: BenchmarkBatchResult, fault_class: FaultClass
) -> bool:
    """Return whether any candidate result carries the given fault class."""
    return any(c.fault_class == fault_class for c in result.candidate_results)


def _infra_error_response(
    env: ServiceEnv,
    req: BenchmarkBatchRequest,
    reason: str,
) -> BenchmarkBatchResult:
    """Build an infra_error response without touching the GPU."""
    from kerlever.benchmarker.types import (  # noqa: PLC0415
        BatchStatus,
        HygieneReport,
        IncumbentAnchor,
        MeasurementContext,
        RunEnvelope,
        VisibleGpu,
    )

    hygiene = HygieneReport(
        gpu_uuid="",
        sm_arch="",
        compute_mode="UNKNOWN",
        reason_on_fail=reason,
    )
    return BenchmarkBatchResult(
        status=BatchStatus.INFRA_ERROR,
        run_envelope=RunEnvelope(
            run_id=req.run_id,
            round_id=req.round_id,
            batch_id=req.batch_id,
            request_id=req.request_id,
            pod_id=env.config.pod_id,
            pod_health=env.pod_health_store.current(),
            ambiguous_failure_count=env.pod_health_store.ambiguous_count(),
            toolchain=env.toolchain,
            visible_gpu=VisibleGpu(gpu_uuid="", pci_bus_id="", sm_arch=""),
        ),
        measurement_context=MeasurementContext(
            artifact_execution_model=req.artifact_execution_model,
            metric_mode=req.metric_mode,
            cache_policy_requested=req.cache_policy,
            cache_policy_effective=req.cache_policy,
            clock_policy=req.clock_policy,
            interleave_enabled=False,
            noise_floor_pct=env.config.thresholds.noise_floor_pct,
            guard_pct=req.problem_spec.objective.regression_guard_pct,
        ),
        hygiene=hygiene,
        incumbent_anchor=IncumbentAnchor(
            incumbent_artifact_id=req.incumbent_ref.artifact_id,
            shape_results=[],
            objective_score=req.incumbent_ref.objective_score,
        ),
        candidate_results=[],
        top_k_profiled=[],
        failure_reason=reason,
    )


__all__ = ["ServiceEnv", "create_app"]
