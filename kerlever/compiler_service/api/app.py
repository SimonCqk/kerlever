"""FastAPI application factory and lifespan hook.

Spec: docs/compiler-service/spec.md §6.12
Design: docs/compiler-service/design.md §5, §12
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from kerlever.compiler_service.adapters import default_registry
from kerlever.compiler_service.api.schemas import (
    HealthzResponse,
    PodStatusResponse,
)
from kerlever.compiler_service.artifact_store import (
    ArtifactStore,
    PinnedRoots,
    RetentionPolicy,
    disk_watermark_snapshot,
)
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.idempotency import IdempotencyRegistry
from kerlever.compiler_service.pod_health import PodHealthTracker
from kerlever.compiler_service.resource_extraction import StaticResourceExtractor
from kerlever.compiler_service.sanitizer import (
    ComputeSanitizerRunner,
    SanitizerPolicy,
)
from kerlever.compiler_service.service import CompilerService, CompilerServiceDeps
from kerlever.compiler_service.static_resource_model import StaticResourceModel
from kerlever.compiler_service.toolchain import (
    CuobjdumpRunner,
    DriverApiAttributes,
    NvccRunner,
    PtxasParser,
    ToolchainProbe,
)
from kerlever.compiler_service.types import (
    ArtifactKind,
    CompileRequest,
    CompileResult,
    PinRole,
    PodHealth,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _iter_file(path: Path, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    """Yield ``path`` in fixed-size binary chunks, closing the FD on exit.

    Starlette's ``StreamingResponse`` iterates whatever body iterable it
    is handed but does **not** close a file handle passed to it directly;
    worse, iterating an ``open("rb")`` object chunks by newline — each
    yield is a line-terminated slice, which is catastrophic for binary
    artifacts (cubin/SASS) that contain many ``\\n`` bytes. This helper
    owns the file via a ``with`` block so the FD is guaranteed to close
    when the generator is exhausted or GC'd, and it reads in 64 KiB
    blocks regardless of the bytes inside.

    The chunk size is intentionally fixed at 64 KiB — a sensible default
    for multi-MB artifacts; configurability is explicitly out of scope.
    """
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Build the FastAPI app + every singleton needed by the request path.

    Deferred import of ``fastapi`` keeps the core package importable without
    the ``[service]`` extra installed.
    """
    try:
        from fastapi import (  # noqa: F401 — optional dep
            FastAPI,
            HTTPException,
            Request,
        )
        from fastapi.responses import JSONResponse, StreamingResponse  # noqa: F401
    except ImportError as exc:  # pragma: no cover — optional extra
        raise RuntimeError(
            "fastapi is required for create_app; install `kerlever[service]`"
        ) from exc

    resolved = config or ServiceConfig.from_env()
    app = FastAPI(
        title="kerlever-compiler-service",
        lifespan=_lifespan_factory(resolved),
    )

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz(request: Request) -> HealthzResponse | JSONResponse:  # noqa: D401
        """Run the toolchain probe + report current pod health."""
        probe_result = ToolchainProbe(resolved).run()
        deps = getattr(request.app.state, "deps", None)
        pod_health = (
            deps.pod_health.snapshot() if deps is not None else PodHealth.HEALTHY
        )
        toolchain = deps.toolchain if deps is not None else None
        if not probe_result.ok:
            body = HealthzResponse(
                status="not_ready",
                ok=False,
                missing=probe_result.missing,
                notes=probe_result.notes,
                toolchain=toolchain,
                pod_health=pod_health,
            ).model_dump()
            return JSONResponse(status_code=503, content=body)
        return HealthzResponse(
            status="ready",
            ok=True,
            missing=[],
            notes=probe_result.notes,
            toolchain=toolchain,
            pod_health=pod_health,
        )

    @app.post("/v1/compile", response_model=CompileResult)
    async def compile_(
        request: Request, body: CompileRequest
    ) -> CompileResult | JSONResponse:  # noqa: D401
        """Compile a candidate and return a structured ``CompileResult``."""
        service: CompilerService = request.app.state.compiler_service
        result = await service.compile(body)
        status_code = (
            503 if result.run_envelope.pod_health is PodHealth.QUARANTINED else 200
        )
        return JSONResponse(
            status_code=status_code, content=result.model_dump(mode="json")
        )

    @app.get("/v1/artifacts/{artifact_id}")
    async def get_artifact(request: Request, artifact_id: str) -> StreamingResponse:  # noqa: D401
        """Stream an artifact's bytes by id (404 on unknown/GC'd).

        Uses ``StreamingResponse`` so multi-MB SASS/cubin artifacts do not
        load into memory at once; ``_iter_file`` chunks by fixed 64 KiB
        blocks (not by newline) and closes the underlying file descriptor
        when the iterator is exhausted — preventing FD leaks on a
        long-running service.
        """
        store: ArtifactStore = request.app.state.deps.artifact_store
        path = await store.path_of(artifact_id)
        if path is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        return StreamingResponse(
            _iter_file(path),
            media_type="application/octet-stream",
        )

    @app.get("/v1/pod-status", response_model=PodStatusResponse)
    async def pod_status(request: Request) -> PodStatusResponse:  # noqa: D401
        """Return pod health + artifact-store disk snapshot."""
        deps: CompilerServiceDeps = request.app.state.deps
        stats = disk_watermark_snapshot(
            deps.artifact_store,
            deps.config.kerlever_artifact_root,
            deps.config.artifact_disk_high_watermark_pct,
        )
        return PodStatusResponse(
            pod_health=deps.pod_health.snapshot(),
            ambiguous_failure_count=deps.pod_health.ambiguous_failure_count,
            toolchain=deps.toolchain,
            disk_used_pct=stats.disk_used_pct,
            artifact_count=stats.count,
            pinned_count=stats.pinned_count,
        )

    return app


def _lifespan_factory(
    config: ServiceConfig,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Return the lifespan context manager bound to ``config``."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        deps = await build_deps(config)
        service = CompilerService(deps)
        app.state.compiler_service = service
        app.state.deps = deps
        yield
        # V1: no persistent state; nothing to flush on shutdown.

    return _lifespan


async def build_deps(config: ServiceConfig) -> CompilerServiceDeps:
    """Build the singleton bag used by both ``create_app`` and the CLI."""
    probe = ToolchainProbe(config)
    probe_result = probe.run()
    if not probe_result.ok:
        sys.stderr.write(probe_result.as_error_json() + "\n")
        raise SystemExit(1)
    toolchain = probe.snapshot(probe_result)

    retention = RetentionPolicy.default()
    pinned = PinnedRoots(roots=config.artifact_pin_roots)
    artifact_store = ArtifactStore(
        root=config.kerlever_artifact_root,
        retention=retention,
        high_watermark_pct=config.artifact_disk_high_watermark_pct,
        pinned_roots=pinned,
    )
    idempotency = IdempotencyRegistry(ttl=config.idempotency_ttl)

    adapter_registry = default_registry()
    compile_semaphore = asyncio.Semaphore(config.cpu_compile_concurrency)
    # V1: single GPU assumed — device 0. A future revision can enumerate
    # multiple devices via the driver API.
    gpu_semaphores = {0: asyncio.Semaphore(config.gpu_run_concurrency)}

    nvcc = NvccRunner(
        probe_result.nvcc_path or config.nvcc_path,
        config,
    )
    cuobjdump = CuobjdumpRunner(
        probe_result.cuobjdump_path or config.cuobjdump_path,
        config,
    )

    # Compile the pod-health probe binary at startup (spec §6.8).
    # INV-CS-012 extension: a pod that cannot compile its own probe is not
    # ready — exit non-zero so the container becomes visibly unhealthy.
    probe_executable_path = await _compile_probe_or_exit(
        config=config,
        nvcc=nvcc,
        artifact_store=artifact_store,
    )

    pod_health = PodHealthTracker(
        ambiguous_limit=config.ambiguous_failure_limit,
        probe_source_path=config.pod_health_probe_path,
        probe_executable_path=probe_executable_path,
    )
    sanitizer_runner = ComputeSanitizerRunner(
        probe_result.sanitizer_path or config.sanitizer_path,
        config,
    )
    sanitizer_policy = SanitizerPolicy(config)
    driver_api = DriverApiAttributes.try_load()
    resource_extractor = StaticResourceExtractor(
        driver_api=driver_api,
        ptxas_parser=PtxasParser(),
        arch_model=StaticResourceModel.default(),
    )

    return CompilerServiceDeps(
        config=config,
        toolchain=toolchain,
        artifact_store=artifact_store,
        pod_health=pod_health,
        idempotency=idempotency,
        adapter_registry=adapter_registry,
        gpu_semaphores=gpu_semaphores,
        compile_semaphore=compile_semaphore,
        nvcc=nvcc,
        cuobjdump=cuobjdump,
        sanitizer_runner=sanitizer_runner,
        sanitizer_policy=sanitizer_policy,
        resource_extractor=resource_extractor,
        pod_id=config.pod_id,
    )


async def _compile_probe_or_exit(
    config: ServiceConfig,
    nvcc: NvccRunner,
    artifact_store: ArtifactStore,
) -> Path:
    """Compile the known-good probe kernel once at startup (spec §6.8 / §10.3).

    The probe is compiled with ``config.probe_target_arch`` and the output
    lives under ``<artifact_root>/probe/probe.out``. On success the binary
    bytes are also registered in the ``ArtifactStore`` under
    ``ArtifactKind.PROBE_BINARY`` and pinned via ``PinRole.PROBE_KERNEL`` so
    GC never drops it (spec §10.3 steps 2–3). On failure the function
    writes a structured payload to ``stderr`` and exits with code 1 —
    INV-CS-012 extension: a pod that cannot compile its own probe is not
    ready.
    """
    import json

    probe_dir = config.kerlever_artifact_root / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_output: Path = probe_dir / "probe.out"

    try:
        result = await nvcc.compile(
            source=config.pod_health_probe_path,
            output=probe_output,
            target_arch=config.probe_target_arch,
            timeout_s=config.compile_timeout_s,
            max_log_bytes=config.max_log_bytes,
        )
    except FileNotFoundError as exc:
        sys.stderr.write(
            json.dumps(
                {
                    "ok": False,
                    "probe_compile": "nvcc_not_found",
                    "error": str(exc),
                },
                sort_keys=True,
            )
            + "\n"
        )
        raise SystemExit(1) from exc

    if result.timed_out or result.returncode != 0 or not probe_output.exists():
        payload = {
            "ok": False,
            "probe_compile": "failed",
            "timed_out": result.timed_out,
            "returncode": result.returncode,
            "stdout": result.stdout_excerpt,
            "stderr": result.stderr_excerpt,
            "command": result.command,
            "source": str(config.pod_health_probe_path),
            "target_arch": config.probe_target_arch,
        }
        sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")
        raise SystemExit(1)

    # Register binary in the artifact store + pin under PROBE_KERNEL so GC
    # never evicts it (spec §10.3 steps 2–3).
    try:
        artifact_id = await artifact_store.write(
            kind=ArtifactKind.PROBE_BINARY,
            data=probe_output.read_bytes(),
            run_id="pod",
            candidate_hash="probe",
        )
        artifact_store.pin(PinRole.PROBE_KERNEL, artifact_id)
    except Exception as exc:  # noqa: BLE001 — best-effort pin
        logger.warning("probe_binary_store_failed", extra={"error": str(exc)})

    return probe_output
