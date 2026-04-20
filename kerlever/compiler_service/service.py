"""``CompilerService`` — top-level orchestrator running the 5 phases.

Spec: docs/compiler-service/spec.md §6
Design: docs/compiler-service/design.md §4.2
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kerlever.compiler_service.adapters import AdapterRegistry
from kerlever.compiler_service.artifact_store import ArtifactStore
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.envelope import PhaseTimer
from kerlever.compiler_service.idempotency import IdempotencyRegistry
from kerlever.compiler_service.phases.phase1_request import Phase1RequestNormalizer
from kerlever.compiler_service.phases.phase2_harness import Phase2HarnessAssembler
from kerlever.compiler_service.phases.phase3_compile import Phase3Compiler
from kerlever.compiler_service.phases.phase4_correctness import (
    Phase4CorrectnessValidator,
)
from kerlever.compiler_service.phases.phase5_output import Phase5ResultAssembler
from kerlever.compiler_service.pod_health import PodHealthTracker
from kerlever.compiler_service.resource_extraction import StaticResourceExtractor
from kerlever.compiler_service.sanitizer import (
    ComputeSanitizerRunner,
    SanitizerPolicy,
)
from kerlever.compiler_service.toolchain import (
    CuobjdumpRunner,
    NvccRunner,
)
from kerlever.compiler_service.types import (
    CompileRequest,
    CompileResult,
    IdempotencyState,
    ToolchainInfo,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompilerServiceDeps:
    """Frozen bag of singletons consumed by the service (design §5.1)."""

    config: ServiceConfig
    toolchain: ToolchainInfo
    artifact_store: ArtifactStore
    pod_health: PodHealthTracker
    idempotency: IdempotencyRegistry
    adapter_registry: AdapterRegistry
    gpu_semaphores: Mapping[int, asyncio.Semaphore]
    compile_semaphore: asyncio.Semaphore
    nvcc: NvccRunner
    cuobjdump: CuobjdumpRunner
    sanitizer_runner: ComputeSanitizerRunner
    sanitizer_policy: SanitizerPolicy
    resource_extractor: StaticResourceExtractor
    pod_id: str


class CompilerService:
    """Runs the 5 phases in strict order; routes every failure to Phase 5.

    Implements: REQ-CS-001..REQ-CS-013 via the bound phases.
    Invariant: INV-CS-015 (Phase 5 is the only ``CompileResult`` ctor).
    """

    def __init__(self, deps: CompilerServiceDeps) -> None:
        self._deps = deps
        self._phase1 = Phase1RequestNormalizer(
            config=deps.config,
            toolchain=deps.toolchain,
            pod_health=deps.pod_health,
            idempotency=deps.idempotency,
            adapter_registry=deps.adapter_registry,
            pod_id=deps.pod_id,
        )
        self._phase2 = Phase2HarnessAssembler(
            config=deps.config,
            artifact_store=deps.artifact_store,
        )
        self._phase3 = Phase3Compiler(
            config=deps.config,
            artifact_store=deps.artifact_store,
            nvcc=deps.nvcc,
            cuobjdump=deps.cuobjdump,
            resource_extractor=deps.resource_extractor,
            compile_semaphore=deps.compile_semaphore,
        )
        self._phase4 = Phase4CorrectnessValidator(
            config=deps.config,
            artifact_store=deps.artifact_store,
            sanitizer_runner=deps.sanitizer_runner,
            sanitizer_policy=deps.sanitizer_policy,
            pod_health=deps.pod_health,
            gpu_semaphores=deps.gpu_semaphores,
        )

    async def compile(self, request: CompileRequest) -> CompileResult:
        """Run Phase 1→5 for a single request.

        Every failure short-circuits forward to ``Phase5ResultAssembler``;
        ``CompileResult`` is constructed in exactly one place (INV-CS-015).

        The Phase 2 workspace ``tempfile.mkdtemp`` directory (holding the
        rendered reference/candidate sources, compiled executables, cubin
        / PTX / SASS dumps, and shape fixtures) is scratch only: every
        artifact has already been copied into the ``ArtifactStore`` by
        the time Phase 5 returns, so this ``finally`` block unconditionally
        removes the workspace to prevent a slow disk leak (design §9.1).
        """
        timer = PhaseTimer()
        phase5 = Phase5ResultAssembler(
            toolchain=self._deps.toolchain,
            pod_health=self._deps.pod_health,
            idempotency=self._deps.idempotency,
            timer=timer,
        )
        id_lock = await self._deps.idempotency.acquire_id_lock(request.request_id)
        # Hoisted into the outer scope so the ``finally`` block can reach
        # the workspace path even if Phase 3/4/5 raise.
        workspace_to_cleanup: Path | None = None

        try:
            async with id_lock:
                phase1_out = await self._phase1.run(request, timer)

                # Replay of a completed entry — refresh envelope and return.
                if phase1_out.idempotency_state is IdempotencyState.REUSED_COMPLETED:
                    return await phase5.from_reused_completed(phase1_out)

                if phase1_out.short_circuit is not None:
                    return await phase5.from_short_circuit(
                        request, phase1_out, phase1_out.short_circuit
                    )

                phase2_out = await self._phase2.run(phase1_out, timer)
                if phase2_out.harness is not None:
                    workspace_to_cleanup = phase2_out.harness.workspace
                if phase2_out.short_circuit is not None:
                    return await phase5.from_short_circuit(
                        request, phase1_out, phase2_out.short_circuit
                    )

                phase3_out = await self._phase3.run(phase2_out, timer)
                if phase3_out.short_circuit is not None:
                    return await phase5.from_short_circuit(
                        request, phase1_out, phase3_out.short_circuit
                    )

                phase4_out = await self._phase4.run(phase3_out, timer)
                return await phase5.assemble(
                    request=request,
                    phase1=phase1_out,
                    phase3=phase3_out,
                    phase4=phase4_out,
                )
        finally:
            self._deps.idempotency.finalize_if_pending(request.request_id)
            try:
                await self._deps.artifact_store.gc_cheap_pass(
                    self._deps.idempotency.referenced_artifact_ids()
                )
            except Exception as exc:  # noqa: BLE001 — best-effort GC
                logger.warning("gc_cheap_pass_failed", extra={"error": str(exc)})
            # Purge expired idempotency entries so the registry doesn't
            # grow without bound (spec §6.10). Best-effort; a walk over an
            # in-memory dict, cheap per-request.
            try:
                self._deps.idempotency.purge_expired()
            except Exception as exc:  # noqa: BLE001 — best-effort purge
                logger.warning("idempotency_purge_failed", extra={"error": str(exc)})
            # Drop the Phase 2 workspace (design §9.1 artifact-store
            # lifecycle): the artifact store owns canonical copies of every
            # file we care about by this point — reference.cu,
            # candidate.cu, .out executables, .cubin, .ptx, .sass, and
            # shape fixtures. Ordered AFTER the idempotency + GC passes so
            # their skip-set computations see a consistent view of the
            # store. Best-effort; never raise.
            if workspace_to_cleanup is not None:
                try:
                    shutil.rmtree(workspace_to_cleanup, ignore_errors=True)
                except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                    logger.warning(
                        "workspace_cleanup_failed",
                        extra={
                            "error": str(exc),
                            "workspace": str(workspace_to_cleanup),
                        },
                    )
