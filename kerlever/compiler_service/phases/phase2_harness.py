"""Phase 2 — harness assembly.

Renders two disjoint harness source files (reference + candidate) per
INV-CS-001. Does not invoke ``nvcc``.

The ``tempfile.mkdtemp`` workspace created here is scratch-only: every
file Phase 3/4 produces inside it (reference.cu, candidate.cu, .out
executables, .cubin, .ptx, .sass, and shape fixtures) is copied into
the ``ArtifactStore`` before Phase 5 returns. The canonical audit trail
lives in the artifact store; the workspace is dropped by
``CompilerService.compile``'s ``finally`` block unconditionally (see
design.md §9.1 "artifact store lifecycle").

Spec: docs/compiler-service/spec.md §6.3
Design: docs/compiler-service/design.md §4.3, §9.1
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from kerlever.compiler_service.adapters.base import OperationAdapter
from kerlever.compiler_service.artifact_store import ArtifactStore
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.envelope import PhaseTimer
from kerlever.compiler_service.phases import PhaseShortCircuit
from kerlever.compiler_service.phases.phase1_request import Phase1Output
from kerlever.compiler_service.types import (
    ArtifactKind,
    CandidateFaultKind,
    CandidateRole,
    CompileResultStatus,
    FailureDetail,
    PhaseName,
)


@dataclass(frozen=True)
class HarnessArtifacts:
    """Filesystem paths and adapter reference for the rendered harnesses."""

    reference_source_path: Path
    candidate_source_path: Path
    workspace: Path
    adapter: OperationAdapter
    reference_source_artifact_id: str
    candidate_source_artifact_id: str


@dataclass(frozen=True)
class Phase2Output:
    """Happy-path payload plus optional short-circuit packet."""

    phase1: Phase1Output
    harness: HarnessArtifacts | None
    short_circuit: PhaseShortCircuit | None = None


class Phase2HarnessAssembler:
    """Renders two disjoint harness sources (reference + candidate).

    Invariant: INV-CS-001 (never merge reference + candidate in one TU)
    """

    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
    ) -> None:
        self._config = config
        self._artifact_store = artifact_store

    async def run(self, phase1: Phase1Output, timer: PhaseTimer) -> Phase2Output:
        """Produce the two harness sources or short-circuit."""
        phase_start = time.monotonic()
        try:
            request = phase1.request
            adapter = phase1.adapter
            resolved_spec = phase1.resolved_execution_spec

            # Bound the inbound source BEFORE rendering anything.
            if len(request.source_code.encode("utf-8")) > self._config.max_source_bytes:
                return self._short_circuit_source_too_large(phase1, "candidate")
            if (
                len(request.reference_source.encode("utf-8"))
                > self._config.max_source_bytes
            ):
                return self._short_circuit_source_too_large(phase1, "reference")

            # Render both harnesses.
            reference_src = adapter.build_harness_source(
                resolved_spec,
                request.problem_spec,
                CandidateRole.REFERENCE,
                request.reference_source,
            )
            candidate_src = adapter.build_harness_source(
                resolved_spec,
                request.problem_spec,
                CandidateRole.CANDIDATE,
                request.source_code,
            )

            if (
                len(reference_src.encode("utf-8")) > self._config.max_source_bytes
                or len(candidate_src.encode("utf-8")) > self._config.max_source_bytes
            ):
                return self._short_circuit_source_too_large(phase1, "harness")

            workspace = Path(
                tempfile.mkdtemp(prefix=f"kerlever-{request.request_id}-", dir=None)
            )
            reference_path = workspace / "reference.cu"
            candidate_path = workspace / "candidate.cu"
            reference_path.write_text(reference_src, encoding="utf-8")
            candidate_path.write_text(candidate_src, encoding="utf-8")

            reference_artifact_id = await self._artifact_store.write(
                kind=ArtifactKind.SOURCE_REFERENCE,
                data=reference_src.encode("utf-8"),
                run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )
            candidate_artifact_id = await self._artifact_store.write(
                kind=ArtifactKind.SOURCE_CANDIDATE,
                data=candidate_src.encode("utf-8"),
                run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )

            return Phase2Output(
                phase1=phase1,
                harness=HarnessArtifacts(
                    reference_source_path=reference_path,
                    candidate_source_path=candidate_path,
                    workspace=workspace,
                    adapter=adapter,
                    reference_source_artifact_id=reference_artifact_id,
                    candidate_source_artifact_id=candidate_artifact_id,
                ),
            )
        finally:
            timer.record(PhaseName.HARNESS_ASSEMBLY, phase_start)

    def _short_circuit_source_too_large(
        self, phase1: Phase1Output, kind: str
    ) -> Phase2Output:
        """Short-circuit when the harness (or source) exceeds byte caps."""
        failure = FailureDetail(
            phase=PhaseName.HARNESS_ASSEMBLY,
            reason=f"source_too_large_{kind}",
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.HARNESS_ASSEMBLY,
            status=CompileResultStatus.INTERFACE_CONTRACT_ERROR,
            candidate_fault_kind=CandidateFaultKind.INTERFACE_CONTRACT_ERROR,
            cuda_error=None,
            failure=failure,
        )
        return Phase2Output(phase1=phase1, harness=None, short_circuit=packet)
