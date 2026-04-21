"""Phase 3 — nvcc compile + static resource extraction.

Spec: docs/compiler-service/spec.md §6.4, §6.5
Design: docs/compiler-service/design.md §4.3
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from kerlever.compiler_service.artifact_store import ArtifactStore
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.envelope import PhaseTimer
from kerlever.compiler_service.phases import PhaseShortCircuit
from kerlever.compiler_service.phases.phase2_harness import Phase2Output
from kerlever.compiler_service.resource_extraction import StaticResourceExtractor
from kerlever.compiler_service.toolchain import (
    CuobjdumpRunner,
    NvccResult,
    NvccRunner,
)
from kerlever.compiler_service.types import (
    ArtifactKind,
    CandidateFaultKind,
    CompileResultStatus,
    FailureDetail,
    PhaseName,
    StaticAnalysisExt,
    SyntaxPatternHit,
)

logger = logging.getLogger(__name__)

_PARSE_PATTERNS = re.compile(
    r"(expected|missing\s+';'|undeclared|syntax\s+error|unclosed|unterminated)",
    re.IGNORECASE,
)
_TOOLCHAIN_PATTERNS = re.compile(
    r"(unsupported\s+arch|cannot\s+find|no\s+such\s+file.*nvcc|unknown\s+target)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompileArtifacts:
    """Artifact ids + paths produced on a successful compile."""

    reference_executable: Path
    candidate_executable: Path
    source_artifact_id: str
    executable_artifact_id: str
    reference_executable_artifact_id: str
    cubin_artifact_id: str | None
    ptx_artifact_id: str | None
    sass_artifact_id: str | None
    compile_log_artifact_id: str


@dataclass(frozen=True)
class Phase3Output:
    """Happy-path payload plus optional short-circuit packet."""

    phase2: Phase2Output
    compile: CompileArtifacts | None
    static_analysis: StaticAnalysisExt | None
    short_circuit: PhaseShortCircuit | None = None


class Phase3Compiler:
    """Two-way ``nvcc`` compile + static resource extraction (INV-CS-001).

    Reference and candidate are always compiled as separate executables.
    """

    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
        nvcc: NvccRunner,
        cuobjdump: CuobjdumpRunner,
        resource_extractor: StaticResourceExtractor,
        compile_semaphore: asyncio.Semaphore,
    ) -> None:
        self._config = config
        self._artifact_store = artifact_store
        self._nvcc = nvcc
        self._cuobjdump = cuobjdump
        self._resource_extractor = resource_extractor
        self._compile_semaphore = compile_semaphore

    async def run(self, phase2: Phase2Output, timer: PhaseTimer) -> Phase3Output:
        """Compile reference + candidate, then extract static facts."""
        phase_start = time.monotonic()
        try:
            if phase2.harness is None:
                return Phase3Output(phase2=phase2, compile=None, static_analysis=None)

            harness = phase2.harness
            request = phase2.phase1.request
            workspace = harness.workspace
            reference_bin = workspace / "reference.out"
            candidate_bin = workspace / "candidate.out"

            async with self._compile_semaphore:
                ref_result = await self._nvcc.compile(
                    source=harness.reference_source_path,
                    output=reference_bin,
                    target_arch=request.target_arch,
                    timeout_s=phase2.phase1.envelope_seed.limits.compile_timeout_s,
                    max_log_bytes=phase2.phase1.envelope_seed.limits.max_log_bytes,
                )
                cand_result = await self._nvcc.compile(
                    source=harness.candidate_source_path,
                    output=candidate_bin,
                    target_arch=request.target_arch,
                    timeout_s=phase2.phase1.envelope_seed.limits.compile_timeout_s,
                    max_log_bytes=phase2.phase1.envelope_seed.limits.max_log_bytes,
                )

            # Reference compile failing is ALWAYS infra — the reference
            # ships with the problem spec and must compile cleanly.
            if ref_result.timed_out:
                return self._short_circuit_timeout(phase2, ref_result, "reference")
            if ref_result.returncode != 0:
                return self._short_circuit_reference_compile_error(phase2, ref_result)

            if cand_result.timed_out:
                return self._short_circuit_timeout(phase2, cand_result, "candidate")
            if cand_result.returncode != 0:
                return self._short_circuit_compile_error(
                    phase2, cand_result, "candidate"
                )

            # Persist successful artifacts.
            exec_artifact_id = await self._artifact_store.write(
                kind=ArtifactKind.EXECUTABLE,
                data=candidate_bin.read_bytes(),
                run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )
            reference_exec_artifact_id = await self._artifact_store.write(
                kind=ArtifactKind.EXECUTABLE,
                data=reference_bin.read_bytes(),
                run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )

            combined_log = (
                f"// reference compile log\n"
                f"{ref_result.stdout_excerpt}\n"
                f"{ref_result.stderr_excerpt}\n"
                f"// candidate compile log\n"
                f"{cand_result.stdout_excerpt}\n"
                f"{cand_result.stderr_excerpt}\n"
            )
            compile_log_artifact_id = await self._artifact_store.write(
                kind=ArtifactKind.COMPILE_LOG,
                data=combined_log.encode("utf-8"),
                run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )

            # Cubin extraction (best-effort; INV-CS-003: missing → None).
            cubin_path = workspace / "candidate.cubin"
            cubin_result = await self._cuobjdump.extract_cubin(
                executable=candidate_bin,
                output=cubin_path,
                target_arch=request.target_arch,
            )
            cubin_artifact_id: str | None = None
            if (
                cubin_result.output_path is not None
                and cubin_result.output_path.exists()
            ):
                try:
                    cubin_artifact_id = await self._artifact_store.write(
                        kind=ArtifactKind.CUBIN,
                        data=cubin_result.output_path.read_bytes(),
                        run_id=request.run_id,
                        candidate_hash=request.candidate_hash,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning("cubin_write_failed", extra={"error": str(exc)})

            # PTX dump (best-effort; INV-CS-003: missing → None).
            ptx_path = workspace / "candidate.ptx"
            ptx_result = await self._cuobjdump.dump_ptx(
                executable=candidate_bin,
                output=ptx_path,
            )
            ptx_artifact_id: str | None = None
            if ptx_result.output_path is not None and ptx_result.output_path.exists():
                try:
                    ptx_artifact_id = await self._artifact_store.write(
                        kind=ArtifactKind.PTX,
                        data=ptx_result.output_path.read_bytes(),
                        run_id=request.run_id,
                        candidate_hash=request.candidate_hash,
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning("ptx_write_failed", extra={"error": str(exc)})

            # SASS extraction (best-effort; missing SASS → None, never fabricated).
            sass_output_path = workspace / "candidate.sass"
            sass_result = await self._cuobjdump.extract_sass(
                executable=candidate_bin,
                output=sass_output_path,
            )
            sass_artifact_id: str | None = None
            if sass_result.output_path is not None and sass_result.output_path.exists():
                sass_artifact_id = await self._artifact_store.write(
                    kind=ArtifactKind.SASS,
                    data=sass_result.output_path.read_bytes(),
                    run_id=request.run_id,
                    candidate_hash=request.candidate_hash,
                )

            resolved_spec = phase2.phase1.resolved_execution_spec
            resource_binary = (
                cubin_path
                if cubin_artifact_id is not None and cubin_path.exists()
                else candidate_bin
            )
            static_analysis = self._resource_extractor.extract(
                binary=resource_binary,
                entrypoint=resolved_spec.entrypoint or "",
                block_dim=resolved_spec.block_dim or (1, 1, 1),
                dynamic_smem_bytes=resolved_spec.dynamic_smem_bytes or 0,
                target_arch=request.target_arch,
                ptxas_stdout=cand_result.stdout_excerpt,
                ptxas_stderr=cand_result.stderr_excerpt,
            )

            # Propagate artifact ids into the analysis record.
            static_analysis = StaticAnalysisExt(
                base=static_analysis.base,
                resource_sources=static_analysis.resource_sources,
                resource_conflicts=static_analysis.resource_conflicts,
                cubin_artifact_id=cubin_artifact_id,
                ptx_artifact_id=ptx_artifact_id,
                sass_artifact_id=sass_artifact_id,
            )

            return Phase3Output(
                phase2=phase2,
                compile=CompileArtifacts(
                    reference_executable=reference_bin,
                    candidate_executable=candidate_bin,
                    source_artifact_id=harness.candidate_source_artifact_id,
                    executable_artifact_id=exec_artifact_id,
                    reference_executable_artifact_id=reference_exec_artifact_id,
                    cubin_artifact_id=cubin_artifact_id,
                    ptx_artifact_id=ptx_artifact_id,
                    sass_artifact_id=sass_artifact_id,
                    compile_log_artifact_id=compile_log_artifact_id,
                ),
                static_analysis=static_analysis,
            )
        finally:
            timer.record(PhaseName.COMPILE, phase_start)

    # ------------------------------------------------------------------
    # Short-circuit helpers
    # ------------------------------------------------------------------

    def _short_circuit_compile_error(
        self, phase2: Phase2Output, result: NvccResult, role: str
    ) -> Phase3Output:
        """Short-circuit on nvcc non-zero exit (spec §6.4)."""
        stderr = result.stderr_excerpt
        if _TOOLCHAIN_PATTERNS.search(stderr):
            pattern = SyntaxPatternHit.TOOLCHAIN
            fault_kind: CandidateFaultKind | None = None
            status = CompileResultStatus.COMPILE_ERROR
        elif _PARSE_PATTERNS.search(stderr):
            pattern = SyntaxPatternHit.PARSE_LEVEL
            fault_kind = CandidateFaultKind.SYNTAX_ERROR
            status = CompileResultStatus.COMPILE_ERROR
        else:
            pattern = SyntaxPatternHit.SEMANTIC
            fault_kind = CandidateFaultKind.SEMANTIC_COMPILE_ERROR
            status = CompileResultStatus.COMPILE_ERROR

        failure = FailureDetail(
            phase=PhaseName.COMPILE,
            command=result.command,
            stdout_excerpt=result.stdout_excerpt,
            stderr_excerpt=result.stderr_excerpt,
            failing_shape_id=None,
            retryable=False,
            reason=f"{role}_compile_error_{pattern.value}",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.COMPILE,
            status=status,
            candidate_fault_kind=fault_kind,
            cuda_error=None,
            failure=failure,
        )
        return Phase3Output(
            phase2=phase2, compile=None, static_analysis=None, short_circuit=packet
        )

    def _short_circuit_reference_compile_error(
        self, phase2: Phase2Output, result: NvccResult
    ) -> Phase3Output:
        """Short-circuit when the reference harness cannot compile.

        The candidate did not cause this failure. The reference source and
        adapter harness are measurement infrastructure for the request, so the
        result must not feed negative evidence into optimization search memory.
        """
        failure = FailureDetail(
            phase=PhaseName.COMPILE,
            command=result.command,
            stdout_excerpt=result.stdout_excerpt,
            stderr_excerpt=result.stderr_excerpt,
            failing_shape_id=None,
            retryable=False,
            reason="reference_compile_error",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.COMPILE,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase3Output(
            phase2=phase2, compile=None, static_analysis=None, short_circuit=packet
        )

    def _short_circuit_timeout(
        self, phase2: Phase2Output, result: NvccResult, role: str
    ) -> Phase3Output:
        """Short-circuit on nvcc wall-clock timeout (attribution: infra)."""
        failure = FailureDetail(
            phase=PhaseName.COMPILE,
            command=result.command,
            stdout_excerpt=result.stdout_excerpt,
            stderr_excerpt=result.stderr_excerpt,
            retryable=True,
            reason=f"{role}_compile_timeout",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.COMPILE,
            status=CompileResultStatus.TIMEOUT,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase3Output(
            phase2=phase2, compile=None, static_analysis=None, short_circuit=packet
        )
