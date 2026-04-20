"""Phase 5 — result assembly.

This module is the **only** place that constructs ``CompileResult``
(INV-CS-015). Every other phase short-circuits by returning a
``PhaseShortCircuit`` packet that the assembler reads here.

Spec: docs/compiler-service/spec.md §6
Design: docs/compiler-service/design.md §4.3, §7
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kerlever.compiler_service.envelope import PhaseTimer, RunEnvelopeBuilder
from kerlever.compiler_service.faults import attribute_fault
from kerlever.compiler_service.idempotency import IdempotencyRegistry
from kerlever.compiler_service.phases import PhaseShortCircuit
from kerlever.compiler_service.phases.phase1_request import Phase1Output
from kerlever.compiler_service.phases.phase3_compile import Phase3Output
from kerlever.compiler_service.phases.phase4_correctness import Phase4Output
from kerlever.compiler_service.pod_health import (
    PodHealthTracker,
    PodHealthTransition,
)
from kerlever.compiler_service.types import (
    ArtifactRefs,
    CandidateFaultKind,
    CompileRequest,
    CompileResult,
    CompileResultStatus,
    CudaErrorKind,
    FailureDetail,
    FaultClass,
    IdempotencyState,
    PhaseName,
    SanitizerTool,
    SyntaxPatternHit,
    ToolchainInfo,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _AssemblyInputs:
    """Combined inputs derived from a short-circuit or a full run."""

    status: CompileResultStatus
    failure: FailureDetail | None
    candidate_fault_kind: CandidateFaultKind | None
    cuda_error: CudaErrorKind | None
    compile_stderr_pattern: SyntaxPatternHit | None
    last_sanitizer_tool: SanitizerTool | None
    pod_health_transitioned: bool


class Phase5ResultAssembler:
    """Assembles the outbound ``CompileResult`` (sole constructor).

    Invariant: INV-CS-015 (single construction site), INV-CS-005
        (status + fault_class consistent), INV-CS-008 (pod_health sampled
        at assembly time)
    """

    def __init__(
        self,
        toolchain: ToolchainInfo,
        pod_health: PodHealthTracker,
        idempotency: IdempotencyRegistry,
        timer: PhaseTimer,
    ) -> None:
        self._toolchain = toolchain
        self._pod_health = pod_health
        self._idempotency = idempotency
        self._timer = timer

    # ------------------------------------------------------------------
    # Short-circuit path
    # ------------------------------------------------------------------

    async def from_short_circuit(
        self,
        request: CompileRequest,
        phase1: Phase1Output,
        packet: PhaseShortCircuit,
    ) -> CompileResult:
        """Assemble a result from a phase's short-circuit packet."""
        phase_start = time.monotonic()
        compile_stderr_pattern = _pattern_from_reason(packet.failure.reason)
        inputs = _AssemblyInputs(
            status=packet.status,
            failure=packet.failure,
            candidate_fault_kind=packet.candidate_fault_kind,
            cuda_error=packet.cuda_error,
            compile_stderr_pattern=compile_stderr_pattern,
            last_sanitizer_tool=None,
            pod_health_transitioned=False,
        )
        result = await self._assemble_shared(
            request=request,
            phase1=phase1,
            phase3=None,
            phase4=None,
            inputs=inputs,
        )
        self._timer.record(PhaseName.OUTPUT, phase_start)
        return result

    # ------------------------------------------------------------------
    # Full-run path
    # ------------------------------------------------------------------

    async def assemble(
        self,
        request: CompileRequest,
        phase1: Phase1Output,
        phase3: Phase3Output,
        phase4: Phase4Output,
    ) -> CompileResult:
        """Assemble a SUCCESS result or a late-phase short-circuit."""
        phase_start = time.monotonic()
        if phase4.short_circuit is not None:
            inputs = _AssemblyInputs(
                status=phase4.short_circuit.status,
                failure=phase4.short_circuit.failure,
                candidate_fault_kind=phase4.short_circuit.candidate_fault_kind,
                cuda_error=phase4.short_circuit.cuda_error,
                compile_stderr_pattern=None,
                last_sanitizer_tool=(
                    phase4.correctness_outcome.last_sanitizer_tool
                    if phase4.correctness_outcome is not None
                    else None
                ),
                pod_health_transitioned=phase4.pod_health_transition is not None,
            )
        else:
            inputs = _AssemblyInputs(
                status=CompileResultStatus.SUCCESS,
                failure=None,
                candidate_fault_kind=None,
                cuda_error=None,
                compile_stderr_pattern=None,
                last_sanitizer_tool=None,
                pod_health_transitioned=phase4.pod_health_transition is not None,
            )
        result = await self._assemble_shared(
            request=request,
            phase1=phase1,
            phase3=phase3,
            phase4=phase4,
            inputs=inputs,
        )
        self._timer.record(PhaseName.OUTPUT, phase_start)
        return result

    # ------------------------------------------------------------------
    # Reused completed replay path
    # ------------------------------------------------------------------

    async def from_reused_completed(
        self,
        phase1: Phase1Output,
    ) -> CompileResult:
        """Return the stored ``CompileResult`` with a refreshed envelope.

        Spec §6.10 requires the returned result to carry the CURRENT
        ``pod_health`` at assembly time, not the stored value.
        """
        stored = phase1.reused_completed_result
        assert stored is not None  # noqa: S101 — guaranteed by caller
        pod_health = self._pod_health.snapshot()
        envelope = RunEnvelopeBuilder.build(
            seed=phase1.envelope_seed,
            timer=self._timer,
            pod_health=pod_health,
            idempotency_state=IdempotencyState.REUSED_COMPLETED,
            previous_attempt_lost=False,
            prior_attempt_observed_phase=None,
        )
        # Replace the envelope + idempotency state on the stored result.
        return stored.model_copy(
            update={
                "run_envelope": envelope,
            }
        )

    # ------------------------------------------------------------------
    # Shared assembly implementation
    # ------------------------------------------------------------------

    async def _assemble_shared(
        self,
        request: CompileRequest,
        phase1: Phase1Output,
        phase3: Phase3Output | None,
        phase4: Phase4Output | None,
        inputs: _AssemblyInputs,
    ) -> CompileResult:
        """Build the final ``CompileResult`` for any path."""
        pod_health = self._pod_health.snapshot()
        pod_transition = _pod_transition_of(phase4)
        pod_health_transitioned = inputs.pod_health_transitioned or (
            pod_transition is not None
        )

        fault_class, candidate_fault_kind = attribute_fault(
            status=inputs.status,
            pod_health_during_request=pod_health,
            pod_health_transitioned=pod_health_transitioned,
            last_sanitizer_tool=inputs.last_sanitizer_tool,
            cuda_error=inputs.cuda_error,
            compile_stderr_pattern=inputs.compile_stderr_pattern,
            explicit_candidate_fault_kind=inputs.candidate_fault_kind,
        )

        # INV-CS-014: when pod transitioned to suspect/quarantined during
        # the request, never keep a candidate_fault_kind.
        if fault_class is FaultClass.AMBIGUOUS_FAULT:
            candidate_fault_kind = None

        envelope = RunEnvelopeBuilder.build(
            seed=phase1.envelope_seed,
            timer=self._timer,
            pod_health=pod_health,
            idempotency_state=phase1.idempotency_state,
            previous_attempt_lost=phase1.previous_attempt_lost,
            prior_attempt_observed_phase=phase1.prior_attempt_observed_phase,
        )

        artifacts = _artifact_refs_of(phase3, phase4)

        static_analysis = phase3.static_analysis if phase3 is not None else None
        correctness = (
            phase4.correctness_outcome.correctness
            if phase4 is not None and phase4.correctness_outcome is not None
            else None
        )

        result = CompileResult(
            status=inputs.status,
            candidate_hash=request.candidate_hash,
            run_envelope=envelope,
            legacy_inferred_execution_spec=phase1.legacy_inferred_execution_spec,
            toolchain=self._toolchain,
            static_analysis=static_analysis,
            correctness=correctness,
            artifacts=artifacts,
            fault_class=fault_class,
            candidate_fault_kind=candidate_fault_kind,
            failure=inputs.failure,
        )

        # Finalize idempotency — skip on replay paths where the entry is
        # already final (REUSED_COMPLETED), handled by the caller.
        if phase1.idempotency_state is not IdempotencyState.REUSED_COMPLETED:
            await self._idempotency.finalize(
                request_id=request.request_id,
                artifact_key=phase1.envelope_seed.artifact_key,
                refs=_collect_artifact_ids(artifacts),
                result=result,
            )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pattern_from_reason(reason: str | None) -> SyntaxPatternHit | None:
    """Infer the compile-stderr pattern from a short-circuit's reason tag."""
    if reason is None:
        return None
    if "parse_level" in reason:
        return SyntaxPatternHit.PARSE_LEVEL
    if "semantic" in reason:
        return SyntaxPatternHit.SEMANTIC
    if "toolchain" in reason:
        return SyntaxPatternHit.TOOLCHAIN
    return None


def _pod_transition_of(
    phase4: Phase4Output | None,
) -> PodHealthTransition | None:
    """Return the Phase 4 transition if present."""
    if phase4 is None:
        return None
    return phase4.pod_health_transition


def _artifact_refs_of(
    phase3: Phase3Output | None,
    phase4: Phase4Output | None,
) -> ArtifactRefs:
    """Collect artifact ids into an ``ArtifactRefs`` bundle."""
    if phase3 is None or phase3.compile is None:
        return ArtifactRefs()
    compile_ = phase3.compile
    sanitizer_ids: list[str] = []
    correctness_log_artifact_id: str | None = None
    if phase4 is not None:
        if phase4.correctness_outcome is not None:
            for outcome in phase4.correctness_outcome.correctness.sanitizer_results:
                if outcome.report_artifact_id is not None:
                    sanitizer_ids.append(outcome.report_artifact_id)
        correctness_log_artifact_id = phase4.correctness_log_artifact_id
    return ArtifactRefs(
        source_artifact_id=compile_.source_artifact_id,
        executable_artifact_id=compile_.executable_artifact_id,
        reference_executable_artifact_id=compile_.reference_executable_artifact_id,
        cubin_artifact_id=compile_.cubin_artifact_id,
        ptx_artifact_id=compile_.ptx_artifact_id,
        sass_artifact_id=compile_.sass_artifact_id,
        compile_log_artifact_id=compile_.compile_log_artifact_id,
        sanitizer_report_artifact_ids=sanitizer_ids,
        correctness_log_artifact_id=correctness_log_artifact_id,
    )


def _collect_artifact_ids(refs: ArtifactRefs) -> list[str]:
    """Flatten ``ArtifactRefs`` into a list of artifact ids for idempotency."""
    ids: list[str] = []
    for field_name in (
        "source_artifact_id",
        "executable_artifact_id",
        "reference_executable_artifact_id",
        "cubin_artifact_id",
        "ptx_artifact_id",
        "sass_artifact_id",
        "compile_log_artifact_id",
        "correctness_log_artifact_id",
    ):
        value = getattr(refs, field_name, None)
        if isinstance(value, str) and value:
            ids.append(value)
    ids.extend(refs.sanitizer_report_artifact_ids)
    return ids
