"""Fault attribution — the spec §6.9 table as a pure function.

``attribute_fault`` is consumed only by ``Phase5ResultAssembler`` per
design §4.7. It never returns a ``(candidate_fault, …)`` attribution
when the pod health transitioned during the request (conservative
attribution rule, INV-CS-014).

Spec: docs/compiler-service/spec.md §6.9
Design: docs/compiler-service/design.md §4.7
"""

from __future__ import annotations

from kerlever.compiler_service.types import (
    CandidateFaultKind,
    CompileResultStatus,
    CudaErrorKind,
    FaultClass,
    PodHealth,
    SanitizerTool,
    SyntaxPatternHit,
)

_AMBIGUOUS_CUDA_ERRORS: frozenset[CudaErrorKind] = frozenset(
    {
        CudaErrorKind.ILLEGAL_ADDRESS,
        CudaErrorKind.LAUNCH_TIMEOUT,
        CudaErrorKind.MISALIGNED_ADDRESS,
        CudaErrorKind.DRIVER_RESET,
    }
)


def _sanitizer_fault_kind(tool: SanitizerTool | None) -> CandidateFaultKind:
    """Map a sanitizer tool that fired FAIL to its candidate fault kind.

    Spec §6.7 failure → candidate_fault_kind table.
    """
    if tool is SanitizerTool.MEMCHECK:
        return CandidateFaultKind.MEMORY_SAFETY_ERROR
    if tool is SanitizerTool.RACECHECK or tool is SanitizerTool.SYNCCHECK:
        return CandidateFaultKind.RACE_OR_SYNC_ERROR
    if tool is SanitizerTool.INITCHECK:
        return CandidateFaultKind.UNINITIALIZED_MEMORY_ERROR
    # Should never happen: caller must pass a real tool for a sanitizer_fail.
    return CandidateFaultKind.MEMORY_SAFETY_ERROR


def attribute_fault(
    *,
    status: CompileResultStatus,
    pod_health_during_request: PodHealth,
    pod_health_transitioned: bool,
    last_sanitizer_tool: SanitizerTool | None,
    cuda_error: CudaErrorKind | None,
    compile_stderr_pattern: SyntaxPatternHit | None,
    explicit_candidate_fault_kind: CandidateFaultKind | None = None,
) -> tuple[FaultClass | None, CandidateFaultKind | None]:
    """Map a finalized ``status`` to ``(fault_class, candidate_fault_kind)``.

    The function implements the §6.9 attribution table. It is the sole
    producer of fault-class labels in the service (INV-CS-005).

    Conservative attribution rule (INV-CS-014): when the pod health
    transitioned during the request, the attribution is downgraded to
    ``AMBIGUOUS_FAULT`` even if the proximate event was candidate-like.

    Implements: REQ-CS-009
    Invariant: INV-CS-005, INV-CS-014
    """
    # SUCCESS → no fault.
    if status is CompileResultStatus.SUCCESS:
        return (None, None)

    # Pod already quarantined → always infra.
    if pod_health_during_request is PodHealth.QUARANTINED:
        return (FaultClass.INFRA_FAULT, None)

    # Conservative attribution: a pod transition during the request forces
    # AMBIGUOUS. This applies BEFORE candidate-ish labels.
    transition_is_ambiguous = (
        cuda_error in _AMBIGUOUS_CUDA_ERRORS
        or pod_health_during_request is PodHealth.SUSPECT
    )
    if pod_health_transitioned and transition_is_ambiguous:
        return (FaultClass.AMBIGUOUS_FAULT, None)

    # Infra statuses (timeout / infra_error) — never candidate.
    if status is CompileResultStatus.INFRA_ERROR:
        return (FaultClass.INFRA_FAULT, None)
    if status is CompileResultStatus.TIMEOUT:
        return (FaultClass.INFRA_FAULT, None)

    # Compile errors.
    if status is CompileResultStatus.COMPILE_ERROR:
        if compile_stderr_pattern is SyntaxPatternHit.TOOLCHAIN:
            return (FaultClass.INFRA_FAULT, None)
        if compile_stderr_pattern is SyntaxPatternHit.PARSE_LEVEL:
            return (FaultClass.CANDIDATE_FAULT, CandidateFaultKind.SYNTAX_ERROR)
        # Default non-toolchain compile errors → semantic.
        return (
            FaultClass.CANDIDATE_FAULT,
            CandidateFaultKind.SEMANTIC_COMPILE_ERROR,
        )

    # Interface contract errors — candidate-side (missing spec, unknown op).
    if status is CompileResultStatus.INTERFACE_CONTRACT_ERROR:
        kind = (
            explicit_candidate_fault_kind or CandidateFaultKind.INTERFACE_CONTRACT_ERROR
        )
        return (FaultClass.CANDIDATE_FAULT, kind)

    # Correctness failures.
    if status is CompileResultStatus.CORRECTNESS_FAIL:
        kind = explicit_candidate_fault_kind or CandidateFaultKind.CORRECTNESS_MISMATCH
        return (FaultClass.CANDIDATE_FAULT, kind)

    # Sanitizer failures — ``last_sanitizer_tool`` decides the sub-kind.
    if status is CompileResultStatus.SANITIZER_FAIL:
        kind = explicit_candidate_fault_kind or _sanitizer_fault_kind(
            last_sanitizer_tool
        )
        return (FaultClass.CANDIDATE_FAULT, kind)

    # Should be unreachable — every CompileResultStatus is handled above.
    return (FaultClass.INFRA_FAULT, None)
