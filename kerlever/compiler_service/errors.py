"""Internal exception hierarchy for the Compiler Service.

These exceptions never surface to HTTP clients — ``Phase5ResultAssembler``
catches them and converts them to a structured ``CompileResult`` with
``status=infra_error`` / ``fault_class=infra_fault`` unless the exception
carries an explicit mapping.

Spec: docs/compiler-service/spec.md
Design: docs/compiler-service/design.md §7
"""

from __future__ import annotations

from kerlever.compiler_service.types import (
    CandidateFaultKind,
    CompileResultStatus,
    FaultClass,
    PhaseName,
)


class CompilerServiceError(Exception):
    """Base class for all internal Compiler Service exceptions.

    Subclasses may carry a status/fault mapping; the assembler falls back to
    ``(INFRA_ERROR, INFRA_FAULT)`` when no mapping is present.
    """

    phase: PhaseName | None = None
    status: CompileResultStatus | None = None
    fault_class: FaultClass | None = None
    candidate_fault_kind: CandidateFaultKind | None = None
    reason: str | None = None


class ToolchainUnavailableError(CompilerServiceError):
    """A required toolchain component is missing or unusable.

    Raised by the toolchain probe and by runners that discover a missing
    tool mid-flight.
    """

    fault_class = FaultClass.INFRA_FAULT


class ArtifactStoreError(CompilerServiceError):
    """Artifact store I/O failure (disk full, unwritable root, etc.)."""

    fault_class = FaultClass.INFRA_FAULT


class DriverApiUnavailableError(CompilerServiceError):
    """``cuda-python`` is not importable or the CUDA driver is unavailable.

    Callers must fall back to ``ptxas`` per INV-CS-003 — they do not treat
    this as a user-facing error.
    """


class InvalidAdapterError(CompilerServiceError):
    """An adapter misbehaved (returned a malformed harness, etc.)."""

    fault_class = FaultClass.INFRA_FAULT


class UnsupportedOperationError(CompilerServiceError):
    """The inbound ``op_name`` is not in the adapter registry.

    Surfaces as ``INTERFACE_CONTRACT_ERROR`` / ``CANDIDATE_FAULT`` /
    ``INTERFACE_CONTRACT_ERROR`` per INV-CS-013.
    """

    status = CompileResultStatus.INTERFACE_CONTRACT_ERROR
    fault_class = FaultClass.CANDIDATE_FAULT
    candidate_fault_kind = CandidateFaultKind.INTERFACE_CONTRACT_ERROR
    reason = "unsupported_operation"


class ProbeFailureError(CompilerServiceError):
    """The known-good probe kernel failed on the pod (spec §6.8)."""

    status = CompileResultStatus.INFRA_ERROR
    fault_class = FaultClass.INFRA_FAULT
