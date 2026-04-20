"""Phase handoff dataclasses shared across phase modules.

Each ``PhaseNOutput`` carries an optional ``short_circuit`` packet. When
set, ``CompilerService.compile`` routes the request directly to Phase 5
(design §7; INV-CS-015).
"""

from __future__ import annotations

from dataclasses import dataclass

from kerlever.compiler_service.types import (
    CandidateFaultKind,
    CompileResultStatus,
    CudaErrorKind,
    FailureDetail,
    PhaseName,
)


@dataclass(frozen=True)
class PhaseShortCircuit:
    """Packet handed to Phase 5 when a phase decides the request is done.

    Phase 5 is the sole ``CompileResult`` constructor (INV-CS-015); every
    early-exit path uses this packet instead of constructing the result
    directly.
    """

    phase: PhaseName
    status: CompileResultStatus
    candidate_fault_kind: CandidateFaultKind | None
    cuda_error: CudaErrorKind | None
    failure: FailureDetail
