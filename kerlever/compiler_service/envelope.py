"""RunEnvelope construction and per-request phase timing.

Spec: docs/compiler-service/spec.md §6.1
Design: docs/compiler-service/design.md §4
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from kerlever.compiler_service.types import (
    IdempotencyState,
    PhaseName,
    PodHealth,
    RequestLimits,
    RunEnvelope,
)


class PhaseTimer:
    """Accumulates phase durations for the RunEnvelope.

    Not thread-safe by design; one instance per request.
    """

    def __init__(self) -> None:
        self._timings_ms: dict[PhaseName, float] = {}

    def record(self, phase: PhaseName, start_monotonic: float) -> None:
        """Record duration (ms) from ``start_monotonic`` to now for ``phase``."""
        duration_ms = (time.monotonic() - start_monotonic) * 1000.0
        self._timings_ms[phase] = duration_ms

    def snapshot(self) -> dict[PhaseName, float]:
        """Return a shallow copy of the current timings."""
        return dict(self._timings_ms)


@dataclass(frozen=True)
class RunEnvelopeSeed:
    """Intermediate envelope material computed by Phase 1.

    Phase 5 combines this with the live ``pod_health`` sample and final
    idempotency state to produce the outbound ``RunEnvelope``.
    """

    run_id: str
    round_id: str
    request_id: str
    candidate_hash: str
    source_hash: str
    problem_spec_hash: str
    launch_spec_hash: str
    toolchain_hash: str
    compile_flags_hash: str
    adapter_version: str
    artifact_key: str
    limits: RequestLimits
    pod_id: str
    gpu_uuid: str


class RunEnvelopeBuilder:
    """Assembles the outbound ``RunEnvelope`` at Phase 5 time.

    The builder reads live pod_health at ``build()`` time so the envelope
    reflects the pod state AFTER any Phase 4 transitions (INV-CS-008).
    """

    @staticmethod
    def build(
        *,
        seed: RunEnvelopeSeed,
        timer: PhaseTimer,
        pod_health: PodHealth,
        idempotency_state: IdempotencyState,
        previous_attempt_lost: bool,
        prior_attempt_observed_phase: PhaseName | None,
    ) -> RunEnvelope:
        """Construct the ``RunEnvelope`` for a completed request."""
        return RunEnvelope(
            run_id=seed.run_id,
            round_id=seed.round_id,
            request_id=seed.request_id,
            candidate_hash=seed.candidate_hash,
            source_hash=seed.source_hash,
            problem_spec_hash=seed.problem_spec_hash,
            launch_spec_hash=seed.launch_spec_hash,
            toolchain_hash=seed.toolchain_hash,
            compile_flags_hash=seed.compile_flags_hash,
            adapter_version=seed.adapter_version,
            artifact_key=seed.artifact_key,
            limits=seed.limits,
            pod_id=seed.pod_id,
            gpu_uuid=seed.gpu_uuid,
            phase_timings_ms=timer.snapshot(),
            pod_health=pod_health,
            idempotency_state=idempotency_state,
            previous_attempt_lost=previous_attempt_lost,
            prior_attempt_observed_phase=prior_attempt_observed_phase,
        )
