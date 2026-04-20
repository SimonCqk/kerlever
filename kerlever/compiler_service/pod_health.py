"""Pod health singleton + known-good probe coordination.

Spec: docs/compiler-service/spec.md §6.8
Design: docs/compiler-service/design.md §10
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from kerlever.compiler_service.types import CudaErrorKind, PodHealth

logger = logging.getLogger(__name__)

TransitionReason = Literal[
    "clean_pass",
    "ambiguous_event",
    "probe_pass",
    "probe_fail",
    "ambiguous_limit_exceeded",
]


class Phase4ClassificationKind(StrEnum):
    """How Phase 4 classified its own outcome for the tracker."""

    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    CANDIDATE_FAILURE = "candidate_failure"


@dataclass(frozen=True)
class Phase4Classification:
    """Feedback from Phase 4 to the tracker."""

    kind: Phase4ClassificationKind
    cuda_error: CudaErrorKind | None = None


@dataclass(frozen=True)
class PodHealthTransition:
    """One pod-health FSM transition record."""

    previous: PodHealth
    current: PodHealth
    reason: TransitionReason


@dataclass(frozen=True)
class ProbeOutcome:
    """Result of running the known-good probe kernel."""

    passed: bool
    detail: str = ""


class PodHealthTracker:
    """Pod-wide health state machine (spec §6.8).

    The tracker is a singleton per service; it holds an internal
    ``asyncio.Lock`` but the lock is NEVER held during probe execution —
    only around the small state-mutation critical sections.

    Implements: REQ-CS-007
    Invariant: INV-CS-008 (snapshot at assembly time),
        INV-CS-014 (ambiguous never morphs into candidate)
    """

    def __init__(
        self,
        ambiguous_limit: int,
        probe_source_path: Path,
        probe_executable_path: Path | None = None,
    ) -> None:
        self._state: PodHealth = PodHealth.HEALTHY
        self._ambiguous_count: int = 0
        self._last_transition: PodHealthTransition | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._ambiguous_limit = ambiguous_limit
        self._probe_source_path = probe_source_path
        self._probe_executable_path = probe_executable_path

    @property
    def probe_source_path(self) -> Path:
        """Path to the known-good probe kernel source."""
        return self._probe_source_path

    @property
    def probe_executable_path(self) -> Path | None:
        """Pre-compiled probe binary; ``None`` means probe disabled in V1."""
        return self._probe_executable_path

    @property
    def ambiguous_failure_count(self) -> int:
        """Cumulative ambiguous fault count across requests."""
        return self._ambiguous_count

    @property
    def last_transition(self) -> PodHealthTransition | None:
        """Most recent state transition, if any."""
        return self._last_transition

    def snapshot(self) -> PodHealth:
        """Return the pod's current health state.

        Safe to call without the lock: reading a single enum is atomic.
        """
        return self._state

    async def needs_probe(self) -> bool:
        """Return True if the next Phase 4 must run the probe first."""
        async with self._lock:
            return self._state is PodHealth.SUSPECT

    async def run_probe_if_needed(
        self,
        runner: Callable[[], Awaitable[ProbeOutcome]],
    ) -> PodHealthTransition | None:
        """Run the probe kernel if the tracker is in ``SUSPECT``.

        The runner coroutine is awaited OUTSIDE the mutation lock so
        concurrent ``snapshot()`` calls are not blocked by a slow probe.
        ``runner`` is responsible for acquiring the GPU semaphore.
        """
        async with self._lock:
            if self._state is not PodHealth.SUSPECT:
                return None

        outcome = await runner()

        async with self._lock:
            previous = self._state
            if outcome.passed:
                self._state = PodHealth.HEALTHY
                self._ambiguous_count = 0
                transition = PodHealthTransition(previous, self._state, "probe_pass")
            else:
                self._state = PodHealth.QUARANTINED
                transition = PodHealthTransition(previous, self._state, "probe_fail")
            self._last_transition = transition
            return transition

    async def record_phase4_outcome(
        self, classification: Phase4Classification
    ) -> PodHealthTransition | None:
        """Apply the FSM transition for a Phase 4 outcome.

        Returns ``None`` when no transition occurred.
        """
        async with self._lock:
            previous = self._state

            if classification.kind is Phase4ClassificationKind.CLEAN:
                # A clean pass resets ambiguous counter but does not force
                # a transition from SUSPECT — the probe is responsible for
                # that edge, not a candidate-driven clean pass.
                if previous is PodHealth.SUSPECT:
                    return None
                self._ambiguous_count = 0
                return None

            if classification.kind is Phase4ClassificationKind.AMBIGUOUS:
                self._ambiguous_count += 1
                if self._ambiguous_count >= self._ambiguous_limit:
                    self._state = PodHealth.QUARANTINED
                    transition = PodHealthTransition(
                        previous, self._state, "ambiguous_limit_exceeded"
                    )
                elif previous is PodHealth.HEALTHY:
                    self._state = PodHealth.SUSPECT
                    transition = PodHealthTransition(
                        previous, self._state, "ambiguous_event"
                    )
                elif previous is PodHealth.SUSPECT:
                    # Already suspect, still ambiguous — stay in suspect.
                    return None
                else:
                    return None
                self._last_transition = transition
                return transition

            # CANDIDATE_FAILURE: deterministic candidate bug on a healthy
            # pod; do not change pod health.
            return None
