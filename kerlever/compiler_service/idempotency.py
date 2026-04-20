"""In-memory idempotency registry.

Spec: docs/compiler-service/spec.md §6.10
Design: docs/compiler-service/design.md §4.5, §11
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from weakref import WeakValueDictionary

from kerlever.compiler_service.types import (
    CompileResult,
    IdempotencyState,
    PhaseName,
)

logger = logging.getLogger(__name__)


class _IntakeState(StrEnum):
    """Internal intake classification (not user-facing)."""

    NEW = "new"
    REUSED_COMPLETED = "reused_completed"
    PRIOR_ATTEMPT_LOST = "prior_attempt_lost"


@dataclass
class IdempotencyEntry:
    """One in-memory record per ``request_id``.

    Lifecycle: created on NEW intake → ``phase_observed`` advances per phase
    → ``completed_at`` and ``compile_result`` set by ``finalize``.
    """

    request_id: str
    started_at: datetime
    phase_observed: PhaseName = PhaseName.REQUEST_NORMALIZATION
    artifact_key: str | None = None
    artifact_refs: list[str] = field(default_factory=list)
    completed_at: datetime | None = None
    compile_result: CompileResult | None = None


@dataclass(frozen=True)
class IdempotencyIntake:
    """Result of ``observe_intake`` — what Phase 1 needs to decide next."""

    state: IdempotencyState
    result: CompileResult | None = None
    prior_attempt_observed_phase: PhaseName | None = None


class IdempotencyRegistry:
    """Per-``request_id`` attempt bookkeeping with TTL and artifact ref tracking.

    V1 is in-memory only. A process restart empties the registry (spec §6.10).

    Implements: REQ-CS-006
    Invariant: INV-CS-009 (reuse asserts artifact_key equality)
    """

    def __init__(self, ttl: timedelta) -> None:
        self._entries: dict[str, IdempotencyEntry] = {}
        self._per_id_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._registry_lock: asyncio.Lock = asyncio.Lock()
        self._ttl = ttl

    async def acquire_id_lock(self, request_id: str) -> asyncio.Lock:
        """Return the per-id lock, creating it if needed.

        Design §11.2: per-``request_id`` lock held for the duration of the
        request so concurrent replays serialize correctly.
        """
        async with self._registry_lock:
            lock = self._per_id_locks.get(request_id)
            if lock is None:
                lock = asyncio.Lock()
                self._per_id_locks[request_id] = lock
        return lock

    async def observe_intake(
        self, request_id: str, current_artifact_key: str
    ) -> IdempotencyIntake:
        """Look up or create the entry; classify the intake (spec §6.10).

        Implements: REQ-CS-006
        Invariant: INV-CS-009 (stale key → treat as NEW)
        """
        async with self._registry_lock:
            entry = self._entries.get(request_id)
            if entry is None:
                self._entries[request_id] = IdempotencyEntry(
                    request_id=request_id,
                    started_at=_utcnow(),
                )
                return IdempotencyIntake(state=IdempotencyState.NEW)

            # Completed entry — check artifact_key equality.
            if entry.compile_result is not None and entry.completed_at is not None:
                if entry.artifact_key == current_artifact_key:
                    return IdempotencyIntake(
                        state=IdempotencyState.REUSED_COMPLETED,
                        result=entry.compile_result,
                    )
                logger.warning(
                    "idempotency_stale_entry",
                    extra={
                        "request_id": request_id,
                        "stored_artifact_key": entry.artifact_key,
                        "current_artifact_key": current_artifact_key,
                    },
                )
                # Stale entry — replace with a fresh NEW record.
                self._entries[request_id] = IdempotencyEntry(
                    request_id=request_id,
                    started_at=_utcnow(),
                )
                return IdempotencyIntake(state=IdempotencyState.NEW)

            # Started but not completed — prior attempt lost.
            return IdempotencyIntake(
                state=IdempotencyState.PRIOR_ATTEMPT_LOST,
                prior_attempt_observed_phase=entry.phase_observed,
            )

    async def record_phase(self, request_id: str, phase: PhaseName) -> None:
        """Advance ``phase_observed`` for ``request_id``.

        Missing entries are ignored — a NEW intake always creates one first.
        """
        async with self._registry_lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry.phase_observed = phase

    async def finalize(
        self,
        request_id: str,
        artifact_key: str,
        refs: Sequence[str],
        result: CompileResult,
    ) -> None:
        """Atomically mark the entry as completed with its final result."""
        async with self._registry_lock:
            entry = self._entries.get(request_id)
            if entry is None:
                entry = IdempotencyEntry(request_id=request_id, started_at=_utcnow())
                self._entries[request_id] = entry
            entry.artifact_key = artifact_key
            entry.artifact_refs = [r for r in refs if r]
            entry.completed_at = _utcnow()
            entry.compile_result = result
            entry.phase_observed = PhaseName.OUTPUT

    def finalize_if_pending(self, request_id: str) -> None:
        """Best-effort cleanup; called from ``finally`` in the service.

        If the request finished normally, this is a no-op because
        ``finalize`` already ran. If an exception escaped, the entry stays
        in ``started`` state so a replay sees ``PRIOR_ATTEMPT_LOST``.
        """
        # This method is intentionally synchronous and lock-free: it is a
        # cleanup hook only, and the registry's ``finalize`` handles every
        # real completion path.
        return None

    def referenced_artifact_ids(self) -> frozenset[str]:
        """Return every artifact id referenced by an in-TTL completed entry.

        Used by ``ArtifactStore`` GC to build its skip set so a live replay
        can never find a dangling reference (INV-CS-007).
        """
        now = _utcnow()
        cutoff = now - self._ttl
        ids: set[str] = set()
        for entry in self._entries.values():
            if entry.completed_at is None:
                continue
            if entry.completed_at < cutoff:
                continue
            ids.update(entry.artifact_refs)
        return frozenset(ids)

    def purge_expired(self) -> int:
        """Drop entries whose ``completed_at`` is older than ``ttl``.

        Returns the number of entries dropped.
        """
        now = _utcnow()
        cutoff = now - self._ttl
        to_drop: list[str] = []
        for request_id, entry in self._entries.items():
            if entry.completed_at is None:
                continue
            if entry.completed_at < cutoff:
                to_drop.append(request_id)
        for request_id in to_drop:
            del self._entries[request_id]
        return len(to_drop)


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)
