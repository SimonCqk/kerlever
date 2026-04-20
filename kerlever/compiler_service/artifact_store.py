"""Artifact store — pod-local filesystem surface with pinning + GC.

Spec: docs/compiler-service/spec.md §6.11
Design: docs/compiler-service/design.md §9
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from kerlever.compiler_service.errors import ArtifactStoreError
from kerlever.compiler_service.types import ArtifactClass, ArtifactKind, PinRole

logger = logging.getLogger(__name__)


# Class → TTL map derived from spec §6.11. Numbers are conservative defaults
# that prevent disk blowup during a multi-day run while leaving plenty of
# room for the orchestrator to pin what it needs.
_DEFAULT_RETENTION: dict[ArtifactClass, timedelta] = {
    ArtifactClass.BASELINE_INCUMBENT: timedelta(days=7),
    ArtifactClass.SUCCESS_TOPK: timedelta(days=3),
    ArtifactClass.SUCCESS_NON_PROFILED: timedelta(hours=6),
    ArtifactClass.COMPILE_FAILURE: timedelta(hours=24),
    ArtifactClass.CORRECTNESS_FAILURE: timedelta(hours=24),
    ArtifactClass.SANITIZER_FAILURE: timedelta(hours=24),
    ArtifactClass.INFRA_OR_AMBIGUOUS_FAILURE: timedelta(hours=6),
}


# Kind → default class map — callers may override at write time.
_KIND_DEFAULT_CLASS: dict[ArtifactKind, ArtifactClass] = {
    ArtifactKind.SOURCE_CANDIDATE: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.SOURCE_REFERENCE: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.EXECUTABLE: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.CUBIN: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.PTX: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.SASS: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.COMPILE_LOG: ArtifactClass.SUCCESS_NON_PROFILED,
    ArtifactKind.SANITIZER_REPORT: ArtifactClass.SANITIZER_FAILURE,
    ArtifactKind.CORRECTNESS_LOG: ArtifactClass.CORRECTNESS_FAILURE,
    ArtifactKind.PROBE_BINARY: ArtifactClass.BASELINE_INCUMBENT,
}


@dataclass(frozen=True)
class RetentionEntry:
    """Retention rule for one artifact class."""

    class_: ArtifactClass
    ttl: timedelta


class RetentionPolicy:
    """Maps artifact class to TTL (spec §6.11)."""

    def __init__(self, overrides: dict[ArtifactClass, timedelta] | None = None) -> None:
        self._map: dict[ArtifactClass, timedelta] = dict(_DEFAULT_RETENTION)
        if overrides:
            self._map.update(overrides)

    def entry_for(self, class_: ArtifactClass) -> RetentionEntry:
        """Return the ``RetentionEntry`` for ``class_``."""
        return RetentionEntry(class_=class_, ttl=self._map[class_])

    def ttl_priority(self) -> list[ArtifactClass]:
        """Classes sorted ascending by TTL — eager GC drops shortest first."""
        return sorted(self._map, key=lambda c: self._map[c])

    @classmethod
    def default(cls) -> RetentionPolicy:
        """Construct the default retention policy."""
        return cls()


class PinnedRoots:
    """In-memory role → {artifact_id} map (spec §6.11)."""

    def __init__(self, roots: frozenset[PinRole]) -> None:
        self._allowed_roles: frozenset[PinRole] = roots
        self._pinned: dict[PinRole, set[str]] = {role: set() for role in roots}

    def pin(self, role: PinRole, artifact_id: str) -> None:
        """Add ``artifact_id`` to the pinned set under ``role``."""
        if role not in self._allowed_roles:
            raise ArtifactStoreError(f"pin role {role} is not enabled")
        self._pinned[role].add(artifact_id)

    def unpin(self, role: PinRole, artifact_id: str) -> None:
        """Remove ``artifact_id`` from ``role``'s pinned set (no-op if absent)."""
        if role in self._pinned:
            self._pinned[role].discard(artifact_id)

    def pinned_ids(self) -> frozenset[str]:
        """Return the union of all pinned artifact ids."""
        out: set[str] = set()
        for ids in self._pinned.values():
            out.update(ids)
        return frozenset(out)


@dataclass
class _StoredArtifact:
    """Internal bookkeeping for one file on disk."""

    artifact_id: str
    kind: ArtifactKind
    class_: ArtifactClass
    path: Path
    bytes_written: int
    written_at: float  # monotonic epoch seconds
    ttl_seconds: float


class ArtifactStore:
    """Pod-local filesystem surface with pinning and class-TTL GC.

    Implements: REQ-CS-012
    Invariant: INV-CS-007 (never evict pinned / referenced artifacts)
    """

    def __init__(
        self,
        root: Path,
        retention: RetentionPolicy,
        high_watermark_pct: float,
        pinned_roots: PinnedRoots,
    ) -> None:
        self._root = root
        self._retention = retention
        self._high_watermark_pct = high_watermark_pct
        self._pinned_roots = pinned_roots
        self._lock: asyncio.Lock = asyncio.Lock()
        self._artifacts: dict[str, _StoredArtifact] = {}
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    async def write(
        self,
        kind: ArtifactKind,
        data: bytes,
        run_id: str,
        candidate_hash: str,
        class_: ArtifactClass | None = None,
    ) -> str:
        """Persist ``data`` and return its artifact id.

        The artifact id is derived from ``(run_id, candidate_hash, kind,
        sha256(data))`` — stable for identical inputs.
        """
        resolved_class = class_ or _KIND_DEFAULT_CLASS[kind]
        content_hash = hashlib.sha256(data).hexdigest()[:32]
        artifact_id = _make_artifact_id(kind, candidate_hash, content_hash)
        target = self._path_for(run_id, candidate_hash, kind, artifact_id)

        async with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            self._artifacts[artifact_id] = _StoredArtifact(
                artifact_id=artifact_id,
                kind=kind,
                class_=resolved_class,
                path=target,
                bytes_written=len(data),
                written_at=time.monotonic(),
                ttl_seconds=self._retention.entry_for(
                    resolved_class
                ).ttl.total_seconds(),
            )
        return artifact_id

    async def write_stream(
        self,
        kind: ArtifactKind,
        source: Path,
        run_id: str,
        candidate_hash: str,
        class_: ArtifactClass | None = None,
    ) -> str:
        """Copy ``source`` into the store and return its artifact id."""
        if not source.exists():
            raise ArtifactStoreError(f"source does not exist: {source}")
        data = source.read_bytes()
        return await self.write(
            kind=kind,
            data=data,
            run_id=run_id,
            candidate_hash=candidate_hash,
            class_=class_,
        )

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    async def read(self, artifact_id: str) -> AsyncIterator[bytes]:
        """Yield bytes from the artifact file.

        Returns an async iterator that yields the content in one chunk
        (files are bounded; V1 does not need streaming granularity).
        """
        path = await self.path_of(artifact_id)
        if path is None:
            raise ArtifactStoreError(f"artifact not found: {artifact_id}")

        async def _gen() -> AsyncIterator[bytes]:
            yield path.read_bytes()

        return _gen()

    async def path_of(self, artifact_id: str) -> Path | None:
        """Return the filesystem path of an artifact, or ``None`` if unknown."""
        async with self._lock:
            entry = self._artifacts.get(artifact_id)
            if entry is None:
                return None
            if not entry.path.exists():
                # Disk drifted — drop the stale map entry.
                del self._artifacts[artifact_id]
                return None
            return entry.path

    def metadata_of(self, artifact_id: str) -> _StoredArtifact | None:
        """Return the in-memory bookkeeping for an artifact (lock-free)."""
        return self._artifacts.get(artifact_id)

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin(self, role: PinRole, artifact_id: str) -> None:
        """Pin ``artifact_id`` under ``role`` (spec §6.11)."""
        self._pinned_roots.pin(role, artifact_id)

    def unpin(self, role: PinRole, artifact_id: str) -> None:
        """Unpin ``artifact_id`` from ``role``."""
        self._pinned_roots.unpin(role, artifact_id)

    def pinned_ids(self) -> frozenset[str]:
        """Return the union of all pinned artifact ids."""
        return self._pinned_roots.pinned_ids()

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    async def gc_cheap_pass(self, referenced_ids: frozenset[str] = frozenset()) -> int:
        """Drop expired, unpinned, unreferenced artifacts.

        ``referenced_ids`` is the skip set from ``IdempotencyRegistry``
        (INV-CS-007). Returns the number of artifacts deleted.
        """
        skip = self._pinned_roots.pinned_ids() | referenced_ids
        now = time.monotonic()
        to_delete: list[str] = []
        async with self._lock:
            for artifact_id, entry in self._artifacts.items():
                if artifact_id in skip:
                    continue
                if now - entry.written_at < entry.ttl_seconds:
                    continue
                to_delete.append(artifact_id)
            deleted = await self._delete_ids_locked(to_delete)
        if deleted and await self._over_watermark():
            # Eager pass; re-collect skip set (pinned may have changed).
            deleted += await self.gc_eager_if_over_watermark(referenced_ids)
        return deleted

    async def gc_eager_if_over_watermark(self, referenced_ids: frozenset[str]) -> int:
        """Drop artifacts by TTL-class priority until below watermark.

        Pinned and referenced ids are ALWAYS skipped.
        """
        if not await self._over_watermark():
            return 0
        skip = self._pinned_roots.pinned_ids() | referenced_ids
        total_deleted = 0
        async with self._lock:
            for class_ in self._retention.ttl_priority():
                candidates = [
                    a.artifact_id
                    for a in self._artifacts.values()
                    if a.class_ is class_ and a.artifact_id not in skip
                ]
                if not candidates:
                    continue
                total_deleted += await self._delete_ids_locked(candidates)
                if not _disk_over_watermark(self._root, self._high_watermark_pct):
                    break
        return total_deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _delete_ids_locked(self, ids: Sequence[str]) -> int:
        """Delete artifacts from disk and map; caller holds ``_lock``."""
        deleted = 0
        for artifact_id in ids:
            entry = self._artifacts.pop(artifact_id, None)
            if entry is None:
                continue
            try:
                if entry.path.exists():
                    entry.path.unlink()
                deleted += 1
            except OSError as exc:  # noqa: PERF203 — filesystem is slow
                logger.warning(
                    "artifact_delete_failed",
                    extra={"artifact_id": artifact_id, "error": str(exc)},
                )
        return deleted

    async def _over_watermark(self) -> bool:
        """Return True if disk utilization exceeds the configured watermark."""
        return _disk_over_watermark(self._root, self._high_watermark_pct)

    def _path_for(
        self,
        run_id: str,
        candidate_hash: str,
        kind: ArtifactKind,
        artifact_id: str,
    ) -> Path:
        """Compute the on-disk path for a new artifact."""
        return (
            self._root
            / _safe(run_id)
            / _safe(candidate_hash)
            / kind.value
            / f"{artifact_id}.bin"
        )


@dataclass
class ArtifactStoreStats:
    """Summary for the ``/v1/pod-status`` response."""

    count: int
    total_bytes: int
    disk_used_pct: float
    pinned_count: int
    unpinned_count: int = field(default=0)


def disk_watermark_snapshot(
    store: ArtifactStore, root: Path, high_watermark_pct: float
) -> ArtifactStoreStats:
    """Return a disk-usage snapshot for observability surfaces."""
    total = 0
    count = 0
    pinned = store.pinned_ids()
    for entry in store._artifacts.values():  # noqa: SLF001 — observability read
        total += entry.bytes_written
        count += 1
    used_pct = _disk_used_pct(root)
    return ArtifactStoreStats(
        count=count,
        total_bytes=total,
        disk_used_pct=used_pct,
        pinned_count=len(pinned),
        unpinned_count=count - len(pinned),
    )


def _safe(component: str) -> str:
    """Return a filesystem-safe version of ``component``."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in component)


def _make_artifact_id(
    kind: ArtifactKind, candidate_hash: str, content_hash: str
) -> str:
    """Construct a stable artifact id."""
    return f"{kind.value}-{candidate_hash[:12]}-{content_hash}"


def _disk_used_pct(root: Path) -> float:
    """Return percent disk usage at ``root``'s filesystem, or 0 if unknown."""
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return 0.0
    if usage.total == 0:
        return 0.0
    return (usage.used / usage.total) * 100.0


def _disk_over_watermark(root: Path, high_watermark_pct: float) -> bool:
    """True if disk utilization exceeds the watermark."""
    return _disk_used_pct(root) >= high_watermark_pct
