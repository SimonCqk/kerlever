"""Benchmarker — GPU lease manager.

Owns an :class:`asyncio.Semaphore(1)` per ``gpu_uuid`` / MIG id.
Guarantees single-writer access to a timed GPU for the duration of a batch.

Spec: docs/benchmarker/spec.md §6.2, INV-BENCH-003
Design: docs/benchmarker/design.md §2.1 lease.py, §5.1
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from kerlever.benchmarker.config import LeaseConfig
from kerlever.benchmarker.types import DeviceInventoryEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetGpuSpec:
    """Requested target device description.

    A request selects the first GPU whose ``sm_arch`` and optional
    ``mig_profile`` match. The :class:`LeaseManager` is responsible for
    tie-breaking by least-recently-used.
    """

    target_gpu: str
    sm_arch: str
    mig_profile: str | None = None


@dataclass(frozen=True)
class LeasedDevice:
    """Handle passed to a batch after successful lease acquisition."""

    ordinal: int
    gpu_uuid: str
    pci_bus_id: str
    sm_arch: str
    mig_profile: str | None = None
    name: str | None = None


class NoCompatibleDeviceError(RuntimeError):
    """Raised when no visible GPU matches the requested target spec.

    Routed to ``status=infra_error`` with reason ``arch_mismatch`` or
    ``mig_profile_mismatch`` (spec §6.2 decision table).
    """


class DeviceInventory:
    """Immutable snapshot of GPUs visible at service startup.

    The inventory is captured once (``telemetry.info_inventory()``) and
    reused for every request. Reopening NVML handles per-request is done
    elsewhere (``telemetry.preflight``) under the per-GPU semaphore.
    """

    def __init__(self, entries: list[DeviceInventoryEntry]) -> None:
        self._entries: list[DeviceInventoryEntry] = list(entries)

    def entries(self) -> list[DeviceInventoryEntry]:
        """Return a shallow copy of the inventory."""
        return list(self._entries)

    def find_compatible(
        self, target: TargetGpuSpec
    ) -> list[DeviceInventoryEntry]:
        """Return inventory entries whose sm_arch and MIG match the target."""
        out: list[DeviceInventoryEntry] = []
        for e in self._entries:
            if e.sm_arch != target.sm_arch:
                continue
            if target.mig_profile is not None and e.mig_profile != target.mig_profile:
                continue
            out.append(e)
        return out


def parse_target(target_gpu: str, sm_arch: str) -> TargetGpuSpec:
    """Parse a request's target_gpu and sm_arch into a :class:`TargetGpuSpec`."""
    return TargetGpuSpec(target_gpu=target_gpu, sm_arch=sm_arch)


class LeaseManager:
    """Per-GPU async semaphore registry.

    Implements: REQ-BENCH-002
    Invariant: INV-BENCH-003 (timed GPU access serialized per device)
    """

    def __init__(
        self, cfg: LeaseConfig, inventory: DeviceInventory
    ) -> None:
        self._cfg = cfg
        self._inventory = inventory
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_used_order: list[str] = []
        self._registry_lock = asyncio.Lock()

    def _key_for(self, entry: DeviceInventoryEntry) -> str:
        """Semaphore key; MIG instances use profile suffix to separate."""
        if entry.mig_profile is not None:
            return f"{entry.gpu_uuid}::{entry.mig_profile}"
        return entry.gpu_uuid

    async def _semaphore_for(self, key: str) -> asyncio.Semaphore:
        """Return the asyncio.Semaphore for ``key``, creating it on demand."""
        async with self._registry_lock:
            sem = self._semaphores.get(key)
            if sem is None:
                sem = asyncio.Semaphore(1)
                self._semaphores[key] = sem
            return sem

    def _pick_lru(
        self, entries: list[DeviceInventoryEntry]
    ) -> DeviceInventoryEntry:
        """Pick the least-recently-used matching device (spec §6.2)."""
        used = {k: i for i, k in enumerate(self._last_used_order)}
        entries_sorted = sorted(
            entries,
            key=lambda e: used.get(self._key_for(e), -1),
        )
        return entries_sorted[0]

    def _touch(self, key: str) -> None:
        """Record key as most-recently-used for LRU preference on next call."""
        if key in self._last_used_order:
            self._last_used_order.remove(key)
        self._last_used_order.append(key)

    @asynccontextmanager
    async def acquire(
        self, target: TargetGpuSpec
    ) -> AsyncIterator[LeasedDevice]:
        """Acquire an exclusive lease matching ``target``.

        The async context blocks while another batch holds the same physical
        GPU / MIG instance. On exit, the semaphore is released whether the
        batch succeeded or raised.

        Args:
            target: Requested device spec.

        Yields:
            A :class:`LeasedDevice` describing the leased GPU.

        Raises:
            NoCompatibleDeviceError: If no visible GPU matches ``target``.

        Implements: REQ-BENCH-002
        Invariant: INV-BENCH-003
        """
        compatible = self._inventory.find_compatible(target)
        if not compatible:
            raise NoCompatibleDeviceError(
                f"no visible GPU matches target sm_arch={target.sm_arch} "
                f"mig_profile={target.mig_profile}"
            )
        entry = self._pick_lru(compatible)
        key = self._key_for(entry)
        sem = await self._semaphore_for(key)
        logger.info(
            "lease.acquire",
            extra={
                "gpu_uuid": entry.gpu_uuid,
                "sm_arch": entry.sm_arch,
                "mig_profile": entry.mig_profile,
                "key": key,
            },
        )
        await sem.acquire()
        try:
            self._touch(key)
            yield LeasedDevice(
                ordinal=entry.ordinal,
                gpu_uuid=entry.gpu_uuid,
                pci_bus_id=entry.pci_bus_id,
                sm_arch=entry.sm_arch,
                mig_profile=entry.mig_profile,
                name=entry.name,
            )
        finally:
            sem.release()
            logger.info(
                "lease.release",
                extra={"gpu_uuid": entry.gpu_uuid, "key": key},
            )


__all__ = [
    "DeviceInventory",
    "LeaseManager",
    "LeasedDevice",
    "NoCompatibleDeviceError",
    "TargetGpuSpec",
    "parse_target",
]
