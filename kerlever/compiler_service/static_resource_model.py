"""Per-arch hardware limits + closed-form occupancy formula.

Spec: docs/compiler-service/spec.md §6.5
Design: docs/compiler-service/design.md §4.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchLimits:
    """Hardware limits for one GPU compute capability."""

    max_warps_per_sm: int
    max_regs_per_sm: int
    max_smem_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int


# Per compute-capability limits. Source: NVIDIA CUDA C Programming Guide
# Appendix H (compute capability → hardware limits table). These values are
# conservative for occupancy estimation; they NEVER synthesize an unknown
# attribute (spec §6.5 "Never-fabricate rule").
_ARCH_LIMITS: dict[str, ArchLimits] = {
    # Ampere: sm_80 (A100) and sm_86 (RTX 30xx).
    "sm_80": ArchLimits(
        max_warps_per_sm=64,
        max_regs_per_sm=65536,
        max_smem_per_sm=167936,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
    ),
    "sm_86": ArchLimits(
        max_warps_per_sm=48,
        max_regs_per_sm=65536,
        max_smem_per_sm=102400,
        max_threads_per_sm=1536,
        max_blocks_per_sm=16,
    ),
    # Ada Lovelace.
    "sm_89": ArchLimits(
        max_warps_per_sm=48,
        max_regs_per_sm=65536,
        max_smem_per_sm=102400,
        max_threads_per_sm=1536,
        max_blocks_per_sm=24,
    ),
    # Hopper.
    "sm_90": ArchLimits(
        max_warps_per_sm=64,
        max_regs_per_sm=65536,
        max_smem_per_sm=232448,
        max_threads_per_sm=2048,
        max_blocks_per_sm=32,
    ),
}


class StaticResourceModel:
    """Per-arch limits lookup + occupancy formula."""

    def __init__(self, overrides: dict[str, ArchLimits] | None = None) -> None:
        self._limits: dict[str, ArchLimits] = dict(_ARCH_LIMITS)
        if overrides:
            self._limits.update(overrides)

    def limits_for(self, target_arch: str) -> ArchLimits | None:
        """Return hardware limits for ``target_arch`` or None if unknown."""
        return self._limits.get(target_arch)

    def compute_occupancy(
        self,
        block_dim: tuple[int, int, int],
        registers_per_thread: int | None,
        smem_bytes_per_block: int | None,
        dynamic_smem_bytes: int,
        target_arch: str,
    ) -> float | None:
        """Closed-form occupancy per spec §6.5.

        Returns ``None`` if any input is missing — the service NEVER
        fabricates a resource fact (INV-CS-003).
        """
        if registers_per_thread is None:
            return None
        if smem_bytes_per_block is None:
            return None
        limits = self.limits_for(target_arch)
        if limits is None:
            return None

        threads_per_block = block_dim[0] * block_dim[1] * block_dim[2]
        if threads_per_block <= 0:
            return None
        warps_per_block = math.ceil(threads_per_block / 32)
        if warps_per_block <= 0:
            return None

        blocks_by_warps = limits.max_warps_per_sm // warps_per_block

        register_footprint = registers_per_thread * threads_per_block
        blocks_by_registers = (
            limits.max_regs_per_sm // register_footprint
            if register_footprint > 0
            else limits.max_blocks_per_sm
        )

        smem_footprint = smem_bytes_per_block + dynamic_smem_bytes
        blocks_by_shared_memory = (
            limits.max_smem_per_sm // smem_footprint
            if smem_footprint > 0
            else limits.max_blocks_per_sm
        )

        blocks_by_threads = limits.max_threads_per_sm // threads_per_block

        active_blocks = min(
            blocks_by_warps,
            blocks_by_registers,
            blocks_by_shared_memory,
            blocks_by_threads,
            limits.max_blocks_per_sm,
        )
        if active_blocks <= 0:
            return 0.0

        occupancy = (active_blocks * warps_per_block / limits.max_warps_per_sm) * 100.0
        return max(0.0, min(100.0, occupancy))

    @classmethod
    def default(cls) -> StaticResourceModel:
        """Return the default per-arch limits model."""
        return cls()
