"""Static resource extraction with provenance.

Spec: docs/compiler-service/spec.md §6.5
Design: docs/compiler-service/design.md §4.5
"""

from __future__ import annotations

import logging
from pathlib import Path

from kerlever.compiler_service.static_resource_model import StaticResourceModel
from kerlever.compiler_service.toolchain import (
    DriverApiAttributes,
    PtxasParser,
)
from kerlever.compiler_service.types import (
    ResourceConflict,
    ResourceSource,
    StaticAnalysis,
    StaticAnalysisExt,
)

logger = logging.getLogger(__name__)


class StaticResourceExtractor:
    """Builds ``StaticAnalysisExt`` with explicit per-fact provenance.

    INV-CS-003: facts are ``None`` when unavailable; no fabricated numbers.
    """

    def __init__(
        self,
        driver_api: DriverApiAttributes | None,
        ptxas_parser: PtxasParser,
        arch_model: StaticResourceModel,
    ) -> None:
        self._driver_api = driver_api
        self._ptxas_parser = ptxas_parser
        self._arch_model = arch_model

    def extract(
        self,
        binary: Path,
        entrypoint: str,
        block_dim: tuple[int, int, int],
        dynamic_smem_bytes: int,
        target_arch: str,
        ptxas_stdout: str,
        ptxas_stderr: str = "",
    ) -> StaticAnalysisExt:
        """Extract and return a ``StaticAnalysisExt`` with provenance.

        The driver-API source is preferred where available; ptxas fallback
        fills gaps. On disagreement, both values are recorded in
        ``resource_conflicts`` and the preferred source wins in the flat
        fields (spec §6.5 source preference table).

        Implements: REQ-CS-001
        Invariant: INV-CS-003 (never fabricate)
        """
        ptxas_report = self._ptxas_parser.parse(ptxas_stdout, ptxas_stderr)

        driver_regs = self._driver_read("read_registers_per_thread", binary, entrypoint)
        driver_smem = self._driver_read("read_static_smem_bytes", binary, entrypoint)

        resource_sources: dict[str, ResourceSource] = {}
        conflicts: list[ResourceConflict] = []

        registers_per_thread = _resolve_fact(
            fact_name="registers_per_thread",
            driver_value=driver_regs,
            ptxas_value=ptxas_report.registers_per_thread,
            resource_sources=resource_sources,
            conflicts=conflicts,
        )

        smem_bytes_per_block = _resolve_fact(
            fact_name="smem_bytes_per_block",
            driver_value=driver_smem,
            ptxas_value=ptxas_report.smem_bytes_per_block,
            resource_sources=resource_sources,
            conflicts=conflicts,
        )

        # Spill facts: ptxas only (spec §6.5 source preference table).
        spill_loads = ptxas_report.spill_loads
        spill_stores = ptxas_report.spill_stores
        resource_sources["spill_loads"] = (
            ResourceSource.PTXAS if spill_loads is not None else ResourceSource.NULL
        )
        resource_sources["spill_stores"] = (
            ResourceSource.PTXAS if spill_stores is not None else ResourceSource.NULL
        )

        occupancy = self._arch_model.compute_occupancy(
            block_dim=block_dim,
            registers_per_thread=registers_per_thread,
            smem_bytes_per_block=smem_bytes_per_block,
            dynamic_smem_bytes=dynamic_smem_bytes,
            target_arch=target_arch,
        )

        base = StaticAnalysis(
            registers_per_thread=registers_per_thread,
            smem_bytes_per_block=smem_bytes_per_block,
            spill_stores=spill_stores,
            spill_loads=spill_loads,
            occupancy_estimate_pct=occupancy,
        )
        return StaticAnalysisExt(
            base=base,
            resource_sources=resource_sources,
            resource_conflicts=conflicts,
        )

    def _driver_read(
        self, method_name: str, binary: Path, entrypoint: str
    ) -> int | None:
        """Call a driver-API read method; return None on any failure."""
        if self._driver_api is None:
            return None
        try:
            read = getattr(self._driver_api, method_name)
            value = read(binary, entrypoint)
        except Exception as exc:  # noqa: BLE001 — best-effort fallback
            logger.warning(
                "driver_api_read_failed",
                extra={"method": method_name, "error": str(exc)},
            )
            return None
        if not isinstance(value, int):
            return None
        return value


def _resolve_fact(
    *,
    fact_name: str,
    driver_value: int | None,
    ptxas_value: int | None,
    resource_sources: dict[str, ResourceSource],
    conflicts: list[ResourceConflict],
) -> int | None:
    """Apply the source preference table for one fact (spec §6.5).

    Preference order: ``cuda_func_attribute`` > ``ptxas`` > ``null``.
    Disagreement is captured in ``conflicts``.
    """
    if driver_value is not None and ptxas_value is not None:
        if driver_value != ptxas_value:
            conflicts.append(
                ResourceConflict(
                    fact=fact_name,
                    sources=[
                        (ResourceSource.CUDA_FUNC_ATTRIBUTE, driver_value),
                        (ResourceSource.PTXAS, ptxas_value),
                    ],
                    preferred_value=driver_value,
                )
            )
        resource_sources[fact_name] = ResourceSource.CUDA_FUNC_ATTRIBUTE
        return driver_value

    if driver_value is not None:
        resource_sources[fact_name] = ResourceSource.CUDA_FUNC_ATTRIBUTE
        return driver_value

    if ptxas_value is not None:
        resource_sources[fact_name] = ResourceSource.PTXAS
        return ptxas_value

    resource_sources[fact_name] = ResourceSource.NULL
    return None
