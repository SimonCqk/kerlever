"""Adapter registry + ``OperationAdapter`` Protocol re-exports.

Spec: docs/compiler-service/spec.md §6.2
Design: docs/compiler-service/design.md §4.4
"""

from __future__ import annotations

from collections.abc import Sequence

from kerlever.compiler_service.adapters.base import (
    InputBundle,
    OperationAdapter,
    ShapeComparisonResult,
)
from kerlever.compiler_service.adapters.elementwise import ElementwiseAdapter
from kerlever.compiler_service.adapters.matmul import MatmulAdapter


class AdapterRegistry:
    """Frozen registry of operation adapters (design §4.4).

    The registry is the only place ``op_name`` is read outside Phase 2
    (INV-CS-013).
    """

    def __init__(self, adapters: Sequence[OperationAdapter]) -> None:
        self._by_name: dict[str, OperationAdapter] = {}
        for adapter in adapters:
            self._by_name[adapter.op_name] = adapter

    def get(self, op_name: str) -> OperationAdapter | None:
        """Return the adapter for ``op_name`` or None if not registered."""
        return self._by_name.get(op_name)

    def names(self) -> frozenset[str]:
        """Return the set of registered op names."""
        return frozenset(self._by_name.keys())


def default_registry() -> AdapterRegistry:
    """Return the V1 registry with matmul + elementwise adapters."""
    return AdapterRegistry([MatmulAdapter(), ElementwiseAdapter()])


__all__ = [
    "AdapterRegistry",
    "ElementwiseAdapter",
    "InputBundle",
    "MatmulAdapter",
    "OperationAdapter",
    "ShapeComparisonResult",
    "default_registry",
]
