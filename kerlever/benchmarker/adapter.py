"""Benchmarker — Operation Adapter Protocol, registry, and V1 built-ins.

Adapters own every operation-specific concern: per-shape device buffer
allocation, deterministic input seeding, grid-dim derivation, launch-arg
construction, reset hooks between timed iterations, and buffer rotation.
The worker harness is adapter-agnostic — it receives closures, never the
adapter object itself (design §3.3).

This module provides:

* :class:`AdapterBuffers` — opaque buffer record carried through the harness.
* :class:`OperationAdapter` — the Protocol every adapter implements.
* :class:`AdapterRegistry` — (abi_name, abi_version) → adapter lookup.
* Module-level :func:`register_adapter`, :func:`get_adapter`,
  :func:`list_registered` — thin wrappers over a module-singleton registry.
* Two V1 built-in adapters:
  - :class:`ElementwiseAddFp32V1` — C = A + B for fp32.
  - :class:`MatmulFp16V1` — C = A @ B for fp16.

Both built-ins register themselves on module import via module-level calls
(spec §6.11, SC-BENCH-012 Non-Goal: no filesystem-scan discovery).

Spec: docs/benchmarker/spec.md §5 Operation Adapter, §6.11
Design: docs/benchmarker/design.md §3.3, §6 External Adapters
Implements: REQ-BENCH-028, REQ-BENCH-032
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from kerlever.benchmarker.cuda_driver import DevicePtr
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    ShapeCase,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdapterBuffers
# ---------------------------------------------------------------------------


@dataclass
class AdapterBuffers:
    """Opaque, adapter-owned record for one (shape, candidate) tuple.

    The adapter is the sole interpreter of these fields; the harness only
    passes the record through to :meth:`OperationAdapter.build_launch_args`
    and :meth:`OperationAdapter.reset_between_iterations`.

    Attributes:
        device_ptrs: Device pointers (one per logical buffer; order is
            adapter-specific).
        host_inputs: Named host-side byte buffers kept for deterministic
            re-seeding and debugging (never consulted inside the timed loop).
        nbytes_per_buffer: Byte size per device pointer, parallel to
            ``device_ptrs``. Used by ``free`` and for diagnostics.
        dtype: The element dtype string (e.g., ``"fp32"``, ``"fp16"``).
        shape_id: The spec §5 ``ShapeCase.shape_id`` this record was allocated
            for.
    """

    device_ptrs: list[DevicePtr] = field(default_factory=list)
    host_inputs: dict[str, bytes] = field(default_factory=dict)
    nbytes_per_buffer: list[int] = field(default_factory=list)
    dtype: str = ""
    shape_id: str = ""


# ---------------------------------------------------------------------------
# OperationAdapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OperationAdapter(Protocol):
    """Per-operation plugin surface (spec §5 Operation Adapter).

    Adapters are stateless aside from registry identity; every method is a
    pure function of its inputs plus the adapter's ABI declaration.

    Implements: REQ-BENCH-028
    """

    abi_name: ClassVar[str]
    abi_version: ClassVar[str]
    iteration_semantics: ClassVar[AdapterIterationSemantics]

    def allocate(
        self,
        shape: ShapeCase,
        dtype: str,
        device: LeasedDevice,
    ) -> AdapterBuffers: ...

    def seed_inputs(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
        seed: int,
    ) -> None: ...

    def build_launch_args(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
    ) -> tuple[object, ...]: ...

    def grid_dim(
        self,
        shape: ShapeCase,
        block_dim: tuple[int, int, int],
    ) -> tuple[int, int, int]: ...

    def useful_bytes(self, shape: ShapeCase) -> int: ...

    def algorithmic_flops(self, shape: ShapeCase) -> int: ...

    def reset_between_iterations(
        self,
        buffers: AdapterBuffers,
        semantics: AdapterIterationSemantics,
    ) -> None: ...

    def rotate_buffers(
        self,
        buffers_pool: list[AdapterBuffers],
    ) -> AdapterBuffers: ...

    def free(self, buffers: AdapterBuffers) -> None: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AdapterRegistry:
    """Simple in-memory (abi_name, abi_version) → adapter lookup.

    V1 does not support version ranges; each ``(abi_name, abi_version)`` tuple
    maps to exactly one adapter instance. Duplicate registration is a warn
    (last one wins) so hot-reload in tests does not crash the service.

    Implements: REQ-BENCH-028
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], OperationAdapter] = {}

    def register(self, adapter: OperationAdapter) -> None:
        """Register ``adapter`` under its declared ``(abi_name, abi_version)``."""
        key = (adapter.abi_name, adapter.abi_version)
        if key in self._store:
            logger.warning(
                "adapter.duplicate_registration",
                extra={"abi_name": key[0], "abi_version": key[1]},
            )
        self._store[key] = adapter

    def get(self, abi_name: str, abi_version: str) -> OperationAdapter | None:
        """Return the registered adapter for ``(abi_name, abi_version)`` or ``None``."""
        return self._store.get((abi_name, abi_version))

    def list_registered(self) -> list[tuple[str, str]]:
        """Return a stable-ordered list of registered ``(abi, version)`` keys."""
        return sorted(self._store.keys())


_REGISTRY = AdapterRegistry()


def register_adapter(adapter: OperationAdapter) -> None:
    """Register ``adapter`` in the module-singleton registry."""
    _REGISTRY.register(adapter)


def get_adapter(abi_name: str, abi_version: str) -> OperationAdapter | None:
    """Look up an adapter from the module-singleton registry."""
    return _REGISTRY.get(abi_name, abi_version)


def list_registered() -> list[tuple[str, str]]:
    """Enumerate registered adapters in the module-singleton registry."""
    return _REGISTRY.list_registered()


# ---------------------------------------------------------------------------
# Helpers shared by built-ins
# ---------------------------------------------------------------------------


def _seeded_rng(seed: int) -> object:
    """Return a ``numpy.random.Generator`` seeded deterministically.

    Lazy-import keeps ``import kerlever.benchmarker.adapter`` cheap on systems
    where numpy takes multiple hundreds of milliseconds to import.
    """
    from numpy.random import PCG64, Generator  # noqa: PLC0415

    # Mask to 64 bits because Python hashes can be negative or wider than
    # PCG64 accepts on some platforms.
    seed64 = seed & 0xFFFF_FFFF_FFFF_FFFF
    return Generator(PCG64(seed64))


def _alloc_device_buffer(nbytes: int) -> DevicePtr:
    """Wrapper over :func:`cuda_driver.mem_alloc` so tests can monkeypatch."""
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    return cd.mem_alloc(nbytes)


def _free_device_buffer(ptr: DevicePtr) -> None:
    """Wrapper over :func:`cuda_driver.mem_free`."""
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    cd.mem_free(ptr)


def _htod(ptr: DevicePtr, host_bytes: bytes) -> None:
    """Wrapper over :func:`cuda_driver.memcpy_htod`."""
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    cd.memcpy_htod(ptr, host_bytes)


def _prod(dims: list[int] | tuple[int, ...]) -> int:
    """Multiplicative reduction over a dims list; returns 1 for an empty list."""
    out = 1
    for d in dims:
        out *= int(d)
    return out


# ---------------------------------------------------------------------------
# Built-in: elementwise_add_fp32_v1
# ---------------------------------------------------------------------------


class ElementwiseAddFp32V1:
    """C = A + B for fp32 tensors of total length N = prod(dims).

    ABI: ``(const float* A, const float* B, float* C, int N)``.

    Implements: REQ-BENCH-028, SC-BENCH-012
    """

    abi_name: ClassVar[str] = "elementwise_add_fp32_v1"
    abi_version: ClassVar[str] = "0.1.0"
    iteration_semantics: ClassVar[AdapterIterationSemantics] = (
        AdapterIterationSemantics.OVERWRITE_PURE
    )

    _ITEM_SIZE: ClassVar[int] = 4  # sizeof(float32)

    def _nbytes(self, shape: ShapeCase) -> int:
        """Return the byte count for one operand buffer."""
        n = _prod(shape.dims)
        return n * self._ITEM_SIZE

    def allocate(
        self,
        shape: ShapeCase,
        dtype: str,
        device: LeasedDevice,
    ) -> AdapterBuffers:
        """Allocate A, B, C on device (all ``N * sizeof(float32)`` bytes)."""
        _ = device  # single-device subprocess; allocation is context-implicit.
        nb = self._nbytes(shape)
        d_a = _alloc_device_buffer(nb)
        d_b = _alloc_device_buffer(nb)
        d_c = _alloc_device_buffer(nb)
        return AdapterBuffers(
            device_ptrs=[d_a, d_b, d_c],
            host_inputs={},
            nbytes_per_buffer=[nb, nb, nb],
            dtype=dtype or "fp32",
            shape_id=shape.shape_id,
        )

    def seed_inputs(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
        seed: int,
    ) -> None:
        """Fill A and B with deterministic uniform-random fp32; zero C."""
        import numpy as np  # noqa: PLC0415

        rng = _seeded_rng(seed)
        n = _prod(shape.dims)
        # The output buffer is zero-initialized rather than skipped so an
        # accidental "read-before-write" kernel does not see stale bytes.
        a = rng.standard_normal(size=n, dtype=np.float32)  # type: ignore[attr-defined]
        b = rng.standard_normal(size=n, dtype=np.float32)  # type: ignore[attr-defined]
        c = np.zeros(n, dtype=np.float32)
        a_bytes = a.tobytes()
        b_bytes = b.tobytes()
        c_bytes = c.tobytes()
        _htod(buffers.device_ptrs[0], a_bytes)
        _htod(buffers.device_ptrs[1], b_bytes)
        _htod(buffers.device_ptrs[2], c_bytes)
        buffers.host_inputs = {"A": a_bytes, "B": b_bytes}

    def build_launch_args(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
    ) -> tuple[object, ...]:
        """Return ``(d_A, d_B, d_C, N)`` per the declared ABI."""
        n = _prod(shape.dims)
        return (
            int(buffers.device_ptrs[0]),
            int(buffers.device_ptrs[1]),
            int(buffers.device_ptrs[2]),
            int(n),
        )

    def grid_dim(
        self,
        shape: ShapeCase,
        block_dim: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Return ``(ceil(N / block_dim.x), 1, 1)``."""
        n = _prod(shape.dims)
        bx = max(1, int(block_dim[0]))
        return (max(1, math.ceil(n / bx)), 1, 1)

    def useful_bytes(self, shape: ShapeCase) -> int:
        """A + B read, C written → 3 × N × 4 bytes."""
        n = _prod(shape.dims)
        return 3 * n * self._ITEM_SIZE

    def algorithmic_flops(self, shape: ShapeCase) -> int:
        """One add per element → N FLOP."""
        return _prod(shape.dims)

    def reset_between_iterations(
        self,
        buffers: AdapterBuffers,
        semantics: AdapterIterationSemantics,
    ) -> None:
        """No-op for ``OVERWRITE_PURE`` (spec §6.11)."""
        _ = buffers, semantics

    def rotate_buffers(
        self,
        buffers_pool: list[AdapterBuffers],
    ) -> AdapterBuffers:
        """Return the next buffer in a deterministic round-robin order."""
        if not buffers_pool:
            raise ValueError("rotate_buffers called with empty pool")
        # V1 adapters do not track rotation state; callers that need
        # rotation own the pool index. We return the first entry so the
        # call remains well-typed; the Phase 3 harness currently does not
        # enable rotation for OVERWRITE_PURE adapters.
        return buffers_pool[0]

    def free(self, buffers: AdapterBuffers) -> None:
        """Release every device pointer in the buffer record."""
        for ptr in buffers.device_ptrs:
            _free_device_buffer(ptr)
        buffers.device_ptrs = []
        buffers.nbytes_per_buffer = []


# ---------------------------------------------------------------------------
# Built-in: matmul_fp16_v1
# ---------------------------------------------------------------------------


class MatmulFp16V1:
    """C = A @ B for fp16 matrices shaped ``[M, N, K]``.

    ABI: ``(const half* A, const half* B, half* C, int M, int N, int K)``.
    ``A`` is ``[M, K]``, ``B`` is ``[K, N]``, ``C`` is ``[M, N]``.

    Implements: REQ-BENCH-028, SC-BENCH-012
    """

    abi_name: ClassVar[str] = "matmul_fp16_v1"
    abi_version: ClassVar[str] = "0.1.0"
    iteration_semantics: ClassVar[AdapterIterationSemantics] = (
        AdapterIterationSemantics.OVERWRITE_PURE
    )

    _ITEM_SIZE: ClassVar[int] = 2  # sizeof(float16)

    def _dims_mnk(self, shape: ShapeCase) -> tuple[int, int, int]:
        """Unpack ``shape.dims`` as ``(M, N, K)``; fail loudly on mis-shaped input."""
        if len(shape.dims) != 3:
            raise ValueError(
                f"matmul_fp16_v1 requires 3-dim shape [M, N, K]; got {shape.dims!r}"
            )
        m, n, k = shape.dims
        return int(m), int(n), int(k)

    def allocate(
        self,
        shape: ShapeCase,
        dtype: str,
        device: LeasedDevice,
    ) -> AdapterBuffers:
        """Allocate fp16 A(M×K), B(K×N), C(M×N)."""
        _ = device
        m, n, k = self._dims_mnk(shape)
        nb_a = m * k * self._ITEM_SIZE
        nb_b = k * n * self._ITEM_SIZE
        nb_c = m * n * self._ITEM_SIZE
        d_a = _alloc_device_buffer(nb_a)
        d_b = _alloc_device_buffer(nb_b)
        d_c = _alloc_device_buffer(nb_c)
        return AdapterBuffers(
            device_ptrs=[d_a, d_b, d_c],
            host_inputs={},
            nbytes_per_buffer=[nb_a, nb_b, nb_c],
            dtype=dtype or "fp16",
            shape_id=shape.shape_id,
        )

    def seed_inputs(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
        seed: int,
    ) -> None:
        """Fill A and B with deterministic uniform-random fp16; zero C."""
        import numpy as np  # noqa: PLC0415

        rng = _seeded_rng(seed)
        m, n, k = self._dims_mnk(shape)
        # Standard-normal sampled in fp32 then cast to fp16; fp16 random
        # numbers with large magnitude often overflow in matmul, so we scale
        # down by 1/sqrt(k) — still deterministic, just better conditioned.
        scale = 1.0 / math.sqrt(max(1, k))
        a32 = rng.standard_normal(size=m * k, dtype=np.float32) * scale  # type: ignore[attr-defined]
        b32 = rng.standard_normal(size=k * n, dtype=np.float32) * scale  # type: ignore[attr-defined]
        a = a32.astype(np.float16)
        b = b32.astype(np.float16)
        c = np.zeros(m * n, dtype=np.float16)
        a_bytes = a.tobytes()
        b_bytes = b.tobytes()
        c_bytes = c.tobytes()
        _htod(buffers.device_ptrs[0], a_bytes)
        _htod(buffers.device_ptrs[1], b_bytes)
        _htod(buffers.device_ptrs[2], c_bytes)
        buffers.host_inputs = {"A": a_bytes, "B": b_bytes}

    def build_launch_args(
        self,
        buffers: AdapterBuffers,
        shape: ShapeCase,
    ) -> tuple[object, ...]:
        """Return ``(d_A, d_B, d_C, M, N, K)`` per the declared ABI."""
        m, n, k = self._dims_mnk(shape)
        return (
            int(buffers.device_ptrs[0]),
            int(buffers.device_ptrs[1]),
            int(buffers.device_ptrs[2]),
            m,
            n,
            k,
        )

    def grid_dim(
        self,
        shape: ShapeCase,
        block_dim: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Tile ``(M, N)`` by the block dim; K is the reduction axis.

        Default tile is 16×16 when the block does not declare its own dims;
        when ``block_dim = (16, 16, 1)`` the output covers the full (M, N)
        grid exactly.
        """
        m, n, _k = self._dims_mnk(shape)
        bx = max(1, int(block_dim[0]))
        by = max(1, int(block_dim[1]))
        gx = max(1, math.ceil(n / bx))
        gy = max(1, math.ceil(m / by))
        return (gx, gy, 1)

    def useful_bytes(self, shape: ShapeCase) -> int:
        """Read A + B, write C → ``(M*K + K*N + M*N) * sizeof(half)``."""
        m, n, k = self._dims_mnk(shape)
        return (m * k + k * n + m * n) * self._ITEM_SIZE

    def algorithmic_flops(self, shape: ShapeCase) -> int:
        """Standard matmul flops: ``2 * M * N * K``."""
        m, n, k = self._dims_mnk(shape)
        return 2 * m * n * k

    def reset_between_iterations(
        self,
        buffers: AdapterBuffers,
        semantics: AdapterIterationSemantics,
    ) -> None:
        """No-op for ``OVERWRITE_PURE`` (spec §6.11)."""
        _ = buffers, semantics

    def rotate_buffers(
        self,
        buffers_pool: list[AdapterBuffers],
    ) -> AdapterBuffers:
        """Return the first buffer in the pool (stateless rotate)."""
        if not buffers_pool:
            raise ValueError("rotate_buffers called with empty pool")
        return buffers_pool[0]

    def free(self, buffers: AdapterBuffers) -> None:
        """Release every device pointer in the buffer record."""
        for ptr in buffers.device_ptrs:
            _free_device_buffer(ptr)
        buffers.device_ptrs = []
        buffers.nbytes_per_buffer = []


# ---------------------------------------------------------------------------
# Auto-registration of V1 built-ins
# ---------------------------------------------------------------------------

register_adapter(ElementwiseAddFp32V1())
register_adapter(MatmulFp16V1())


__all__ = [
    "AdapterBuffers",
    "AdapterRegistry",
    "ElementwiseAddFp32V1",
    "MatmulFp16V1",
    "OperationAdapter",
    "get_adapter",
    "list_registered",
    "register_adapter",
]
