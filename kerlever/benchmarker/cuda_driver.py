"""Benchmarker — cuda-python Driver API facade.

All ``cuda.bindings.driver`` symbols are accessed via lazy imports inside
functions. No raw ctypes escape this module; callers receive opaque
dataclass handles (``CudaContext`` / ``CudaModule`` / ``CudaFunction`` /
``CudaEvent`` / ``CudaStream``) and typed Python scalar returns.

The design allows ``import kerlever.benchmarker.main`` to succeed on
non-GPU dev machines by deferring binding access until first use.

Spec: docs/benchmarker/spec.md §6.3, §6.6
Design: docs/benchmarker/design.md §6.1 cuda_driver.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NewType

from kerlever.benchmarker.types import CacheConfig, FunctionAttribute

DevicePtr = NewType("DevicePtr", int)
"""Opaque device pointer (``CUdeviceptr``) carried as a Python int.

Only :mod:`kerlever.benchmarker.adapter`, :func:`worker._load_candidates`,
and :func:`profile_child.main` are expected to construct or free a
``DevicePtr``. The harness and other modules treat it as an opaque value.
"""

logger = logging.getLogger(__name__)


class CudaDriverError(RuntimeError):
    """Raised for any non-SUCCESS CUresult returned by a driver API call.

    Callers convert this to per-candidate ``runtime_fault`` inside the
    harness or bubble it up to the worker's top-level handler where the
    fault classifier decides ``candidate_fault`` vs ``ambiguous_fault``.
    """

    def __init__(self, code: int, symbol: str, message: str) -> None:
        self.code = code
        self.symbol = symbol
        super().__init__(f"CUDA {symbol}({code}): {message}")


class FuncAttr(StrEnum):
    """Function attributes we set on loaded candidates.

    Values correspond to driver API ``CUfunction_attribute`` enums and are
    resolved lazily in :func:`_funcattr_enum`.
    """

    MAX_DYNAMIC_SHARED_SIZE_BYTES = "MAX_DYNAMIC_SHARED_SIZE_BYTES"
    PREFERRED_SHARED_CARVEOUT = "PREFERRED_SHARED_MEMORY_CARVEOUT"


@dataclass(frozen=True)
class ModuleLoadOptions:
    """Options passed to ``cuModuleLoadDataEx``.

    Exposes a minimal option set; extend only when a measurement path needs
    a new knob (YAGNI-first per AGENTS.md rule 2).
    """

    max_registers: int | None = None
    target_from_context: bool = True


@dataclass(frozen=True)
class CudaContext:
    """Opaque handle around a primary CUDA context."""

    handle: Any
    device_ordinal: int


@dataclass(frozen=True)
class CudaModule:
    """Opaque handle around a loaded module."""

    handle: Any


@dataclass(frozen=True)
class CudaFunction:
    """Opaque handle around a resolved kernel function."""

    handle: Any
    entrypoint: str


@dataclass(frozen=True)
class CudaEvent:
    """Opaque handle around a CUDA event."""

    handle: Any


@dataclass(frozen=True)
class CudaStream:
    """Opaque handle around a CUDA stream."""

    handle: Any


def _cuda() -> Any:
    """Lazily import ``cuda.bindings.driver``.

    Falls back to the legacy ``cuda.cuda`` package name when the user has
    the older cuda-python release installed (12.3 line).

    Raises:
        CudaDriverError: If neither package is importable.
    """
    try:
        from cuda.bindings import driver  # noqa: PLC0415

        return driver
    except ImportError:  # pragma: no cover - environment-dependent
        try:
            from cuda import cuda  # noqa: PLC0415

            return cuda
        except ImportError as exc:
            raise CudaDriverError(
                -1, "CUDA_DRIVER_UNAVAILABLE", f"cuda-python not installed: {exc}"
            ) from exc


def _check(rc: Any) -> None:
    """Validate a driver API return tuple/status.

    ``cuda-python`` returns ``(CUresult, *values)`` tuples. We inspect the
    first element; non-SUCCESS raises :class:`CudaDriverError`.
    """
    cuda_mod = _cuda()
    status = rc[0] if isinstance(rc, tuple) else rc
    try:
        ok = cuda_mod.CUresult.CUDA_SUCCESS
    except Exception as exc:  # pragma: no cover - stub shapes differ
        raise CudaDriverError(
            -1, "CUDA_ENUM_MISSING", f"cannot resolve CUDA_SUCCESS: {exc}"
        ) from exc
    if status == ok:
        return
    code = int(getattr(status, "value", -1))
    symbol = str(status)
    raise CudaDriverError(code, symbol, f"cuda call returned {symbol}")


def _unpack_one(rc: Any) -> Any:
    """Return the single payload value from a ``(status, value)`` tuple."""
    _check(rc)
    if isinstance(rc, tuple) and len(rc) >= 2:
        return rc[1]
    return None


def init() -> None:
    """Initialize the CUDA driver API (``cuInit(0)``)."""
    cuda_mod = _cuda()
    rc = cuda_mod.cuInit(0)
    _check(rc)


def create_primary_context(device_ordinal: int) -> CudaContext:
    """Retain and bind the primary context for ``device_ordinal``."""
    cuda_mod = _cuda()
    device = _unpack_one(cuda_mod.cuDeviceGet(device_ordinal))
    handle = _unpack_one(cuda_mod.cuDevicePrimaryCtxRetain(device))
    _check(cuda_mod.cuCtxSetCurrent(handle))
    return CudaContext(handle=handle, device_ordinal=device_ordinal)


def destroy_primary_context(ctx: CudaContext) -> None:
    """Release the primary context retained by :func:`create_primary_context`."""
    cuda_mod = _cuda()
    device = _unpack_one(cuda_mod.cuDeviceGet(ctx.device_ordinal))
    try:
        _check(cuda_mod.cuDevicePrimaryCtxRelease(device))
    except CudaDriverError as exc:
        # Intentionally log and return — tearing down a poisoned context can
        # hang and the worker is about to ``os._exit`` anyway (design §4.3).
        logger.warning("cuda.ctx_release.failed", extra={"error": str(exc)})


def load_module(cubin: bytes, options: ModuleLoadOptions | None) -> CudaModule:
    """Load a cubin/module via ``cuModuleLoadDataEx``.

    ``options`` is accepted for interface completeness; the V1 load path uses
    no non-default options. Adding options here is the designated extension
    point for architecture tuning — callers should not reach for ctypes.
    """
    _ = options  # Reserved for ModuleLoadOption tuning; V1 uses defaults.
    cuda_mod = _cuda()
    handle = _unpack_one(
        cuda_mod.cuModuleLoadDataEx(cubin, 0, [], [])
    )
    return CudaModule(handle=handle)


def get_function(module: CudaModule, entrypoint: str) -> CudaFunction:
    """Resolve ``entrypoint`` inside ``module`` via ``cuModuleGetFunction``."""
    cuda_mod = _cuda()
    handle = _unpack_one(
        cuda_mod.cuModuleGetFunction(module.handle, entrypoint.encode("utf-8"))
    )
    return CudaFunction(handle=handle, entrypoint=entrypoint)


_FUNC_ATTR_ENUM_NAMES: dict[FunctionAttribute, str] = {
    FunctionAttribute.CACHE_MODE_CA: "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES: (
        "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"
    ),
    FunctionAttribute.PREFERRED_SHARED_MEMORY_CARVEOUT: (
        "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"
    ),
    FunctionAttribute.REQUIRED_CLUSTER_WIDTH: (
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH"
    ),
    FunctionAttribute.REQUIRED_CLUSTER_HEIGHT: (
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT"
    ),
    FunctionAttribute.REQUIRED_CLUSTER_DEPTH: (
        "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH"
    ),
    FunctionAttribute.NON_PORTABLE_CLUSTER_SIZE_ALLOWED: (
        "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED"
    ),
}


def _funcattr_enum(attr: FuncAttr | FunctionAttribute) -> Any:
    """Resolve :class:`FuncAttr`/:class:`FunctionAttribute` to a driver enum.

    The legacy :class:`FuncAttr` StrEnum is kept for existing call sites; new
    code uses the typed :class:`FunctionAttribute` IntEnum from
    :mod:`kerlever.benchmarker.types` (spec §6.11, design §6.1).
    """
    cuda_mod = _cuda()
    try:
        enum_cls = cuda_mod.CUfunction_attribute
    except AttributeError as exc:  # pragma: no cover
        raise CudaDriverError(
            -1, "CU_FUNC_ATTR_MISSING", f"cannot resolve function attr enum: {exc}"
        ) from exc
    if isinstance(attr, FunctionAttribute):
        name = _FUNC_ATTR_ENUM_NAMES[attr]
        try:
            return getattr(enum_cls, name)
        except AttributeError as exc:  # pragma: no cover
            raise CudaDriverError(
                -1,
                "CU_FUNC_ATTR_MISSING",
                f"cuda-python enum missing {name}: {exc}",
            ) from exc
    if attr == FuncAttr.MAX_DYNAMIC_SHARED_SIZE_BYTES:
        return enum_cls.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    return enum_cls.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT


def set_function_attribute(
    fn: CudaFunction,
    attr: FuncAttr | FunctionAttribute,
    value: int,
) -> int:
    """Set a function attribute and read back the observed value.

    Wraps ``cuFuncSetAttribute`` + ``cuFuncGetAttribute`` so the envelope
    always records the driver-clamped value (REQ-BENCH-008, REQ-BENCH-029).

    Implements: REQ-BENCH-008, REQ-BENCH-029
    """
    cuda_mod = _cuda()
    enum_value = _funcattr_enum(attr)
    _check(
        cuda_mod.cuFuncSetAttribute(fn.handle, enum_value, value)
    )
    observed = _unpack_one(
        cuda_mod.cuFuncGetAttribute(enum_value, fn.handle)
    )
    return int(observed)


def get_function_attribute(
    fn: CudaFunction, attr: FuncAttr | FunctionAttribute
) -> int:
    """Return the current value of a function attribute via ``cuFuncGetAttribute``.

    Implements: REQ-BENCH-008, REQ-BENCH-029
    """
    cuda_mod = _cuda()
    enum_value = _funcattr_enum(attr)
    observed = _unpack_one(
        cuda_mod.cuFuncGetAttribute(enum_value, fn.handle)
    )
    return int(observed)


def _cache_config_enum(cfg: CacheConfig) -> Any:
    """Translate :class:`CacheConfig` to a driver API ``CUfunc_cache`` value."""
    cuda_mod = _cuda()
    enum_cls = cuda_mod.CUfunc_cache
    return {
        CacheConfig.PREFER_NONE: enum_cls.CU_FUNC_CACHE_PREFER_NONE,
        CacheConfig.PREFER_SHARED: enum_cls.CU_FUNC_CACHE_PREFER_SHARED,
        CacheConfig.PREFER_L1: enum_cls.CU_FUNC_CACHE_PREFER_L1,
        CacheConfig.PREFER_EQUAL: enum_cls.CU_FUNC_CACHE_PREFER_EQUAL,
    }[cfg]


def set_cache_config(fn: CudaFunction, cfg: CacheConfig) -> None:
    """Apply a L1/shared cache preference via ``cuFuncSetCacheConfig``."""
    cuda_mod = _cuda()
    _check(
        cuda_mod.cuFuncSetCacheConfig(fn.handle, _cache_config_enum(cfg))
    )


def create_event() -> CudaEvent:
    """Create a CUDA event for timing.

    Implements: REQ-BENCH-006
    """
    cuda_mod = _cuda()
    handle = _unpack_one(cuda_mod.cuEventCreate(0))
    return CudaEvent(handle=handle)


def destroy_event(event: CudaEvent) -> None:
    """Destroy a CUDA event."""
    cuda_mod = _cuda()
    try:
        _check(cuda_mod.cuEventDestroy(event.handle))
    except CudaDriverError as exc:
        logger.warning("cuda.event_destroy.failed", extra={"error": str(exc)})


def create_stream() -> CudaStream:
    """Create a non-blocking stream for benchmark kernel launches."""
    cuda_mod = _cuda()
    handle = _unpack_one(cuda_mod.cuStreamCreate(0))
    return CudaStream(handle=handle)


def destroy_stream(stream: CudaStream) -> None:
    """Destroy a CUDA stream."""
    cuda_mod = _cuda()
    try:
        _check(cuda_mod.cuStreamDestroy(stream.handle))
    except CudaDriverError as exc:
        logger.warning("cuda.stream_destroy.failed", extra={"error": str(exc)})


def event_record(event: CudaEvent, stream: CudaStream) -> None:
    """Record an event on a stream via ``cuEventRecord``.

    Implements: REQ-BENCH-006
    """
    cuda_mod = _cuda()
    _check(cuda_mod.cuEventRecord(event.handle, stream.handle))


def event_synchronize(event: CudaEvent) -> None:
    """Block until ``event`` has completed."""
    cuda_mod = _cuda()
    _check(cuda_mod.cuEventSynchronize(event.handle))


def event_elapsed_ms(start: CudaEvent, stop: CudaEvent) -> float:
    """Return ``cuEventElapsedTime(stop, start)`` in milliseconds.

    Caller is responsible for synchronizing ``stop`` first.

    Implements: REQ-BENCH-006
    """
    cuda_mod = _cuda()
    event_synchronize(stop)
    ms = _unpack_one(
        cuda_mod.cuEventElapsedTime(start.handle, stop.handle)
    )
    return float(ms)


def stream_synchronize(stream: CudaStream) -> None:
    """Block until all outstanding work on ``stream`` has completed.

    Implements: REQ-BENCH-006
    """
    cuda_mod = _cuda()
    _check(cuda_mod.cuStreamSynchronize(stream.handle))


def mem_alloc(nbytes: int) -> DevicePtr:
    """Allocate ``nbytes`` of device memory via ``cuMemAlloc``.

    Returns:
        An opaque ``DevicePtr`` (int). Free with :func:`mem_free`.
    """
    if nbytes < 0:
        raise CudaDriverError(
            -1, "CUDA_ERROR_INVALID_VALUE", f"nbytes must be >= 0 (got {nbytes})"
        )
    cuda_mod = _cuda()
    handle = _unpack_one(cuda_mod.cuMemAlloc(nbytes))
    # cuda-python returns a CUdeviceptr wrapper; coerce to int for the NewType.
    return DevicePtr(int(handle))


def mem_free(ptr: DevicePtr) -> None:
    """Free a device pointer allocated by :func:`mem_alloc`."""
    cuda_mod = _cuda()
    try:
        _check(cuda_mod.cuMemFree(int(ptr)))
    except CudaDriverError as exc:
        # Teardown path — log and swallow so the caller's finally block does
        # not mask a primary exception. Matches destroy_stream / destroy_event.
        logger.warning("cuda.mem_free.failed", extra={"error": str(exc)})


def memcpy_htod(ptr: DevicePtr, host_bytes: bytes) -> None:
    """Copy a host bytes buffer into device memory at ``ptr``.

    Uses the synchronous ``cuMemcpyHtoD`` because adapter seeding happens
    outside any timed region (spec §6.3).
    """
    cuda_mod = _cuda()
    _check(
        cuda_mod.cuMemcpyHtoD(int(ptr), host_bytes, len(host_bytes))
    )


def memcpy_dtoh(ptr: DevicePtr, nbytes: int) -> bytes:
    """Copy ``nbytes`` from device memory at ``ptr`` into a fresh host buffer."""
    cuda_mod = _cuda()
    buf = bytearray(nbytes)
    _check(cuda_mod.cuMemcpyDtoH(buf, int(ptr), nbytes))
    return bytes(buf)


def launch(
    fn: CudaFunction,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    smem: int,
    stream: CudaStream,
    args: tuple[Any, ...],
) -> None:
    """Launch a kernel via ``cuLaunchKernel``.

    ``args`` is passed straight to cuda-python; the adapter layer is
    responsible for producing a well-typed argument tuple matching the
    kernel's ABI. We do not introspect argument types here.

    Implements: REQ-BENCH-006
    """
    cuda_mod = _cuda()
    gx, gy, gz = grid
    bx, by, bz = block
    _check(
        cuda_mod.cuLaunchKernel(
            fn.handle,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            smem,
            stream.handle,
            args,
            0,
        )
    )


__all__ = [
    "CacheConfig",
    "CudaContext",
    "CudaDriverError",
    "CudaEvent",
    "CudaFunction",
    "CudaModule",
    "CudaStream",
    "DevicePtr",
    "FuncAttr",
    "FunctionAttribute",
    "ModuleLoadOptions",
    "create_event",
    "create_primary_context",
    "create_stream",
    "destroy_event",
    "destroy_primary_context",
    "destroy_stream",
    "event_elapsed_ms",
    "event_record",
    "event_synchronize",
    "get_function",
    "get_function_attribute",
    "init",
    "launch",
    "load_module",
    "mem_alloc",
    "mem_free",
    "memcpy_dtoh",
    "memcpy_htod",
    "set_cache_config",
    "set_function_attribute",
    "stream_synchronize",
]
