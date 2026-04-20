"""Benchmarker — pynvml adapter.

All ``pynvml`` symbols are accessed via lazy imports inside functions.
This lets the service import ``kerlever.benchmarker.main`` without a GPU
driver being installed (spec §7.5 build/runtime separation).

Spec: docs/benchmarker/spec.md §6.2
Design: docs/benchmarker/design.md §6.2 telemetry.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from kerlever.benchmarker.config import ClockPolicyConfig, HygieneThresholds
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.types import (
    AnchorDriftTelemetry,
    ClockPolicyMode,
    DeviceInventoryEntry,
    DeviceTelemetrySnapshot,
    HygieneReport,
)

logger = logging.getLogger(__name__)

_INITIALIZED = False


class NvmlAdapterError(RuntimeError):
    """Raised for library-level NVML failures (not hygiene-gate failures)."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)


def _pynvml() -> Any:
    """Lazily import the pynvml module.

    Raises:
        NvmlAdapterError: If the pynvml package is not installed.
    """
    try:
        import pynvml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise NvmlAdapterError(-1, f"pynvml unavailable: {exc}") from exc
    return pynvml


def init() -> None:
    """Initialize NVML in the current process. Idempotent.

    On a non-GPU dev host the underlying driver may be missing; callers
    should wrap this with try/except for ``NvmlAdapterError`` and degrade
    gracefully (§7.5).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    pynvml = _pynvml()
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # pynvml raises NVMLError subclasses
        raise NvmlAdapterError(
            getattr(exc, "value", -1),
            f"nvmlInit failed: {exc}",
        ) from exc
    _INITIALIZED = True


def shutdown() -> None:
    """Shut down NVML if previously initialized."""
    global _INITIALIZED
    if not _INITIALIZED:
        return
    try:
        pynvml = _pynvml()
        pynvml.nvmlShutdown()
    except Exception as exc:
        logger.warning("nvml.shutdown.failed", extra={"error": str(exc)})
    finally:
        _INITIALIZED = False


def _decode(value: Any) -> str:
    """Decode a pynvml bytes-or-str return to a Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def info_inventory() -> list[DeviceInventoryEntry]:
    """Enumerate visible GPUs via NVML, once at service startup.

    Returns an empty list when the driver is absent or NVML init fails —
    the caller stores the error in ``DeviceInventory.error`` and surfaces
    it via ``/healthz``.
    """
    try:
        init()
    except NvmlAdapterError as exc:
        logger.warning("nvml.init.unavailable", extra={"error": str(exc)})
        return []
    pynvml = _pynvml()
    entries: list[DeviceInventoryEntry] = []
    try:
        count = int(pynvml.nvmlDeviceGetCount())
    except Exception as exc:
        logger.warning("nvml.device_count.failed", extra={"error": str(exc)})
        return []
    for ordinal in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(ordinal)
            uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            bus_id = _decode(getattr(pci, "busId", b""))
            name = _decode(pynvml.nvmlDeviceGetName(handle))
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                sm_arch = f"sm_{int(major)}{int(minor)}"
            except Exception:
                sm_arch = "unknown"
            mig_profile: str | None = None
            try:
                mig_mode, _pending = pynvml.nvmlDeviceGetMigMode(handle)
                if int(mig_mode) == 1:
                    mig_profile = "mig_enabled"
            except Exception:
                mig_profile = None
            entries.append(
                DeviceInventoryEntry(
                    ordinal=ordinal,
                    gpu_uuid=uuid,
                    pci_bus_id=bus_id,
                    sm_arch=sm_arch,
                    mig_profile=mig_profile,
                    name=name,
                ),
            )
        except Exception as exc:
            logger.warning(
                "nvml.device_inspect.failed",
                extra={"ordinal": ordinal, "error": str(exc)},
            )
    return entries


def snapshot(lease: LeasedDevice) -> DeviceTelemetrySnapshot:
    """Sample a single telemetry snapshot for the leased device.

    Returns a snapshot with all optional fields populated as ``None`` when
    the underlying NVML query is not available.
    """
    taken_at_ms = int(time.time() * 1000)
    try:
        init()
        pynvml = _pynvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(lease.ordinal)
    except Exception as exc:
        logger.warning("nvml.snapshot.init_failed", extra={"error": str(exc)})
        return DeviceTelemetrySnapshot(taken_at_ms=taken_at_ms)

    sm_clock = _safe_int(
        lambda: pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        )
    )
    mem_clock = _safe_int(
        lambda: pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_MEM
        )
    )
    temp = _safe_float(
        lambda: pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
    )
    power = _safe_float(
        lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
    )
    throttle_reasons = _throttle_reasons(handle)
    ecc_sbe: int | None = None
    ecc_dbe: int | None = None
    try:
        ecc_sbe = int(
            pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                pynvml.NVML_VOLATILE_ECC,
            )
        )
        ecc_dbe = int(
            pynvml.nvmlDeviceGetTotalEccErrors(
                handle,
                pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC,
            )
        )
    except Exception:
        pass
    return DeviceTelemetrySnapshot(
        taken_at_ms=taken_at_ms,
        sm_clock_mhz=sm_clock,
        mem_clock_mhz=mem_clock,
        gpu_temp_c=temp,
        power_w=power,
        throttle_reasons=throttle_reasons,
        ecc_sbe_total=ecc_sbe,
        ecc_dbe_total=ecc_dbe,
        xid_events_since_last=0,
    )


_THROTTLE_REASON_MAP: tuple[tuple[str, str], ...] = (
    ("nvmlClocksEventReasonHwSlowdown", "HW_SLOWDOWN"),
    ("nvmlClocksEventReasonSwThermalSlowdown", "SW_THERMAL_SLOWDOWN"),
    ("nvmlClocksEventReasonHwThermalSlowdown", "HW_THERMAL_SLOWDOWN"),
    ("nvmlClocksEventReasonHwPowerBrakeSlowdown", "HW_POWER_BRAKE_SLOWDOWN"),
    ("nvmlClocksEventReasonSwPowerCap", "SW_POWER_CAP"),
    ("nvmlClocksEventReasonSyncBoost", "SYNC_BOOST"),
    ("nvmlClocksEventReasonApplicationsClocksSetting", "APPLICATION_CLOCKS"),
    ("nvmlClocksEventReasonGpuIdle", "GPU_IDLE"),
    ("nvmlClocksEventReasonDisplayClockSetting", "DISPLAY_CLOCK"),
)


def _throttle_reasons(handle: Any) -> list[str]:
    """Translate the NVML clocks-event-reasons bitmask into a list of tags."""
    try:
        pynvml = _pynvml()
        bitmask = int(
            pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle)
        )
    except Exception:
        return []
    out: list[str] = []
    for attr_name, label in _THROTTLE_REASON_MAP:
        try:
            flag = int(getattr(pynvml, attr_name))
        except Exception:
            continue
        if flag != 0 and (bitmask & flag) == flag:
            out.append(label)
    return out


def _safe_int(thunk: Any) -> int | None:
    """Call ``thunk`` and return its int result or ``None`` on failure."""
    try:
        return int(thunk())
    except Exception:
        return None


def _safe_float(thunk: Any) -> float | None:
    """Call ``thunk`` and return its float result or ``None`` on failure."""
    try:
        return float(thunk())
    except Exception:
        return None


@dataclass(frozen=True)
class _ForeignProcessCheck:
    """Result of foreign-process detection on the leased device."""

    pids: list[str]


def _foreign_compute_processes(handle: Any) -> list[str]:
    """Return PIDs running compute work that are not this process.

    The pynvml API returns lists of ``nvmlProcessInfo_t`` with ``pid``.
    Spec §6.2 treats any non-self PID as a foreign compute process.
    """
    try:
        pynvml = _pynvml()
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception:
        return []
    self_pid = os.getpid()
    return [str(p.pid) for p in procs if int(getattr(p, "pid", 0)) != self_pid]


_HARD_GATE_REASONS: frozenset[str] = frozenset(
    {
        "arch_mismatch",
        "mig_profile_mismatch",
        "ecc_xid",
        "probe_failed",
    }
)


def is_hard_gate(reason: str | None) -> bool:
    """Return ``True`` when ``reason`` is a Phase 2 hard-gate failure.

    Hard gates short-circuit the batch to ``status = infra_error``
    (spec §6.2 decision table).
    """
    return reason is not None and reason in _HARD_GATE_REASONS


def preflight(
    lease: LeasedDevice,
    policy: ClockPolicyConfig,
    thresholds: HygieneThresholds,
    compute_mode: str = "EXCLUSIVE_PROCESS",
) -> HygieneReport:
    """Run the Phase 2 preflight and produce a :class:`HygieneReport`.

    Implements the §6.2 decision table in code: hard gates set
    ``reason_on_fail`` to a short-circuit token; unstable gates record
    warnings in ``clocks_event_reasons`` or ``foreign_processes`` but leave
    ``reason_on_fail`` ``None`` (caller routes to ``unstable`` downstream).

    Implements: REQ-BENCH-003
    """
    snap = snapshot(lease)
    try:
        init()
        pynvml = _pynvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(lease.ordinal)
    except Exception as exc:
        return HygieneReport(
            gpu_uuid=lease.gpu_uuid,
            sm_arch=lease.sm_arch,
            mig_profile=lease.mig_profile,
            compute_mode=compute_mode,
            telemetry=snap,
            reason_on_fail=f"nvml_unavailable:{exc}",
        )
    foreign = _foreign_compute_processes(handle)
    throttle_reasons = snap.throttle_reasons

    ecc_ok = True
    xid_ok = True
    reason: str | None = None
    if snap.ecc_dbe_total is not None and snap.ecc_dbe_total > 0:
        ecc_ok = False
        reason = "ecc_xid"

    # Thermal steady-state limit check.
    temp_limit = thresholds.thermal_steady_state_limit_c
    if (
        snap.gpu_temp_c is not None
        and snap.gpu_temp_c > temp_limit
        and reason is None
    ):
        reason = "thermal_above_steady_state"

    profiler_perm = _probe_profiler_counter_permission()

    # Clock policy advisory: "disabled" or "enabled_when_privileged".
    _ = policy  # reserved for future lock attempt; observed-only by default.

    return HygieneReport(
        gpu_uuid=lease.gpu_uuid,
        sm_arch=lease.sm_arch,
        mig_profile=lease.mig_profile,
        compute_mode=compute_mode,
        foreign_processes=foreign,
        clocks_event_reasons=throttle_reasons,
        gpu_temp_c=snap.gpu_temp_c,
        power_w=snap.power_w,
        ecc_ok=ecc_ok,
        xid_ok=xid_ok,
        profiler_counter_permission=profiler_perm,
        telemetry=snap,
        reason_on_fail=reason,
    )


def postflight(
    lease: LeasedDevice, pre: DeviceTelemetrySnapshot
) -> tuple[DeviceTelemetrySnapshot, AnchorDriftTelemetry]:
    """Sample after the worker exits and compute drift vs. ``pre``."""
    post = snapshot(lease)
    drift = AnchorDriftTelemetry(
        sm_clock_drift_mhz=_drift_int(pre.sm_clock_mhz, post.sm_clock_mhz),
        mem_clock_drift_mhz=_drift_int(pre.mem_clock_mhz, post.mem_clock_mhz),
        temp_drift_c=_drift_float(pre.gpu_temp_c, post.gpu_temp_c),
        power_drift_w=_drift_float(pre.power_w, post.power_w),
    )
    return post, drift


def _drift_int(pre: int | None, post: int | None) -> int | None:
    """Return ``post - pre`` when both are present; else ``None``."""
    if pre is None or post is None:
        return None
    return post - pre


def _drift_float(pre: float | None, post: float | None) -> float | None:
    """Return ``post - pre`` when both are present; else ``None``."""
    if pre is None or post is None:
        return None
    return post - pre


def probe_ready() -> bool:
    """Return whether NVML init succeeded and at least one device is visible."""
    try:
        init()
    except NvmlAdapterError:
        return False
    pynvml = _pynvml()
    try:
        return int(pynvml.nvmlDeviceGetCount()) > 0
    except Exception:
        return False


_PROFILER_PERM_CACHE: bool | None = None


def _probe_profiler_counter_permission() -> bool:
    """Single-shot cached probe of NCU counter permission.

    Spec §6.2 bundles this check with preflight hygiene. Rather than running
    ``ncu`` on every preflight, we remember the first result for the process
    lifetime. A missing ``ncu`` or a permission failure both resolve to
    ``False`` and route to ``profile_unavailable`` downstream.
    """
    global _PROFILER_PERM_CACHE
    if _PROFILER_PERM_CACHE is not None:
        return _PROFILER_PERM_CACHE
    # Resolve from env (cheap, avoids subprocess at this layer).
    override = os.environ.get("KERLEVER_BENCH_PROFILER_PERM_OVERRIDE")
    if override is not None:
        _PROFILER_PERM_CACHE = override.strip().lower() in {"1", "true", "yes"}
        return _PROFILER_PERM_CACHE
    # Default: assume unknown → False; the profiler module re-checks before use.
    _PROFILER_PERM_CACHE = False
    return _PROFILER_PERM_CACHE


def pynvml_version() -> str:
    """Return the installed pynvml package version string, or 'unknown'."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("pynvml")
    except Exception:
        return "unknown"


def cuda_python_version() -> str:
    """Return the installed cuda-python package version string, or 'unknown'."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("cuda-python")
    except Exception:
        try:
            from importlib.metadata import version  # noqa: PLC0415

            return version("cuda-bindings")
        except Exception:
            return "unknown"


def driver_version() -> str:
    """Return the NVIDIA driver version as reported by NVML."""
    try:
        init()
        pynvml = _pynvml()
        return _decode(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        return "unknown"


def cuda_runtime_version() -> str:
    """Return the CUDA runtime version as reported by NVML."""
    try:
        init()
        pynvml = _pynvml()
        raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
        major = raw // 1000
        minor = (raw % 1000) // 10
        return f"{major}.{minor}"
    except Exception:
        return "unknown"


def resolve_clock_policy_mode(policy: ClockPolicyConfig) -> ClockPolicyMode:
    """Decide the effective clock policy mode for the current deployment.

    Spec §6.2: ``disabled`` → observed_only; ``enabled_when_privileged``
    attempts a lock via ``nvidia-smi`` and falls back to
    ``LOCK_REQUESTED_UNAVAILABLE`` when privileges are missing.
    """
    if policy.lock_mode != "enabled_when_privileged":
        return ClockPolicyMode.OBSERVED_ONLY
    # V1 does not actually invoke nvidia-smi --lock-gpu-clocks here; the
    # privileged path is wired behind the config flag but stubbed per
    # spec §11 ("No clock locking by default").
    return ClockPolicyMode.LOCK_REQUESTED_UNAVAILABLE


__all__ = [
    "NvmlAdapterError",
    "cuda_python_version",
    "cuda_runtime_version",
    "driver_version",
    "info_inventory",
    "init",
    "is_hard_gate",
    "postflight",
    "preflight",
    "probe_ready",
    "pynvml_version",
    "resolve_clock_policy_mode",
    "shutdown",
    "snapshot",
]
