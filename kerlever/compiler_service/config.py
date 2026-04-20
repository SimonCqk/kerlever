"""Compiler Service configuration (env-driven singleton).

Spec: docs/compiler-service/spec.md §6.13
Design: docs/compiler-service/design.md §5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from kerlever.compiler_service.types import PinRole, SanitizerTool


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _env_timedelta_seconds(name: str, default: timedelta) -> timedelta:
    """Parse an integer-seconds env var into a ``timedelta``.

    Operators express the idempotency TTL in seconds for parity with
    other Kerlever knobs (benchmarker, navigator); parsing into
    ``timedelta`` here keeps the downstream API unchanged.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return timedelta(seconds=int(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be an int (seconds), got {raw!r}") from exc


def _default_pin_roots() -> frozenset[PinRole]:
    return frozenset(
        {
            PinRole.BASELINE,
            PinRole.INCUMBENT,
            PinRole.ACTIVE_BENCHMARK_BATCH,
            PinRole.ACTIVE_PROFILE_BATCH,
            PinRole.PROBE_KERNEL,
        }
    )


def _default_compile_flags() -> tuple[str, ...]:
    return ("-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v")


def _default_escalation_tools() -> tuple[SanitizerTool, ...]:
    return (SanitizerTool.RACECHECK, SanitizerTool.SYNCCHECK, SanitizerTool.INITCHECK)


@dataclass(frozen=True)
class ServiceConfig:
    """Immutable snapshot of service tunables.

    Every consumer reads only from this bag; values are resolved once at
    startup. Per-request overrides live on ``CompileRequest.limits``.

    Spec: §6.13
    """

    # External tool paths (resolved by the toolchain probe).
    nvcc_path: Path = field(default_factory=lambda: Path("/usr/local/cuda/bin/nvcc"))
    cuobjdump_path: Path = field(
        default_factory=lambda: Path("/usr/local/cuda/bin/cuobjdump")
    )
    sanitizer_path: Path = field(
        default_factory=lambda: Path("/usr/local/cuda/bin/compute-sanitizer")
    )
    nvidia_smi_path: Path = field(default_factory=lambda: Path("/usr/bin/nvidia-smi"))

    # Artifact store.
    kerlever_artifact_root: Path = field(
        default_factory=lambda: Path("/var/lib/kerlever/artifacts")
    )
    artifact_disk_high_watermark_pct: float = 85.0
    artifact_pin_roots: frozenset[PinRole] = field(default_factory=_default_pin_roots)
    max_artifact_bytes: int = 32 * 1024 * 1024  # 32 MiB ceiling for any single artifact

    # Timeouts (seconds).
    compile_timeout_s: float = 60.0
    correctness_timeout_s: float = 120.0
    sanitizer_timeout_s: float = 300.0
    cuobjdump_timeout_s: float = 30.0

    # Byte caps.
    max_source_bytes: int = 256 * 1024
    max_log_bytes: int = 64 * 1024

    # Concurrency.
    cpu_compile_concurrency: int = 4
    gpu_run_concurrency: int = 1  # per visible GPU

    # Sanitizer policy.
    sanitizer_default_tool: SanitizerTool = SanitizerTool.MEMCHECK
    sanitizer_escalation_tools: tuple[SanitizerTool, ...] = field(
        default_factory=_default_escalation_tools
    )

    # Compile flags applied to both reference and candidate (spec §6.4).
    default_compile_flags: tuple[str, ...] = field(
        default_factory=_default_compile_flags
    )

    # Pod health.
    ambiguous_failure_limit: int = 3
    pod_health_probe_path: Path = field(
        default_factory=lambda: (
            Path(__file__).parent / "reference_kernels" / "vec_add.cu"
        )
    )
    # Target arch used to compile the probe kernel at startup (spec §6.8,
    # INV-CS-012 extension: a pod that cannot compile its own probe is not
    # ready). Conservative default matches Ampere/A100.
    probe_target_arch: str = "sm_80"
    # Wall-clock bound for one probe invocation. The probe kernel is a tiny
    # vec_add; 10 seconds is generous and keeps the operator UX snappy.
    probe_timeout_s: float = 10.0

    # Idempotency.
    idempotency_ttl: timedelta = field(default_factory=lambda: timedelta(hours=24))

    # Pod identity.
    pod_id: str = "unknown-pod"

    # Service default tolerances (used as a last resort in tolerance
    # resolution, spec §6.6).
    service_default_float_tolerance: float = 1e-4
    service_default_int_tolerance: float = 0.0

    # Adapter version baked into artifact_key (design §Adapter surface).
    # The adapter itself also exposes its version; this is the fallback for
    # cases without an adapter-resolved request path.
    service_adapter_version: str = "v1"

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Build a config from process environment variables.

        Only documented knobs are read; unknown env vars are ignored.
        """
        defaults = cls()
        return cls(
            nvcc_path=_env_path("KERLEVER_NVCC_PATH", defaults.nvcc_path),
            cuobjdump_path=_env_path(
                "KERLEVER_CUOBJDUMP_PATH", defaults.cuobjdump_path
            ),
            sanitizer_path=_env_path(
                "KERLEVER_SANITIZER_PATH", defaults.sanitizer_path
            ),
            nvidia_smi_path=_env_path(
                "KERLEVER_NVIDIA_SMI_PATH", defaults.nvidia_smi_path
            ),
            kerlever_artifact_root=_env_path(
                "KERLEVER_ARTIFACT_ROOT", defaults.kerlever_artifact_root
            ),
            artifact_disk_high_watermark_pct=_env_float(
                "KERLEVER_ARTIFACT_DISK_HIGH_WATERMARK_PCT",
                defaults.artifact_disk_high_watermark_pct,
            ),
            compile_timeout_s=_env_float(
                "KERLEVER_COMPILE_TIMEOUT_S", defaults.compile_timeout_s
            ),
            correctness_timeout_s=_env_float(
                "KERLEVER_CORRECTNESS_TIMEOUT_S", defaults.correctness_timeout_s
            ),
            sanitizer_timeout_s=_env_float(
                "KERLEVER_SANITIZER_TIMEOUT_S", defaults.sanitizer_timeout_s
            ),
            max_source_bytes=_env_int(
                "KERLEVER_MAX_SOURCE_BYTES", defaults.max_source_bytes
            ),
            max_log_bytes=_env_int("KERLEVER_MAX_LOG_BYTES", defaults.max_log_bytes),
            cpu_compile_concurrency=_env_int(
                "KERLEVER_CPU_COMPILE_CONCURRENCY",
                defaults.cpu_compile_concurrency,
            ),
            gpu_run_concurrency=_env_int(
                "KERLEVER_GPU_RUN_CONCURRENCY", defaults.gpu_run_concurrency
            ),
            ambiguous_failure_limit=_env_int(
                "KERLEVER_AMBIGUOUS_FAILURE_LIMIT",
                defaults.ambiguous_failure_limit,
            ),
            probe_target_arch=_env_str(
                "KERLEVER_PROBE_TARGET_ARCH", defaults.probe_target_arch
            ),
            probe_timeout_s=_env_float(
                "KERLEVER_PROBE_TIMEOUT_S", defaults.probe_timeout_s
            ),
            idempotency_ttl=_env_timedelta_seconds(
                "KERLEVER_IDEMPOTENCY_TTL_SECONDS", defaults.idempotency_ttl
            ),
            max_artifact_bytes=_env_int(
                "KERLEVER_MAX_ARTIFACT_BYTES", defaults.max_artifact_bytes
            ),
            cuobjdump_timeout_s=_env_float(
                "KERLEVER_CUOBJDUMP_TIMEOUT_S", defaults.cuobjdump_timeout_s
            ),
            pod_id=_env_str("KERLEVER_POD_ID", defaults.pod_id),
        )
