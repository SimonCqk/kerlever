"""Benchmarker — deployment configuration (env-backed).

Anything that changes **measurement semantics** comes from the request
(spec §9.1). Anything that is a **pod-level deployment choice** comes from
this module (spec §9.2). The split exists so envelope hashes stay complete
and reproducible across pods.

Spec: docs/benchmarker/spec.md §9
Design: docs/benchmarker/design.md §9
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _env_str(name: str, default: str) -> str:
    """Read an env var as a plain string with default."""
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    """Read an env var as int; fall back to default on missing/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read an env var as float; fall back to default on missing/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read an env var as bool (1/true/yes); fall back to default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HygieneThresholds:
    """Numeric thresholds for hygiene and noise evaluation (spec §6.8)."""

    noise_floor_pct: float = 0.01
    measurement_cv_warn_pct: float = 2.0
    measurement_cv_fail_pct: float = 5.0
    p95_p50_ratio_warn: float = 1.5
    anchor_drift_warn_pct: float = 0.02
    anchor_drift_fail_pct: float = 0.05
    thermal_steady_state_limit_c: float = 80.0


@dataclass(frozen=True)
class CalibrationConfig:
    """Fast-benchmark warmup / repetition / calibration tunables (spec §6.8)."""

    warmup_min_runs: int = 5
    min_timed_batch_ms: float = 1.0
    max_timed_batch_ms: float = 200.0
    repetitions: int = 30
    max_iterations_per_sample: int = 1024
    min_p95_samples: int = 20
    anchor_every_n_samples: int = 4
    max_interleave_block_len: int = 6
    bench_rerun_limit: int = 1


@dataclass(frozen=True)
class ClockPolicyConfig:
    """Clock-lock deployment policy (spec §6.2, §9.2)."""

    lock_mode: str = "disabled"  # "disabled" or "enabled_when_privileged"


@dataclass(frozen=True)
class ArtifactStoreConfig:
    """Pod-local artifact-store configuration (spec §6.8 ARTIFACT_RETENTION)."""

    root: Path = Path("/var/lib/kerlever/bench")
    retention_seconds: int = 24 * 60 * 60


@dataclass(frozen=True)
class ProfilerConfig:
    """NCU / NSYS binary locations and policy (spec §6.6, §9.2)."""

    ncu_bin: str = "/usr/local/cuda/bin/ncu"
    nsys_bin: str = "/usr/local/cuda/bin/nsys"
    ncu_profile_set: str = "focused"
    profile_timeout_s: float = 300.0
    include_incumbent: bool = True


@dataclass(frozen=True)
class LeaseConfig:
    """Lease-manager deployment config."""

    # Reserved for future lease tuning; currently no user-visible knobs.
    acquire_timeout_s: float = 60.0


@dataclass(frozen=True)
class SupervisorConfig:
    """Supervisor-level timeouts and limits (spec §6.8, §9.2)."""

    batch_timeout_s: float = 1800.0
    shutdown_grace_s: float = 30.0
    ambiguous_failure_limit: int = 3
    kernel_timeout_ms: int = 10000


@dataclass(frozen=True)
class BenchmarkerConfig:
    """Top-level Benchmarker service configuration.

    Implements: REQ-BENCH-023, REQ-BENCH-027
    """

    bind_host: str = "0.0.0.0"
    bind_port: int = 8080
    log_level: str = "INFO"
    pod_id: str = "pod-local"
    service_version: str = "0.1.0"
    build_hash: str | None = None
    thresholds: HygieneThresholds = field(default_factory=HygieneThresholds)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    clock_policy: ClockPolicyConfig = field(default_factory=ClockPolicyConfig)
    artifact: ArtifactStoreConfig = field(default_factory=ArtifactStoreConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    lease: LeaseConfig = field(default_factory=LeaseConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    supported_adapter_abis: tuple[str, ...] = ()
    # Comma-separated Python import paths loaded at service startup so their
    # import-time side effect registers out-of-tree adapters (design §9 /
    # spec §6.11 REQ-BENCH-028). V1 built-ins always auto-register; this list
    # only adds custom adapters.
    adapter_registry_modules: tuple[str, ...] = ()

    @staticmethod
    def from_env() -> BenchmarkerConfig:
        """Load a ``BenchmarkerConfig`` from environment variables.

        Missing env vars fall back to spec §6.8 defaults.

        Implements: REQ-BENCH-023
        """
        pod_id = _env_str("KERLEVER_BENCH_POD_ID", os.uname().nodename)
        thresholds = HygieneThresholds(
            noise_floor_pct=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_NOISE_FLOOR_PCT", 0.01
            ),
            measurement_cv_warn_pct=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_MEASUREMENT_CV_WARN_PCT", 2.0
            ),
            measurement_cv_fail_pct=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_MEASUREMENT_CV_FAIL_PCT", 5.0
            ),
            p95_p50_ratio_warn=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_P95_P50_RATIO_WARN", 1.5
            ),
            anchor_drift_warn_pct=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_ANCHOR_DRIFT_WARN_PCT", 0.02
            ),
            anchor_drift_fail_pct=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_ANCHOR_DRIFT_FAIL_PCT", 0.05
            ),
            thermal_steady_state_limit_c=_env_float(
                "KERLEVER_BENCH_THRESHOLDS_THERMAL_STEADY_STATE_LIMIT", 80.0
            ),
        )
        calibration = CalibrationConfig(
            warmup_min_runs=_env_int("KERLEVER_BENCH_WARMUP_MIN_RUNS", 5),
            min_timed_batch_ms=_env_float("KERLEVER_BENCH_MIN_TIMED_BATCH_MS", 1.0),
            max_timed_batch_ms=_env_float("KERLEVER_BENCH_MAX_TIMED_BATCH_MS", 200.0),
            repetitions=_env_int("KERLEVER_BENCH_REPETITIONS", 30),
            max_iterations_per_sample=_env_int(
                "KERLEVER_BENCH_MAX_ITERATIONS_PER_SAMPLE", 1024
            ),
            min_p95_samples=_env_int("KERLEVER_BENCH_MIN_P95_SAMPLES", 20),
            anchor_every_n_samples=_env_int(
                "KERLEVER_BENCH_ANCHOR_EVERY_N_SAMPLES", 4
            ),
            max_interleave_block_len=_env_int(
                "KERLEVER_BENCH_MAX_INTERLEAVE_BLOCK_LEN", 6
            ),
            bench_rerun_limit=_env_int("KERLEVER_BENCH_RERUN_LIMIT", 1),
        )
        clock_policy = ClockPolicyConfig(
            lock_mode=_env_str("KERLEVER_BENCH_CLOCK_LOCK_POLICY", "disabled"),
        )
        artifact = ArtifactStoreConfig(
            root=Path(
                _env_str("KERLEVER_BENCH_ARTIFACT_ROOT", "/var/lib/kerlever/bench")
            ),
            retention_seconds=_env_int(
                "KERLEVER_BENCH_ARTIFACT_RETENTION_S", 24 * 60 * 60
            ),
        )
        profiler = ProfilerConfig(
            ncu_bin=_env_str("KERLEVER_BENCH_NCU_BIN", "/usr/local/cuda/bin/ncu"),
            nsys_bin=_env_str("KERLEVER_BENCH_NSYS_BIN", "/usr/local/cuda/bin/nsys"),
            ncu_profile_set=_env_str("KERLEVER_BENCH_NCU_PROFILE_SET", "focused"),
            profile_timeout_s=_env_float("KERLEVER_BENCH_PROFILE_TIMEOUT_S", 300.0),
            include_incumbent=_env_bool(
                "KERLEVER_BENCH_INCLUDE_INCUMBENT_PROFILE", True
            ),
        )
        lease = LeaseConfig(
            acquire_timeout_s=_env_float("KERLEVER_BENCH_LEASE_TIMEOUT_S", 60.0),
        )
        supervisor = SupervisorConfig(
            batch_timeout_s=_env_float("KERLEVER_BENCH_BATCH_TIMEOUT_S", 1800.0),
            shutdown_grace_s=_env_float("KERLEVER_BENCH_SHUTDOWN_GRACE_S", 30.0),
            ambiguous_failure_limit=_env_int(
                "KERLEVER_BENCH_AMBIGUOUS_FAILURE_LIMIT", 3
            ),
            kernel_timeout_ms=_env_int("KERLEVER_BENCH_KERNEL_TIMEOUT_MS", 10000),
        )
        adapters_raw = _env_str("KERLEVER_BENCH_SUPPORTED_ADAPTER_ABIS", "")
        adapters = tuple(a.strip() for a in adapters_raw.split(",") if a.strip())
        adapter_mods_raw = _env_str("KERLEVER_BENCH_ADAPTER_REGISTRY_MODULES", "")
        adapter_mods = tuple(
            m.strip() for m in adapter_mods_raw.split(",") if m.strip()
        )
        return BenchmarkerConfig(
            bind_host=_env_str("KERLEVER_BENCH_BIND_HOST", "0.0.0.0"),
            bind_port=_env_int("KERLEVER_BENCH_BIND_PORT", 8080),
            log_level=_env_str("KERLEVER_BENCH_LOG_LEVEL", "INFO"),
            pod_id=pod_id,
            service_version=_env_str("KERLEVER_BENCH_SERVICE_VERSION", "0.1.0"),
            build_hash=os.environ.get("KERLEVER_BENCH_BUILD_HASH"),
            thresholds=thresholds,
            calibration=calibration,
            clock_policy=clock_policy,
            artifact=artifact,
            profiler=profiler,
            lease=lease,
            supervisor=supervisor,
            supported_adapter_abis=adapters,
            adapter_registry_modules=adapter_mods,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON handoff to the worker."""
        return {
            "bind_host": self.bind_host,
            "bind_port": self.bind_port,
            "log_level": self.log_level,
            "pod_id": self.pod_id,
            "service_version": self.service_version,
            "build_hash": self.build_hash,
            "thresholds": {
                "noise_floor_pct": self.thresholds.noise_floor_pct,
                "measurement_cv_warn_pct": self.thresholds.measurement_cv_warn_pct,
                "measurement_cv_fail_pct": self.thresholds.measurement_cv_fail_pct,
                "p95_p50_ratio_warn": self.thresholds.p95_p50_ratio_warn,
                "anchor_drift_warn_pct": self.thresholds.anchor_drift_warn_pct,
                "anchor_drift_fail_pct": self.thresholds.anchor_drift_fail_pct,
                "thermal_steady_state_limit_c": (
                    self.thresholds.thermal_steady_state_limit_c
                ),
            },
            "calibration": {
                "warmup_min_runs": self.calibration.warmup_min_runs,
                "min_timed_batch_ms": self.calibration.min_timed_batch_ms,
                "max_timed_batch_ms": self.calibration.max_timed_batch_ms,
                "repetitions": self.calibration.repetitions,
                "max_iterations_per_sample": self.calibration.max_iterations_per_sample,
                "min_p95_samples": self.calibration.min_p95_samples,
                "anchor_every_n_samples": self.calibration.anchor_every_n_samples,
                "max_interleave_block_len": self.calibration.max_interleave_block_len,
                "bench_rerun_limit": self.calibration.bench_rerun_limit,
            },
            "clock_policy": {"lock_mode": self.clock_policy.lock_mode},
            "artifact": {
                "root": str(self.artifact.root),
                "retention_seconds": self.artifact.retention_seconds,
            },
            "profiler": {
                "ncu_bin": self.profiler.ncu_bin,
                "nsys_bin": self.profiler.nsys_bin,
                "ncu_profile_set": self.profiler.ncu_profile_set,
                "profile_timeout_s": self.profiler.profile_timeout_s,
                "include_incumbent": self.profiler.include_incumbent,
            },
            "lease": {"acquire_timeout_s": self.lease.acquire_timeout_s},
            "supervisor": {
                "batch_timeout_s": self.supervisor.batch_timeout_s,
                "shutdown_grace_s": self.supervisor.shutdown_grace_s,
                "ambiguous_failure_limit": self.supervisor.ambiguous_failure_limit,
                "kernel_timeout_ms": self.supervisor.kernel_timeout_ms,
            },
            "supported_adapter_abis": list(self.supported_adapter_abis),
            "adapter_registry_modules": list(self.adapter_registry_modules),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BenchmarkerConfig:
        """Inverse of :meth:`to_dict`; used by the worker to rehydrate."""
        thresholds_raw = dict(data.get("thresholds", {}))
        calibration_raw = dict(data.get("calibration", {}))
        clock_raw = dict(data.get("clock_policy", {}))
        artifact_raw = dict(data.get("artifact", {}))
        profiler_raw = dict(data.get("profiler", {}))
        lease_raw = dict(data.get("lease", {}))
        sup_raw = dict(data.get("supervisor", {}))
        return BenchmarkerConfig(
            bind_host=str(data.get("bind_host", "0.0.0.0")),
            bind_port=int(data.get("bind_port", 8080)),
            log_level=str(data.get("log_level", "INFO")),
            pod_id=str(data.get("pod_id", "pod-local")),
            service_version=str(data.get("service_version", "0.1.0")),
            build_hash=data.get("build_hash"),
            thresholds=HygieneThresholds(
                noise_floor_pct=float(thresholds_raw.get("noise_floor_pct", 0.01)),
                measurement_cv_warn_pct=float(
                    thresholds_raw.get("measurement_cv_warn_pct", 2.0)
                ),
                measurement_cv_fail_pct=float(
                    thresholds_raw.get("measurement_cv_fail_pct", 5.0)
                ),
                p95_p50_ratio_warn=float(
                    thresholds_raw.get("p95_p50_ratio_warn", 1.5)
                ),
                anchor_drift_warn_pct=float(
                    thresholds_raw.get("anchor_drift_warn_pct", 0.02)
                ),
                anchor_drift_fail_pct=float(
                    thresholds_raw.get("anchor_drift_fail_pct", 0.05)
                ),
                thermal_steady_state_limit_c=float(
                    thresholds_raw.get("thermal_steady_state_limit_c", 80.0)
                ),
            ),
            calibration=CalibrationConfig(
                warmup_min_runs=int(calibration_raw.get("warmup_min_runs", 5)),
                min_timed_batch_ms=float(
                    calibration_raw.get("min_timed_batch_ms", 1.0)
                ),
                max_timed_batch_ms=float(
                    calibration_raw.get("max_timed_batch_ms", 200.0)
                ),
                repetitions=int(calibration_raw.get("repetitions", 30)),
                max_iterations_per_sample=int(
                    calibration_raw.get("max_iterations_per_sample", 1024)
                ),
                min_p95_samples=int(calibration_raw.get("min_p95_samples", 20)),
                anchor_every_n_samples=int(
                    calibration_raw.get("anchor_every_n_samples", 4)
                ),
                max_interleave_block_len=int(
                    calibration_raw.get("max_interleave_block_len", 6)
                ),
                bench_rerun_limit=int(
                    calibration_raw.get("bench_rerun_limit", 1)
                ),
            ),
            clock_policy=ClockPolicyConfig(
                lock_mode=str(clock_raw.get("lock_mode", "disabled")),
            ),
            artifact=ArtifactStoreConfig(
                root=Path(
                    str(artifact_raw.get("root", "/var/lib/kerlever/bench"))
                ),
                retention_seconds=int(
                    artifact_raw.get("retention_seconds", 24 * 60 * 60)
                ),
            ),
            profiler=ProfilerConfig(
                ncu_bin=str(profiler_raw.get("ncu_bin", "/usr/local/cuda/bin/ncu")),
                nsys_bin=str(profiler_raw.get("nsys_bin", "/usr/local/cuda/bin/nsys")),
                ncu_profile_set=str(profiler_raw.get("ncu_profile_set", "focused")),
                profile_timeout_s=float(profiler_raw.get("profile_timeout_s", 300.0)),
                include_incumbent=bool(profiler_raw.get("include_incumbent", True)),
            ),
            lease=LeaseConfig(
                acquire_timeout_s=float(lease_raw.get("acquire_timeout_s", 60.0)),
            ),
            supervisor=SupervisorConfig(
                batch_timeout_s=float(sup_raw.get("batch_timeout_s", 1800.0)),
                shutdown_grace_s=float(sup_raw.get("shutdown_grace_s", 30.0)),
                ambiguous_failure_limit=int(
                    sup_raw.get("ambiguous_failure_limit", 3)
                ),
                kernel_timeout_ms=int(sup_raw.get("kernel_timeout_ms", 10000)),
            ),
            supported_adapter_abis=tuple(data.get("supported_adapter_abis", ())),
            adapter_registry_modules=tuple(
                data.get("adapter_registry_modules", ())
            ),
        )


__all__ = [
    "ArtifactStoreConfig",
    "BenchmarkerConfig",
    "CalibrationConfig",
    "ClockPolicyConfig",
    "HygieneThresholds",
    "LeaseConfig",
    "ProfilerConfig",
    "SupervisorConfig",
]
