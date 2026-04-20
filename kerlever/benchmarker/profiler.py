"""Benchmarker — Phase 6b: NCU subprocess adapter.

Wraps ``ncu`` as a subprocess, parses the JSON report into
:class:`RawProfileMetric` entries with mandatory provenance, and normalizes
into compact :class:`ProfileMetrics`. Missing metrics are ``null``, never
fabricated (INV-BENCH-009).

Spec: docs/benchmarker/spec.md §6.6
Design: docs/benchmarker/design.md §6.3 profiler.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from kerlever.benchmarker.config import ProfilerConfig
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    HygieneReport,
    NormalizedProfileMetricProvenance,
    ProfileMetrics,
    ProfilerName,
    ProfileUnavailableReason,
    RawProfileMetric,
    ReplayMode,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NcuRunResult:
    """Return of :func:`run_ncu`; callers inspect this instead of catching."""

    returncode: int
    stdout: str
    stderr: str
    report_path: Path | None
    timed_out: bool


def _as_text(value: object) -> str:
    """Decode bytes-or-str-or-None into a safe string for logs."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _build_cmdline(
    cfg: ProfilerConfig,
    target_cmd: list[str],
    nvtx_range: str,
    set_name: str,
    replay_mode: ReplayMode,
    report_out: Path,
) -> list[str]:
    """Assemble the full ``ncu`` argv tuple per spec §6.6.

    The NVTX filter pins the profiler to exactly one measured launch
    (INV-BENCH-008). ``--launch-count 1`` is redundant safety in case the
    NVTX range accidentally expands over multiple kernels.
    """
    return [
        cfg.ncu_bin,
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        nvtx_range,
        "--launch-count",
        "1",
        "--set",
        set_name,
        "--replay-mode",
        replay_mode.value,
        "--export",
        str(report_out),
        "--",
        *target_cmd,
    ]


def run_ncu(
    cfg: ProfilerConfig,
    target_cmd: list[str],
    nvtx_range: str,
    set_name: str,
    replay_mode: ReplayMode,
    report_out: Path,
    timeout_s: float,
) -> NcuRunResult:
    """Run ``ncu`` as a subprocess.

    Never raises for profiler-level failures; the caller inspects the
    :class:`NcuRunResult` and maps returncode / timed_out / missing-report
    to a typed :class:`ProfileUnavailableReason`.

    Implements: REQ-BENCH-017, REQ-BENCH-018
    """
    if shutil.which(cfg.ncu_bin) is None and not Path(cfg.ncu_bin).exists():
        return NcuRunResult(
            returncode=127,
            stdout="",
            stderr=f"ncu binary not found at {cfg.ncu_bin}",
            report_path=None,
            timed_out=False,
        )
    cmd = _build_cmdline(cfg, target_cmd, nvtx_range, set_name, replay_mode, report_out)
    logger.info(
        "profiler.ncu.invoke",
        extra={"cmd": cmd, "timeout_s": timeout_s, "report_out": str(report_out)},
    )
    try:
        completed = subprocess.run(  # noqa: S603 — argv list, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return NcuRunResult(
            returncode=-1,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            report_path=None,
            timed_out=True,
        )
    return NcuRunResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        report_path=report_out if report_out.exists() else None,
        timed_out=False,
    )


def _run_ncu_import(
    cfg: ProfilerConfig, report_path: Path, timeout_s: float
) -> tuple[int, str, str]:
    """Dump raw metrics as JSON via ``ncu --import --print-metrics-json``."""
    cmd = [
        cfg.ncu_bin,
        "--import",
        str(report_path),
        "--print-metrics-json",
    ]
    completed = subprocess.run(  # noqa: S603 — argv list
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def parse_report(
    cfg: ProfilerConfig,
    report_path: Path,
    architecture: str,
    profiler_version: str,
    timeout_s: float = 60.0,
) -> list[RawProfileMetric]:
    """Parse an ``.ncu-rep`` report into :class:`RawProfileMetric` entries.

    Returns an empty list on parse failure. Missing metric values are
    encoded as ``RawProfileMetric(value=None, ...)`` — they are never
    skipped or fabricated (INV-BENCH-009).

    The spec allows two NCU JSON shapes: either a top-level list of
    ``{name, value, unit, section}`` objects, or a nested
    ``{metrics: [...]}`` object. Both are handled.

    Implements: REQ-BENCH-018
    Invariant: INV-BENCH-009
    """
    if not report_path.exists():
        return []
    rc, stdout, stderr = _run_ncu_import(cfg, report_path, timeout_s)
    if rc != 0:
        logger.warning(
            "profiler.ncu.import_failed",
            extra={"returncode": rc, "stderr": stderr[:400]},
        )
        return []
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.warning("profiler.ncu.json_decode", extra={"error": str(exc)})
        return []
    entries: list[dict[str, object]] = []
    if isinstance(payload, list):
        entries = [e for e in payload if isinstance(e, dict)]
    elif isinstance(payload, dict):
        raw = payload.get("metrics")
        if isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, dict)]
    out: list[RawProfileMetric] = []
    for e in entries:
        name = str(e.get("name") or e.get("metric_name") or "")
        if not name:
            continue
        raw_value = e.get("value")
        value: float | int | None
        if isinstance(raw_value, bool):
            value = int(raw_value)
        elif isinstance(raw_value, (int, float)):
            value = raw_value
        elif raw_value is None:
            value = None
        else:
            try:
                value = float(str(raw_value))
            except ValueError:
                value = None
        unit_raw = e.get("unit")
        unit = str(unit_raw) if isinstance(unit_raw, (str, int, float)) else None
        section_raw = e.get("section") or e.get("collection_section")
        section = str(section_raw) if section_raw is not None else None
        out.append(
            RawProfileMetric(
                metric_name=name,
                value=value,
                unit=unit,
                architecture=architecture,
                profiler_name=ProfilerName.NCU,
                profiler_version=profiler_version,
                collection_section=section,
            )
        )
    return out


_NORMALIZATION_MAP: dict[str, tuple[str, ...]] = {
    "achieved_occupancy_pct": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "dram_throughput_pct_of_peak": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "sm_throughput_pct_of_peak": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "l2_hit_rate_pct": ("lts__t_sectors_hit_rate.pct",),
    "warp_stall_memory_dependency_pct": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    ),
    "warp_stall_exec_dependency_pct": (
        "smsp__average_warps_issue_stalled_execution_dependency_per_issue_active.ratio",
    ),
    "tensor_core_utilization_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    ),
}


def normalize(
    raw: list[RawProfileMetric],
    arch: str,
    profiler_version: str,
) -> tuple[ProfileMetrics, dict[str, NormalizedProfileMetricProvenance]]:
    """Normalize raw NCU metrics into compact :class:`ProfileMetrics`.

    Each normalized field either resolves to a ``float`` value or stays at
    ``None`` when the source metric is missing / null. Provenance is always
    emitted and always records the source metric names — even when the
    normalized value is ``None``.

    Implements: REQ-BENCH-018
    Invariant: INV-BENCH-009 (null propagates; never fabricated)
    """
    by_name: dict[str, RawProfileMetric] = {r.metric_name: r for r in raw}
    fields: dict[str, float | None] = {}
    provenance: dict[str, NormalizedProfileMetricProvenance] = {}
    for compact_name, sources in _NORMALIZATION_MAP.items():
        value: float | None = None
        for src in sources:
            candidate = by_name.get(src)
            if candidate is not None and candidate.value is not None:
                value = float(candidate.value)
                break
        fields[compact_name] = value
        provenance[compact_name] = NormalizedProfileMetricProvenance(
            source_metrics=list(sources),
            architecture=arch,
            profiler_version=profiler_version,
        )
    metrics = ProfileMetrics(
        achieved_occupancy_pct=fields["achieved_occupancy_pct"],
        dram_throughput_pct_of_peak=fields["dram_throughput_pct_of_peak"],
        sm_throughput_pct_of_peak=fields["sm_throughput_pct_of_peak"],
        l2_hit_rate_pct=fields["l2_hit_rate_pct"],
        warp_stall_memory_dependency_pct=fields["warp_stall_memory_dependency_pct"],
        warp_stall_exec_dependency_pct=fields["warp_stall_exec_dependency_pct"],
        tensor_core_utilization_pct=fields["tensor_core_utilization_pct"],
        arithmetic_intensity_flop_per_byte=None,
    )
    return metrics, provenance


def resolve_unavailable_reason(
    err: NcuRunResult | None,
    hygiene: HygieneReport,
    semantics: AdapterIterationSemantics,
) -> ProfileUnavailableReason | None:
    """Map a failure mode to the typed :class:`ProfileUnavailableReason`.

    Implements: REQ-BENCH-017
    Invariant: INV-BENCH-009 (no synthetic reason)
    """
    if semantics == AdapterIterationSemantics.NOT_REPEATABLE:
        return ProfileUnavailableReason.ADAPTER_NOT_REPEATABLE
    if semantics == AdapterIterationSemantics.REQUIRES_FULL_INPUT_RESET:
        return ProfileUnavailableReason.PROFILER_REPLAY_REFUSED
    if err is None:
        if not hygiene.profiler_counter_permission:
            return ProfileUnavailableReason.PROFILER_PERMISSION_DENIED
        return None
    if err.timed_out:
        return ProfileUnavailableReason.PROFILER_TIMEOUT
    if err.returncode == 127:
        return ProfileUnavailableReason.PROFILER_BINARY_MISSING
    if err.returncode != 0:
        if "permission" in err.stderr.lower():
            return ProfileUnavailableReason.PROFILER_PERMISSION_DENIED
        if "architecture" in err.stderr.lower():
            return ProfileUnavailableReason.ARCH_MISMATCH
        return ProfileUnavailableReason.PROFILER_REPLAY_REFUSED
    return None


def ncu_ready(cfg: ProfilerConfig) -> bool:
    """Return whether the ``ncu`` binary exists and ``ncu --version`` works."""
    if shutil.which(cfg.ncu_bin) is None and not Path(cfg.ncu_bin).exists():
        return False
    try:
        result = subprocess.run(  # noqa: S603
            [cfg.ncu_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def ncu_version(cfg: ProfilerConfig) -> str | None:
    """Return the ``ncu --version`` output first-line or ``None`` if missing."""
    if not ncu_ready(cfg):
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [cfg.ncu_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (result.stdout or result.stderr or "").strip().splitlines()
    return out[0] if out else None


def build_nvtx_range(
    run_id: str, batch_id: str, candidate_hash: str, shape_id: str
) -> str:
    """Construct the canonical NVTX range name per spec §6.6 / INV-BENCH-008."""
    return f"kerlever/{run_id}/{batch_id}/{candidate_hash}/{shape_id}/profile"


__all__ = [
    "NcuRunResult",
    "build_nvtx_range",
    "ncu_ready",
    "ncu_version",
    "normalize",
    "parse_report",
    "resolve_unavailable_reason",
    "run_ncu",
]
