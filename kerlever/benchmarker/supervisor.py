"""Benchmarker — batch supervisor in the service process.

Owns the per-batch worker subprocess lifecycle:

1. Stage ``config.json`` + ``request.json`` on disk.
2. Run preflight hygiene via ``telemetry.preflight`` (``asyncio.to_thread``).
3. Hard-gate short-circuit on ``arch_mismatch`` / ``ecc_xid`` / MIG mismatch.
4. Spawn ``python -m kerlever.benchmarker.worker`` with an asyncio.timeout.
5. Run postflight, read ``result.json``, attribute worker exit, return.

Never pools or reuses workers (REQ-BENCH-025, INV-BENCH-012).

Spec: docs/benchmarker/spec.md §6.2, §6.7
Design: docs/benchmarker/design.md §3.1, §4.2, §3.3
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.fault import BatchOutcomeSignal, attribute, signal_name
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.telemetry import (
    cuda_python_version,
    cuda_runtime_version,
    driver_version,
    is_hard_gate,
    postflight,
    preflight,
    pynvml_version,
)
from kerlever.benchmarker.telemetry import snapshot as telemetry_snapshot
from kerlever.benchmarker.types import (
    BatchStatus,
    BenchmarkBatchRequest,
    BenchmarkBatchResult,
    FaultClass,
    HygieneReport,
    IncumbentAnchor,
    MeasurementContext,
    PodHealth,
    RunEnvelope,
    ToolchainIdentity,
    VisibleGpu,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerExit:
    """Supervisor view of the child worker's termination state."""

    returncode: int | None
    signal: int | None
    timed_out: bool
    stderr_tail: str


@dataclass(frozen=True)
class WorkerFailure:
    """Supervisor-side failure wrapper for an unreadable result file."""

    reason: str


@dataclass(frozen=True)
class FinalizedBatch:
    """Supervisor's final assembly for one batch."""

    result: BenchmarkBatchResult
    pod_health: PodHealth
    ambiguous_count: int


class Supervisor:
    """Per-request batch orchestrator.

    Implements: REQ-BENCH-025, REQ-BENCH-020 (pod-health bookkeeping)
    Invariant: INV-BENCH-012 (disposable subprocess contains GPU faults)
    """

    def __init__(
        self,
        cfg: BenchmarkerConfig,
    ) -> None:
        self._cfg = cfg

    async def run_batch(
        self,
        req: BenchmarkBatchRequest,
        device: LeasedDevice,
        pod_health: PodHealth,
        ambiguous_count: int,
    ) -> FinalizedBatch:
        """Run a full benchmark batch end-to-end.

        Preflight hygiene runs before the worker is spawned; a hard gate
        short-circuits to a structured ``infra_error`` response (spec §6.2).
        Otherwise a worker subprocess is spawned with a config+request file
        pair; the supervisor awaits exit under ``asyncio.timeout`` and maps
        the exit to a final :class:`BatchStatus`.

        The staging directory at ``<artifact_root>/staging/<batch_id>/`` is
        cleaned up in a ``finally`` block — even when the worker crashes or
        times out (REQ-BENCH-035, INV-BENCH-016). The
        ``KEEP_STAGED_ARTIFACTS`` env var overrides cleanup for debugging.
        """
        pre_hygiene = await asyncio.to_thread(
            preflight, device, self._cfg.clock_policy, self._cfg.thresholds
        )
        if is_hard_gate(pre_hygiene.reason_on_fail):
            logger.warning(
                "supervisor.hard_gate",
                extra={
                    "reason": pre_hygiene.reason_on_fail,
                    "gpu_uuid": device.gpu_uuid,
                },
            )
            return FinalizedBatch(
                result=self._hard_gate_result(req, device, pre_hygiene),
                pod_health=PodHealth.QUARANTINED
                if pre_hygiene.reason_on_fail == "ecc_xid"
                else pod_health,
                ambiguous_count=ambiguous_count,
            )

        staging_dir, req_path, cfg_path, res_path = self._stage_worker_inputs(req)
        try:
            proc = await self._spawn_worker(
                req_path=req_path,
                res_path=res_path,
                cfg_path=cfg_path,
                device=device,
            )
            exit_info = await self._await_worker(
                proc, self._cfg.supervisor.batch_timeout_s
            )

            post_snapshot, _drift = await asyncio.to_thread(
                postflight,
                device,
                pre_hygiene.telemetry or telemetry_snapshot(device),
            )

            result_or_failure = self._read_worker_result(res_path)
            fault_class, batch_status = self._attribute_worker_exit(
                exit_info, result_or_failure
            )

            outcome_signal = BatchOutcomeSignal(
                had_ambiguous_fault=fault_class == FaultClass.AMBIGUOUS_FAULT,
                had_infra_fault=fault_class == FaultClass.INFRA_FAULT,
                had_candidate_fault=fault_class == FaultClass.CANDIDATE_FAULT,
                worker_timed_out=exit_info.timed_out,
            )
            _ = outcome_signal  # consumed by the service's PodHealthStore

            final_result = self._finalize(
                req=req,
                device=device,
                pre_hygiene=pre_hygiene,
                post_snapshot=post_snapshot,
                result_or_failure=result_or_failure,
                batch_status=batch_status,
                exit_info=exit_info,
                pod_health=pod_health,
                ambiguous_count=ambiguous_count,
            )
            return FinalizedBatch(
                result=final_result,
                pod_health=pod_health,
                ambiguous_count=ambiguous_count,
            )
        finally:
            # REQ-BENCH-035 / INV-BENCH-016: remove the staged directory so
            # no staged artifact outlives its batch. KEEP_STAGED_ARTIFACTS=1
            # (or "true") preserves the directory for debugging. Cleanup
            # failures are logged and swallowed — they must not override a
            # successful batch result.
            self._cleanup_staging(staging_dir, batch_id=req.batch_id)

    def _stage_worker_inputs(
        self, req: BenchmarkBatchRequest
    ) -> tuple[Path, Path, Path, Path]:
        """Persist request + config under ``<artifact_root>/staging/<batch_id>/``.

        The staging directory lives under ``staging/<batch_id>/`` so the
        ``finally`` block in :meth:`run_batch` can recursively remove every
        artifact produced by the batch — worker request/config/result plus
        the NCU reports written into ``staging/<batch_id>/ncu/`` by the
        worker (spec REQ-BENCH-035 / INV-BENCH-016 / SC-BENCH-016).

        Returns:
            ``(staging_dir, req_path, cfg_path, res_path)``.

        Implements: REQ-BENCH-035
        Invariant: INV-BENCH-016
        """
        staging_dir = self._cfg.artifact.root / "staging" / req.batch_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        req_path = staging_dir / "request.json"
        cfg_path = staging_dir / "config.json"
        res_path = staging_dir / "result.json"
        req_path.write_text(req.model_dump_json())
        cfg_path.write_text(json.dumps(self._cfg.to_dict()))
        return staging_dir, req_path, cfg_path, res_path

    def _cleanup_staging(self, staging_dir: Path, *, batch_id: str) -> None:
        """Remove the staging directory at batch completion.

        Honors ``KEEP_STAGED_ARTIFACTS`` (``"1"`` or ``"true"``) for dev
        debugging: the directory is retained and a ``supervisor.staging.retained``
        log is emitted. Otherwise the tree is removed with ``shutil.rmtree``;
        a failure (e.g., open file handle on Windows CI) is logged as
        ``supervisor.staging.cleanup_failed`` and swallowed so the caller's
        batch result is preserved.

        Implements: REQ-BENCH-035
        Invariant: INV-BENCH-016
        """
        keep_raw = os.environ.get("KEEP_STAGED_ARTIFACTS", "")
        if keep_raw == "1" or keep_raw.strip().lower() == "true":
            logger.info(
                "supervisor.staging.retained",
                extra={
                    "batch_id": batch_id,
                    "staging_dir": str(staging_dir),
                    "reason": "KEEP_STAGED_ARTIFACTS env var set",
                },
            )
            return
        try:
            shutil.rmtree(staging_dir, ignore_errors=False)
        except FileNotFoundError:
            # Already cleaned (e.g., hard-gate path never created the dir).
            return
        except OSError as exc:
            logger.warning(
                "supervisor.staging.cleanup_failed",
                extra={
                    "batch_id": batch_id,
                    "staging_dir": str(staging_dir),
                    "error": str(exc),
                },
            )

    async def _spawn_worker(
        self,
        req_path: Path,
        res_path: Path,
        cfg_path: Path,
        device: LeasedDevice,
    ) -> asyncio.subprocess.Process:
        """Spawn ``python -m kerlever.benchmarker.worker`` with file-based IPC."""
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "kerlever.benchmarker.worker",
            "--config",
            str(cfg_path),
            "--request-file",
            str(req_path),
            "--result-file",
            str(res_path),
            "--device-uuid",
            device.gpu_uuid,
            "--device-ordinal",
            str(device.ordinal),
            env=self._child_env(device),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def _child_env(self, device: LeasedDevice) -> dict[str, str]:
        """Return the subprocess env with CUDA_VISIBLE_DEVICES pinned."""
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(device.ordinal)
        return env

    async def _await_worker(
        self, proc: asyncio.subprocess.Process, timeout_s: float
    ) -> WorkerExit:
        """Await worker exit under ``asyncio.timeout``.

        On timeout, sends SIGTERM → waits 5 s → SIGKILL. After SIGKILL we
        still attempt to read ``result.json`` so a partial flush survives
        (design §4.2).
        """
        try:
            async with asyncio.timeout(timeout_s):
                stdout, stderr = await proc.communicate()
        except TimeoutError:
            proc.terminate()
            try:
                async with asyncio.timeout(5.0):
                    _stdout, _stderr = await proc.communicate()
            except TimeoutError:
                proc.kill()
                try:
                    _stdout, _stderr = await proc.communicate()
                except Exception:
                    _stdout, _stderr = b"", b""
            return WorkerExit(
                returncode=proc.returncode,
                signal=None,
                timed_out=True,
                stderr_tail="timeout",
            )
        rc = proc.returncode
        sig: int | None = None
        if rc is not None and rc < 0:
            sig = -rc
            rc = None
        stderr_text = stderr.decode("utf-8", "replace") if stderr else ""
        if stderr_text:
            logger.info(
                "supervisor.worker.stderr",
                extra={"tail": stderr_text[-400:]},
            )
        if stdout:
            logger.info(
                "supervisor.worker.stdout",
                extra={"tail": stdout.decode("utf-8", "replace")[-200:]},
            )
        return WorkerExit(
            returncode=rc,
            signal=sig,
            timed_out=False,
            stderr_tail=stderr_text[-400:],
        )

    def _read_worker_result(
        self, res_path: Path
    ) -> BenchmarkBatchResult | WorkerFailure:
        """Parse the worker's result.json; classify failures structurally."""
        if not res_path.exists():
            return WorkerFailure(reason="result_file_missing")
        try:
            text = res_path.read_text()
        except OSError as exc:
            return WorkerFailure(reason=f"result_read_error:{exc}")
        if not text.strip():
            return WorkerFailure(reason="result_file_empty")
        try:
            return BenchmarkBatchResult.model_validate_json(text)
        except Exception as exc:  # Pydantic ValidationError or JSON error
            return WorkerFailure(reason=f"result_parse_error:{type(exc).__name__}")

    def _attribute_worker_exit(
        self,
        exit_info: WorkerExit,
        result_or_failure: BenchmarkBatchResult | WorkerFailure,
    ) -> tuple[FaultClass, BatchStatus]:
        """Translate (exit, result) into fault class + batch status (design §3.3)."""
        if exit_info.timed_out:
            return FaultClass.AMBIGUOUS_FAULT, BatchStatus.TIMEOUT
        if exit_info.signal is not None:
            return FaultClass.AMBIGUOUS_FAULT, BatchStatus.INFRA_ERROR
        rc = exit_info.returncode
        if isinstance(result_or_failure, WorkerFailure):
            fault = attribute(
                exc=RuntimeError(result_or_failure.reason),
                exit_signal=None,
                exit_code=rc,
                pod_health=PodHealth.HEALTHY,
            )
            return fault, BatchStatus.INFRA_ERROR
        if rc == 0:
            # Result's status is the source of truth.
            return FaultClass.CANDIDATE_FAULT, result_or_failure.status
        if rc == 1:
            return FaultClass.CANDIDATE_FAULT, BatchStatus.PARTIAL
        return FaultClass.AMBIGUOUS_FAULT, BatchStatus.INFRA_ERROR

    def _finalize(
        self,
        req: BenchmarkBatchRequest,
        device: LeasedDevice,
        pre_hygiene: HygieneReport,
        post_snapshot: object,  # DeviceTelemetrySnapshot
        result_or_failure: BenchmarkBatchResult | WorkerFailure,
        batch_status: BatchStatus,
        exit_info: WorkerExit,
        pod_health: PodHealth,
        ambiguous_count: int,
    ) -> BenchmarkBatchResult:
        """Assemble the final BenchmarkBatchResult returned to the client."""
        if isinstance(result_or_failure, BenchmarkBatchResult):
            result = result_or_failure
            # Overlay the supervisor-observed hygiene + post snapshot.
            result = result.model_copy(
                update={
                    "hygiene": pre_hygiene,
                    "status": batch_status
                    if batch_status != BatchStatus.SUCCESS
                    else result.status,
                }
            )
            _ = post_snapshot
            return result
        # WorkerFailure → synthesize an infra_error result.
        return BenchmarkBatchResult(
            status=batch_status,
            run_envelope=_run_envelope_from_supervisor(
                req, self._cfg, device, pod_health, ambiguous_count
            ),
            measurement_context=MeasurementContext(
                artifact_execution_model=req.artifact_execution_model,
                metric_mode=req.metric_mode,
                cache_policy_requested=req.cache_policy,
                cache_policy_effective=req.cache_policy,
                clock_policy=req.clock_policy,
                interleave_enabled=False,
                noise_floor_pct=self._cfg.thresholds.noise_floor_pct,
                guard_pct=req.problem_spec.objective.regression_guard_pct,
            ),
            hygiene=pre_hygiene,
            incumbent_anchor=IncumbentAnchor(
                incumbent_artifact_id=req.incumbent_ref.artifact_id,
                shape_results=[],
                objective_score=req.incumbent_ref.objective_score,
            ),
            candidate_results=[],
            top_k_profiled=[],
            failure_reason=(
                f"{result_or_failure.reason}"
                + (
                    f":signal={signal_name(exit_info.signal)}"
                    if exit_info.signal is not None
                    else ""
                )
            ),
        )

    def _hard_gate_result(
        self,
        req: BenchmarkBatchRequest,
        device: LeasedDevice,
        pre_hygiene: HygieneReport,
    ) -> BenchmarkBatchResult:
        """Construct the infra_error response for a Phase 2 hard gate."""
        return BenchmarkBatchResult(
            status=BatchStatus.INFRA_ERROR,
            run_envelope=_run_envelope_from_supervisor(
                req, self._cfg, device, PodHealth.HEALTHY, 0
            ),
            measurement_context=MeasurementContext(
                artifact_execution_model=req.artifact_execution_model,
                metric_mode=req.metric_mode,
                cache_policy_requested=req.cache_policy,
                cache_policy_effective=req.cache_policy,
                clock_policy=req.clock_policy,
                interleave_enabled=False,
                noise_floor_pct=self._cfg.thresholds.noise_floor_pct,
                guard_pct=req.problem_spec.objective.regression_guard_pct,
            ),
            hygiene=pre_hygiene,
            incumbent_anchor=IncumbentAnchor(
                incumbent_artifact_id=req.incumbent_ref.artifact_id,
                shape_results=[],
                objective_score=req.incumbent_ref.objective_score,
            ),
            candidate_results=[],
            top_k_profiled=[],
            failure_reason=pre_hygiene.reason_on_fail,
        )


def _run_envelope_from_supervisor(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
    pod_health: PodHealth,
    ambiguous_count: int,
) -> RunEnvelope:
    """Build the :class:`RunEnvelope` from supervisor-side toolchain identity."""
    return RunEnvelope(
        run_id=req.run_id,
        round_id=req.round_id,
        batch_id=req.batch_id,
        request_id=req.request_id,
        pod_id=cfg.pod_id,
        pod_health=pod_health,
        ambiguous_failure_count=ambiguous_count,
        toolchain=ToolchainIdentity(
            driver_version=driver_version(),
            cuda_runtime_version=cuda_runtime_version(),
            cuda_python_version=cuda_python_version(),
            pynvml_version=pynvml_version(),
            ncu_version=None,
        ),
        visible_gpu=VisibleGpu(
            gpu_uuid=device.gpu_uuid,
            pci_bus_id=device.pci_bus_id,
            sm_arch=device.sm_arch,
            mig_profile=device.mig_profile,
        ),
    )


def ensure_artifact_root(cfg: BenchmarkerConfig) -> None:
    """Create the artifact root directory tree if missing.

    Called from the service lifespan; falls back to a temp dir when the
    configured root is not writable (e.g., dev host without /var/lib).
    """
    try:
        cfg.artifact.root.mkdir(parents=True, exist_ok=True)
        (cfg.artifact.root / "ncu").mkdir(exist_ok=True)
        (cfg.artifact.root / "worker").mkdir(exist_ok=True)
        # REQ-BENCH-035: staging dirs are created per-batch under here; we
        # pre-create the top-level so a read-only fallback is detected early.
        (cfg.artifact.root / "staging").mkdir(exist_ok=True)
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / "kerlever_bench"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "supervisor.artifact_root.fallback",
            extra={"requested": str(cfg.artifact.root), "fallback": str(fallback)},
        )


__all__ = [
    "FinalizedBatch",
    "Supervisor",
    "WorkerExit",
    "WorkerFailure",
    "ensure_artifact_root",
]
