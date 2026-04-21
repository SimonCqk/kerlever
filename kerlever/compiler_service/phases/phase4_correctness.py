"""Phase 4 — correctness validation + sanitizer gate.

Runs under a per-GPU ``asyncio.Semaphore(1)`` (INV-CS-010). Iterates the
``ProblemSpec.shape_cases``, compares against the reference executable,
and — only on a clean pass — runs the sanitizer gate.

Spec: docs/compiler-service/spec.md §6.6, §6.7, §6.8
Design: docs/compiler-service/design.md §4.3, §10
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kerlever.compiler_service.adapters.base import OperationAdapter
from kerlever.compiler_service.artifact_store import ArtifactStore
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.envelope import PhaseTimer
from kerlever.compiler_service.idempotency import IdempotencyRegistry
from kerlever.compiler_service.identity import problem_spec_hash
from kerlever.compiler_service.phases import PhaseShortCircuit
from kerlever.compiler_service.phases.phase3_compile import Phase3Output
from kerlever.compiler_service.pod_health import (
    Phase4Classification,
    Phase4ClassificationKind,
    PodHealthTracker,
    PodHealthTransition,
    ProbeOutcome,
)
from kerlever.compiler_service.sanitizer import (
    ComputeSanitizerRunner,
    SanitizerPolicy,
)
from kerlever.compiler_service.types import (
    ArtifactKind,
    CandidateFaultKind,
    ComparisonMode,
    CompileResultStatus,
    CorrectnessResultExt,
    CudaErrorKind,
    FailureDetail,
    OracleKind,
    PhaseName,
    SanitizerOutcome,
    SanitizerStatus,
    SanitizerTool,
)
from kerlever.types import CorrectnessResult, ProblemSpec, ShapeCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectnessOutcome:
    """Holds the accumulated correctness + sanitizer results."""

    correctness: CorrectnessResultExt
    pod_health_transition: PodHealthTransition | None
    last_sanitizer_tool: SanitizerTool | None = None
    cuda_error: CudaErrorKind | None = None
    candidate_fault_kind: CandidateFaultKind | None = None


@dataclass(frozen=True)
class Phase4Output:
    """Happy-path payload plus optional short-circuit packet."""

    phase3: Phase3Output
    correctness_outcome: CorrectnessOutcome | None
    short_circuit: PhaseShortCircuit | None = None
    pod_health_transition: PodHealthTransition | None = None
    correctness_log_artifact_id: str | None = None


@dataclass
class _ShapeExecutionResult:
    """Internal bookkeeping for one shape's run of ref + candidate."""

    shape_id: str
    passed: bool
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    nan_or_inf: bool = False
    runtime_error: bool = False
    cuda_error: CudaErrorKind | None = None
    timed_out: bool = False


@dataclass
class _ShapeLogEntry:
    """Per-shape record for the correctness log artifact."""

    shape_id: str
    ref_returncode: int
    ref_timed_out: bool
    cand_returncode: int
    cand_timed_out: bool
    passed: bool
    stderr_excerpt: str


@dataclass
class _CorrectnessAccumulator:
    """Accumulates per-shape pass/fail while iterating the shape cases."""

    passed: bool = True
    failing_shape_ids: list[str] = field(default_factory=list)
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    saw_nan_or_inf: bool = False
    runtime_error_shape: str | None = None
    runtime_error_role: str | None = None
    cuda_error: CudaErrorKind | None = None
    timed_out_shape: str | None = None
    timed_out_role: str | None = None
    shape_logs: list[_ShapeLogEntry] = field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedToleranceParams:
    """Phase 4's resolved comparison parameters, passed to ``_run_all_shapes``."""

    tolerance: float
    tolerance_source: object  # ToleranceSource at runtime
    comparison_mode: object  # ComparisonMode at runtime


class Phase4CorrectnessValidator:
    """Runs correctness under a per-GPU semaphore, then the sanitizer gate.

    Implements: REQ-CS-003, REQ-CS-004
    Invariant: INV-CS-010, INV-CS-004, INV-CS-006, INV-CS-011
    """

    def __init__(
        self,
        config: ServiceConfig,
        artifact_store: ArtifactStore,
        sanitizer_runner: ComputeSanitizerRunner,
        sanitizer_policy: SanitizerPolicy,
        pod_health: PodHealthTracker,
        idempotency: IdempotencyRegistry,
        gpu_semaphores: Mapping[int, asyncio.Semaphore],
    ) -> None:
        self._config = config
        self._artifact_store = artifact_store
        self._sanitizer_runner = sanitizer_runner
        self._sanitizer_policy = sanitizer_policy
        self._pod_health = pod_health
        self._idempotency = idempotency
        self._gpu_semaphores = gpu_semaphores

    async def run(self, phase3: Phase3Output, timer: PhaseTimer) -> Phase4Output:
        """Iterate shapes under the GPU semaphore and emit an outcome."""
        phase_start = time.monotonic()
        try:
            if phase3.compile is None or phase3.static_analysis is None:
                return Phase4Output(phase3=phase3, correctness_outcome=None)

            request = phase3.phase2.phase1.request
            adapter = phase3.phase2.harness.adapter if phase3.phase2.harness else None
            if adapter is None:
                return Phase4Output(phase3=phase3, correctness_outcome=None)

            gpu_index = _select_gpu_index(self._gpu_semaphores)
            semaphore = self._gpu_semaphores.get(gpu_index)
            if semaphore is None:
                return self._short_circuit_infra(
                    phase3,
                    reason="no_visible_gpu",
                )

            # Resolve tolerance + comparison mode BEFORE running shapes so
            # the actual comparison uses the right tolerance (spec §6.6).
            resolved_tolerance, tolerance_source = _resolve_tolerance(
                request.problem_spec,
                adapter,
                self._config,
            )
            comparison_mode = adapter.comparison_mode(request.problem_spec.dtype)
            resolved = _ResolvedToleranceParams(
                tolerance=resolved_tolerance,
                tolerance_source=tolerance_source,
                comparison_mode=comparison_mode,
            )

            # Pod health probe (spec §6.8 / SCN-CS-007-02..04): run the
            # known-good probe before candidate work if the pod is SUSPECT.
            probe_transition = await self._maybe_run_probe(semaphore)
            if probe_transition is not None and probe_transition.reason == "probe_fail":
                return self._short_circuit_probe_fail(phase3, probe_transition)

            sanitizer_results: list[SanitizerOutcome] | None = None
            async with semaphore:
                accumulator = await self._run_all_shapes(
                    phase3=phase3,
                    adapter=adapter,
                    request_problem_spec=request.problem_spec,
                    workspace=phase3.phase2.harness.workspace
                    if phase3.phase2.harness
                    else Path("."),
                    resolved=resolved,
                )
                if (
                    accumulator.timed_out_shape is None
                    and accumulator.runtime_error_shape is None
                    and accumulator.passed
                ):
                    await self._idempotency.record_phase(
                        request.request_id, PhaseName.SANITIZER
                    )
                    sanitizer_results = await self._run_sanitizer_gate(
                        request_source=request.source_code,
                        problem_spec=request.problem_spec,
                        adapter=adapter,
                        candidate_exec=phase3.compile.candidate_executable,
                        workspace=phase3.phase2.harness.workspace
                        if phase3.phase2.harness
                        else Path("."),
                        request_run_id=request.run_id,
                        candidate_hash=request.candidate_hash,
                        timeout_s=phase3.phase2.phase1.envelope_seed.limits.sanitizer_timeout_s,
                        saw_nan_or_inf=accumulator.saw_nan_or_inf,
                    )

            # Update pod health. Timeouts / CUDA errors are ambiguous.
            transition = await self._report_to_tracker(accumulator)

            # Write the bounded correctness log once per request (spec §6.4).
            correctness_log_artifact_id = await self._write_correctness_log(
                accumulator=accumulator,
                request_run_id=request.run_id,
                candidate_hash=request.candidate_hash,
            )

            # Correctness short-circuits.
            if accumulator.timed_out_shape is not None:
                return self._short_circuit_timeout(
                    phase3,
                    accumulator,
                    resolved_tolerance,
                    tolerance_source,
                    comparison_mode,
                    transition,
                    probe_transition,
                    correctness_log_artifact_id,
                )
            if accumulator.runtime_error_shape is not None:
                return self._short_circuit_runtime_error(
                    phase3,
                    accumulator,
                    resolved_tolerance,
                    tolerance_source,
                    comparison_mode,
                    transition,
                    probe_transition,
                    correctness_log_artifact_id,
                )
            if not accumulator.passed:
                return self._short_circuit_correctness_fail(
                    phase3,
                    accumulator,
                    resolved_tolerance,
                    tolerance_source,
                    comparison_mode,
                    transition,
                    probe_transition,
                    correctness_log_artifact_id,
                )

            # Sanitizer gate — correctness passed. It ran inside the same
            # per-GPU semaphore as value correctness.
            if sanitizer_results is None:
                sanitizer_results = []

            correctness_ext = _build_correctness_ext(
                accumulator,
                resolved_tolerance,
                tolerance_source,
                comparison_mode,
                sanitizer_results,
            )

            last_fail = _last_sanitizer_fail(sanitizer_results)
            last_timeout = _last_sanitizer_timeout(sanitizer_results)
            if last_timeout is not None:
                return self._short_circuit_sanitizer_timeout(
                    phase3,
                    correctness_ext,
                    last_timeout,
                    probe_transition,
                    correctness_log_artifact_id,
                )
            if last_fail is not None:
                return self._short_circuit_sanitizer_fail(
                    phase3,
                    correctness_ext,
                    last_fail,
                    transition,
                    probe_transition,
                    correctness_log_artifact_id,
                )

            # All clean. Signal a clean pass to pod health.
            transition = await self._pod_health.record_phase4_outcome(
                Phase4Classification(kind=Phase4ClassificationKind.CLEAN)
            )

            merged_transition = transition or probe_transition
            return Phase4Output(
                phase3=phase3,
                correctness_outcome=CorrectnessOutcome(
                    correctness=correctness_ext,
                    pod_health_transition=merged_transition,
                    last_sanitizer_tool=None,
                ),
                pod_health_transition=merged_transition,
                correctness_log_artifact_id=correctness_log_artifact_id,
            )
        finally:
            timer.record(PhaseName.CORRECTNESS, phase_start)

    # ------------------------------------------------------------------
    # Per-shape execution
    # ------------------------------------------------------------------

    async def _run_all_shapes(
        self,
        phase3: Phase3Output,
        adapter: OperationAdapter,
        request_problem_spec: ProblemSpec,
        workspace: Path,
        resolved: _ResolvedToleranceParams,
    ) -> _CorrectnessAccumulator:
        """Run all shape cases and accumulate results (spec §6.6).

        ``resolved`` carries the tolerance and comparison-mode already
        resolved from ``ShapeCase.correctness_tolerance`` → adapter default
        → service default. This method uses them for the actual per-shape
        comparison so the check is not hardcoded to ``0.0`` on float
        dtypes.
        """
        accumulator = _CorrectnessAccumulator()
        assert phase3.compile is not None  # noqa: S101 — caller-guaranteed

        for shape in request_problem_spec.shape_cases:
            seed = _shape_seed(request_problem_spec, shape)
            bundle = adapter.allocate_inputs(request_problem_spec, shape, seed)

            shape_dir = workspace / "shapes" / shape.shape_id
            shape_dir.mkdir(parents=True, exist_ok=True)

            a_path = shape_dir / "A.bin"
            b_path = shape_dir / "B.bin"
            ref_out = shape_dir / "ref.bin"
            cand_out = shape_dir / "cand.bin"
            a_path.write_bytes(bundle.buffers.get("A", b""))
            b_path.write_bytes(bundle.buffers.get("B", b""))

            argv_ref = _shape_argv(
                phase3.compile.reference_executable, a_path, b_path, ref_out, shape
            )
            argv_cand = _shape_argv(
                phase3.compile.candidate_executable, a_path, b_path, cand_out, shape
            )
            timeout = self._config.correctness_timeout_s
            timeout = (
                phase3.phase2.phase1.envelope_seed.limits.correctness_timeout_s
                or timeout
            )

            ref_result = await _run_executable(argv_ref, timeout=timeout)
            if ref_result.timed_out or ref_result.returncode != 0:
                # Reference failure is infra — but we surface as timeout or
                # infra through the accumulator flags.
                if ref_result.timed_out:
                    accumulator.timed_out_shape = shape.shape_id
                    accumulator.timed_out_role = "reference"
                else:
                    accumulator.runtime_error_shape = shape.shape_id
                    accumulator.runtime_error_role = "reference"
                    accumulator.cuda_error = ref_result.cuda_error
                accumulator.shape_logs.append(
                    _ShapeLogEntry(
                        shape_id=shape.shape_id,
                        ref_returncode=ref_result.returncode,
                        ref_timed_out=ref_result.timed_out,
                        cand_returncode=0,
                        cand_timed_out=False,
                        passed=False,
                        stderr_excerpt=_bound_stderr(
                            ref_result.stderr_excerpt, "reference"
                        ),
                    )
                )
                continue

            cand_result = await _run_executable(argv_cand, timeout=timeout)
            if cand_result.timed_out:
                accumulator.timed_out_shape = shape.shape_id
                accumulator.timed_out_role = "candidate"
                accumulator.shape_logs.append(
                    _ShapeLogEntry(
                        shape_id=shape.shape_id,
                        ref_returncode=ref_result.returncode,
                        ref_timed_out=False,
                        cand_returncode=cand_result.returncode,
                        cand_timed_out=True,
                        passed=False,
                        stderr_excerpt=_bound_stderr(
                            cand_result.stderr_excerpt, "candidate"
                        ),
                    )
                )
                continue
            if cand_result.returncode != 0:
                accumulator.runtime_error_shape = shape.shape_id
                accumulator.runtime_error_role = "candidate"
                accumulator.cuda_error = cand_result.cuda_error
                accumulator.shape_logs.append(
                    _ShapeLogEntry(
                        shape_id=shape.shape_id,
                        ref_returncode=ref_result.returncode,
                        ref_timed_out=False,
                        cand_returncode=cand_result.returncode,
                        cand_timed_out=False,
                        passed=False,
                        stderr_excerpt=_bound_stderr(
                            cand_result.stderr_excerpt, "candidate"
                        ),
                    )
                )
                continue

            comparison = adapter.compare_outputs(
                request_problem_spec,
                shape,
                ref_out,
                cand_out,
                tolerance=resolved.tolerance,
                comparison_mode=_as_comparison_mode(resolved.comparison_mode),
            )
            if not comparison.passed:
                accumulator.passed = False
                accumulator.failing_shape_ids.append(shape.shape_id)
                if comparison.max_abs_error is not None and (
                    accumulator.max_abs_error is None
                    or comparison.max_abs_error > accumulator.max_abs_error
                ):
                    accumulator.max_abs_error = comparison.max_abs_error
                if comparison.max_rel_error is not None and (
                    accumulator.max_rel_error is None
                    or comparison.max_rel_error > accumulator.max_rel_error
                ):
                    accumulator.max_rel_error = comparison.max_rel_error
            if comparison.max_abs_error is not None and _is_non_finite(
                comparison.max_abs_error
            ):
                accumulator.saw_nan_or_inf = True
            accumulator.shape_logs.append(
                _ShapeLogEntry(
                    shape_id=shape.shape_id,
                    ref_returncode=ref_result.returncode,
                    ref_timed_out=False,
                    cand_returncode=cand_result.returncode,
                    cand_timed_out=False,
                    passed=comparison.passed,
                    stderr_excerpt="",
                )
            )

        return accumulator

    # ------------------------------------------------------------------
    # Sanitizer gate
    # ------------------------------------------------------------------

    async def _run_sanitizer_gate(
        self,
        request_source: str,
        problem_spec: ProblemSpec,
        adapter: OperationAdapter,
        candidate_exec: Path,
        workspace: Path,
        request_run_id: str,
        candidate_hash: str,
        timeout_s: float | None,
        saw_nan_or_inf: bool,
    ) -> list[SanitizerOutcome]:
        """Run ``memcheck`` + any escalation tools (spec §6.7)."""
        tools = self._sanitizer_policy.decide(
            candidate_source=request_source,
            problem_spec=problem_spec,
            saw_nan_or_inf=saw_nan_or_inf,
            adapter_high_risk_shapes=frozenset(
                adapter.high_risk_shape_ids(problem_spec)
            ),
        )
        smallest = self._sanitizer_policy.smallest_shape(problem_spec.shape_cases)

        outcomes: list[SanitizerOutcome] = []
        for tool in tools:
            shape_dir = workspace / "shapes" / smallest.shape_id
            report_path = shape_dir / f"sanitizer-{tool.value}.report"
            out_path = shape_dir / f"sanitizer-{tool.value}.out"
            harness_args = _shape_argv(
                candidate_exec,
                shape_dir / "A.bin",
                shape_dir / "B.bin",
                out_path,
                smallest,
            )[1:]
            outcome = await self._sanitizer_runner.run(
                tool=tool,
                executable=candidate_exec,
                shape=smallest,
                input_dir=candidate_exec.parent,
                harness_args=harness_args,
                timeout_s=timeout_s or self._config.sanitizer_timeout_s,
                report_path=report_path,
            )
            if outcome.status is SanitizerStatus.FAIL and report_path.exists():
                try:
                    report_artifact_id = await self._artifact_store.write(
                        kind=ArtifactKind.SANITIZER_REPORT,
                        data=report_path.read_bytes()[
                            : self._config.max_artifact_bytes
                        ],
                        run_id=request_run_id,
                        candidate_hash=candidate_hash,
                    )
                    outcome = outcome.model_copy(
                        update={"report_artifact_id": report_artifact_id}
                    )
                except Exception as exc:  # noqa: BLE001 — best-effort evidence
                    logger.warning(
                        "sanitizer_report_write_failed",
                        extra={"tool": tool.value, "error": str(exc)},
                    )
            outcomes.append(outcome)
            if outcome.status is SanitizerStatus.TIMEOUT:
                break
            if outcome.status is SanitizerStatus.FAIL:
                break
        return outcomes

    # ------------------------------------------------------------------
    # Probe invocation
    # ------------------------------------------------------------------

    async def _maybe_run_probe(
        self,
        semaphore: asyncio.Semaphore,
    ) -> PodHealthTransition | None:
        """Run the known-good probe if the pod is SUSPECT.

        Delegates to ``PodHealthTracker.run_probe_if_needed`` with a closure
        that acquires the same per-device GPU semaphore ``_run_all_shapes``
        uses, invokes the pre-compiled probe executable, and decodes the
        exit code into a ``ProbeOutcome``.

        Implements: SCN-CS-007-02, SCN-CS-007-03, SCN-CS-007-04
        """
        if not await self._pod_health.needs_probe():
            return None

        probe_executable = self._pod_health.probe_executable_path
        if probe_executable is None:
            logger.warning("pod_health_probe_missing_executable")
            return None

        config = self._config

        async def runner() -> ProbeOutcome:
            async with semaphore:
                outcome = await _run_executable(
                    [str(probe_executable)],
                    timeout=config.probe_timeout_s,
                )
            if outcome.timed_out:
                return ProbeOutcome(passed=False, detail="probe_timeout")
            if outcome.returncode != 0:
                return ProbeOutcome(
                    passed=False,
                    detail=outcome.stderr_excerpt[:512] or "nonzero_exit",
                )
            return ProbeOutcome(passed=True, detail="")

        return await self._pod_health.run_probe_if_needed(runner)

    # ------------------------------------------------------------------
    # Correctness log artifact
    # ------------------------------------------------------------------

    async def _write_correctness_log(
        self,
        accumulator: _CorrectnessAccumulator,
        request_run_id: str,
        candidate_hash: str,
    ) -> str | None:
        """Persist a bounded plain-text correctness log (spec §6.4).

        The log records per-shape exit codes and truncated stderr excerpts
        so the Coding Agent can audit why a candidate failed. Empty logs
        (no shapes ran) are skipped — ``None`` is returned instead.
        """
        if not accumulator.shape_logs:
            return None
        try:
            payload = _format_correctness_log(
                accumulator.shape_logs, self._config.max_log_bytes
            )
            return await self._artifact_store.write(
                kind=ArtifactKind.CORRECTNESS_LOG,
                data=payload.encode("utf-8"),
                run_id=request_run_id,
                candidate_hash=candidate_hash,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort log write
            logger.warning("correctness_log_write_failed", extra={"error": str(exc)})
            return None

    # ------------------------------------------------------------------
    # Pod-health reporting
    # ------------------------------------------------------------------

    async def _report_to_tracker(
        self, accumulator: _CorrectnessAccumulator
    ) -> PodHealthTransition | None:
        """Convert accumulator state to a classification for the tracker."""
        ambiguous = (
            accumulator.cuda_error is not None
            or accumulator.timed_out_shape is not None
        )
        if ambiguous:
            return await self._pod_health.record_phase4_outcome(
                Phase4Classification(
                    kind=Phase4ClassificationKind.AMBIGUOUS,
                    cuda_error=accumulator.cuda_error,
                )
            )
        if accumulator.runtime_error_role == "reference":
            return None
        if accumulator.failing_shape_ids or accumulator.runtime_error_shape:
            return await self._pod_health.record_phase4_outcome(
                Phase4Classification(
                    kind=Phase4ClassificationKind.CANDIDATE_FAILURE,
                )
            )
        return None

    # ------------------------------------------------------------------
    # Short-circuit helpers
    # ------------------------------------------------------------------

    def _short_circuit_infra(self, phase3: Phase3Output, reason: str) -> Phase4Output:
        """Short-circuit when infrastructure is missing."""
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            reason=reason,
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=None,
            short_circuit=packet,
        )

    def _short_circuit_probe_fail(
        self,
        phase3: Phase3Output,
        probe_transition: PodHealthTransition,
    ) -> Phase4Output:
        """Short-circuit when the known-good probe failed (SCN-CS-007-03).

        The probe result is attributed to infra_fault regardless of the
        candidate — the pod is quarantined by ``PodHealthTracker``; no
        candidate work must occur on this request.
        """
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            reason="probe_failed_pod_quarantined",
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=None,
            short_circuit=packet,
            pod_health_transition=probe_transition,
        )

    def _short_circuit_timeout(
        self,
        phase3: Phase3Output,
        accumulator: _CorrectnessAccumulator,
        tolerance: float,
        tolerance_source: _ToleranceSourceType,
        comparison_mode: _ComparisonModeType,
        transition: PodHealthTransition | None,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit on a correctness-phase timeout (infra)."""
        del tolerance, tolerance_source, comparison_mode
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            failing_shape_id=accumulator.timed_out_shape,
            retryable=True,
            reason="correctness_timeout",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.TIMEOUT,
            candidate_fault_kind=None,
            cuda_error=accumulator.cuda_error,
            failure=failure,
        )
        merged = transition or probe_transition
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=None,
            short_circuit=packet,
            pod_health_transition=merged,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )

    def _short_circuit_runtime_error(
        self,
        phase3: Phase3Output,
        accumulator: _CorrectnessAccumulator,
        tolerance: float,
        tolerance_source: _ToleranceSourceType,
        comparison_mode: _ComparisonModeType,
        transition: PodHealthTransition | None,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit on a runtime error from reference or candidate."""
        del tolerance, tolerance_source, comparison_mode
        if accumulator.runtime_error_role == "reference":
            return self._short_circuit_reference_runtime_error(
                phase3,
                accumulator,
                transition,
                probe_transition,
                correctness_log_artifact_id,
            )
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            failing_shape_id=accumulator.runtime_error_shape,
            retryable=False,
            reason="candidate_runtime_error",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.CORRECTNESS_FAIL,
            candidate_fault_kind=CandidateFaultKind.CANDIDATE_RUNTIME_ERROR,
            cuda_error=accumulator.cuda_error,
            failure=failure,
        )
        merged = transition or probe_transition
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=None,
            short_circuit=packet,
            pod_health_transition=merged,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )

    def _short_circuit_reference_runtime_error(
        self,
        phase3: Phase3Output,
        accumulator: _CorrectnessAccumulator,
        transition: PodHealthTransition | None,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit when the reference executable fails during correctness."""
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            failing_shape_id=accumulator.runtime_error_shape,
            retryable=False,
            reason="reference_runtime_error",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=accumulator.cuda_error,
            failure=failure,
        )
        merged = transition or probe_transition
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=None,
            short_circuit=packet,
            pod_health_transition=merged,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )

    def _short_circuit_correctness_fail(
        self,
        phase3: Phase3Output,
        accumulator: _CorrectnessAccumulator,
        tolerance: float,
        tolerance_source: _ToleranceSourceType,
        comparison_mode: _ComparisonModeType,
        transition: PodHealthTransition | None,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit on a shape correctness mismatch."""
        correctness = _build_correctness_ext(
            accumulator,
            tolerance,
            tolerance_source,
            comparison_mode,
            sanitizer_results=[],
        )
        failure = FailureDetail(
            phase=PhaseName.CORRECTNESS,
            failing_shape_id=(
                accumulator.failing_shape_ids[0]
                if accumulator.failing_shape_ids
                else None
            ),
            retryable=False,
            reason="correctness_mismatch",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.CORRECTNESS,
            status=CompileResultStatus.CORRECTNESS_FAIL,
            candidate_fault_kind=CandidateFaultKind.CORRECTNESS_MISMATCH,
            cuda_error=None,
            failure=failure,
        )
        merged = transition or probe_transition
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=CorrectnessOutcome(
                correctness=correctness,
                pod_health_transition=merged,
                last_sanitizer_tool=None,
                candidate_fault_kind=CandidateFaultKind.CORRECTNESS_MISMATCH,
            ),
            short_circuit=packet,
            pod_health_transition=merged,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )

    def _short_circuit_sanitizer_fail(
        self,
        phase3: Phase3Output,
        correctness: CorrectnessResultExt,
        failing_outcome: SanitizerOutcome,
        transition: PodHealthTransition | None,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit on any sanitizer FAIL."""
        from kerlever.compiler_service.faults import _sanitizer_fault_kind

        fault_kind = _sanitizer_fault_kind(failing_outcome.tool)
        failure = FailureDetail(
            phase=PhaseName.SANITIZER,
            failing_shape_id=failing_outcome.shape_id,
            retryable=False,
            reason=f"sanitizer_{failing_outcome.tool.value}_fail",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.SANITIZER,
            status=CompileResultStatus.SANITIZER_FAIL,
            candidate_fault_kind=fault_kind,
            cuda_error=None,
            failure=failure,
        )
        merged = transition or probe_transition
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=CorrectnessOutcome(
                correctness=correctness,
                pod_health_transition=merged,
                last_sanitizer_tool=failing_outcome.tool,
                candidate_fault_kind=fault_kind,
            ),
            short_circuit=packet,
            pod_health_transition=merged,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )

    def _short_circuit_sanitizer_timeout(
        self,
        phase3: Phase3Output,
        correctness: CorrectnessResultExt,
        outcome: SanitizerOutcome,
        probe_transition: PodHealthTransition | None,
        correctness_log_artifact_id: str | None,
    ) -> Phase4Output:
        """Short-circuit on a sanitizer wall-clock timeout."""
        failure = FailureDetail(
            phase=PhaseName.SANITIZER,
            failing_shape_id=outcome.shape_id,
            retryable=True,
            reason=f"sanitizer_{outcome.tool.value}_timeout",
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.SANITIZER,
            status=CompileResultStatus.TIMEOUT,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase4Output(
            phase3=phase3,
            correctness_outcome=CorrectnessOutcome(
                correctness=correctness,
                pod_health_transition=probe_transition,
                last_sanitizer_tool=outcome.tool,
            ),
            short_circuit=packet,
            pod_health_transition=probe_transition,
            correctness_log_artifact_id=correctness_log_artifact_id,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Private aliases used only inside this module to keep helper signatures
# short. They resolve to the enums from ``.types``.
_ToleranceSourceType = object
_ComparisonModeType = object


@dataclass(frozen=True)
class _ExecutableOutcome:
    """Return shape from ``_run_executable``."""

    returncode: int
    timed_out: bool
    cuda_error: CudaErrorKind | None
    stdout_excerpt: str
    stderr_excerpt: str


async def _run_executable(
    argv: list[str],
    timeout: float,
) -> _ExecutableOutcome:
    """Run a correctness/probe executable and classify its outcome."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return _ExecutableOutcome(
            returncode=-1,
            timed_out=False,
            cuda_error=None,
            stdout_excerpt="",
            stderr_excerpt="executable not found",
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        with contextlib.suppress(Exception):
            await process.wait()
        return _ExecutableOutcome(
            returncode=-1,
            timed_out=True,
            cuda_error=CudaErrorKind.LAUNCH_TIMEOUT,
            stdout_excerpt="",
            stderr_excerpt="timed out",
        )

    rc = process.returncode or 0
    cuda_error = _classify_cuda_error(stderr_bytes.decode("utf-8", errors="replace"))
    return _ExecutableOutcome(
        returncode=rc,
        timed_out=False,
        cuda_error=cuda_error,
        stdout_excerpt=stdout_bytes.decode("utf-8", errors="replace")[:4096],
        stderr_excerpt=stderr_bytes.decode("utf-8", errors="replace")[:4096],
    )


def _classify_cuda_error(stderr: str) -> CudaErrorKind | None:
    """Classify known CUDA error strings from harness stderr."""
    lowered = stderr.lower()
    if "illegal memory access" in lowered or "illegal address" in lowered:
        return CudaErrorKind.ILLEGAL_ADDRESS
    if "launch timeout" in lowered:
        return CudaErrorKind.LAUNCH_TIMEOUT
    if "misaligned" in lowered:
        return CudaErrorKind.MISALIGNED_ADDRESS
    if "driver" in lowered and "reset" in lowered:
        return CudaErrorKind.DRIVER_RESET
    return None


def _shape_argv(
    executable: Path,
    a_path: Path,
    b_path: Path,
    out_path: Path,
    shape: ShapeCase,
) -> list[str]:
    """Build the argv to invoke a rendered harness executable."""
    args = [str(executable), str(a_path), str(b_path), str(out_path)]
    args.extend(str(int(dim)) for dim in shape.dims)
    return args


def _shape_seed(problem_spec: ProblemSpec, shape: ShapeCase) -> int:
    """Stable deterministic input seed for one correctness shape."""
    payload = "\0".join(
        (
            problem_spec_hash(problem_spec),
            shape.shape_id,
            problem_spec.dtype,
            "kerlever_correctness",
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _resolve_tolerance(
    problem_spec: ProblemSpec,
    adapter: OperationAdapter,
    config: ServiceConfig,
) -> tuple[float, _ToleranceSourceType]:
    """Resolve correctness tolerance per spec §6.6."""
    from kerlever.compiler_service.types import ToleranceSource

    # Iterate shape cases: if any sets an explicit tolerance, use it.
    for shape in problem_spec.shape_cases:
        if shape.correctness_tolerance is not None:
            return (shape.correctness_tolerance, ToleranceSource.SHAPE_CASE)
    adapter_default = adapter.default_tolerance(problem_spec.dtype)
    if adapter_default is not None:
        return (adapter_default, ToleranceSource.ADAPTER_DTYPE_DEFAULT)
    if problem_spec.dtype in ("fp16", "float16", "fp32", "float32"):
        return (config.service_default_float_tolerance, ToleranceSource.SERVICE_DEFAULT)
    return (config.service_default_int_tolerance, ToleranceSource.SERVICE_DEFAULT)


def _build_correctness_ext(
    accumulator: _CorrectnessAccumulator,
    tolerance: float,
    tolerance_source: _ToleranceSourceType,
    comparison_mode: _ComparisonModeType,
    sanitizer_results: list[SanitizerOutcome],
) -> CorrectnessResultExt:
    """Construct the ``CorrectnessResultExt`` from accumulated state."""
    from kerlever.compiler_service.types import ComparisonMode, ToleranceSource

    base = CorrectnessResult(
        passed=accumulator.passed and not accumulator.runtime_error_shape,
        failing_shape_ids=list(accumulator.failing_shape_ids),
        max_abs_error=accumulator.max_abs_error,
        max_rel_error=accumulator.max_rel_error,
    )
    ts: ToleranceSource = (
        tolerance_source
        if isinstance(tolerance_source, ToleranceSource)
        else ToleranceSource.SERVICE_DEFAULT
    )
    cm: ComparisonMode = (
        comparison_mode
        if isinstance(comparison_mode, ComparisonMode)
        else ComparisonMode.TOLERANCE
    )
    return CorrectnessResultExt(
        base=base,
        oracle_kind=OracleKind.REFERENCE_KERNEL,
        comparison_mode=cm,
        tolerance_source=ts,
        tolerance_value=tolerance,
        sanitizer_results=sanitizer_results,
    )


def _last_sanitizer_fail(
    results: list[SanitizerOutcome],
) -> SanitizerOutcome | None:
    """Return the last FAIL outcome, if any."""
    for outcome in results:
        if outcome.status is SanitizerStatus.FAIL:
            return outcome
    return None


def _last_sanitizer_timeout(
    results: list[SanitizerOutcome],
) -> SanitizerOutcome | None:
    """Return the last TIMEOUT outcome, if any."""
    for outcome in results:
        if outcome.status is SanitizerStatus.TIMEOUT:
            return outcome
    return None


def _is_non_finite(value: float) -> bool:
    """Return True if ``value`` is NaN or Inf."""
    return value != value or value in (float("inf"), float("-inf"))


def _select_gpu_index(gpu_semaphores: Mapping[int, asyncio.Semaphore]) -> int:
    """Return the first-visible GPU index; V1 uses device 0 when available."""
    if not gpu_semaphores:
        return 0
    return next(iter(gpu_semaphores.keys()))


def _bound_stderr(stderr: str, role: str) -> str:
    """Return a short labeled excerpt suitable for the correctness log."""
    if not stderr:
        return ""
    head = stderr[:1024]
    return f"[{role}] {head}"


def _format_correctness_log(entries: list[_ShapeLogEntry], max_bytes: int) -> str:
    """Format per-shape entries into a bounded plain-text log."""
    lines: list[str] = ["# kerlever correctness log"]
    for entry in entries:
        lines.append(
            f"shape={entry.shape_id} "
            f"ref_rc={entry.ref_returncode} ref_timeout={entry.ref_timed_out} "
            f"cand_rc={entry.cand_returncode} cand_timeout={entry.cand_timed_out} "
            f"passed={entry.passed}"
        )
        if entry.stderr_excerpt:
            lines.append(entry.stderr_excerpt)
    body = "\n".join(lines)
    if len(body.encode("utf-8")) <= max_bytes:
        return body
    # Truncate on byte boundary; keep a marker so readers can tell.
    truncated = body.encode("utf-8")[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n[truncated]\n"


def _as_comparison_mode(value: object) -> ComparisonMode:
    """Narrow an opaque alias back to ``ComparisonMode`` for adapter calls."""
    if isinstance(value, ComparisonMode):
        return value
    return ComparisonMode.TOLERANCE
