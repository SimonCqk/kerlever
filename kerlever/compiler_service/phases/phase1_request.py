"""Phase 1 — request normalization and interface resolution.

Implements spec §6.1. Never touches the GPU.

Spec: docs/compiler-service/spec.md §6.1
Design: docs/compiler-service/design.md §4.3
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from kerlever.compiler_service.adapters import AdapterRegistry
from kerlever.compiler_service.adapters.base import (
    InputBundle,
    ShapeComparisonResult,
)
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.envelope import PhaseTimer, RunEnvelopeSeed
from kerlever.compiler_service.idempotency import (
    IdempotencyIntake,
    IdempotencyRegistry,
)
from kerlever.compiler_service.identity import (
    artifact_key as compute_artifact_key,
)
from kerlever.compiler_service.identity import (
    compile_flags_hash,
    launch_spec_hash,
    problem_spec_hash,
    source_hash,
)
from kerlever.compiler_service.phases import PhaseShortCircuit
from kerlever.compiler_service.pod_health import PodHealthTracker
from kerlever.compiler_service.types import (
    CandidateFaultKind,
    CandidateRole,
    ComparisonMode,
    CompileRequest,
    CompileResult,
    CompileResultStatus,
    FailureDetail,
    IdempotencyState,
    KernelExecutionSpec,
    MetadataMode,
    PhaseName,
    PodHealth,
    RequestLimits,
    ToolchainInfo,
)
from kerlever.types import ProblemSpec, ShapeCase

if TYPE_CHECKING:
    from kerlever.compiler_service.adapters.base import OperationAdapter

_GLOBAL_FUNC_RE = re.compile(
    r"__global__\s+\w[\w\s\*<>,]*\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE
)


@dataclass(frozen=True)
class Phase1Output:
    """Happy-path payload plus optional short-circuit packet.

    Phase 5 reads ``reused_completed_result`` when ``idempotency_state`` is
    ``REUSED_COMPLETED``; in that case the cached ``CompileResult`` is
    refreshed (``pod_health`` re-sampled) and returned.
    """

    request: CompileRequest
    envelope_seed: RunEnvelopeSeed
    resolved_execution_spec: KernelExecutionSpec
    legacy_inferred_execution_spec: bool
    adapter: OperationAdapter
    idempotency_state: IdempotencyState
    previous_attempt_lost: bool = False
    prior_attempt_observed_phase: PhaseName | None = None
    reused_completed_result: CompileResult | None = None
    pod_health_at_intake: PodHealth = PodHealth.HEALTHY
    short_circuit: PhaseShortCircuit | None = None


class Phase1RequestNormalizer:
    """Runs request normalization, idempotency lookup, and interface resolution.

    Implements: REQ-CS-005, REQ-CS-006, REQ-CS-011
    Invariant: INV-CS-002, INV-CS-009
    """

    def __init__(
        self,
        config: ServiceConfig,
        toolchain: ToolchainInfo,
        pod_health: PodHealthTracker,
        idempotency: IdempotencyRegistry,
        adapter_registry: AdapterRegistry,
        pod_id: str,
    ) -> None:
        self._config = config
        self._toolchain = toolchain
        self._pod_health = pod_health
        self._idempotency = idempotency
        self._adapter_registry = adapter_registry
        self._pod_id = pod_id

    async def run(self, request: CompileRequest, timer: PhaseTimer) -> Phase1Output:
        """Normalize the request; short-circuit on quarantine / mismatches."""
        phase_start = time.monotonic()
        try:
            await self._idempotency.record_phase(
                request.request_id, PhaseName.REQUEST_NORMALIZATION
            )

            pod_snapshot = self._pod_health.snapshot()

            # Pod quarantined → short-circuit without touching anything else.
            if pod_snapshot is PodHealth.QUARANTINED:
                return self._short_circuit_quarantined(request, pod_snapshot)

            # Resolve adapter (pre-check so we can compute artifact_key with
            # the correct adapter_version).
            adapter = self._adapter_registry.get(request.problem_spec.op_name)
            if adapter is None:
                return self._short_circuit_unsupported_op(request, pod_snapshot)

            # Interface resolution (spec §6.1).
            resolved_spec, legacy_inferred, spec_error = self._resolve_interface(
                request, adapter
            )
            if spec_error is not None:
                return self._short_circuit_interface_contract(
                    request, pod_snapshot, adapter, spec_error
                )

            # Compute content hashes + artifact_key.
            limits = self._resolve_limits(request.limits)
            compile_flags = list(self._config.default_compile_flags) + [
                f"-arch={request.target_arch}"
            ]
            seed = RunEnvelopeSeed(
                run_id=request.run_id,
                round_id=request.round_id,
                request_id=request.request_id,
                candidate_hash=request.candidate_hash,
                source_hash=source_hash(request.source_code),
                problem_spec_hash=problem_spec_hash(request.problem_spec),
                launch_spec_hash=launch_spec_hash(resolved_spec),
                toolchain_hash=self._toolchain.toolchain_hash,
                compile_flags_hash=compile_flags_hash(compile_flags),
                adapter_version=adapter.adapter_version(),
                artifact_key=compute_artifact_key(
                    source_hash=source_hash(request.source_code),
                    problem_spec_hash=problem_spec_hash(request.problem_spec),
                    launch_spec_hash=launch_spec_hash(resolved_spec),
                    target_arch=request.target_arch,
                    toolchain_hash=self._toolchain.toolchain_hash,
                    compile_flags_hash=compile_flags_hash(compile_flags),
                    adapter_version=adapter.adapter_version(),
                    legacy_inferred_execution_spec=legacy_inferred,
                ),
                limits=limits,
                pod_id=self._pod_id,
                gpu_uuid=self._toolchain.gpu_uuid,
            )

            intake = await self._observe_intake(request, seed.artifact_key)

            if intake.state is IdempotencyState.REUSED_COMPLETED:
                return Phase1Output(
                    request=request,
                    envelope_seed=seed,
                    resolved_execution_spec=resolved_spec,
                    legacy_inferred_execution_spec=legacy_inferred,
                    adapter=adapter,
                    idempotency_state=IdempotencyState.REUSED_COMPLETED,
                    reused_completed_result=intake.result,
                    pod_health_at_intake=pod_snapshot,
                )

            if intake.state is IdempotencyState.PRIOR_ATTEMPT_LOST:
                return self._short_circuit_prior_attempt_lost(
                    request,
                    seed,
                    resolved_spec,
                    legacy_inferred,
                    adapter,
                    pod_snapshot,
                    intake.prior_attempt_observed_phase,
                )

            return Phase1Output(
                request=request,
                envelope_seed=seed,
                resolved_execution_spec=resolved_spec,
                legacy_inferred_execution_spec=legacy_inferred,
                adapter=adapter,
                idempotency_state=IdempotencyState.NEW,
                pod_health_at_intake=pod_snapshot,
            )
        finally:
            timer.record(PhaseName.REQUEST_NORMALIZATION, phase_start)

    # ------------------------------------------------------------------
    # Interface resolution
    # ------------------------------------------------------------------

    def _resolve_interface(
        self, request: CompileRequest, adapter: OperationAdapter
    ) -> tuple[KernelExecutionSpec, bool, str | None]:
        """Return ``(resolved_spec, legacy_inferred, error_or_none)``.

        INV-CS-002: legacy inference only runs when ``legacy_compatibility``
        is true.
        """
        spec = request.execution_spec
        fields_complete = (
            spec.entrypoint is not None
            and spec.block_dim is not None
            and spec.dynamic_smem_bytes is not None
            and spec.abi_name is not None
            and spec.abi_version is not None
        )

        if not request.legacy_compatibility:
            if not fields_complete:
                return spec, False, "missing_execution_spec_field"
            # Normal path — validate metadata_mode.
            if spec.metadata_mode is not MetadataMode.EXPLICIT:
                return spec, False, "metadata_mode_must_be_explicit"
            return spec, False, None

        # Legacy path: infer what is missing.
        if fields_complete:
            return spec, False, None

        match = _GLOBAL_FUNC_RE.findall(request.source_code)
        # There must be exactly one __global__ function for legacy inference.
        if len(match) != 1:
            return spec, False, "legacy_inference_requires_single_global"

        entrypoint = spec.entrypoint or match[0]
        block_dim = spec.block_dim or adapter.default_block_dim(request.problem_spec)
        dynamic_smem = (
            spec.dynamic_smem_bytes if spec.dynamic_smem_bytes is not None else 0
        )
        abi_name, abi_version = adapter.abi_contract()
        resolved = KernelExecutionSpec(
            entrypoint=entrypoint,
            block_dim=block_dim,
            dynamic_smem_bytes=dynamic_smem,
            abi_name=spec.abi_name or abi_name,
            abi_version=spec.abi_version or abi_version,
            metadata_mode=MetadataMode.LEGACY_INFERRED,
        )
        return resolved, True, None

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    async def _observe_intake(
        self, request: CompileRequest, artifact_key: str
    ) -> IdempotencyIntake:
        """Delegate to the idempotency registry with the computed key."""
        return await self._idempotency.observe_intake(request.request_id, artifact_key)

    def _resolve_limits(self, override: RequestLimits | None) -> RequestLimits:
        """Merge request overrides with service defaults."""
        if override is None:
            return RequestLimits(
                compile_timeout_s=self._config.compile_timeout_s,
                correctness_timeout_s=self._config.correctness_timeout_s,
                sanitizer_timeout_s=self._config.sanitizer_timeout_s,
                max_source_bytes=self._config.max_source_bytes,
                max_log_bytes=self._config.max_log_bytes,
            )
        return RequestLimits(
            compile_timeout_s=override.compile_timeout_s
            or self._config.compile_timeout_s,
            correctness_timeout_s=override.correctness_timeout_s
            or self._config.correctness_timeout_s,
            sanitizer_timeout_s=override.sanitizer_timeout_s
            or self._config.sanitizer_timeout_s,
            max_source_bytes=override.max_source_bytes or self._config.max_source_bytes,
            max_log_bytes=override.max_log_bytes or self._config.max_log_bytes,
        )

    # ------------------------------------------------------------------
    # Short-circuit constructors — each produces a Phase1Output that
    # carries a PhaseShortCircuit packet for Phase 5.
    # ------------------------------------------------------------------

    def _empty_seed(
        self, request: CompileRequest, pod_health: PodHealth
    ) -> RunEnvelopeSeed:
        """A best-effort envelope seed used by early-exit short-circuits.

        Content hashes are computed from the raw request; ``artifact_key``
        is best-effort (uses the service adapter_version fallback when no
        adapter was resolved).
        """
        del pod_health
        src_hash = source_hash(request.source_code)
        ps_hash = problem_spec_hash(request.problem_spec)
        ls_hash = launch_spec_hash(request.execution_spec)
        flags = [
            *self._config.default_compile_flags,
            f"-arch={request.target_arch}",
        ]
        cf_hash = compile_flags_hash(flags)
        return RunEnvelopeSeed(
            run_id=request.run_id,
            round_id=request.round_id,
            request_id=request.request_id,
            candidate_hash=request.candidate_hash,
            source_hash=src_hash,
            problem_spec_hash=ps_hash,
            launch_spec_hash=ls_hash,
            toolchain_hash=self._toolchain.toolchain_hash,
            compile_flags_hash=cf_hash,
            adapter_version=self._config.service_adapter_version,
            artifact_key=compute_artifact_key(
                source_hash=src_hash,
                problem_spec_hash=ps_hash,
                launch_spec_hash=ls_hash,
                target_arch=request.target_arch,
                toolchain_hash=self._toolchain.toolchain_hash,
                compile_flags_hash=cf_hash,
                adapter_version=self._config.service_adapter_version,
                legacy_inferred_execution_spec=False,
            ),
            limits=self._resolve_limits(request.limits),
            pod_id=self._pod_id,
            gpu_uuid=self._toolchain.gpu_uuid,
        )

    def _short_circuit_quarantined(
        self, request: CompileRequest, pod_health: PodHealth
    ) -> Phase1Output:
        """Short-circuit when the pod is already quarantined."""
        seed = self._empty_seed(request, pod_health)
        failure = FailureDetail(
            phase=PhaseName.REQUEST_NORMALIZATION,
            reason="pod_quarantined",
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.REQUEST_NORMALIZATION,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase1Output(
            request=request,
            envelope_seed=seed,
            resolved_execution_spec=request.execution_spec,
            legacy_inferred_execution_spec=False,
            adapter=_NullAdapter(),
            idempotency_state=IdempotencyState.NEW,
            pod_health_at_intake=pod_health,
            short_circuit=packet,
        )

    def _short_circuit_unsupported_op(
        self, request: CompileRequest, pod_health: PodHealth
    ) -> Phase1Output:
        """Short-circuit for an unknown op_name (INV-CS-013)."""
        seed = self._empty_seed(request, pod_health)
        failure = FailureDetail(
            phase=PhaseName.REQUEST_NORMALIZATION,
            reason="unsupported_operation",
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.REQUEST_NORMALIZATION,
            status=CompileResultStatus.INTERFACE_CONTRACT_ERROR,
            candidate_fault_kind=CandidateFaultKind.INTERFACE_CONTRACT_ERROR,
            cuda_error=None,
            failure=failure,
        )
        return Phase1Output(
            request=request,
            envelope_seed=seed,
            resolved_execution_spec=request.execution_spec,
            legacy_inferred_execution_spec=False,
            adapter=_NullAdapter(),
            idempotency_state=IdempotencyState.NEW,
            pod_health_at_intake=pod_health,
            short_circuit=packet,
        )

    def _short_circuit_interface_contract(
        self,
        request: CompileRequest,
        pod_health: PodHealth,
        adapter: OperationAdapter,
        reason: str,
    ) -> Phase1Output:
        """Short-circuit for a missing/incomplete execution spec."""
        seed = self._empty_seed(request, pod_health)
        failure = FailureDetail(
            phase=PhaseName.REQUEST_NORMALIZATION,
            reason=reason,
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.REQUEST_NORMALIZATION,
            status=CompileResultStatus.INTERFACE_CONTRACT_ERROR,
            candidate_fault_kind=CandidateFaultKind.INTERFACE_CONTRACT_ERROR,
            cuda_error=None,
            failure=failure,
        )
        return Phase1Output(
            request=request,
            envelope_seed=seed,
            resolved_execution_spec=request.execution_spec,
            legacy_inferred_execution_spec=False,
            adapter=adapter,
            idempotency_state=IdempotencyState.NEW,
            pod_health_at_intake=pod_health,
            short_circuit=packet,
        )

    def _short_circuit_prior_attempt_lost(
        self,
        request: CompileRequest,
        seed: RunEnvelopeSeed,
        resolved_spec: KernelExecutionSpec,
        legacy_inferred: bool,
        adapter: OperationAdapter,
        pod_health: PodHealth,
        prior_phase: PhaseName | None,
    ) -> Phase1Output:
        """Short-circuit for an in-flight prior attempt (spec §6.10)."""
        failure = FailureDetail(
            phase=PhaseName.REQUEST_NORMALIZATION,
            reason="prior_attempt_lost_before_durability",
            retryable=False,
        )
        packet = PhaseShortCircuit(
            phase=PhaseName.REQUEST_NORMALIZATION,
            status=CompileResultStatus.INFRA_ERROR,
            candidate_fault_kind=None,
            cuda_error=None,
            failure=failure,
        )
        return Phase1Output(
            request=request,
            envelope_seed=seed,
            resolved_execution_spec=resolved_spec,
            legacy_inferred_execution_spec=legacy_inferred,
            adapter=adapter,
            idempotency_state=IdempotencyState.PRIOR_ATTEMPT_LOST,
            previous_attempt_lost=True,
            prior_attempt_observed_phase=prior_phase,
            pod_health_at_intake=pod_health,
            short_circuit=packet,
        )


class _NullAdapter:
    """Placeholder adapter used by early short-circuits before resolution.

    It satisfies the ``OperationAdapter`` Protocol so ``Phase1Output`` can
    hold the adapter slot even on early-exit paths; every short-circuit
    path skips Phase 2+ entirely, so the methods are never called.
    """

    op_name: ClassVar[str] = "null"

    def adapter_version(self) -> str:  # pragma: no cover — never called
        return "null"

    def abi_contract(self) -> tuple[str, str]:  # pragma: no cover
        return ("null", "0")

    def default_block_dim(
        self, problem_spec: ProblemSpec
    ) -> tuple[int, int, int]:  # pragma: no cover
        del problem_spec
        return (1, 1, 1)

    def default_tolerance(self, dtype: str) -> float:  # pragma: no cover
        del dtype
        return 0.0

    def comparison_mode(self, dtype: str) -> ComparisonMode:  # pragma: no cover
        del dtype
        return ComparisonMode.TOLERANCE

    def high_risk_shape_ids(
        self, problem_spec: ProblemSpec
    ) -> set[str]:  # pragma: no cover
        del problem_spec
        return set()

    def allocate_inputs(
        self, problem_spec: ProblemSpec, shape: ShapeCase, seed: int
    ) -> InputBundle:  # pragma: no cover
        del problem_spec, shape, seed
        raise RuntimeError("null adapter called")

    def build_harness_source(
        self,
        execution_spec: KernelExecutionSpec,
        problem_spec: ProblemSpec,
        role: CandidateRole,
        kernel_source: str,
    ) -> str:  # pragma: no cover
        del execution_spec, problem_spec, role, kernel_source
        raise RuntimeError("null adapter called")

    def compare_outputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        reference_output: Path,
        candidate_output: Path,
        tolerance: float,
        comparison_mode: ComparisonMode,
    ) -> ShapeComparisonResult:  # pragma: no cover
        del problem_spec, shape, reference_output, candidate_output
        del tolerance, comparison_mode
        raise RuntimeError("null adapter called")
