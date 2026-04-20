"""Benchmarker — Phase 1: request normalization and envelope construction.

Pure Python. No GPU, no I/O beyond filesystem stat of the cubin URI.

Spec: docs/benchmarker/spec.md §6.1
Design: docs/benchmarker/design.md §2.1 normalize.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.telemetry import (
    cuda_runtime_version,
    driver_version,
    resolve_clock_policy_mode,
)
from kerlever.benchmarker.types import (
    ArtifactExecutionModel,
    BenchmarkBatchRequest,
    CachePolicy,
    CachePolicyBlock,
    CandidateArtifactRef,
    ClockPolicy,
    FaultClass,
    MeasurementEnvelope,
    RepeatPolicy,
    WarmupPolicy,
)

logger = logging.getLogger(__name__)


class NormalizationError(RuntimeError):
    """Top-level Phase 1 failure that should short-circuit to infra_error.

    ``reason`` matches the spec §6.1 closed set of validation failure tokens.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class InfraFault:
    """Per-candidate reject note carried into the final result."""

    candidate_hash: str
    reason: str
    fault_class: FaultClass


@dataclass(frozen=True)
class NormalizedBatch:
    """Phase 1 output held in memory until Phase 3.

    All scalar fields are derived from the request + device identity; no GPU
    call has happened yet.
    """

    envelope_per_candidate: dict[str, MeasurementEnvelope]
    admit_candidates: list[CandidateArtifactRef]
    reject_candidates: dict[str, InfraFault] = field(default_factory=dict)
    interleave_seed_per_shape: dict[str, int] = field(default_factory=dict)
    interleave_enabled: bool = False
    requested_cache_policy: CachePolicy = CachePolicy.WARM_SAME_BUFFERS
    effective_cache_policy: CachePolicy = CachePolicy.WARM_SAME_BUFFERS
    cache_policy_reason: str | None = None
    clock_policy: ClockPolicy = field(
        default_factory=lambda: ClockPolicy(mode=resolve_clock_policy_mode)  # type: ignore[arg-type]
    )


def _sha256_canonical(obj: object) -> str:
    """Stable SHA-256 of canonical-JSON encoding of ``obj``."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _interleave_seed(
    run_id: str, batch_id: str, shape_id: str
) -> int:
    """Deterministic PCG64 seed per (run_id, batch_id, shape_id).

    The token ``"kerlever_benchmark_order"`` is fixed per spec §6.1 so a
    retry with the same ``batch_id`` reproduces the realized block order
    (INV-BENCH-004). We hash with blake2b for a uniform 64-bit value.
    """
    key = f"{run_id}|{batch_id}|{shape_id}|kerlever_benchmark_order"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _module_artifact_hash(candidate: CandidateArtifactRef) -> str:
    """Stable hash of the module artifact identity for envelope comparability.

    Uses a combination of ``artifact_id``, ``source_hash``, and
    ``launch_spec_hash`` so two uploads of the same binary with different
    metadata still hash distinctly.
    """
    return hashlib.sha256(
        f"{candidate.artifact_id}|{candidate.source_hash}|{candidate.launch_spec_hash}"
        .encode()
    ).hexdigest()


_URI_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")


def _validate_cubin_uri(uri: str) -> str | None:
    """Validate a cubin URI per V1 contract (spec §6.1, REQ-BENCH-031).

    V1 supports only absolute POSIX paths to readable files. URI schemes
    (``s3:``, ``file:``, ``http:``, ``https:``, etc.) and relative paths
    are rejected outright.

    Returns:
        ``None`` when the URI is acceptable; otherwise the closed-set
        rejection reason token.

    Implements: REQ-BENCH-031
    """
    if not uri:
        return "cubin_uri_not_readable"
    if _URI_SCHEME_RE.match(uri):
        return "cubin_uri_unsupported_scheme"
    path = Path(uri)
    if not path.is_absolute():
        return "cubin_uri_not_absolute"
    if not path.is_file():
        return "cubin_uri_not_readable"
    return None


def normalize_request(
    req: BenchmarkBatchRequest,
    cfg: BenchmarkerConfig,
    device: LeasedDevice,
) -> NormalizedBatch:
    """Validate ``req`` and build one :class:`MeasurementEnvelope` per candidate.

    Follows spec §6.1 validation steps in order and fails fast on hard
    violations. Unknown candidates and missing correctness are recorded
    as per-candidate rejects without aborting the batch.

    Implements: REQ-BENCH-001, REQ-BENCH-008, INV-BENCH-001
    """
    if req.artifact_execution_model != ArtifactExecutionModel.COMMON_HARNESS_CUBIN:
        raise NormalizationError(
            "unsupported_artifact_execution_model",
            f"got {req.artifact_execution_model}",
        )
    if not req.candidate_module_artifact_refs:
        raise NormalizationError("empty_batch", "no candidate refs provided")
    if not req.run_id:
        raise NormalizationError("missing_run_id", "run_id is required")

    # Operation adapter compat: V1 admits any registered ABI matching request.
    admitted_abis = cfg.supported_adapter_abis
    if admitted_abis and req.operation_adapter_abi not in admitted_abis:
        raise NormalizationError(
            "unsupported_adapter_abi",
            f"adapter {req.operation_adapter_abi!r} not in "
            f"configured list {admitted_abis!r}",
        )

    # REQ-BENCH-028/SC-BENCH-012: the (abi, version) must resolve in the
    # adapter registry. We import lazily so this module stays cheap to
    # import on non-GPU dev hosts.
    from kerlever.benchmarker.adapter import get_adapter  # noqa: PLC0415

    if get_adapter(
        req.operation_adapter_abi, req.operation_adapter_version
    ) is None:
        raise NormalizationError(
            "adapter_not_registered",
            f"adapter {req.operation_adapter_abi!r} @ "
            f"{req.operation_adapter_version!r} is not registered",
        )

    # REQ-BENCH-031: incumbent's cubin_uri (when present) must also satisfy
    # the V1 absolute-readable-path contract.
    if req.incumbent_ref.cubin_uri is not None:
        reason = _validate_cubin_uri(req.incumbent_ref.cubin_uri)
        if reason is not None:
            raise NormalizationError(
                reason,
                f"incumbent cubin_uri invalid: {req.incumbent_ref.cubin_uri!r}",
            )

    # Top-K / top-M guard (spec §6.1 step 7).
    if req.top_k_profile < 0 or req.top_m_profile_shift_potential < 0:
        raise NormalizationError(
            "invalid_profile_selection",
            f"top_k={req.top_k_profile} top_m={req.top_m_profile_shift_potential}",
        )

    problem_hash = _sha256_canonical(req.problem_spec.model_dump())
    objective_hash = _sha256_canonical(req.problem_spec.objective.model_dump())
    shape_ids = [s.shape_id for s in req.objective_shape_cases]

    interleave_seed_per_shape = {
        s.shape_id: _interleave_seed(req.run_id, req.batch_id, s.shape_id)
        for s in req.objective_shape_cases
    }

    interleave_enabled = len(req.candidate_module_artifact_refs) >= 2

    requested_cache = req.cache_policy
    if (
        interleave_enabled
        and requested_cache == CachePolicy.WARM_SAME_BUFFERS
    ):
        effective_cache = CachePolicy.WARM_ROTATING_BUFFERS
        cache_reason: str | None = "interleaved_batch_requires_rotation"
    else:
        effective_cache = requested_cache
        cache_reason = None

    cache_block = CachePolicyBlock(
        requested=requested_cache,
        effective=effective_cache,
        reason=cache_reason,
    )

    resolved_clock = ClockPolicy(
        mode=resolve_clock_policy_mode(cfg.clock_policy),
        requested_sm_clock_mhz=req.clock_policy.requested_sm_clock_mhz,
        requested_mem_clock_mhz=req.clock_policy.requested_mem_clock_mhz,
    )

    warmup_policy = WarmupPolicy(
        min_runs=cfg.calibration.warmup_min_runs,
        cache_state="touched",
    )
    repeat_policy_default = RepeatPolicy(
        repetitions=cfg.calibration.repetitions,
        iterations_per_sample=1,
        min_timed_batch_ms=cfg.calibration.min_timed_batch_ms,
        max_timed_batch_ms=cfg.calibration.max_timed_batch_ms,
    )

    drv_ver = driver_version()
    rt_ver = cuda_runtime_version()

    seen_hashes: set[str] = set()
    admit: list[CandidateArtifactRef] = []
    reject: dict[str, InfraFault] = {}
    envelopes: dict[str, MeasurementEnvelope] = {}

    for ref in req.candidate_module_artifact_refs:
        if ref.candidate_hash in seen_hashes:
            logger.warning(
                "normalize.duplicate_candidate",
                extra={"candidate_hash": ref.candidate_hash},
            )
            continue
        seen_hashes.add(ref.candidate_hash)
        if ref.correctness is None or not ref.correctness.passed:
            reject[ref.candidate_hash] = InfraFault(
                candidate_hash=ref.candidate_hash,
                reason="correctness_not_passed",
                fault_class=FaultClass.INFRA_FAULT,
            )
            continue
        cubin_reason = _validate_cubin_uri(ref.cubin_uri)
        if cubin_reason is not None:
            # REQ-BENCH-031: unsupported schemes + relative paths fail the
            # whole batch (spec §6.1). An unreadable absolute path is a
            # per-candidate reject so the rest of the batch can proceed.
            if cubin_reason == "cubin_uri_not_readable":
                reject[ref.candidate_hash] = InfraFault(
                    candidate_hash=ref.candidate_hash,
                    reason="cubin_uri_not_readable",
                    fault_class=FaultClass.INFRA_FAULT,
                )
                continue
            raise NormalizationError(
                cubin_reason,
                f"cubin_uri invalid for {ref.candidate_hash}: {ref.cubin_uri!r}",
            )
        envelope = MeasurementEnvelope(
            run_id=req.run_id,
            round_id=req.round_id,
            batch_id=req.batch_id,
            request_id=req.request_id,
            candidate_hash=ref.candidate_hash,
            artifact_id=ref.artifact_id,
            source_hash=ref.source_hash,
            launch_spec_hash=ref.launch_spec_hash,
            toolchain_hash=ref.toolchain_hash,
            module_artifact_hash=_module_artifact_hash(ref),
            artifact_execution_model=req.artifact_execution_model,
            problem_spec_hash=problem_hash,
            objective_hash=objective_hash,
            shape_ids=shape_ids,
            operation_adapter_abi=req.operation_adapter_abi,
            operation_adapter_version=req.operation_adapter_version,
            target_gpu=req.problem_spec.target_gpu,
            gpu_uuid=device.gpu_uuid,
            pci_bus_id=device.pci_bus_id,
            mig_profile=device.mig_profile,
            sm_arch=device.sm_arch,
            driver_version=drv_ver,
            cuda_runtime_version=rt_ver,
            metric_mode=req.metric_mode,
            function_attribute_policy_requested=req.function_attribute_policy,
            function_attribute_policy_observed=req.function_attribute_policy,
            warmup_policy=warmup_policy,
            repeat_policy=repeat_policy_default,
            cache_policy=cache_block,
            clock_policy=resolved_clock,
            interleave_seed=(
                interleave_seed_per_shape[shape_ids[0]] if shape_ids else None
            ),
        )
        envelopes[ref.candidate_hash] = envelope
        admit.append(ref)

    return NormalizedBatch(
        envelope_per_candidate=envelopes,
        admit_candidates=admit,
        reject_candidates=reject,
        interleave_seed_per_shape=interleave_seed_per_shape,
        interleave_enabled=interleave_enabled,
        requested_cache_policy=requested_cache,
        effective_cache_policy=effective_cache,
        cache_policy_reason=cache_reason,
        clock_policy=resolved_clock,
    )


__all__ = [
    "InfraFault",
    "NormalizationError",
    "NormalizedBatch",
    "normalize_request",
]
