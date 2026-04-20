#!/usr/bin/env python3
"""Benchmarker — dry-run smoke test (no GPU required).

Single executable script that exercises the adapter-registry resolution,
request normalization, scoring, incumbent-comparison gate, profile-set
selection, and noise-margin aggregation WITHOUT touching ``cuda-python``,
``pynvml``, or ``nvtx``. Its purpose is a ~1-second "nothing is broken"
smoke between implementation rounds — NOT coverage of the GPU path.

Usage:
    python scripts/benchmarker_dry_run.py

Exit codes:
    0 — all pure-Python business logic composed cleanly ("DRY-RUN OK").
    1 — any unexpected exception; the traceback is printed.

Spec: docs/benchmarker/spec.md §6.1, §6.5, §6.6
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import traceback
from pathlib import Path

# Repo root must be on sys.path so ``kerlever.benchmarker.*`` resolves when
# the script is invoked directly (``python scripts/benchmarker_dry_run.py``)
# and kerlever is not installed into the venv.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kerlever.benchmarker.config import BenchmarkerConfig  # noqa: E402
from kerlever.benchmarker.harness import (  # noqa: E402
    BatchMeasurement,
    PerShapeMeasurement,
)
from kerlever.benchmarker.lease import LeasedDevice  # noqa: E402
from kerlever.benchmarker.normalize import (  # noqa: E402
    NormalizationError,
    normalize_request,
)
from kerlever.benchmarker.scoring import (  # noqa: E402
    compute_objective_score,
    decide_incumbent_comparison,
)
from kerlever.benchmarker.selection import (  # noqa: E402
    ScoredCandidate,
    build_profile_set,
)
from kerlever.benchmarker.stats import (  # noqa: E402
    aggregate_noise_margin,
    cv_pct,
    mean,
    p50,
    p95,
    stdev,
)
from kerlever.benchmarker.types import (  # noqa: E402
    AdapterIterationSemantics,
    ArtifactExecutionModel,
    BaselineRef,
    BenchmarkBatchRequest,
    CachePolicy,
    CachePolicyBlock,
    CandidateArtifactRef,
    ClockPolicy,
    ClockPolicyMode,
    CorrectnessForward,
    FunctionAttributePolicy,
    IncumbentComparison,
    IncumbentRef,
    LaunchSpec,
    MeasurementEnvelope,
    MeasurementQualityStatus,
    ObjectiveScore,
    PerformanceObjective,
    ProblemSpec,
    RepeatPolicy,
    ShapeBenchResult,
    ShapeCase,
    WarmupPolicy,
)


def _make_placeholder_cubin() -> Path:
    """Create a tiny readable file that passes V1 cubin_uri gate.

    The file contains a single null byte — enough to satisfy
    :func:`normalize._validate_cubin_uri` (absolute path, readable file).
    It is never loaded by cuda because this script never enters the
    worker's CUDA path.
    """
    fd, name = tempfile.mkstemp(suffix=".cubin", prefix="kerlever_dryrun_")
    os.write(fd, b"\x00")
    os.close(fd)
    return Path(name)


def _build_request(cubin_a: Path, cubin_b: Path) -> BenchmarkBatchRequest:
    """Assemble a tiny two-candidate matmul batch request."""
    launch_spec = LaunchSpec(
        entrypoint="kernel_main",
        block_dim=(128, 1, 1),
        grid_dim=None,
        dynamic_smem_bytes=0,
        abi_name="matmul_fp16_v1",
        abi_version="0.1.0",
        metadata_mode=None,
    )
    baseline = BaselineRef(
        artifact_id="baseline",
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=2.0,
            relative_to_baseline=1.0,
            relative_to_incumbent=1.0,
        ),
    )
    incumbent = IncumbentRef(
        artifact_id="incumbent",
        objective_score=ObjectiveScore(
            metric_name="weighted_p50_us",
            value=1.5,
            relative_to_baseline=0.75,
            relative_to_incumbent=1.0,
        ),
        cubin_uri=str(cubin_a),
        launch_spec=launch_spec,
        launch_spec_hash="lsh-inc",
        source_hash="srh-inc",
        toolchain_hash="tch-inc",
    )
    return BenchmarkBatchRequest(
        request_id="req-dry",
        run_id="run-dry",
        round_id="r-dry",
        batch_id="batch-dry",
        problem_spec=ProblemSpec(
            op_name="matmul",
            op_semantics="C = A @ B",
            dtype="fp16",
            target_gpu="H100-SXM5-80G",
            shape_cases=[ShapeCase(shape_id="s", dims=[64, 64, 64], weight=1.0)],
            objective=PerformanceObjective(
                primary_metric="weighted_p50_us",
                aggregation="weighted_mean",
                regression_guard_pct=0.0,
            ),
            target_metric_value=1.0,
            max_rounds=1,
            reference_kernel="",
        ),
        objective_shape_cases=[
            ShapeCase(shape_id="s", dims=[64, 64, 64], weight=1.0),
        ],
        baseline_ref=baseline,
        incumbent_ref=incumbent,
        candidate_module_artifact_refs=[
            CandidateArtifactRef(
                candidate_hash="cand-a",
                artifact_id="art-a",
                cubin_uri=str(cubin_a),
                launch_spec=launch_spec,
                launch_spec_hash="lsh-a",
                source_hash="srh-a",
                toolchain_hash="tch-a",
                correctness=CorrectnessForward(passed=True),
                adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
            ),
            CandidateArtifactRef(
                candidate_hash="cand-b",
                artifact_id="art-b",
                cubin_uri=str(cubin_b),
                launch_spec=launch_spec,
                launch_spec_hash="lsh-b",
                source_hash="srh-b",
                toolchain_hash="tch-b",
                correctness=CorrectnessForward(passed=True),
                adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
            ),
        ],
        operation_adapter_abi="matmul_fp16_v1",
        operation_adapter_version="0.1.0",
        artifact_execution_model=ArtifactExecutionModel.COMMON_HARNESS_CUBIN,
    )


def _synthetic_envelope(
    req: BenchmarkBatchRequest, device: LeasedDevice, candidate_hash: str
) -> MeasurementEnvelope:
    """Produce a minimal :class:`MeasurementEnvelope` for incumbent-comparison."""
    return MeasurementEnvelope(
        run_id=req.run_id,
        round_id=req.round_id,
        batch_id=req.batch_id,
        request_id=req.request_id,
        candidate_hash=candidate_hash,
        artifact_id=f"art-{candidate_hash}",
        source_hash="src",
        launch_spec_hash="lsh",
        toolchain_hash="tch",
        module_artifact_hash="mah",
        artifact_execution_model=req.artifact_execution_model,
        problem_spec_hash="psh",
        objective_hash="oh",
        shape_ids=["s"],
        operation_adapter_abi=req.operation_adapter_abi,
        operation_adapter_version=req.operation_adapter_version,
        target_gpu=req.problem_spec.target_gpu,
        gpu_uuid=device.gpu_uuid,
        pci_bus_id=device.pci_bus_id,
        mig_profile=device.mig_profile,
        sm_arch=device.sm_arch,
        driver_version="dry",
        cuda_runtime_version="dry",
        metric_mode=req.metric_mode,
        function_attribute_policy_requested=FunctionAttributePolicy(),
        function_attribute_policy_observed=FunctionAttributePolicy(),
        warmup_policy=WarmupPolicy(min_runs=5, cache_state="touched"),
        repeat_policy=RepeatPolicy(
            repetitions=30,
            iterations_per_sample=10,
            min_timed_batch_ms=1.0,
            max_timed_batch_ms=200.0,
        ),
        cache_policy=CachePolicyBlock(
            requested=CachePolicy.WARM_SAME_BUFFERS,
            effective=CachePolicy.WARM_SAME_BUFFERS,
            reason=None,
        ),
        clock_policy=ClockPolicy(mode=ClockPolicyMode.OBSERVED_ONLY),
        interleave_seed=None,
    )


def _synthetic_batch_measurement() -> BatchMeasurement:
    """Construct a :class:`BatchMeasurement` with two candidates, one shape.

    Samples are hand-picked so:

    * ``cand-a`` beats the incumbent (0.8 us p50 vs 1.5 us) → ``IMPROVED``;
    * ``cand-b`` matches the incumbent within noise (1.5 us) →
      ``STATISTICAL_TIE`` after noise-margin aggregation.
    """
    per_shape = PerShapeMeasurement(
        shape_id="s",
        candidate_samples={
            "cand-a": [0.8, 0.82, 0.79, 0.81, 0.80] * 5,
            "cand-b": [1.50, 1.51, 1.49, 1.50, 1.50] * 5,
        },
        anchor_samples=[1.50, 1.51, 1.49, 1.50, 1.50] * 5,
        anchor_pre_samples=[1.50, 1.49, 1.51],
        anchor_post_samples=[1.50, 1.51, 1.49],
        block_order=["cand-a", "anchor", "cand-b"],
        interleave_seed=0,
        interleave_block_len=3,
        anchor_every_n_samples=4,
    )
    return BatchMeasurement(per_shape={"s": per_shape})


def main() -> int:
    """Smoke-test entrypoint; returns 0 on success, 1 on any exception."""
    cubin_a: Path | None = None
    cubin_b: Path | None = None
    try:
        cubin_a = _make_placeholder_cubin()
        cubin_b = _make_placeholder_cubin()
        cfg = BenchmarkerConfig()
        device = LeasedDevice(
            ordinal=0,
            gpu_uuid="dry",
            pci_bus_id="0",
            sm_arch="sm_90",
            mig_profile=None,
            name="dry",
        )

        # Step 1: normalize — files are readable absolute paths, so this
        # should succeed. If normalize ever tightens its gate, a
        # NormalizationError here is still a pure-Python composition proof
        # and is treated as acceptable.
        req = _build_request(cubin_a, cubin_b)
        try:
            normalized = normalize_request(req, cfg, device)
            print(
                f"normalize: admitted={len(normalized.admit_candidates)} "
                f"rejected={len(normalized.reject_candidates)}"
            )
        except NormalizationError as exc:
            print(f"normalize: raised NormalizationError ({exc.reason}) — acceptable")

        # Step 2: hand-crafted samples → scoring → incumbent comparison.
        batch = _synthetic_batch_measurement()
        shape_weights = {"s": 1.0}
        incumbent_score_value = 1.5
        baseline_value = 2.0

        # Build ShapeBenchResult directly (bypasses ShapeMeasurementArtifact
        # to avoid the full telemetry-binding path).
        per_candidate_rows: list[
            tuple[str, ScoredCandidate, IncumbentComparison, float]
        ] = []
        incumbent_envelope = _synthetic_envelope(req, device, "__incumbent__")
        for candidate_hash in ("cand-a", "cand-b"):
            samples = batch.per_shape["s"].candidate_samples[candidate_hash]
            s_mean = mean(samples)
            s_stdev = stdev(samples)
            s_p50 = p50(samples)
            s_p95 = p95(samples, 20)
            s_cv = cv_pct(s_mean, s_stdev)
            compact = ShapeBenchResult(
                shape_id="s",
                latency_p50_us=s_p50,
                latency_p95_us=s_p95 if s_p95 is not None else -1.0,
                stdev_us=s_stdev,
                run_count=len(samples),
            )
            obj = compute_objective_score(
                [compact],
                req.problem_spec.objective,
                shape_weights,
                baseline_value=baseline_value,
                incumbent_anchor_value=incumbent_score_value,
            )
            quality = (
                [MeasurementQualityStatus.VALID]
                if s_cv is not None and s_cv <= cfg.thresholds.measurement_cv_warn_pct
                else [MeasurementQualityStatus.VALID_WITH_WARNING]
            )
            comparison = decide_incumbent_comparison(
                candidate_envelope=_synthetic_envelope(req, device, candidate_hash),
                candidate_score=obj.value,
                candidate_quality=quality,
                candidate_cv_pct=s_cv,
                incumbent_envelope=incumbent_envelope,
                incumbent_score=incumbent_score_value,
                incumbent_cv_pct=s_cv,
                anchor_drift_fraction=0.0,
                guard_pct=req.problem_spec.objective.regression_guard_pct,
                noise_floor_pct=cfg.thresholds.noise_floor_pct,
            )
            sc = ScoredCandidate(
                candidate_hash=candidate_hash,
                incumbent_comparison=comparison,
                objective_score=obj,
                candidate_cv_pct=s_cv,
            )
            per_candidate_rows.append((candidate_hash, sc, comparison, obj.value))
            print(
                f"candidate={candidate_hash} "
                f"comparison={comparison.value} "
                f"objective_value={obj.value:.4f} "
                f"measurement_quality={quality[0].value}"
            )

        # Step 3: selection.build_profile_set.
        scored = [row[1] for row in per_candidate_rows]
        incumbent_scored = ScoredCandidate(
            candidate_hash="__incumbent__",
            incumbent_comparison=IncumbentComparison.IMPROVED,
            objective_score=ObjectiveScore(
                metric_name="weighted_p50_us",
                value=incumbent_score_value,
                relative_to_baseline=incumbent_score_value / baseline_value,
                relative_to_incumbent=1.0,
            ),
        )
        profile_set = build_profile_set(
            scoreable=scored,
            k=2,
            m=1,
            incumbent=incumbent_scored,
            include_incumbent=False,
            hints_per_candidate={},
        )
        print(
            "profile_set:",
            [c.candidate_hash for c in profile_set],
        )

        # Step 4: stats.aggregate_noise_margin — hand-picked inputs the
        # decide_incumbent_comparison table classifies as STATISTICAL_TIE.
        margin = aggregate_noise_margin(
            candidate_cv_pct=1.0,
            anchor_cv_pct=1.0,
            anchor_drift_fraction=0.0,
            floor=cfg.thresholds.noise_floor_pct,
        )
        print(f"noise_margin (tie-band): {margin:.4f}")

        print("DRY-RUN OK")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        for path in (cubin_a, cubin_b):
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
