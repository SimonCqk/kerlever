"""Pure-Python tests for benchmarker service invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.cuda_driver import CudaFunction, CudaStream
from kerlever.benchmarker.harness import HarnessConfig, execute_batch
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.normalize import NormalizationError, normalize_request
from kerlever.benchmarker.plan import CalibratedPlan, LoadedCandidate, SamplePlan
from kerlever.benchmarker.profile_child import _profile_seed
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    ArtifactExecutionModel,
    BaselineRef,
    BenchmarkBatchRequest,
    CachePolicy,
    CandidateArtifactRef,
    ClockPolicy,
    ClockPolicyMode,
    CorrectnessForward,
    FunctionAttributePolicy,
    IncumbentRef,
    LaunchSpec,
    MetricMode,
    ObjectiveScore,
    PerformanceObjective,
    ProblemSpec,
    ShapeCase,
)
from kerlever.benchmarker.worker import _adapter_seed


def _cubin(tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.cubin"
    path.write_bytes(b"\0")
    return str(path)


def _launch_spec() -> LaunchSpec:
    return LaunchSpec(
        entrypoint="kernel_main",
        block_dim=(128, 1, 1),
        grid_dim=None,
        dynamic_smem_bytes=0,
        abi_name="elementwise_add_fp32_v1",
        abi_version="0.1.0",
    )


def _score(value: float) -> ObjectiveScore:
    return ObjectiveScore(
        metric_name="weighted_p50_us",
        value=value,
        relative_to_baseline=1.0,
        relative_to_incumbent=1.0,
    )


def _request(
    tmp_path: Path,
    *,
    include_incumbent: bool = True,
) -> BenchmarkBatchRequest:
    shape = ShapeCase(shape_id="s", dims=[1024], weight=1.0)
    launch = _launch_spec()
    incumbent = IncumbentRef(
        artifact_id="inc",
        objective_score=_score(1.0),
    )
    if include_incumbent:
        incumbent = incumbent.model_copy(
            update={
                "cubin_uri": _cubin(tmp_path, "inc"),
                "launch_spec": launch,
                "launch_spec_hash": "lsh-inc",
                "source_hash": "src-inc",
                "toolchain_hash": "tool-inc",
            }
        )
    return BenchmarkBatchRequest(
        request_id="req",
        run_id="run",
        batch_id="batch",
        problem_spec=ProblemSpec(
            op_name="add",
            op_semantics="C = A + B",
            dtype="fp32",
            target_gpu="A100",
            shape_cases=[shape],
            objective=PerformanceObjective(
                primary_metric="weighted_p50_us",
                aggregation="weighted_mean",
                regression_guard_pct=0.0,
            ),
            target_metric_value=1.0,
            max_rounds=1,
            reference_kernel="",
        ),
        objective_shape_cases=[shape],
        baseline_ref=BaselineRef(artifact_id="base", objective_score=_score(2.0)),
        incumbent_ref=incumbent,
        candidate_module_artifact_refs=[
            CandidateArtifactRef(
                candidate_hash="cand-a",
                artifact_id="art-a",
                cubin_uri=_cubin(tmp_path, "cand-a"),
                launch_spec=launch,
                launch_spec_hash="lsh-a",
                source_hash="src-a",
                toolchain_hash="tool-a",
                correctness=CorrectnessForward(passed=True),
                adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
            ),
            CandidateArtifactRef(
                candidate_hash="cand-b",
                artifact_id="art-b",
                cubin_uri=_cubin(tmp_path, "cand-b"),
                launch_spec=launch,
                launch_spec_hash="lsh-b",
                source_hash="src-b",
                toolchain_hash="tool-b",
                correctness=CorrectnessForward(passed=True),
                adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
            ),
        ],
        operation_adapter_abi="elementwise_add_fp32_v1",
        operation_adapter_version="0.1.0",
        artifact_execution_model=ArtifactExecutionModel.COMMON_HARNESS_CUBIN,
        metric_mode=MetricMode.DEVICE_KERNEL_US,
        clock_policy=ClockPolicy(mode=ClockPolicyMode.OBSERVED_ONLY),
    )


def _device() -> LeasedDevice:
    return LeasedDevice(
        ordinal=0,
        gpu_uuid="GPU-test",
        pci_bus_id="0000:00:00.0",
        sm_arch="sm_80",
    )


def test_normalize_requires_real_incumbent(tmp_path: Path) -> None:
    req = _request(tmp_path, include_incumbent=False)

    with pytest.raises(NormalizationError) as exc:
        normalize_request(req, BenchmarkerConfig(), _device())

    assert exc.value.reason == "incumbent_artifact_required"


def test_normalize_promotes_interleaved_cache_policy(tmp_path: Path) -> None:
    req = _request(tmp_path)

    normalized = normalize_request(req, BenchmarkerConfig(), _device())

    assert normalized.requested_cache_policy == CachePolicy.WARM_SAME_BUFFERS
    assert normalized.effective_cache_policy == CachePolicy.WARM_ROTATING_BUFFERS
    assert normalized.cache_policy_reason == "interleaved_batch_requires_rotation"


def test_profile_seed_matches_worker_seed() -> None:
    assert _adapter_seed("run", "batch", "shape", "cand") == _profile_seed(
        "run", "batch", "shape", "cand"
    )


def test_execute_batch_uses_shape_aware_grid_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = ShapeCase(shape_id="s", dims=[1024])
    cand = LoadedCandidate(
        candidate_hash="cand",
        function=CudaFunction(handle=object(), entrypoint="kernel_main"),
        launch_args_factory=None,
        adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
        function_attribute_policy_observed=FunctionAttributePolicy(),
        block_dim=(128, 1, 1),
        grid_dim=None,
        dynamic_smem_bytes=0,
    )
    incumbent = LoadedCandidate(
        candidate_hash="__incumbent__",
        function=CudaFunction(handle=object(), entrypoint="kernel_main"),
        launch_args_factory=None,
        adapter_iteration_semantics=AdapterIterationSemantics.OVERWRITE_PURE,
        function_attribute_policy_observed=FunctionAttributePolicy(),
        block_dim=(128, 1, 1),
        grid_dim=None,
        dynamic_smem_bytes=0,
    )
    plan = CalibratedPlan(
        sample_plans={
            ("cand", "s"): SamplePlan(
                iterations_per_sample=1,
                repetitions=1,
                warmup_count=0,
            )
        },
        requested_cache_policy=CachePolicy.WARM_SAME_BUFFERS,
        effective_cache_policy=CachePolicy.WARM_SAME_BUFFERS,
        cache_policy_reason=None,
        metric_mode=MetricMode.DEVICE_KERNEL_US,
        adapter_iteration_semantics_per_candidate={
            "cand": AdapterIterationSemantics.OVERWRITE_PURE
        },
        function_attribute_policy_observed_per_candidate={
            "cand": FunctionAttributePolicy()
        },
    )
    grids: list[tuple[int, int, int]] = []

    def fake_run_sample(*args: Any, **_kwargs: Any) -> float:
        grids.append(args[6])
        return 1.0

    monkeypatch.setattr("kerlever.benchmarker.harness.run_sample", fake_run_sample)

    execute_batch(
        plan=plan,
        candidates=[cand],
        incumbent=incumbent,
        shapes=[shape],
        seeds={"s": 1},
        cfg=HarnessConfig(repetitions=1),
        stream=CudaStream(handle=object()),
        build_args=lambda _cand, _shape: (),
        resolve_grid_dim=lambda c, _shape: (7, 1, 1)
        if c.candidate_hash == "cand"
        else (3, 1, 1),
    )

    assert (7, 1, 1) in grids
    assert (3, 1, 1) in grids
