"""Pure-Python tests for compiler service invariants."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from kerlever.compiler_service.adapters.elementwise import ElementwiseAdapter
from kerlever.compiler_service.config import ServiceConfig
from kerlever.compiler_service.idempotency import IdempotencyRegistry
from kerlever.compiler_service.phases.phase3_compile import Phase3Compiler
from kerlever.compiler_service.phases.phase4_correctness import _shape_seed
from kerlever.compiler_service.sanitizer import ComputeSanitizerRunner
from kerlever.compiler_service.toolchain import DriverApiAttributes, NvccResult
from kerlever.compiler_service.types import (
    CompileResultStatus,
    IdempotencyState,
    PhaseName,
    SanitizerStatus,
    SanitizerTool,
)
from kerlever.types import PerformanceObjective, ProblemSpec, ShapeCase


def _problem_spec(*, dtype: str = "fp32") -> ProblemSpec:
    return ProblemSpec(
        op_name="elementwise",
        op_semantics="C = A + B",
        dtype=dtype,
        target_gpu="A100",
        shape_cases=[ShapeCase(shape_id="s", dims=[128])],
        objective=PerformanceObjective(
            primary_metric="weighted_p50_us",
            aggregation="weighted_mean",
            regression_guard_pct=0.0,
        ),
        target_metric_value=1.0,
        max_rounds=1,
        reference_kernel="__global__ void kernel() {}",
    )


@pytest.mark.asyncio
async def test_sanitizer_runner_passes_harness_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def fake_exec(*argv: str, **_kwargs: Any) -> FakeProcess:
        captured.extend(argv)
        return FakeProcess()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    executable = tmp_path / "candidate.out"
    executable.write_bytes(b"")
    runner = ComputeSanitizerRunner(
        Path("/usr/local/cuda/bin/compute-sanitizer"), ServiceConfig()
    )
    shape = ShapeCase(shape_id="s", dims=[16, 16, 16])

    outcome = await runner.run(
        tool=SanitizerTool.MEMCHECK,
        executable=executable,
        shape=shape,
        input_dir=tmp_path,
        harness_args=("A.bin", "B.bin", "C.bin", "16", "16", "16"),
        report_path=tmp_path / "memcheck.report",
    )

    assert outcome.status is SanitizerStatus.PASS
    assert captured[-7:] == [
        str(executable),
        "A.bin",
        "B.bin",
        "C.bin",
        "16",
        "16",
        "16",
    ]


def test_shape_seed_is_stable_and_problem_dependent() -> None:
    shape = ShapeCase(shape_id="s", dims=[128])
    spec = _problem_spec(dtype="fp32")

    assert _shape_seed(spec, shape) == _shape_seed(spec, shape)
    assert _shape_seed(spec, shape) != _shape_seed(
        _problem_spec(dtype="float32"), shape
    )


def test_reference_compile_error_is_infra_not_candidate() -> None:
    compiler = Phase3Compiler.__new__(Phase3Compiler)
    result = compiler._short_circuit_reference_compile_error(
        phase2=object(),  # type: ignore[arg-type]
        result=NvccResult(
            returncode=1,
            stdout_excerpt="",
            stderr_excerpt="reference failed",
            truncated=False,
            command="nvcc reference.cu",
            timed_out=False,
        ),
    )

    assert result.short_circuit is not None
    assert result.short_circuit.status is CompileResultStatus.INFRA_ERROR
    assert result.short_circuit.candidate_fault_kind is None
    assert result.short_circuit.failure.reason == "reference_compile_error"


@pytest.mark.asyncio
async def test_idempotency_tracks_observed_phase() -> None:
    registry = IdempotencyRegistry(ttl=timedelta(hours=1))

    intake = await registry.observe_intake("req", "artifact-key")
    assert intake.state is IdempotencyState.NEW

    await registry.record_phase("req", PhaseName.COMPILE)
    replay = await registry.observe_intake("req", "artifact-key")

    assert replay.state is IdempotencyState.PRIOR_ATTEMPT_LOST
    assert replay.prior_attempt_observed_phase is PhaseName.COMPILE


def test_elementwise_rejects_non_fp32_dtype() -> None:
    adapter = ElementwiseAdapter()

    assert adapter.validate_problem_spec(_problem_spec(dtype="fp32")) is None
    assert (
        adapter.validate_problem_spec(_problem_spec(dtype="int32"))
        == "unsupported_elementwise_dtype"
    )


def test_driver_api_attributes_reads_cuda_function_attribute(tmp_path: Path) -> None:
    class Success:
        value = 0
        name = "CUDA_SUCCESS"

    class FunctionAttribute:
        CU_FUNC_ATTRIBUTE_NUM_REGS = "regs"

    class FakeDriver:
        CUfunction_attribute = FunctionAttribute

        def __init__(self) -> None:
            self.function_name: bytes | None = None
            self.unloaded = False

        def cuInit(self, _flags: int) -> tuple[Success]:  # noqa: N802
            return (Success(),)

        def cuModuleLoadData(self, data: bytes) -> tuple[Success, str]:  # noqa: N802
            assert data == b"cubin"
            return (Success(), "module")

        def cuModuleGetFunction(  # noqa: N802
            self, module: str, name: bytes
        ) -> tuple[Success, str]:
            assert module == "module"
            self.function_name = name
            return (Success(), "function")

        def cuFuncGetAttribute(  # noqa: N802
            self, attribute: str, function: str
        ) -> tuple[Success, int]:
            assert attribute == "regs"
            assert function == "function"
            return (Success(), 64)

        def cuModuleUnload(self, module: str) -> tuple[Success]:  # noqa: N802
            assert module == "module"
            self.unloaded = True
            return (Success(),)

    cubin = tmp_path / "candidate.cubin"
    cubin.write_bytes(b"cubin")
    fake_driver = FakeDriver()
    attrs = DriverApiAttributes(fake_driver)

    assert attrs.read_registers_per_thread(cubin, "kernel_main") == 64
    assert fake_driver.function_name == b"kernel_main"
    assert fake_driver.unloaded is True
