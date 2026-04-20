"""Benchmarker — profile child subprocess entrypoint.

Invoked by NCU for each ``(candidate, profile_shape)`` tuple the worker
selected in Phase 6. The child's single responsibility is to replay exactly
one measured launch inside one NVTX range so NCU can profile it.

The child is **never** the worker: it gets its own CUDA context, loads the
same cubin again, and exits with a typed non-zero code on any failure
(spec §6.12, design §3.5 / §4.4 / §8.3).

Exit codes (design §4.4 / spec §6.12):

* ``0`` — success; `.ncu-rep` is valid upstream.
* ``1`` — CUDA launch/sync failure during the measured loop.
* ``2`` — uncaught Python exception.
* ``11`` — function-attribute policy apply failed.
* ``12`` — adapter resolution failed.

Spec: docs/benchmarker/spec.md §6.12
Design: docs/benchmarker/design.md §3.5, §4.4, §8.3
Implements: REQ-BENCH-030, SC-BENCH-013
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from kerlever.benchmarker.config import BenchmarkerConfig
from kerlever.benchmarker.lease import LeasedDevice
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    BenchmarkBatchRequest,
    CandidateArtifactRef,
    FunctionAttribute,
    IncumbentRef,
)

logger = logging.getLogger(__name__)


EX_OK = 0
EX_CUDA_FAILED = 1
EX_UNCAUGHT = 2
EX_FUNC_ATTR_APPLY_FAILED = 11
EX_ADAPTER_UNREGISTERED = 12


@dataclass(frozen=True)
class ProfileChildArgs:
    """Parsed profile-child argv."""

    config_file: Path
    request_file: Path
    candidate_hash: str
    shape_id: str
    nvtx_range: str
    iterations: int
    device_ordinal: int
    device_uuid: str


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the profile child.

    Split out so ``python -m kerlever.benchmarker.profile_child --help`` runs
    without touching CUDA or NVML — useful for CI smoke tests (REQ-BENCH-030
    verification requires the module import cleanly on non-GPU hosts).
    """
    parser = argparse.ArgumentParser(
        prog="kerlever.benchmarker.profile_child",
        description=(
            "Kerlever Benchmarker profile child — replays one measured "
            "kernel launch inside one NVTX range for NCU to profile."
        ),
    )
    parser.add_argument("--config-file", required=True, type=Path)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--candidate-hash", required=True, type=str)
    parser.add_argument("--shape-id", required=True, type=str)
    parser.add_argument("--nvtx-range", required=True, type=str)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--device-ordinal", required=True, type=int)
    parser.add_argument("--device-uuid", required=True, type=str)
    return parser


def _parse_argv(argv: list[str]) -> ProfileChildArgs:
    """Parse argv into a :class:`ProfileChildArgs` record."""
    parser = _build_parser()
    ns = parser.parse_args(argv)
    return ProfileChildArgs(
        config_file=ns.config_file,
        request_file=ns.request_file,
        candidate_hash=ns.candidate_hash,
        shape_id=ns.shape_id,
        nvtx_range=ns.nvtx_range,
        iterations=ns.iterations,
        device_ordinal=ns.device_ordinal,
        device_uuid=ns.device_uuid,
    )


def _load_config(path: Path) -> BenchmarkerConfig:
    """Rehydrate the config serialized by the worker."""
    return BenchmarkerConfig.from_dict(json.loads(path.read_text()))


def _load_request(path: Path) -> BenchmarkBatchRequest:
    """Pydantic-parse the request serialized by the worker."""
    return BenchmarkBatchRequest.model_validate_json(path.read_text())


def _find_candidate(
    req: BenchmarkBatchRequest, candidate_hash: str
) -> CandidateArtifactRef | IncumbentRef:
    """Return the candidate ref (or incumbent) matching ``candidate_hash``.

    The sentinel ``__incumbent__`` selects the request's incumbent ref so the
    profile child can profile the incumbent anchor too (spec §6.6 allows the
    incumbent in the profile set when ``include_incumbent = True``).
    """
    if candidate_hash == "__incumbent__":
        return req.incumbent_ref
    for ref in req.candidate_module_artifact_refs:
        if ref.candidate_hash == candidate_hash:
            return ref
    raise ValueError(f"unknown candidate_hash: {candidate_hash!r}")


def _find_shape(req: BenchmarkBatchRequest, shape_id: str) -> object:
    """Find the ``ShapeCase`` in the request by id."""
    for shape in list(req.objective_shape_cases) + list(req.profile_shape_cases):
        if shape.shape_id == shape_id:
            return shape
    raise ValueError(f"unknown shape_id: {shape_id!r}")


def _apply_function_attributes(
    fn: object,
    ref: CandidateArtifactRef | IncumbentRef,
) -> None:
    """Apply the requested function-attribute policy.

    Only attributes whose field is non-null on the candidate's launch spec
    (for candidates) or on the incumbent's launch spec (for the incumbent)
    are applied. Any :class:`CudaDriverError` raised here exits with
    ``EX_FUNC_ATTR_APPLY_FAILED`` so NCU reports ``profiler_replay_refused``
    upstream (spec §6.12).

    Implements: REQ-BENCH-029 (V1 scope: ``max_dynamic_shared_memory_size``
        only; other FunctionAttributePolicy fields deferred per spec §9)
    """
    from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415

    launch_spec = ref.launch_spec
    if launch_spec is None:
        return
    dynamic_smem = launch_spec.dynamic_smem_bytes
    if dynamic_smem and dynamic_smem > 0:
        cd.set_function_attribute(
            fn,  # type: ignore[arg-type]
            FunctionAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
            int(dynamic_smem),
        )


def _import_registry_modules(cfg: BenchmarkerConfig) -> None:
    """Re-load out-of-tree adapter modules declared in config.

    V1 built-ins register automatically on ``import kerlever.benchmarker.adapter``
    so we rely on that side-effect rather than calling a dedicated
    ``register_builtin_adapters`` function.
    """
    for mod_path in cfg.adapter_registry_modules:
        try:
            importlib.import_module(mod_path)
        except Exception as exc:  # noqa: BLE001 — logged and re-raised as needed
            logger.error(
                "profile_child.adapter_plugin_import_failed",
                extra={"module": mod_path, "error": str(exc)},
            )
            raise


def _profile_seed(
    run_id: str, batch_id: str, shape_id: str, candidate_hash: str
) -> int:
    """Reproduce the worker's profile-seed hash so inputs match the measurement."""
    return hash(
        (run_id, batch_id, shape_id, candidate_hash, "kerlever_profile_seed")
    )


def _nvtx_push(name: str) -> None:
    """Push an NVTX range with a safe fallback when the library is absent."""
    try:
        from nvtx import push_range  # noqa: PLC0415

        push_range(message=name)
    except Exception:
        logger.debug("nvtx.push.unavailable", extra={"range": name})


def _nvtx_pop() -> None:
    """Pop the most recent NVTX range."""
    try:
        from nvtx import pop_range  # noqa: PLC0415

        pop_range()
    except Exception:
        logger.debug("nvtx.pop.unavailable")


def main(argv: list[str]) -> NoReturn:
    """Profile-child entrypoint.

    Flow (spec §6.12):

    1. Parse argv, load config + request.
    2. Resolve adapter — exit 12 on miss.
    3. Init telemetry + CUDA; create primary context + stream.
    4. Read cubin + load module + resolve entrypoint.
    5. Apply function-attribute policy — exit 11 on apply failure.
    6. Adapter allocate + seed_inputs.
    7. Build launch args + grid; run untimed warmup.
    8. NVTX push → ``iterations`` launches → NVTX pop.
    9. Stream-synchronize; free buffers; destroy context.

    Implements: REQ-BENCH-030
    Invariant: INV-BENCH-012 (os._exit — no CPython atexit / finalizers run)
    """
    try:
        args = _parse_argv(argv)
    except SystemExit:
        # argparse has already printed a diagnostic; exit as uncaught.
        raise
    except BaseException as exc:  # noqa: BLE001 — exit path, no classification
        logger.error(
            "profile_child.argv_parse_failed", extra={"error": str(exc)}
        )
        os._exit(EX_UNCAUGHT)

    try:
        cfg = _load_config(args.config_file)
        logging.basicConfig(level=cfg.log_level)
        _import_registry_modules(cfg)
        # The import side-effect of adapter.py registers V1 built-ins.
        from kerlever.benchmarker import adapter as adapter_mod  # noqa: PLC0415

        req = _load_request(args.request_file)
        ref = _find_candidate(req, args.candidate_hash)
        shape = _find_shape(req, args.shape_id)

        operation = adapter_mod.get_adapter(
            req.operation_adapter_abi, req.operation_adapter_version
        )
        if operation is None:
            logger.error(
                "profile_child.adapter_unregistered",
                extra={
                    "abi": req.operation_adapter_abi,
                    "version": req.operation_adapter_version,
                },
            )
            os._exit(EX_ADAPTER_UNREGISTERED)

        # Lazy imports so ``--help`` works without cuda-python / nvtx / pynvml.
        from kerlever.benchmarker import cuda_driver as cd  # noqa: PLC0415
        from kerlever.benchmarker.telemetry import (
            init as telemetry_init,  # noqa: PLC0415
        )
        from kerlever.benchmarker.telemetry import (
            shutdown as telemetry_shutdown,  # noqa: PLC0415
        )

        try:
            telemetry_init()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "profile_child.nvml_init_skipped",
                extra={"error": str(exc)},
            )

        cd.init()
        ctx = cd.create_primary_context(args.device_ordinal)
        stream = cd.create_stream()

        launch_spec = ref.launch_spec
        if launch_spec is None:
            logger.error(
                "profile_child.missing_launch_spec",
                extra={"candidate_hash": args.candidate_hash},
            )
            os._exit(EX_ADAPTER_UNREGISTERED)
        cubin_uri = ref.cubin_uri
        if cubin_uri is None:
            logger.error(
                "profile_child.missing_cubin_uri",
                extra={"candidate_hash": args.candidate_hash},
            )
            os._exit(EX_ADAPTER_UNREGISTERED)
        cubin_bytes = Path(cubin_uri).read_bytes()
        module = cd.load_module(cubin_bytes, None)
        fn = cd.get_function(module, launch_spec.entrypoint)

        try:
            _apply_function_attributes(fn, ref)
        except cd.CudaDriverError as exc:
            logger.error(
                "profile_child.function_attribute_apply_failed",
                extra={
                    "candidate_hash": args.candidate_hash,
                    "error": str(exc),
                },
            )
            os._exit(EX_FUNC_ATTR_APPLY_FAILED)

        device = LeasedDevice(
            ordinal=args.device_ordinal,
            gpu_uuid=args.device_uuid,
            pci_bus_id="unknown",
            sm_arch="unknown",
        )
        buffers = operation.allocate(shape, req.problem_spec.dtype, device)  # type: ignore[arg-type]
        seed = _profile_seed(
            req.run_id, req.batch_id, args.shape_id, args.candidate_hash
        )
        operation.seed_inputs(buffers, shape, seed)  # type: ignore[arg-type]
        launch_args = operation.build_launch_args(buffers, shape)  # type: ignore[arg-type]
        grid = operation.grid_dim(shape, launch_spec.block_dim)  # type: ignore[arg-type]
        block_dim = launch_spec.block_dim
        dynamic_smem = launch_spec.dynamic_smem_bytes

        # Untimed warmup — adapter semantics dispatch honored (INV-BENCH-013).
        semantics = operation.iteration_semantics
        warmup_count = max(0, cfg.calibration.warmup_min_runs)
        for i in range(warmup_count):
            if (
                i > 0
                and semantics != AdapterIterationSemantics.OVERWRITE_PURE
            ):
                operation.reset_between_iterations(buffers, semantics)
            cd.launch(
                fn, grid, block_dim, dynamic_smem, stream, tuple(launch_args)
            )
        cd.stream_synchronize(stream)

        # NVTX-wrapped timed launches — this is the one NCU profiles.
        try:
            _nvtx_push(args.nvtx_range)
            try:
                for i in range(max(1, args.iterations)):
                    if (
                        i > 0
                        and semantics
                        != AdapterIterationSemantics.OVERWRITE_PURE
                    ):
                        operation.reset_between_iterations(buffers, semantics)
                    cd.launch(
                        fn,
                        grid,
                        block_dim,
                        dynamic_smem,
                        stream,
                        tuple(launch_args),
                    )
            finally:
                _nvtx_pop()
            cd.stream_synchronize(stream)
        except cd.CudaDriverError as exc:
            logger.error(
                "profile_child.launch_failed",
                extra={
                    "candidate_hash": args.candidate_hash,
                    "shape_id": args.shape_id,
                    "error": str(exc),
                },
            )
            os._exit(EX_CUDA_FAILED)

        operation.free(buffers)
        cd.destroy_stream(stream)
        cd.destroy_primary_context(ctx)
        # Best-effort NVML teardown — a failure here cannot undo the
        # successful measurement already recorded on the NCU side.
        with contextlib.suppress(Exception):
            telemetry_shutdown()
        os._exit(EX_OK)
    except SystemExit:
        # argparse or explicit sys.exit flows fall through cleanly.
        raise
    except BaseException as exc:  # noqa: BLE001 — uncaught top-level
        logger.error(
            "profile_child.uncaught",
            extra={"error": f"{type(exc).__name__}:{exc}"},
        )
        os._exit(EX_UNCAUGHT)


if __name__ == "__main__":  # pragma: no cover — direct subprocess entry
    main(sys.argv[1:])
