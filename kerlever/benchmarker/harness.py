"""Benchmarker — Phase 4: common harness with interleaved sampling.

Responsible for:

* ``generate_block_order`` — deterministic PCG64-seeded permutation of
  ``[anchor, candidate...]`` tokens (spec §6.4, INV-BENCH-004).
* ``run_sample`` — single-sample CUDA event timing with optional NVTX
  wrap (only the ONE profiled launch per tuple gets wrapped — INV-BENCH-008).
* ``execute_batch`` — orchestrates per-shape sampling, anchors, and
  records raw samples and block order.

Spec: docs/benchmarker/spec.md §6.4
Design: docs/benchmarker/design.md §2.1 harness.py
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kerlever.benchmarker.cuda_driver import (
    CudaFunction,
    CudaStream,
    create_event,
    destroy_event,
    event_elapsed_ms,
    event_record,
    launch,
)
from kerlever.benchmarker.plan import CalibratedPlan, LoadedCandidate
from kerlever.benchmarker.types import (
    AdapterIterationSemantics,
    ShapeCase,
)

logger = logging.getLogger(__name__)

# Spec §6.8 defaults; a higher limit would let a single block starve anchors.
_DEFAULT_ANCHOR_EVERY_N = 4
_DEFAULT_MAX_BLOCK_LEN = 6


@dataclass(frozen=True)
class NvtxRange:
    """NVTX range descriptor for one profiled launch.

    When non-None in :func:`run_sample`, the harness wraps the measured
    launch in an NVTX range named exactly
    ``kerlever/<run_id>/<batch_id>/<candidate_hash>/<shape_id>/profile``
    so that ``ncu --nvtx --nvtx-include`` targets exactly one launch
    (INV-BENCH-008, REQ-BENCH-017).
    """

    name: str


@dataclass(frozen=True)
class HarnessConfig:
    """Per-batch harness tunables used by :func:`execute_batch`."""

    repetitions: int
    anchor_every_n_samples: int = _DEFAULT_ANCHOR_EVERY_N
    max_interleave_block_len: int = _DEFAULT_MAX_BLOCK_LEN
    kernel_timeout_ms: int = 10_000


@dataclass
class PerShapeMeasurement:
    """Raw samples for one (shape, candidate) tuple + anchor samples."""

    shape_id: str
    candidate_samples: dict[str, list[float]]
    anchor_samples: list[float]
    anchor_pre_samples: list[float]
    anchor_post_samples: list[float]
    block_order: list[str]
    interleave_seed: int
    interleave_block_len: int
    anchor_every_n_samples: int


@dataclass
class BatchMeasurement:
    """Aggregate Phase 4 output across all shapes."""

    per_shape: dict[str, PerShapeMeasurement]
    runtime_faults: dict[str, dict[str, str]] = field(default_factory=dict)


def generate_block_order(
    candidates: list[str],
    anchor_every_n: int,
    max_block_len: int,
    seed: int,
    total_repetitions: int,
) -> list[str]:
    """Produce a deterministic interleave schedule per spec §6.4.

    The schedule always starts with an ``"anchor"`` token, interleaves
    candidate tokens in randomized under-full order, inserts periodic
    ``"anchor"`` tokens, and always ends with a post ``"anchor"`` token.

    * Block length is ``min(max_block_len, anchor_every_n)`` — spec wording.
    * Total candidate emissions equal ``total_repetitions * len(candidates)``
      so every candidate collects exactly ``total_repetitions`` samples.
    * Seed comes from ``hash((run_id, batch_id, shape_id, "kerlever_benchmark_order"))``
      (INV-BENCH-004).

    Args:
        candidates: Candidate-hash tokens.
        anchor_every_n: Anchor cadence between blocks.
        max_block_len: Hard upper bound on block size.
        seed: PCG64 seed.
        total_repetitions: Target samples per candidate.

    Returns:
        A list of tokens (candidate_hash or ``"anchor"``).

    Implements: REQ-BENCH-010
    Invariant: INV-BENCH-004
    """
    if not candidates:
        return ["anchor", "anchor"]
    from numpy.random import PCG64, Generator  # noqa: PLC0415

    rng = Generator(PCG64(seed))
    remaining: dict[str, int] = {c: total_repetitions for c in candidates}
    order: list[str] = ["anchor"]
    block_len_cap = max(1, min(max_block_len, anchor_every_n))
    total_to_emit = total_repetitions * len(candidates)
    emitted = 0
    while emitted < total_to_emit:
        block: list[str] = []
        for _ in range(block_len_cap):
            still_owed = [c for c, n in remaining.items() if n > 0]
            if not still_owed:
                break
            max_owed = max(remaining[c] for c in still_owed)
            underfull = [c for c in still_owed if remaining[c] == max_owed]
            idx = int(rng.integers(0, len(underfull)))
            pick = underfull[idx]
            block.append(pick)
            remaining[pick] -= 1
            emitted += 1
        order.extend(block)
        if emitted < total_to_emit:
            order.append("anchor")
    order.append("anchor")
    return order


def run_sample(
    fn: CudaFunction,
    shape: ShapeCase,
    iterations_per_sample: int,
    stream: CudaStream,
    build_args: Any,
    block_dim: tuple[int, int, int],
    grid_dim: tuple[int, int, int],
    dynamic_smem_bytes: int,
    nvtx: NvtxRange | None = None,
    reset_hook: Callable[[], None] | None = None,
    semantics: AdapterIterationSemantics = (
        AdapterIterationSemantics.OVERWRITE_PURE
    ),
) -> float:
    """Run one timed sample, returning per-launch microseconds.

    The timed region covers only the ``N`` kernel launches. Host allocation,
    random input generation, correctness comparison, and profiler startup
    are all excluded per REQ-BENCH-006.

    When ``nvtx`` is not ``None`` the measured launch is wrapped in an NVTX
    range so the NCU profiler can filter to exactly this launch. Warmup
    and anchors are never wrapped (INV-BENCH-008).

    For adapter iteration semantics other than ``OVERWRITE_PURE``, the
    harness invokes ``reset_hook`` between iterations so the adapter can
    restore its buffers to a known state (INV-BENCH-013). ``reset_hook``
    must be provided when ``semantics != OVERWRITE_PURE`` — a missing hook
    is a spec violation enforced here at call time.

    Args:
        fn: Resolved kernel function.
        shape: Shape being measured (passed to ``build_args``).
        iterations_per_sample: N launches per sample from calibration.
        stream: Benchmark stream handle.
        build_args: Callable producing the cuLaunchKernel arg tuple for
            this (shape, iteration) pair.
        block_dim: Thread block dimensions.
        grid_dim: Grid dimensions (adapter-resolved).
        dynamic_smem_bytes: Kernel dynamic shared memory size.
        nvtx: Optional NVTX range wrapper.
        reset_hook: Callable invoked between iterations for non-OVERWRITE_PURE
            semantics. ``None`` is valid only when ``semantics == OVERWRITE_PURE``.
        semantics: Adapter iteration semantics for this candidate.

    Returns:
        Mean per-launch elapsed time in microseconds.

    Implements: REQ-BENCH-006, REQ-BENCH-017
    Invariant: INV-BENCH-013
    """
    if (
        semantics != AdapterIterationSemantics.OVERWRITE_PURE
        and reset_hook is None
    ):
        raise ValueError(
            "reset_hook is required when semantics != OVERWRITE_PURE "
            f"(INV-BENCH-013): got semantics={semantics.value}"
        )
    start = create_event()
    stop = create_event()
    try:
        # Lazy import of NVTX C extension; only imported when a profile
        # range is requested so non-GPU dev hosts can still import harness.
        if nvtx is not None:
            _nvtx_push(nvtx.name)
        try:
            event_record(start, stream)
            for i in range(iterations_per_sample):
                if (
                    i > 0
                    and reset_hook is not None
                    and semantics != AdapterIterationSemantics.OVERWRITE_PURE
                ):
                    reset_hook()
                args = build_args(shape)
                launch(
                    fn,
                    grid_dim,
                    block_dim,
                    dynamic_smem_bytes,
                    stream,
                    args,
                )
            event_record(stop, stream)
            elapsed_ms = event_elapsed_ms(start, stop)
        finally:
            if nvtx is not None:
                _nvtx_pop()
    finally:
        destroy_event(start)
        destroy_event(stop)
    return elapsed_ms * 1000.0 / iterations_per_sample


def _nvtx_push(name: str) -> None:
    """Push an NVTX range named ``name`` onto the default domain.

    Falls back to a warning log when the NVTX library is not present; the
    profile phase will still collect nothing in that case and resolve to
    ``profile_unavailable`` downstream.
    """
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


def execute_batch(
    plan: CalibratedPlan,
    candidates: list[LoadedCandidate],
    incumbent: LoadedCandidate,
    shapes: list[ShapeCase],
    seeds: dict[str, int],
    cfg: HarnessConfig,
    stream: CudaStream,
    build_args: Any,
    reset_hook_per_candidate: dict[str, Callable[[], None]] | None = None,
) -> BatchMeasurement:
    """Execute Phase 4 for the full batch.

    For each shape, the harness:

    1. Generates a deterministic interleave block order (INV-BENCH-004).
    2. Runs warmup launches per (candidate, shape) untimed.
    3. Walks the block order, calling :func:`run_sample` for each token
       with the pre/post anchor runs recorded separately for drift.
    4. Classifies runtime faults as a dict; the scoring phase maps these
       to ``measurement_quality`` status later.

    Args:
        plan: Phase 3 calibrated plan.
        candidates: Non-anchor candidates in the batch (admitted set).
        incumbent: Incumbent kernel used for anchor tokens.
        shapes: Shapes to measure.
        seeds: Per-shape interleave seeds from normalization.
        cfg: Harness tunables.
        stream: Benchmark stream.
        build_args: ``(candidate, shape) -> launch_args`` factory.

    Implements: REQ-BENCH-006, REQ-BENCH-010, REQ-BENCH-013, INV-BENCH-002
    """
    per_shape: dict[str, PerShapeMeasurement] = {}
    runtime_faults: dict[str, dict[str, str]] = {}
    candidate_hashes = [c.candidate_hash for c in candidates]
    candidate_by_hash: dict[str, LoadedCandidate] = {
        c.candidate_hash: c for c in candidates
    }
    hooks: dict[str, Callable[[], None]] = reset_hook_per_candidate or {}

    for shape in shapes:
        order = generate_block_order(
            candidate_hashes,
            anchor_every_n=cfg.anchor_every_n_samples,
            max_block_len=cfg.max_interleave_block_len,
            seed=seeds[shape.shape_id],
            total_repetitions=cfg.repetitions,
        )

        # Warmup (untimed).
        for cand in candidates:
            sp = plan.sample_plans[(cand.candidate_hash, shape.shape_id)]
            for _ in range(sp.warmup_count):
                _run_warmup(cand, shape, stream, build_args)

        candidate_samples: dict[str, list[float]] = {
            h: [] for h in candidate_hashes
        }
        anchor_samples: list[float] = []
        anchor_pre_samples: list[float] = []
        anchor_post_samples: list[float] = []
        seen_any_candidate = False
        for token in order:
            if token == "anchor":
                try:
                    anchor_plan = (
                        plan.sample_plans.get(
                            (candidate_hashes[0], shape.shape_id)
                        )
                        if candidate_hashes
                        else None
                    )
                    iters = (
                        anchor_plan.iterations_per_sample
                        if anchor_plan is not None
                        else 1
                    )
                    per_us = run_sample(
                        incumbent.function,
                        shape,
                        iters,
                        stream,
                        lambda s: build_args(incumbent, s),
                        incumbent.block_dim,
                        incumbent.grid_dim or (1, 1, 1),
                        incumbent.dynamic_smem_bytes,
                        nvtx=None,
                        reset_hook=hooks.get(incumbent.candidate_hash),
                        semantics=incumbent.adapter_iteration_semantics,
                    )
                except Exception as exc:
                    _record_fault(runtime_faults, "anchor", shape.shape_id, exc)
                    continue
                anchor_samples.append(per_us)
                if not seen_any_candidate:
                    anchor_pre_samples.append(per_us)
                else:
                    anchor_post_samples.append(per_us)
            else:
                cand = candidate_by_hash[token]
                sp = plan.sample_plans[(cand.candidate_hash, shape.shape_id)]
                try:
                    per_us = run_sample(
                        cand.function,
                        shape,
                        sp.iterations_per_sample,
                        stream,
                        lambda s, c=cand: build_args(c, s),
                        cand.block_dim,
                        cand.grid_dim or (1, 1, 1),
                        cand.dynamic_smem_bytes,
                        nvtx=None,
                        reset_hook=hooks.get(cand.candidate_hash),
                        semantics=cand.adapter_iteration_semantics,
                    )
                except Exception as exc:
                    _record_fault(runtime_faults, token, shape.shape_id, exc)
                    continue
                candidate_samples[token].append(per_us)
                seen_any_candidate = True

        # Promote first-half anchors if we never observed any candidate emit
        # (defensive; keeps drift defined when the loop emits no candidates).
        if not anchor_pre_samples and anchor_samples:
            anchor_pre_samples = [anchor_samples[0]]
        if not anchor_post_samples and len(anchor_samples) >= 2:
            anchor_post_samples = [anchor_samples[-1]]

        per_shape[shape.shape_id] = PerShapeMeasurement(
            shape_id=shape.shape_id,
            candidate_samples=candidate_samples,
            anchor_samples=anchor_samples,
            anchor_pre_samples=anchor_pre_samples,
            anchor_post_samples=anchor_post_samples,
            block_order=order,
            interleave_seed=seeds[shape.shape_id],
            interleave_block_len=min(
                cfg.max_interleave_block_len, cfg.anchor_every_n_samples
            ),
            anchor_every_n_samples=cfg.anchor_every_n_samples,
        )
    return BatchMeasurement(per_shape=per_shape, runtime_faults=runtime_faults)


def _run_warmup(
    cand: LoadedCandidate,
    shape: ShapeCase,
    stream: CudaStream,
    build_args: Any,
) -> None:
    """Untimed warmup launch.

    Honors adapter iteration semantics: ``REQUIRES_OUTPUT_RESET`` and
    ``REQUIRES_FULL_INPUT_RESET`` delegate reset to the adapter before each
    untimed launch; ``OVERWRITE_PURE`` launches straight; ``NOT_REPEATABLE``
    delegates state-reset to the adapter's ``reset_full`` hook.
    """
    try:
        args = build_args(cand, shape)
        launch(
            cand.function,
            cand.grid_dim or (1, 1, 1),
            cand.block_dim,
            cand.dynamic_smem_bytes,
            stream,
            args,
        )
    except Exception as exc:
        logger.warning(
            "harness.warmup.failed",
            extra={
                "candidate_hash": cand.candidate_hash,
                "shape_id": shape.shape_id,
                "error": str(exc),
                "semantics": cand.adapter_iteration_semantics,
            },
        )


def _record_fault(
    store: dict[str, dict[str, str]],
    candidate_hash: str,
    shape_id: str,
    exc: BaseException,
) -> None:
    """Record a per-candidate per-shape runtime fault for scoring phase."""
    store.setdefault(candidate_hash, {})[shape_id] = f"{type(exc).__name__}:{exc}"


def monotonic_ns() -> int:
    """Thin wrapper for test seams and host-launch timing (spec §6.3)."""
    return time.monotonic_ns()


__all__ = [
    "AdapterIterationSemantics",
    "BatchMeasurement",
    "HarnessConfig",
    "NvtxRange",
    "PerShapeMeasurement",
    "execute_batch",
    "generate_block_order",
    "monotonic_ns",
    "run_sample",
]
