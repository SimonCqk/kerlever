"""``ElementwiseAdapter`` — minimal elementwise op adapter (V1 skeleton).

Exists so the adapter interface is exercised by two concrete adapters
(INV-CS-013); no matmul-specific code may leak into the core.

V1 non-goal (documented): this adapter is restricted to **two-operand
arithmetic ops** with the ABI ``(A, B, C, N)`` — i.e. ``C = f(A, B)``
element-wise. Unary and n-ary elementwise ops are out of scope for V1;
a dedicated adapter task will extend the surface once the two-operand
path has stabilised. The rendered harness hard-codes the four-argument
signature; submitting a kernel with a different arity will fail at
Phase 4 launch time with a candidate runtime error.

Spec: docs/compiler-service/spec.md §6.2 "ElementwiseAdapter semantics"
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import ClassVar

from kerlever.compiler_service.adapters.base import (
    InputBundle,
    ShapeComparisonResult,
)
from kerlever.compiler_service.types import (
    CandidateRole,
    ComparisonMode,
    KernelExecutionSpec,
)
from kerlever.types import ProblemSpec, ShapeCase

_INT_DTYPES: frozenset[str] = frozenset({"int8", "int16", "int32", "int64"})


class ElementwiseAdapter:
    """Skeleton elementwise adapter (ABI: ``A, B, C, N``)."""

    op_name: ClassVar[str] = "elementwise"
    _ABI_NAME: ClassVar[str] = "elementwise_v1"
    _ABI_VERSION: ClassVar[str] = "1.0"
    _ADAPTER_VERSION: ClassVar[str] = "elementwise-v1.0"

    def adapter_version(self) -> str:
        """Return the adapter-version string baked into ``artifact_key``."""
        return self._ADAPTER_VERSION

    def abi_contract(self) -> tuple[str, str]:
        """Return ``(abi_name, abi_version)``."""
        return (self._ABI_NAME, self._ABI_VERSION)

    def default_block_dim(self, problem_spec: ProblemSpec) -> tuple[int, int, int]:
        """Return the legacy-inference default block dim (spec §6.2)."""
        del problem_spec
        return (256, 1, 1)

    def default_tolerance(self, dtype: str) -> float:
        """Per-dtype tolerance default (spec §6.2)."""
        if dtype in ("fp16", "float16"):
            return 1e-2
        if dtype in ("fp32", "float32"):
            return 1e-5
        if dtype in _INT_DTYPES:
            return 0.0
        return 1e-4

    def comparison_mode(self, dtype: str) -> ComparisonMode:
        """Integers use exact; floats use tolerance (spec §6.2 / INV-CS-011)."""
        if dtype in _INT_DTYPES:
            return ComparisonMode.EXACT
        return ComparisonMode.TOLERANCE

    def high_risk_shape_ids(self, problem_spec: ProblemSpec) -> set[str]:
        """Elementwise has no a priori high-risk shapes in V1."""
        del problem_spec
        return set()

    def allocate_inputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        seed: int,
    ) -> InputBundle:
        """Generate deterministic pseudo-random elementwise inputs."""
        if len(shape.dims) != 1:
            raise ValueError(f"elementwise shape requires [N] dims, got {shape.dims}")
        n = shape.dims[0]
        dtype_bytes = _dtype_bytes(problem_spec.dtype)
        import random

        rng = random.Random(seed)
        buf_a = bytes(rng.randrange(0, 256) for _ in range(n * dtype_bytes))
        buf_b = bytes(rng.randrange(0, 256) for _ in range(n * dtype_bytes))
        buf_c = bytes(n * dtype_bytes)
        return InputBundle(
            shape_id=shape.shape_id,
            buffers={"A": buf_a, "B": buf_b, "C": buf_c},
        )

    def build_harness_source(
        self,
        execution_spec: KernelExecutionSpec,
        problem_spec: ProblemSpec,
        role: CandidateRole,
        kernel_source: str,
    ) -> str:
        """Render a minimal elementwise harness."""
        del problem_spec, role
        entrypoint = execution_spec.entrypoint or "elementwise"
        block = execution_spec.block_dim or (256, 1, 1)
        dynamic_smem = execution_spec.dynamic_smem_bytes or 0
        return _render_elementwise_harness(
            entrypoint=entrypoint,
            block_dim=block,
            dynamic_smem_bytes=dynamic_smem,
            kernel_source=kernel_source,
        )

    def compare_outputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        reference_output: Path,
        candidate_output: Path,
        tolerance: float,
        comparison_mode: ComparisonMode,
    ) -> ShapeComparisonResult:
        """Compare two output files byte-for-byte (exact) or float-wise."""
        ref_bytes = reference_output.read_bytes()
        cand_bytes = candidate_output.read_bytes()
        if len(ref_bytes) != len(cand_bytes):
            return ShapeComparisonResult(
                shape_id=shape.shape_id,
                passed=False,
                max_abs_error=None,
                max_rel_error=None,
            )
        if comparison_mode is ComparisonMode.EXACT:
            passed = ref_bytes == cand_bytes
            abs_err: float | None = 0.0 if passed else None
            return ShapeComparisonResult(
                shape_id=shape.shape_id,
                passed=passed,
                max_abs_error=abs_err,
                max_rel_error=abs_err,
            )
        if problem_spec.dtype in ("fp32", "float32"):
            max_abs, max_rel = _compare_float32(ref_bytes, cand_bytes)
            passed = max_abs is not None and max_abs <= tolerance
            return ShapeComparisonResult(
                shape_id=shape.shape_id,
                passed=passed,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
            )
        # Fallback: byte-equal check.
        passed = ref_bytes == cand_bytes
        return ShapeComparisonResult(
            shape_id=shape.shape_id,
            passed=passed,
            max_abs_error=0.0 if passed else None,
            max_rel_error=0.0 if passed else None,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dtype_bytes(dtype: str) -> int:
    """Byte width of a supported scalar dtype."""
    if dtype in ("fp16", "float16"):
        return 2
    if dtype in ("fp32", "float32"):
        return 4
    if dtype in ("int8",):
        return 1
    if dtype in ("int16",):
        return 2
    if dtype in ("int32",):
        return 4
    if dtype in ("int64",):
        return 8
    raise ValueError(f"unsupported elementwise dtype: {dtype}")


def _render_elementwise_harness(
    *,
    entrypoint: str,
    block_dim: tuple[int, int, int],
    dynamic_smem_bytes: int,
    kernel_source: str,
) -> str:
    """Render a minimal elementwise harness (skeleton)."""
    bx, by, bz = block_dim
    return f"""// Auto-generated elementwise harness (skeleton).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

{kernel_source}

int main(int argc, char** argv) {{
    if (argc != 5) {{
        std::fprintf(stderr, "usage: %s A_path B_path C_out_path N\\n", argv[0]);
        return 2;
    }}
    const int N = std::atoi(argv[4]);
    std::vector<float> h_a(N), h_b(N), h_c(N, 0);
    FILE* fa = std::fopen(argv[1], "rb");
    FILE* fb = std::fopen(argv[2], "rb");
    if (!fa || !fb) return 2;
    std::fread(h_a.data(), sizeof(float), N, fa);
    std::fread(h_b.data(), sizeof(float), N, fb);
    std::fclose(fa);
    std::fclose(fb);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block({bx}, {by}, {bz});
    dim3 grid((N + {bx} - 1) / {bx}, 1, 1);

    if ({dynamic_smem_bytes} > 0) {{
        cudaFuncSetAttribute(
            {entrypoint},
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            {dynamic_smem_bytes});
    }}

    {entrypoint}<<<grid, block, {dynamic_smem_bytes}>>>(d_a, d_b, d_c, N);
    if (cudaGetLastError() != cudaSuccess) return 3;
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;

    cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* fc = std::fopen(argv[3], "wb");
    if (!fc) return 2;
    std::fwrite(h_c.data(), sizeof(float), N, fc);
    std::fclose(fc);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}}
"""


def _compare_float32(
    ref_bytes: bytes, cand_bytes: bytes
) -> tuple[float | None, float | None]:
    """Compare two fp32 byte buffers returning max-abs / max-rel errors."""
    count = len(ref_bytes) // 4
    if count == 0:
        return (None, None)
    refs = struct.unpack(f"{count}f", ref_bytes[: count * 4])
    cands = struct.unpack(f"{count}f", cand_bytes[: count * 4])
    max_abs = 0.0
    max_rel = 0.0
    for r, c in zip(refs, cands, strict=False):
        diff = abs(r - c)
        if diff > max_abs:
            max_abs = diff
        denom = max(abs(r), 1e-12)
        rel = diff / denom
        if rel > max_rel:
            max_rel = rel
    return (max_abs, max_rel)
