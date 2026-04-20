"""``MatmulAdapter`` — reference matrix-multiply operation adapter.

Implements ``OperationAdapter`` for ``op_name="matmul"`` per spec §6.2.
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

_CUDA_SCALAR_TYPE: dict[str, str] = {
    "fp16": "half",
    "float16": "half",
    "fp32": "float",
    "float32": "float",
}


class MatmulAdapter:
    """Matmul operation adapter (ABI: ``A, B, C, M, N, K``).

    Spec: §6.2 "MatmulAdapter semantics".
    """

    op_name: ClassVar[str] = "matmul"
    _ABI_NAME: ClassVar[str] = "matmul_v1"
    _ABI_VERSION: ClassVar[str] = "1.0"
    _ADAPTER_VERSION: ClassVar[str] = "matmul-v1.0"

    def adapter_version(self) -> str:
        """Return the adapter-version string baked into ``artifact_key``."""
        return self._ADAPTER_VERSION

    def abi_contract(self) -> tuple[str, str]:
        """Return ``(abi_name, abi_version)``."""
        return (self._ABI_NAME, self._ABI_VERSION)

    def default_block_dim(self, problem_spec: ProblemSpec) -> tuple[int, int, int]:
        """Return the legacy-inference default block dim (spec §6.2)."""
        del problem_spec
        return (16, 16, 1)

    def default_tolerance(self, dtype: str) -> float:
        """Per-dtype tolerance default (spec §6.2)."""
        if dtype in ("fp16", "float16"):
            return 1e-2
        if dtype in ("fp32", "float32"):
            return 1e-4
        return 1e-4

    def comparison_mode(self, dtype: str) -> ComparisonMode:
        """Float dtypes compare by tolerance (spec §6.2 / INV-CS-011)."""
        del dtype
        return ComparisonMode.TOLERANCE

    def high_risk_shape_ids(self, problem_spec: ProblemSpec) -> set[str]:
        """Matmul has no a priori high-risk shapes in V1."""
        del problem_spec
        return set()

    def allocate_inputs(
        self,
        problem_spec: ProblemSpec,
        shape: ShapeCase,
        seed: int,
    ) -> InputBundle:
        """Generate deterministic pseudo-random matmul inputs.

        Produces finite pseudo-random floats in ``[-1.0, 1.0]`` packed into
        the requested dtype (fp16 or fp32). Using raw random bytes would
        surface NaN/Inf bit patterns (~8% for fp16) and make tolerance
        comparisons degenerate; the adapter therefore generates values in
        a safe numeric range and encodes them explicitly.
        """
        if len(shape.dims) != 3:
            raise ValueError(f"matmul shape requires [M, N, K] dims, got {shape.dims}")
        m, n, k = shape.dims
        dtype = problem_spec.dtype
        dtype_bytes = _dtype_bytes(dtype)
        rng = _DeterministicRNG(seed)
        buf_a = rng.finite_floats_bytes(m * k, dtype)
        buf_b = rng.finite_floats_bytes(k * n, dtype)
        # Output buffer is zero-initialized — harness writes into it.
        buf_c = bytes(m * n * dtype_bytes)
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
        """Render a deterministic standalone matmul harness.

        The rendered program reads binary inputs from files, launches the
        kernel with the resolved execution spec, and writes outputs to a
        binary file — all paths come from argv so the correctness phase
        can point each executable at the right per-shape directory.
        """
        entrypoint = execution_spec.entrypoint or "matmul"
        block = execution_spec.block_dim or (16, 16, 1)
        dynamic_smem = execution_spec.dynamic_smem_bytes or 0
        scalar_type = _CUDA_SCALAR_TYPE.get(problem_spec.dtype, "float")
        role_tag = role.value
        return _render_matmul_harness(
            scalar_type=scalar_type,
            entrypoint=entrypoint,
            block_dim=block,
            dynamic_smem_bytes=dynamic_smem,
            kernel_source=kernel_source,
            role_tag=role_tag,
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
        """Compare two binary outputs element-wise.

        V1 reads the two output files as bytes and compares them pairwise
        using the dtype layout. The comparison is deterministic and does
        not invoke the GPU.
        """
        dtype = problem_spec.dtype
        ref_bytes = reference_output.read_bytes()
        cand_bytes = candidate_output.read_bytes()
        if len(ref_bytes) != len(cand_bytes):
            return ShapeComparisonResult(
                shape_id=shape.shape_id,
                passed=False,
                max_abs_error=None,
                max_rel_error=None,
            )
        max_abs, max_rel = _compare_scalars(
            ref_bytes, cand_bytes, dtype, tolerance, comparison_mode
        )
        tolerance_ok = (
            max_abs is None
            or (comparison_mode is ComparisonMode.EXACT and max_abs == 0.0)
            or (comparison_mode is ComparisonMode.TOLERANCE and max_abs <= tolerance)
        )
        return ShapeComparisonResult(
            shape_id=shape.shape_id,
            passed=tolerance_ok,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
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
    raise ValueError(f"unsupported matmul dtype: {dtype}")


class _DeterministicRNG:
    """Seeded pseudo-random generator with dtype-aware finite output.

    Produces finite floats in ``[-1.0, 1.0]`` and encodes them to the
    requested dtype. Using raw random bytes would routinely produce NaN
    / Inf bit patterns on float dtypes (~8% of fp16 patterns are NaN),
    which collapses tolerance comparisons into ``nan != nan``. This
    generator is pure Python; no numpy dependency.
    """

    def __init__(self, seed: int) -> None:
        import random

        self._random = random.Random(seed)

    def finite_floats_bytes(self, count: int, dtype: str) -> bytes:
        """Return ``count`` finite floats in [-1, 1] encoded per ``dtype``."""
        if count <= 0:
            return b""
        if dtype in ("fp32", "float32"):
            floats = [self._random.uniform(-1.0, 1.0) for _ in range(count)]
            return struct.pack(f"<{count}f", *floats)
        if dtype in ("fp16", "float16"):
            out = bytearray(count * 2)
            for i in range(count):
                value = self._random.uniform(-1.0, 1.0)
                out[i * 2 : i * 2 + 2] = _float_to_fp16(value)
            return bytes(out)
        raise ValueError(f"unsupported matmul dtype for input gen: {dtype}")


def _render_matmul_harness(
    *,
    scalar_type: str,
    entrypoint: str,
    block_dim: tuple[int, int, int],
    dynamic_smem_bytes: int,
    kernel_source: str,
    role_tag: str,
) -> str:
    """Render a minimal standalone matmul harness program."""
    bx, by, bz = block_dim
    return f"""// Auto-generated matmul harness ({role_tag}).
// Compiled as a standalone executable — per INV-CS-001, reference and
// candidate never share a translation unit.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

{kernel_source}

static void die(const char* msg, cudaError_t err) {{
    std::fprintf(stderr, "harness_error: %s (%s)\\n", msg, cudaGetErrorString(err));
    std::exit(2);
}}

int main(int argc, char** argv) {{
    if (argc != 7) {{
        std::fprintf(stderr, "usage: %s A_path B_path C_out_path M N K\\n", argv[0]);
        return 2;
    }}
    const char* a_path = argv[1];
    const char* b_path = argv[2];
    const char* c_out_path = argv[3];
    const int M = std::atoi(argv[4]);
    const int N = std::atoi(argv[5]);
    const int K = std::atoi(argv[6]);

    std::vector<{scalar_type}> h_a(M * K);
    std::vector<{scalar_type}> h_b(K * N);
    std::vector<{scalar_type}> h_c(M * N, {scalar_type}(0));

    FILE* fa = std::fopen(a_path, "rb");
    FILE* fb = std::fopen(b_path, "rb");
    if (!fa || !fb) {{ std::fprintf(stderr, "cannot open input files\\n"); return 2; }}
    std::fread(h_a.data(), sizeof({scalar_type}), h_a.size(), fa);
    std::fread(h_b.data(), sizeof({scalar_type}), h_b.size(), fb);
    std::fclose(fa);
    std::fclose(fb);

    {scalar_type}* d_a = nullptr;
    {scalar_type}* d_b = nullptr;
    {scalar_type}* d_c = nullptr;
    cudaError_t err = cudaMalloc(&d_a, h_a.size() * sizeof({scalar_type}));
    if (err != cudaSuccess) die("cudaMalloc A", err);
    err = cudaMalloc(&d_b, h_b.size() * sizeof({scalar_type}));
    if (err != cudaSuccess) die("cudaMalloc B", err);
    err = cudaMalloc(&d_c, h_c.size() * sizeof({scalar_type}));
    if (err != cudaSuccess) die("cudaMalloc C", err);

    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof({scalar_type}),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof({scalar_type}),
               cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, h_c.size() * sizeof({scalar_type}));

    dim3 block({bx}, {by}, {bz});
    dim3 grid((N + {bx} - 1) / {bx}, (M + {by} - 1) / {by}, 1);

    if ({dynamic_smem_bytes} > 0) {{
        cudaFuncSetAttribute(
            {entrypoint},
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            {dynamic_smem_bytes});
    }}

    {entrypoint}<<<grid, block, {dynamic_smem_bytes}>>>(d_a, d_b, d_c, M, N, K);
    err = cudaGetLastError();
    if (err != cudaSuccess) die("launch", err);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) die("synchronize", err);

    cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof({scalar_type}),
               cudaMemcpyDeviceToHost);

    FILE* fc = std::fopen(c_out_path, "wb");
    if (!fc) {{ std::fprintf(stderr, "cannot open output file\\n"); return 2; }}
    std::fwrite(h_c.data(), sizeof({scalar_type}), h_c.size(), fc);
    std::fclose(fc);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}}
"""


def _compare_scalars(
    ref_bytes: bytes,
    cand_bytes: bytes,
    dtype: str,
    tolerance: float,
    mode: ComparisonMode,
) -> tuple[float | None, float | None]:
    """Compare two scalar buffers and return (max_abs_error, max_rel_error)."""
    del tolerance, mode  # tolerance check happens in the caller
    if dtype in ("fp32", "float32"):
        fmt_char = "f"
        width = 4
    elif dtype in ("fp16", "float16"):
        return _compare_fp16(ref_bytes, cand_bytes)
    else:
        return (None, None)

    count = len(ref_bytes) // width
    if count == 0:
        return (None, None)
    ref_floats = struct.unpack(f"{count}{fmt_char}", ref_bytes[: count * width])
    cand_floats = struct.unpack(f"{count}{fmt_char}", cand_bytes[: count * width])
    max_abs = 0.0
    max_rel = 0.0
    for r, c in zip(ref_floats, cand_floats, strict=False):
        diff = abs(r - c)
        if diff > max_abs:
            max_abs = diff
        denom = max(abs(r), 1e-12)
        rel = diff / denom
        if rel > max_rel:
            max_rel = rel
    return (max_abs, max_rel)


def _compare_fp16(
    ref_bytes: bytes, cand_bytes: bytes
) -> tuple[float | None, float | None]:
    """Compare two fp16 byte buffers returning max-abs / max-rel errors."""
    count = len(ref_bytes) // 2
    if count == 0:
        return (None, None)
    refs = [_fp16_to_float(ref_bytes[i * 2 : i * 2 + 2]) for i in range(count)]
    cands = [_fp16_to_float(cand_bytes[i * 2 : i * 2 + 2]) for i in range(count)]
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


def _float_to_fp16(value: float) -> bytes:
    """Encode a finite Python float as a 2-byte little-endian IEEE 754 half.

    The caller is responsible for ensuring ``value`` is finite and within
    the fp16 normal range (``|x| < 2^16``); the matmul adapter only calls
    this with uniformly sampled values in ``[-1, 1]``, which are always
    representable as normal halfs.
    """
    if value != value:  # NaN guard, though adapter never passes one.
        return b"\x00\x7e"
    sign_bit = 0x8000 if value < 0.0 else 0
    magnitude = -value if value < 0.0 else value
    if magnitude == 0.0:
        return (sign_bit).to_bytes(2, byteorder="little")
    # Decompose magnitude = frac * 2^exp with 1 <= frac < 2.
    import math

    mantissa, exp2 = math.frexp(magnitude)
    # frexp returns mantissa in [0.5, 1); shift to [1, 2) and adjust exponent.
    mantissa *= 2.0
    exp2 -= 1
    biased_exp = exp2 + 15
    if biased_exp >= 0x1F:
        # Overflow → saturate to largest normal fp16 (~65504.0).
        bits = sign_bit | (0x1E << 10) | 0x3FF
        return bits.to_bytes(2, byteorder="little")
    if biased_exp <= 0:
        # Subnormal (or underflow to zero).
        # Subnormal formula: value = (frac / 1024) * 2^-14
        sub = magnitude / (2**-14)
        frac_int = int(round(sub * 1024.0))
        if frac_int >= 1024:
            # Rounded up into the smallest normal.
            bits = sign_bit | (1 << 10)
            return bits.to_bytes(2, byteorder="little")
        bits = sign_bit | (frac_int & 0x3FF)
        return bits.to_bytes(2, byteorder="little")
    frac_int = int(round((mantissa - 1.0) * 1024.0))
    if frac_int >= 1024:
        frac_int = 0
        biased_exp += 1
        if biased_exp >= 0x1F:
            bits = sign_bit | (0x1E << 10) | 0x3FF
            return bits.to_bytes(2, byteorder="little")
    bits = sign_bit | ((biased_exp & 0x1F) << 10) | (frac_int & 0x3FF)
    return bits.to_bytes(2, byteorder="little")


def _fp16_to_float(value: bytes) -> float:
    """Convert a 2-byte little-endian IEEE 754 half to Python float."""
    if len(value) != 2:
        return 0.0
    bits = int.from_bytes(value, byteorder="little")
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        if frac == 0:
            return -0.0 if sign else 0.0
        # Subnormal.
        subnormal: float = (frac / 1024.0) * (2**-14)
        return -subnormal if sign else subnormal
    if exp == 0x1F:
        if frac == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    magnitude: float = (1.0 + frac / 1024.0) * (2 ** (exp - 15))
    return -magnitude if sign else magnitude
