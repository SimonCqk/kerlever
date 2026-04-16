"""Coding Agent hardware — GPU constraint table and lookup.

Provides hardcoded specifications for supported GPU architectures.
Unknown GPUs receive conservative default values that are safe for all
NVIDIA GPUs.

Spec: docs/coding-agent/spec.md §6.1
"""

from __future__ import annotations

from kerlever.coding_agent.types import GPUSpec

GPU_SPECS: dict[str, GPUSpec] = {
    "V100": GPUSpec(
        arch="sm_70",
        smem_per_sm_kb=96,
        max_smem_per_block_kb=96,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=0.9,
        l2_cache_mb=6,
        supports_cp_async=False,
        supports_tma=False,
        supports_fp8=False,
        tensor_core_types=["fp16"],
    ),
    "A100": GPUSpec(
        arch="sm_80",
        smem_per_sm_kb=164,
        max_smem_per_block_kb=163,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=2.0,
        l2_cache_mb=40,
        supports_cp_async=True,
        supports_tma=False,
        supports_fp8=False,
        tensor_core_types=["fp16", "bf16", "tf32", "fp64", "int8", "int4"],
    ),
    "H100": GPUSpec(
        arch="sm_90",
        smem_per_sm_kb=228,
        max_smem_per_block_kb=227,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=64,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=3.35,
        l2_cache_mb=50,
        supports_cp_async=True,
        supports_tma=True,
        supports_fp8=True,
        tensor_core_types=["fp16", "bf16", "tf32", "fp64", "int8", "fp8"],
    ),
    "T4": GPUSpec(
        arch="sm_75",
        smem_per_sm_kb=64,
        max_smem_per_block_kb=64,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=32,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=0.3,
        l2_cache_mb=4,
        supports_cp_async=False,
        supports_tma=False,
        supports_fp8=False,
        tensor_core_types=["fp16", "int8", "int4"],
    ),
    "L40": GPUSpec(
        arch="sm_89",
        smem_per_sm_kb=100,
        max_smem_per_block_kb=99,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=48,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=0.864,
        l2_cache_mb=96,
        supports_cp_async=True,
        supports_tma=False,
        supports_fp8=True,
        tensor_core_types=["fp16", "bf16", "tf32", "int8", "fp8"],
    ),
    "RTX4090": GPUSpec(
        arch="sm_89",
        smem_per_sm_kb=100,
        max_smem_per_block_kb=99,
        registers_per_sm=65536,
        max_registers_per_thread=255,
        max_warps_per_sm=48,
        max_threads_per_block=1024,
        hbm_bandwidth_tbps=1.008,
        l2_cache_mb=72,
        supports_cp_async=True,
        supports_tma=False,
        supports_fp8=True,
        tensor_core_types=["fp16", "bf16", "tf32", "int8", "fp8"],
    ),
}

_DEFAULT_GPU_SPEC = GPUSpec(
    arch="sm_70",
    smem_per_sm_kb=48,
    max_smem_per_block_kb=48,
    registers_per_sm=65536,
    max_registers_per_thread=255,
    max_warps_per_sm=64,
    max_threads_per_block=1024,
    hbm_bandwidth_tbps=0.9,
    l2_cache_mb=6,
    supports_cp_async=False,
    supports_tma=False,
    supports_fp8=False,
    tensor_core_types=["fp16"],
)


def get_gpu_spec(target_gpu: str) -> GPUSpec:
    """Look up the GPU specification for the given target GPU name.

    Normalizes the input by stripping whitespace and converting to uppercase.
    Returns conservative default values for unrecognized GPU names.

    Args:
        target_gpu: GPU name string (e.g., "A100", "h100", " V100 ").

    Returns:
        The GPUSpec for the target GPU, or conservative defaults if not found.

    Implements: REQ-CA-004, REQ-CA-011
    Invariant: INV-CA-005 (GPU constraint lookup never fails)
    """
    normalized = target_gpu.strip().upper()
    return GPU_SPECS.get(normalized, _DEFAULT_GPU_SPEC)


def format_gpu_spec(spec: GPUSpec) -> str:
    """Format a GPUSpec into a human-readable summary for the LLM system prompt.

    Advanced features that are not supported are omitted to avoid confusing
    the LLM into generating code that uses unsupported features.

    Args:
        spec: The GPU specification to format.

    Returns:
        Multi-line string summarizing the GPU constraints.

    Implements: REQ-CA-004
    """
    lines = [
        f"Architecture: {spec.arch}",
        f"Shared memory per SM: {spec.smem_per_sm_kb} KB",
        f"Max shared memory per block: {spec.max_smem_per_block_kb} KB",
        f"Registers per SM: {spec.registers_per_sm}",
        f"Max registers per thread: {spec.max_registers_per_thread}",
        f"Max warps per SM: {spec.max_warps_per_sm}",
        f"Max threads per block: {spec.max_threads_per_block}",
        f"HBM bandwidth: {spec.hbm_bandwidth_tbps} TB/s",
        f"L2 cache: {spec.l2_cache_mb} MB",
    ]

    if spec.supports_cp_async:
        lines.append("cp.async: Supported (Ampere+)")
    else:
        lines.append("cp.async: Not available")

    if spec.supports_tma:
        lines.append("TMA: Supported (Hopper+)")
    else:
        lines.append("TMA: Not available")

    if spec.supports_fp8:
        lines.append("FP8: Supported (Hopper+)")
    else:
        lines.append("FP8: Not available")

    if spec.tensor_core_types:
        lines.append(f"Tensor Core types: {', '.join(spec.tensor_core_types)}")

    return "\n".join(lines)
