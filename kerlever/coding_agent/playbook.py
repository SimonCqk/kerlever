"""Coding Agent playbook — 6-layer CUDA optimization knowledge base.

Provides structured optimization techniques organized by bottleneck type
and hardware generation. The playbook is queried per generation call to
inject relevant techniques into the LLM system prompt.

Spec: docs/coding-agent/spec.md §6.2
"""

from __future__ import annotations

import re

from kerlever.coding_agent.types import GPUSpec, PlaybookLayer, PlaybookTechnique

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

LAYER_1 = PlaybookLayer(
    layer_number=1,
    name="Block/Grid Configuration",
    techniques=[
        PlaybookTechnique(
            name="block_size_tuning",
            layer=1,
            applicable_when="Always",
            expected_gain="10-50%",
            template=(
                "Try 128, 256, 512 threads per block; must be a multiple of 32 "
                "(warp size)."
            ),
            caveats=("Larger blocks reduce occupancy if register pressure is high."),
        ),
        PlaybookTechnique(
            name="grid_sizing",
            layer=1,
            applicable_when="Always",
            expected_gain="10-50%",
            template=(
                "Grid dim = ceiling division of problem size by block size; "
                "account for wave quantization (grid should be a multiple of "
                "SM count)."
            ),
            caveats="Over-provisioning grid wastes scheduling overhead.",
        ),
        PlaybookTechnique(
            name="launch_bounds_declaration",
            layer=1,
            applicable_when="Always",
            expected_gain="10-50%",
            template=(
                "__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) "
                "on every kernel."
            ),
            caveats=("Without this, the compiler may spill registers aggressively."),
        ),
    ],
)

LAYER_2 = PlaybookLayer(
    layer_number=2,
    name="Memory Access Optimization",
    techniques=[
        PlaybookTechnique(
            name="coalesced_access",
            layer=2,
            applicable_when="Strided or scattered global memory access",
            expected_gain="10-30%",
            template=(
                "Consecutive threads access consecutive addresses; transpose "
                "access patterns where needed."
            ),
            caveats="May require data layout changes.",
        ),
        PlaybookTechnique(
            name="shared_memory_tiling",
            layer=2,
            applicable_when="Repeated global memory reads of same data",
            expected_gain="10-30%",
            template=(
                "Load tile into __shared__ memory, __syncthreads(), compute "
                "from shared."
            ),
            caveats=("Shared memory capacity is limited per block (check GPUSpec)."),
        ),
        PlaybookTechnique(
            name="vectorized_loads",
            layer=2,
            applicable_when="Aligned, contiguous memory access",
            expected_gain="10-30%",
            template=("Use float4/int4 for 128-bit transactions per thread."),
            caveats=("Requires 16-byte alignment; not applicable to scattered access."),
        ),
        PlaybookTechnique(
            name="async_copy",
            layer=2,
            applicable_when="Bulk data movement from global to shared (Ampere+)",
            expected_gain="10-30%",
            template=("cp.async with __pipeline_memop_async / memcpy_async."),
            caveats="Only on GPUs with supports_cp_async = True.",
        ),
        PlaybookTechnique(
            name="bank_conflict_avoidance",
            layer=2,
            applicable_when=(
                "Shared memory access with stride that is a multiple of 32"
            ),
            expected_gain="10-30%",
            template=(
                "Pad shared memory declaration (+1 column) or use swizzled indexing."
            ),
            caveats="Extra shared memory usage from padding.",
        ),
    ],
)

LAYER_3 = PlaybookLayer(
    layer_number=3,
    name="Compute Optimization",
    techniques=[
        PlaybookTechnique(
            name="mixed_precision",
            layer=3,
            applicable_when=(
                "Accumulation allows reduced precision for intermediate results"
            ),
            expected_gain="5-15%",
            template=("FP16/BF16 for computation, FP32 for accumulation."),
            caveats=("Requires careful handling of numerical stability."),
        ),
        PlaybookTechnique(
            name="tensor_core_utilization",
            layer=3,
            applicable_when=(
                "Matrix operations with compatible shapes (multiples of 16)"
            ),
            expected_gain="5-15%",
            template=(
                "wmma / mma.sync intrinsics; ensure tile dimensions align with "
                "TC requirements."
            ),
            caveats=("Shape alignment is mandatory; not all operations map to TC."),
        ),
        PlaybookTechnique(
            name="loop_unrolling",
            layer=3,
            applicable_when="Inner loops with known trip count",
            expected_gain="5-15%",
            template="#pragma unroll or #pragma unroll N.",
            caveats=("Over-unrolling increases register pressure and code size."),
        ),
        PlaybookTechnique(
            name="fma_utilization",
            layer=3,
            applicable_when="Multiply-add sequences",
            expected_gain="5-15%",
            template=(
                "Use fma() or let compiler generate FMA by structuring "
                "a * b + c patterns."
            ),
            caveats=(
                "Compiler usually handles this, but manual can help with "
                "complex expressions."
            ),
        ),
    ],
)

LAYER_4 = PlaybookLayer(
    layer_number=4,
    name="Advanced Techniques",
    techniques=[
        PlaybookTechnique(
            name="thread_coarsening",
            layer=4,
            applicable_when=(
                "Each thread processes only one element; occupancy is sufficient"
            ),
            expected_gain="5-20%",
            template=(
                "Each thread processes K elements in a loop; reduces thread "
                "scheduling overhead."
            ),
            caveats="Too much coarsening reduces parallelism.",
        ),
        PlaybookTechnique(
            name="kernel_fusion",
            layer=4,
            applicable_when=(
                "Multiple sequential kernels with global memory round-trips"
            ),
            expected_gain="5-20%",
            template=(
                "Merge kernels into one; intermediate results stay in registers "
                "or shared memory."
            ),
            caveats="Increases kernel complexity and register pressure.",
        ),
        PlaybookTechnique(
            name="persistent_kernels",
            layer=4,
            applicable_when=("Grid-level load imbalance or many small tasks"),
            expected_gain="5-20%",
            template=(
                "Launch exactly SM_count blocks; each block loops over work "
                "items from a global queue."
            ),
            caveats=("Requires careful synchronization for the work queue."),
        ),
        PlaybookTechnique(
            name="split_k",
            layer=4,
            applicable_when=(
                "Parallelism limited along M/N dimensions in matmul-like operations"
            ),
            expected_gain="5-20%",
            template=(
                "Partition the K dimension across blocks; reduce partial results."
            ),
            caveats=("Requires a second reduction kernel or atomic operations."),
        ),
    ],
)

# Layer 5 is split into architecture-specific sub-groups
_AMPERE_TECHNIQUES = [
    PlaybookTechnique(
        name="ampere_cp_async",
        layer=5,
        applicable_when="Bulk global-to-shared data movement (Ampere+)",
        expected_gain="5-15%",
        template=("cp.async.cg.shared.global with pipeline stages."),
        caveats="Requires supports_cp_async = True.",
    ),
    PlaybookTechnique(
        name="ampere_l2_persistence",
        layer=5,
        applicable_when="Working set fits in L2 (Ampere+)",
        expected_gain="5-15%",
        template=("cudaAccessPolicyWindow to pin data in L2 cache."),
        caveats="Only effective when data reuse fits L2 size.",
    ),
]

_HOPPER_TECHNIQUES = [
    PlaybookTechnique(
        name="hopper_tma",
        layer=5,
        applicable_when="Tensor data movement (Hopper+)",
        expected_gain="5-15%",
        template=(
            "TMA descriptors for bulk async copy with hardware address generation."
        ),
        caveats="Requires supports_tma = True.",
    ),
    PlaybookTechnique(
        name="hopper_clusters",
        layer=5,
        applicable_when="Cross-SM shared memory access (Hopper+)",
        expected_gain="5-15%",
        template=("Thread block clusters with distributed shared memory."),
        caveats="Requires sm_90+; changes programming model.",
    ),
    PlaybookTechnique(
        name="hopper_fp8",
        layer=5,
        applicable_when="Inference or training with FP8 tolerance (Hopper+)",
        expected_gain="5-15%",
        template="FP8 tensor core operations.",
        caveats=("Requires supports_fp8 = True; numerical precision tradeoff."),
    ),
]

# Layer 6: kernel-specific algorithms
_LAYER_6_ALGORITHMS: dict[str, PlaybookTechnique] = {
    "matmul": PlaybookTechnique(
        name="matmul_goto_algorithm",
        layer=6,
        applicable_when="matmul, gemm, batched_matmul",
        expected_gain="Significant (algorithmic)",
        template=(
            "Hierarchical tiling: thread-block tile -> warp tile -> thread "
            "tile; M-N-K loop ordering; register-level accumulation."
        ),
        caveats="Requires careful tile size selection for occupancy.",
    ),
    "gemm": PlaybookTechnique(
        name="matmul_goto_algorithm",
        layer=6,
        applicable_when="matmul, gemm, batched_matmul",
        expected_gain="Significant (algorithmic)",
        template=(
            "Hierarchical tiling: thread-block tile -> warp tile -> thread "
            "tile; M-N-K loop ordering; register-level accumulation."
        ),
        caveats="Requires careful tile size selection for occupancy.",
    ),
    "batched_matmul": PlaybookTechnique(
        name="matmul_goto_algorithm",
        layer=6,
        applicable_when="matmul, gemm, batched_matmul",
        expected_gain="Significant (algorithmic)",
        template=(
            "Hierarchical tiling: thread-block tile -> warp tile -> thread "
            "tile; M-N-K loop ordering; register-level accumulation."
        ),
        caveats="Requires careful tile size selection for occupancy.",
    ),
    "attention": PlaybookTechnique(
        name="attention_online_softmax",
        layer=6,
        applicable_when="attention, flash_attention",
        expected_gain="Significant (algorithmic)",
        template=(
            "Online softmax (Flash Attention): compute softmax statistics in "
            "a single pass; never materialize the full attention matrix; tile "
            "along sequence dimension."
        ),
        caveats="Complex implementation; requires careful numerical handling.",
    ),
    "flash_attention": PlaybookTechnique(
        name="attention_online_softmax",
        layer=6,
        applicable_when="attention, flash_attention",
        expected_gain="Significant (algorithmic)",
        template=(
            "Online softmax (Flash Attention): compute softmax statistics in "
            "a single pass; never materialize the full attention matrix; tile "
            "along sequence dimension."
        ),
        caveats="Complex implementation; requires careful numerical handling.",
    ),
    "sum": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "mean": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "max": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "min": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "norm": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "reduction": PlaybookTechnique(
        name="reduction_warp_shuffle",
        layer=6,
        applicable_when="sum, mean, max, min, norm, reduction",
        expected_gain="Significant (algorithmic)",
        template=(
            "Warp-level shuffle reduction (__shfl_down_sync); block-level tree "
            "reduction; grid-level atomic or multi-pass."
        ),
        caveats="Warp shuffle requires all threads in warp to participate.",
    ),
    "layernorm": PlaybookTechnique(
        name="normalization_welford",
        layer=6,
        applicable_when="layernorm, batchnorm, rmsnorm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Welford's online algorithm for numerically stable single-pass "
            "mean/variance; fused normalization+activation."
        ),
        caveats="Single-pass requires careful accumulator management.",
    ),
    "batchnorm": PlaybookTechnique(
        name="normalization_welford",
        layer=6,
        applicable_when="layernorm, batchnorm, rmsnorm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Welford's online algorithm for numerically stable single-pass "
            "mean/variance; fused normalization+activation."
        ),
        caveats="Single-pass requires careful accumulator management.",
    ),
    "rmsnorm": PlaybookTechnique(
        name="normalization_welford",
        layer=6,
        applicable_when="layernorm, batchnorm, rmsnorm",
        expected_gain="Significant (algorithmic)",
        template=(
            "Welford's online algorithm for numerically stable single-pass "
            "mean/variance; fused normalization+activation."
        ),
        caveats="Single-pass requires careful accumulator management.",
    ),
    "conv2d": PlaybookTechnique(
        name="convolution_algorithms",
        layer=6,
        applicable_when="conv2d, depthwise_conv",
        expected_gain="Significant (algorithmic)",
        template=(
            "Implicit GEMM (no explicit im2col), Winograd for small filter sizes."
        ),
        caveats="Winograd only effective for small filters (3x3, 5x5).",
    ),
    "depthwise_conv": PlaybookTechnique(
        name="convolution_algorithms",
        layer=6,
        applicable_when="conv2d, depthwise_conv",
        expected_gain="Significant (algorithmic)",
        template=(
            "Implicit GEMM (no explicit im2col), Winograd for small filter sizes."
        ),
        caveats="Winograd only effective for small filters (3x3, 5x5).",
    ),
}

# Keywords that map directions to playbook layers
_MEMORY_KEYWORDS = re.compile(r"memory|bandwidth|coalescing|cache", re.IGNORECASE)
_COMPUTE_KEYWORDS = re.compile(r"compute|throughput|flops|arithmetic", re.IGNORECASE)
_ADVANCED_KEYWORDS = re.compile(
    r"fusion|coarsening|persistent|structural", re.IGNORECASE
)


def get_relevant_playbook(
    direction: str,
    gpu_spec: GPUSpec,
    op_name: str,
) -> list[PlaybookLayer]:
    """Query the playbook for techniques relevant to the given context.

    Always includes Layer 1 (universal). Includes additional layers based
    on the direction keywords, GPU architecture features, and operation type.

    Args:
        direction: Optimization direction from the directive.
        gpu_spec: Target GPU specification for arch-specific techniques.
        op_name: Operation name for kernel-specific algorithms.

    Returns:
        List of relevant PlaybookLayer objects. Always non-empty.

    Implements: REQ-CA-003
    Invariant: INV-CA-004 (playbook query never returns empty)
    """
    layers: list[PlaybookLayer] = [LAYER_1]
    direction_matched = False

    # Step 2: Match direction against bottleneck categories
    if _MEMORY_KEYWORDS.search(direction):
        layers.append(LAYER_2)
        direction_matched = True

    if _COMPUTE_KEYWORDS.search(direction):
        layers.append(LAYER_3)
        direction_matched = True

    if _ADVANCED_KEYWORDS.search(direction):
        layers.append(LAYER_4)
        direction_matched = True

    # Step 3: Architecture-specific Layer 5
    layer_5_techniques: list[PlaybookTechnique] = []
    if gpu_spec.supports_cp_async:
        layer_5_techniques.extend(_AMPERE_TECHNIQUES)
    if gpu_spec.supports_tma:
        layer_5_techniques.extend(_HOPPER_TECHNIQUES)

    if layer_5_techniques:
        layers.append(
            PlaybookLayer(
                layer_number=5,
                name="Architecture-Specific Optimizations",
                techniques=layer_5_techniques,
            )
        )

    # Step 4: Kernel-specific Layer 6
    op_lower = op_name.lower()
    if op_lower in _LAYER_6_ALGORITHMS:
        technique = _LAYER_6_ALGORITHMS[op_lower]
        layers.append(
            PlaybookLayer(
                layer_number=6,
                name="Kernel-Specific Algorithms",
                techniques=[technique],
            )
        )

    # Step 5: If no direction-specific layers matched (step 2), Layer 1 is
    # already included as the universal fallback. No additional action needed.
    _ = direction_matched  # Acknowledge the variable is used for logic above.

    return layers


def format_playbook_layers(layers: list[PlaybookLayer]) -> str:
    """Format playbook layers into a human-readable string for the LLM prompt.

    Args:
        layers: The playbook layers to format.

    Returns:
        Multi-line string with technique details.
    """
    sections: list[str] = []
    for layer in layers:
        section_lines = [
            f"### Layer {layer.layer_number}: {layer.name}",
        ]
        for tech in layer.techniques:
            section_lines.append(f"- **{tech.name}**")
            section_lines.append(f"  When: {tech.applicable_when}")
            section_lines.append(f"  Expected gain: {tech.expected_gain}")
            section_lines.append(f"  Approach: {tech.template}")
            section_lines.append(f"  Caveats: {tech.caveats}")
        sections.append("\n".join(section_lines))
    return "\n\n".join(sections)
