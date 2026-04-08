# Coding Agent Implementation Plan

## Context

Coding Agent 是 CUDA kernel 代码的生成器 — 接收 Strategy Navigator 的 directive，产出 N 个 kernel 候选。它是整个系统中唯一直接产出 CUDA 代码的模块。

核心设计理念（来自研究）：
- AutoKernel 的 6 层优化 playbook 显著提升生成质量
- STARK/CUDA-LLM 的迭代纠错模式与我们的架构匹配
- 硬件约束参数化对跨架构优化至关重要
- 通用优化（__restrict__, __launch_bounds__, coalescing）应无条件应用

## Decisions Made

- 内建 6 层 CUDA 优化 playbook，作为 system prompt 的结构化知识库
- Agent 内部做基本代码校验（regex-level: __global__, __launch_bounds__, __restrict__, 括号匹配）
- 内置硬编码 GPU 约束表（A100/H100/V100 等常见 GPU 的 smem/register/TC/async copy 约束）
- LLM 用 Anthropic Claude SDK via LLMClientProtocol（复用公共 llm_client.py）

## File Manifest

```
src/kerlever/coding_agent/
    __init__.py              # 公共 API: CodingAgent class
    config.py                # CodingAgentConfig: 代码生成参数
    playbook.py              # 6 层 CUDA 优化 playbook (结构化知识库)
    hardware.py              # GPU 硬件约束表 (A100/H100/V100 etc.)
    prompt_builder.py        # system prompt + user prompt 组装
    code_validator.py        # regex-level CUDA 代码校验
    generator.py             # LLM 代码生成 + 解析 + 重试逻辑
    types.py                 # CodingAgent 内部类型
tests/test_coding_agent/
    __init__.py
    test_playbook.py         # playbook 查询测试
    test_hardware.py         # GPU 约束查询测试
    test_prompt_builder.py   # prompt 组装测试
    test_code_validator.py   # 代码校验测试
    test_generator.py        # 代码生成 (stub LLM) 测试
    test_coding_agent.py     # 端到端集成测试
```

## Architecture

### 生成流程

```
StrategyDirective + ProblemSpec + current_best_source
    │
    ▼
Step 1: Resolve context
    - 从 directive 提取: mode, direction, sub_mode, num_candidates, constraints
    - 查 GPU 约束表: 目标 GPU 的 smem/register/TC/async 能力
    - 查 playbook: direction 对应的优化技术 + 代码模板
    │
    ▼
Step 2: Build prompts (per candidate)
    - system prompt: CUDA 专家角色 + 6 层 playbook (相关层) + GPU 约束 + 代码规范
    - user prompt: op 语义 + shapes/dtype + direction + base kernel (if exploit)
    - 每个候选 prompt 略有变化 (temperature/variation hint)
    │
    ▼
Step 3: Generate (parallel LLM calls)
    - N 次 LLM 调用 (asyncio.TaskGroup)
    - 解析 CUDA 代码块 (```cuda ... ```)
    - 基本校验 (code_validator)
    │
    ▼
Step 4: Validate & assemble
    - 校验通过 → 构建 KernelCandidate
    - 校验失败 → retry 一次 (带具体错误反馈)
    - 二次失败 → skip 该候选
    - 计算 code_hash (SHA256)
    │
    ▼
Output: list[KernelCandidate]
```

### 6 层 Playbook (playbook.py)

按 bottleneck 类型和优化层级组织的结构化知识库：

```
Layer 1: Block/Grid Configuration (通用, 10-50% 收益)
    - block_size_tuning: 128/256/512, 必须是 32 的倍数
    - grid_sizing: ceiling division, wave quantization awareness
    - __launch_bounds__ 声明

Layer 2: Memory Access Optimization (memory-bound, 10-30%)
    - coalesced_access: 连续线程访问连续地址
    - shared_memory_tiling: tile 到 smem, __syncthreads()
    - vectorized_loads: float4/int4, 128-bit transactions
    - async_copy: cp.async (Ampere+) / TMA (Hopper+)
    - bank_conflict_avoidance: padding (+1) / swizzling

Layer 3: Compute Optimization (compute-bound, 5-15%)
    - mixed_precision: FP16 compute, FP32 accumulate
    - tensor_core_utilization: wmma/mma.sync, tile alignment
    - loop_unrolling: #pragma unroll
    - fma_utilization: fused multiply-add

Layer 4: Advanced Techniques (5-20%)
    - thread_coarsening: 每线程处理多元素
    - kernel_fusion: 合并多 kernel 减少全局内存往返
    - persistent_kernels: grid = SM count, 线程循环处理工作
    - split_k: K 维度并行化

Layer 5: Architecture-Specific (5-15%)
    - ampere: cp.async, L2 persistence, 164KB smem
    - hopper: TMA, clusters, distributed smem, 228KB smem, FP8
    - register_spilling_to_smem: CUDA 13+ PTX pragma

Layer 6: Kernel-Specific Algorithms
    - matmul: Goto's algorithm, hierarchical tiling
    - attention: online softmax (Flash Attention)
    - reduction: warp shuffle, block-level tree reduction
    - normalization: Welford's online algorithm
    - conv: im2col, implicit GEMM, Winograd
```

每个技术条目包含: 名称、适用条件、预期收益范围、代码模板/示例、注意事项。

### GPU 约束表 (hardware.py)

```python
GPU_SPECS = {
    "A100": GPUSpec(
        arch="sm_80", smem_per_sm_kb=164, max_smem_per_block_kb=163,
        registers_per_sm=65536, max_registers_per_thread=255,
        max_warps_per_sm=64, max_threads_per_block=1024,
        hbm_bandwidth_tbps=2.0, l2_cache_mb=40,
        supports_cp_async=True, supports_tma=False,
        supports_fp8=False, tensor_core_types=["fp16","bf16","tf32","fp64","int8","int4"],
    ),
    "H100": GPUSpec(
        arch="sm_90", smem_per_sm_kb=228, max_smem_per_block_kb=227,
        registers_per_sm=65536, max_registers_per_thread=255,
        max_warps_per_sm=64, max_threads_per_block=1024,
        hbm_bandwidth_tbps=3.35, l2_cache_mb=50,
        supports_cp_async=True, supports_tma=True,
        supports_fp8=True, tensor_core_types=["fp16","bf16","tf32","fp64","int8","fp8"],
    ),
    ...
}
```

### 代码校验 (code_validator.py)

regex-level 的基本检查，不编译：

1. `__global__` 函数存在
2. `__launch_bounds__` 存在（warn if missing, not fail）
3. `__restrict__` 在指针参数上（warn if missing）
4. 括号/花括号平衡
5. 不含明显的 host-only API（malloc, printf 非 debug, std:: 等）
6. 内核签名匹配 ProblemSpec 的 dtype (e.g., half* for float16)
7. 不含空 kernel body（至少有赋值或计算语句）

校验结果: list[CodeIssue(severity, message)]。severity=error 时 reject。

### Prompt 设计 (prompt_builder.py)

**System prompt 结构:**
```
你是 CUDA kernel 优化专家。

## 代码规范 (必须遵守)
- __launch_bounds__(maxThreads, minBlocks) 必须声明
- 所有指针参数使用 __restrict__ 和 const (只读参数)
- block size 必须是 32 的倍数
- bounds-check 所有全局内存访问
- 使用 ceiling division: (N + BLOCK_SIZE - 1) / BLOCK_SIZE

## 目标 GPU 约束
{gpu_spec_summary}

## 优化 Playbook (当前方向相关)
{relevant_playbook_layers}

## 输出格式
返回一个 ```cuda 代码块，包含完整的 kernel 函数。不要包含 host 代码。
```

**User prompt 结构 (按 sub_mode 变化):**

EXPLOIT/LOCAL_REWRITE:
```
优化方向: {direction}
当前最优 kernel:
```cuda
{current_best_source}
```
任务: 对上述 kernel 做局部重写，目标是 {direction}。
约束: {hard_constraints}
```

EXPLOIT/PARAM_SEARCH:
```
优化方向: {direction}
当前最优 kernel: {current_best_source}
参数搜索范围: {search_range}
任务: 生成一个变体，尝试参数组合 {specific_params_for_this_candidate}。
```

EXPLORE/DE_NOVO:
```
目标操作: {op_semantics}
输入 shapes: {shapes}, dtype: {dtype}
任务: 从头实现一个高性能的 {op_name} kernel。
优化方向: {direction}
```

EXPLORE/RECOMBINATION:
```
Parent A: {parent_a_source}
Parent B: {parent_b_source}
Gene map: {gene_map}
任务: 组合 Parent A 的 {gene_a} 和 Parent B 的 {gene_b}。
```

### CodingAgent 类

```python
class CodingAgent:
    def __init__(self, llm_client: LLMClientProtocol, config: CodingAgentConfig | None = None):
        ...

    async def generate(self, problem_spec, directive, current_best_source) -> list[KernelCandidate]:
        gpu_spec = get_gpu_spec(problem_spec.target_gpu)
        playbook_context = get_relevant_playbook(directive.direction, directive.mode)
        system_prompt = build_system_prompt(gpu_spec, playbook_context)

        candidates = []
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(
                self._generate_one(system_prompt, problem_spec, directive, current_best_source, i)
            ) for i in range(directive.num_candidates)]
        for task in tasks:
            result = task.result()
            if result is not None:
                candidates.append(result)
        return candidates
```

## Implementation Sequence

1. `coding_agent/types.py` — CodeIssue, GenerationResult, GPUSpec
2. `coding_agent/hardware.py` + test_hardware.py — GPU 约束表
3. `coding_agent/playbook.py` + test_playbook.py — 6 层 playbook
4. `coding_agent/code_validator.py` + test_code_validator.py — regex 校验
5. `coding_agent/config.py` — CodingAgentConfig
6. `coding_agent/prompt_builder.py` + test_prompt_builder.py — prompt 组装
7. `coding_agent/generator.py` + test_generator.py — LLM 生成 + 解析 + 重试
8. `coding_agent/__init__.py` + test_coding_agent.py — CodingAgent 类
9. 验证: ruff, mypy, pytest, orchestrator e2e 仍通过

## SC-* Success Criteria

- **SC-1**: DE_NOVO 模式生成 N 个 CUDA kernel 候选，每个包含 __global__ 函数
- **SC-2**: EXPLOIT/LOCAL_REWRITE 模式在 current_best 基础上生成变体
- **SC-3**: playbook 根据 direction 返回相关的优化技术层
- **SC-4**: GPU 约束表对 A100/H100 返回正确的 smem/register/TC 信息
- **SC-5**: code_validator 检出缺少 __global__ 的代码、不平衡的括号
- **SC-6**: LLM 生成失败时 retry 一次，二次失败 skip 该候选
- **SC-7**: 每个 KernelCandidate 有唯一 code_hash、正确的 intent_tag 和 parent lineage
- **SC-8**: 现有 142 测试仍全部通过
- **SC-9**: mypy --strict 和 ruff check 通过

## Critical Path

`CodingAgent(stub_llm).generate(matmul_spec, exploit_directive, reference_kernel)` → 返回 N 个 KernelCandidate，每个包含合法 CUDA 代码。

## Failure Modes

- LLM 不返回 ```cuda 代码块 → 缓解: 尝试从裸文本中提取 __global__ 函数
- LLM 生成的代码超长 → 缓解: max_tokens 限制 + 截断警告
- 所有候选都校验失败 → 缓解: 返回空列表，Orchestrator 在下一轮 handle
- Playbook 对未知 direction 无匹配 → 缓解: 返回通用优化 checklist (Layer 1)
- GPU 约束表无目标 GPU → 缓解: 返回保守默认值 (48KB smem, 255 registers)

## Non-Goals

- 不做真实 CUDA 编译（GPU pod 的事）
- 不做 AST-level 代码分析
- 不做 kernel 参数 auto-tuning（那是 PARAM_SEARCH + GPU pipeline 的组合）
- 不做 multi-model LLM 支持
- 不做代码缓存/去重（Orchestrator 的 code_hash 已处理）
