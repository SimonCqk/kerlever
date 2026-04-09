# Architecture Migration: Align Modules with Corrected Data Contracts

## Context

`docs/architecture.md` 和 `docs/bitter-lessons.md` 经过架构审查后更新，指出了 5 个必须修正的架构弱点：

1. **Baseline 未测量** — 循环从声明值而非测量值开始
2. **策略信号单位不一致** — 绝对延迟 delta 与相对增益混用
3. **搜索记忆太弱** — tabu 是扁平字符串列表，不是 typed 记录
4. **Profile 证据丢失** — 只保留 bottleneck tags，丢失 metrics 和 rule_trace
5. **目标太标量** — 单个 latency 数值代替多形状工作负载目标

这些修正相互关联（BaselineArtifact 依赖 ObjectiveScore，TabuEntry 依赖 AttemptRecord），因此一次性全部迁移。

### 用户决策

- **迁移范围**: 5 个 correction 全部一次做完
- **YAML 兼容**: 直接迁移到新格式，不保留向后兼容
- **Bootstrap**: 在 spec 中定义行为，代码暂不实现（等 GPU Pipeline 就绪后实现）

---

## Phase 1: Spec 更新（并行 4 个 Architect Agent）

先更新所有 spec 文档，验证一致性后再动代码。每个 Architect 负责一个模块的 spec.md。

### 1A. Orchestrator Spec 更新 (`docs/orchestrator/spec.md`)

这是最大的变更，因为 Orchestrator spec 的 §5 Interfaces 定义了所有共享类型。

**§2 Requirements 变更:**
- 新增 REQ-ORCH-009: Baseline Bootstrap（spec 描述行为，标注 V2 实现）
- REQ-ORCH-003: Global best tracking 从 scalar latency 改为 objective score 比较
- REQ-ORCH-001: 终止条件改为 `incumbent.objective_score.value <= target_metric_value`

**§5 Interfaces / 共享类型变更:**

新增类型（来自 architecture.md）:
- `ShapeCase(shape_id, dims, weight, correctness_tolerance, profile)`
- `PerformanceObjective(primary_metric, aggregation, regression_guard_pct)`
- `StaticAnalysis(registers_per_thread, smem_bytes_per_block, spill_stores, spill_loads, occupancy_estimate_pct)`
- `ShapeBenchResult(shape_id, latency_p50_us, latency_p95_us, stdev_us, run_count)`
- `ObjectiveScore(metric_name, value, relative_to_baseline, relative_to_incumbent)`
- `BaselineArtifact(kernel_hash, source_code, compile_artifact, benchmark_results, objective_score, profile_bundle)`
- `CandidateIntent(direction, mode, sub_mode, rationale)`
- `AttemptRecord(round_number, candidate_hash, base_kernel_hash, direction, sub_mode, outcome, objective_score)`
- `TabuEntry(base_kernel_hash, direction, sub_mode, round_number, expires_after_round)`
- `CorrectnessResult(passed, failing_shape_ids, max_abs_error, max_rel_error)`
- `BenchmarkBundle(shape_results, objective_score, regressed_vs_incumbent)`
- `ProfileMetrics(achieved_occupancy_pct, dram_throughput_pct_of_peak, ... 8 fields)`
- `BottleneckAssessment(tags, primary_tag, evidence, rule_trace)`
- `ProfileBundle(shape_id, metrics, assessment)`

修改类型:
- `ProblemSpec`: `shapes` → `shape_cases: list[ShapeCase]`，删除 `baseline_perf_us`/`target_perf_us`/`tolerance`，新增 `objective: PerformanceObjective`、`target_metric_value: float`
- `KernelCandidate`: `intent_tag` → `intent: CandidateIntent`，`parent_hash` → `parent_hashes: list[str]`
- `EvaluationResult`: 分离 compile_status/static_analysis/correctness/benchmark/profile
- `RoundSummary`: 新增 `best_objective_score`、`abs_gain_vs_prev_best_us`、`rel_gain_vs_prev_best`，删除 `best_latency_us`/`improvement_over_prev_best`
- `OptimizationState`: 新增 `baseline: BaselineArtifact`、`incumbent: BaselineArtifact`，`tabu_list` → `attempts: list[AttemptRecord]` + `tabu_entries: list[TabuEntry]`，`bottleneck_history` → `list[BottleneckAssessment]`

Protocol 签名变更:
- `GPUPipelineProtocol.evaluate()`: 参数从 `current_best_latency_us: float` 改为 `baseline: BaselineArtifact, incumbent: BaselineArtifact`
- 其余 Protocol 签名不变（但消费的类型字段变了）

**§6 Behavioral Specification 变更:**
- 6.1 新增 Stage 0: Baseline Bootstrap（spec-only，标注 "V2: 当 GPU Pipeline 实现后启用"）
- 6.4 Global Best Update: 用 `objective_score.value` 代替 `latency_us`
- 6.5 Termination: `incumbent.objective_score.value <= target_metric_value`
- 6.6 Tabu Management: 写入 `AttemptRecord` + `TabuEntry`，不再是 flat string append
- 6.7 RoundSummary: 计算 abs 和 rel gains

### 1B. Navigator Spec 更新 (`docs/navigator/spec.md`)

**§5 输入类型引用更新:**
- OptimizationState 字段变更（baseline/incumbent/attempts/tabu_entries/bottleneck_history）
- RoundSummary 字段变更（rel_gain_vs_prev_best 代替 improvement_over_prev_best）

**§6 Behavioral Specification 变更:**
- 6.1 Signal Computation:
  - `avg_delta` 改用 `rel_gain_vs_prev_best`（相对增益，不是绝对 delta）
  - `stable_bottleneck` 从 `list[list[str]]` 改为 `list[BottleneckAssessment]`，读 `primary_tag`
  - `direction_attempt_counts` 从 RoundSummary 改为从 `AttemptRecord` 统计
  - `exhausted_directions` 同上
- 6.2 Gate Logic:
  - Gate 3 Near Target: `incumbent.objective_score.value <= target_metric_value / target_threshold`（用 objective ratio）
- 6.3 LLM Reasoning:
  - Context assembly 使用 `rel_gain_vs_prev_best`、`BottleneckAssessment.evidence`
  - Tabu validation 使用 typed `TabuEntry` 匹配
- 6.4 UCB1:
  - `avg_perf_gain` 改用 `rel_gain_vs_prev_best`（相对增益）
- 6.5 Assembly:
  - Tabu filter 使用 `TabuEntry(base_hash, direction, sub_mode)` 而非 flat string
  - Output tabu 字段类型更新

### 1C. Coding Agent Spec 更新 (`docs/coding-agent/spec.md`)

**§5 接口类型引用更新:**
- `KernelCandidate` 字段变更: `intent: CandidateIntent` 代替 `intent_tag: str`，`parent_hashes: list[str]` 代替 `parent_hash: str | None`
- `ProblemSpec` 字段变更: `shape_cases` 代替 `shapes`

**§6 Behavioral Specification 变更:**
- 6.3 Prompt Builder: DE_NOVO prompt 使用 `shape_cases[].dims` 代替 `shapes`
- 6.5 Generation Flow Step 5: Candidate assembly 使用 `CandidateIntent(direction, mode, sub_mode, rationale)` 代替 `intent_tag`
- 6.5 Generation Flow Step 5: `parent_hashes = [directive.base_kernel_hash]` 代替 `parent_hash = directive.base_kernel_hash`

### 1D. Spec Builder Spec 更新 (`docs/spec-builder/spec.md`)

**§2 Requirements 变更:**
- REQ-SB-001: Batch validation 适配新 ProblemSpec 结构

**§6 Behavioral Specification 变更:**
- 6.2 Deterministic Checks:
  - Check 3: `shape_cases` 验证（shape_id 唯一、dims 正整数、weight > 0、correctness_tolerance 合理）
  - Check 5: `target_metric_value > 0`，`objective.primary_metric` 合法，`objective.aggregation` 合法，`objective.regression_guard_pct >= 0`
  - 删除: baseline_perf_us/target_perf_us/tolerance 验证
- 6.5 Interactive Mode: 新字段 schema（shape_cases, objective, target_metric_value）
- YAML 格式示例更新

---

## Phase 2: Code 更新

Spec 通过审查后，进入代码实现。

### 2A. Foundation: types.py + protocols.py + problem_spec.py（必须先做）

这是所有模块的基础。按 architecture.md 的数据合约更新 types.py 中的所有类型。

**文件:**
- `src/kerlever/types.py` — 全部类型重写
- `src/kerlever/protocols.py` — GPUPipelineProtocol 签名更新
- `src/kerlever/problem_spec.py` — load 逻辑适配新 ProblemSpec
- `examples/matmul_spec.yaml` — 更新为新 YAML 格式

### 2B. Orchestrator 更新（依赖 2A）

**文件:**
- `src/kerlever/orchestrator.py` — 主循环适配新类型：
  - Bootstrap stage（spec-only: 暂用声明值构建初始 BaselineArtifact）
  - Global best → incumbent 比较使用 objective_score
  - Termination 使用 objective_score
  - 每轮记录 AttemptRecord
  - TabuEntry 管理（带 expiry）
  - BottleneckAssessment 收集
  - RoundSummary 计算 abs + rel gains
- `src/kerlever/state.py` — 适配新 OptimizationState 结构
- `src/kerlever/stubs.py` — 全部 stub 适配新类型
- `tests/test_orchestrator.py` — 所有测试适配新类型
- `tests/test_state.py` — state persistence 测试适配
- `tests/test_problem_spec.py` — ProblemSpec loading 测试适配

### 2C. Navigator 更新（依赖 2A，可与 2B/2D/2E 并行）

**文件:**
- `src/kerlever/navigator/types.py` — DerivedSignals 适配（avg_delta 为 relative）
- `src/kerlever/navigator/signals.py` — 用 rel_gain，读 BottleneckAssessment.primary_tag，从 AttemptRecord 统计
- `src/kerlever/navigator/gates.py` — near-target 用 objective ratio
- `src/kerlever/navigator/llm_reasoning.py` — context assembly 使用新类型
- `src/kerlever/navigator/ucb1.py` — 用 rel_gain
- `src/kerlever/navigator/assembly.py` — typed TabuEntry 过滤
- `tests/test_navigator/` — 全部测试文件适配

### 2D. Coding Agent 更新（依赖 2A，可与 2B/2C/2E 并行）

**文件:**
- `src/kerlever/coding_agent/__init__.py` — generate() 适配新 KernelCandidate
- `src/kerlever/coding_agent/generator.py` — CandidateIntent + parent_hashes 构建
- `src/kerlever/coding_agent/prompt_builder.py` — shape_cases 代替 shapes
- `src/kerlever/coding_agent/code_validator.py` — ProblemSpec.dtype 访问路径不变
- `tests/test_coding_agent/` — 测试适配

### 2E. Spec Builder 更新（依赖 2A，可与 2B/2C/2D 并行）

**文件:**
- `src/kerlever/spec_builder/__init__.py` — 适配新 ProblemSpec
- `src/kerlever/spec_builder/deterministic.py` — 6 个检查类别全部重写
- `src/kerlever/spec_builder/llm_judge.py` — 新 ProblemSpec 序列化
- `src/kerlever/spec_builder/interactive.py` — 新字段 schema
- `src/kerlever/spec_builder/__main__.py` — CLI 适配
- `tests/test_spec_builder/` — 全部测试适配

---

## Execution Strategy

```
Phase 1: Spec 更新
    ├── Architect: Orchestrator spec (1A)  ─┐
    ├── Architect: Navigator spec (1B)     ─┤ 并行
    ├── Architect: Coding Agent spec (1C)  ─┤
    └── Architect: Spec Builder spec (1D)  ─┘
    ↓ Manager 验证所有 spec 一致性
    ↓ 用户确认

Phase 2: Code 更新
    Step 1: Coding Agent → Foundation types (2A)
    Step 2:
    ├── Coding Agent: Orchestrator (2B)  ─┐
    ├── Coding Agent: Navigator (2C)     ─┤ 并行
    ├── Coding Agent: Coding Agent (2D)  ─┤
    └── Coding Agent: Spec Builder (2E)  ─┘
    ↓ Manager: ruff + mypy + pytest 全量验证
    ↓ 修复 → 迭代直到通过
```

---

## Bootstrap 处理方案（临时方案）

由于 GPU Pipeline 仍是 stub，Bootstrap 阶段暂不实现真实逻辑。临时方案：

1. Spec 中完整定义 Bootstrap 行为（Stage 0: resolve → compile → correctness → benchmark → profile → seed state）
2. Orchestrator 代码中：用声明式初始值构建 `BaselineArtifact`，不调用 GPU Pipeline
3. 具体：从 ProblemSpec 的 `reference_kernel` 构建 source_code，hash 计算，benchmark/profile 字段用合理默认值
4. Stub 中：StubGPUPipeline 新增 `bootstrap()` 方法返回模拟的 BaselineArtifact
5. 标注 TODO: "当 GPU Pipeline 实现后，替换为真实 bootstrap"

---

## Verification

Phase 1 后:
- 读每个 spec，确认 SC-* 覆盖、SCN-* GIVEN/WHEN/THEN 格式、REQ-* 与 architecture.md 一致
- 交叉检查: 一个 spec 引用的类型与 orchestrator spec §5 定义一致

Phase 2 后:
- `ruff check .` — lint 通过
- `mypy --strict .` — 类型检查通过
- `pytest` — 所有测试通过（测试会同步更新）
- 手动检查: `examples/matmul_spec.yaml` 能被 load_problem_spec 正确加载
- 检查无 dead code、TODO（除了标注的 bootstrap TODO）

---

## SC-* Success Criteria

- **SC-1**: 所有 spec.md 反映 architecture.md 的 5 个 correction
- **SC-2**: ProblemSpec 使用 ShapeCase + PerformanceObjective + target_metric_value
- **SC-3**: OptimizationState 包含 baseline/incumbent BaselineArtifact
- **SC-4**: RoundSummary 同时存储 abs_gain 和 rel_gain
- **SC-5**: Navigator signals 使用 relative gains，gates 使用 objective ratio
- **SC-6**: TabuEntry 和 AttemptRecord 替代 flat tabu_list
- **SC-7**: BottleneckAssessment 替代 list[list[str]]
- **SC-8**: EvaluationResult 分离 compile/correctness/benchmark/profile
- **SC-9**: 全部 243+ 测试通过（测试同步更新）
- **SC-10**: mypy --strict + ruff check 通过

## Critical Path

`load_problem_spec("matmul_spec.yaml")` → 新格式 ProblemSpec → `Orchestrator.run()` 正常运行到结束 → 所有 round 产生完整的 AttemptRecord/TabuEntry/BottleneckAssessment。

## Failure Modes

- 类型迁移不完全: 某个模块仍引用旧字段名 → mypy --strict 会捕获
- Spec 交叉不一致: Orchestrator spec 定义的类型与 Navigator spec 引用的不同 → Phase 1 交叉检查
- 测试遗漏: 旧测试 hardcode 了旧类型字段 → pytest 会报错
- Bootstrap 临时方案泄漏: 代码依赖 baseline 的真实 benchmark 数据 → 确保 stub 返回合理默认值

## Non-Goals

- 不实现真实 GPU Pipeline（仍为 stub）
- 不实现真实 Cross-Candidate Analyzer（仍为 stub）
- 不实现真实 Baseline Bootstrap 执行（spec-only + 临时声明值方案）
- 不更新 README（这次只更新 spec + code）
