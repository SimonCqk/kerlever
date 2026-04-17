# Kerlever — Agent Guidelines

## What This System Is

An agent-driven CUDA kernel optimization loop. The system generates kernel candidates, evaluates them on real GPU hardware, and uses structured profiling feedback to navigate the optimization search space.

```
generate → compile → verify → benchmark → profile → analyze → navigate → generate
```

4 LLM agents + 3 deterministic services. See `docs/architecture.md` for the full DAG.

---

## Principles

### 1. First Principles Over Pattern Matching

Do not optimize by guessing or by blindly applying "common CUDA optimizations." Every optimization decision must trace back to a quantifiable bottleneck.

**Wrong**: "This kernel is slow, let's add shared memory tiling."
**Right**: "Global memory throughput is 280 GB/s (35% of peak 800 GB/s). Memory access pattern shows stride-32 access across warps — uncoalesced. Tiling into shared memory eliminates the strided global access."

The question is always: **what specific hardware resource is underutilized, and why?**

### 2. No Over-Engineering

Build the simplest thing that closes the loop. Complexity is added only when a concrete failure forces it.

- Don't build abstractions for hypothetical future needs
- Don't add configuration for parameters that have one sensible value
- Don't split modules until a single module demonstrably cannot handle the scope
- Prefer 50 lines of straightforward code over 20 lines of clever code

### 3. Evolve Existing Structure Before Adding New Surface Area

When updating docs or code, first locate the owning section, module, or abstraction and improve it in place. Do not append loosely related material just because it is faster.

- Prefer clarifying, extending, or reorganizing existing sections over adding parallel sections
- Add a new section or module only when the current structure has no natural home for the change
- Avoid duplicate explanations, overlapping ownership, and disconnected notes
- Keep documentation professional, precise, and logically coherent from top to bottom

Large rewrites require explicit human alignment before discarding the current structure. Explain what would be replaced, why incremental edits are insufficient, and what the new structure will make clearer.

### 4. Structured Data, Not Natural Language

Agents communicate through structured formats (JSON directives, typed dataclasses), not prose. Profiling metrics are numbers, not descriptions of numbers. Optimization directions are enum tags, not sentences.

### 5. Deterministic Where Possible, LLM Where Necessary

Gate checks, tabu filtering, threshold comparisons, trend computation — these are code, not LLM calls. LLM reasoning is reserved for genuinely ambiguous decisions: tradeoff analysis between conflicting bottlenecks, creative structural changes, cross-candidate semantic diff.

### 6. Grounded in Facts

Every claim about kernel behavior must be backed by measurement. "Should be faster" is not a valid statement — "measured 1.2ms → 0.9ms on A100, shape [4096, 4096], p50 over 100 runs" is.

---

## Mathematical & Performance Analysis Methods

Optimization decisions must be informed by quantitative analysis, not intuition. The following methods are first-class tools in this system.

### Roofline Model

Determines whether a kernel is compute-bound or memory-bound by relating arithmetic intensity (FLOPs/byte) to hardware ceilings.

```
Arithmetic Intensity = Total FLOPs / Total Bytes Transferred
Attainable Perf = min(Peak FLOPS, Peak Bandwidth × Arithmetic Intensity)
```

If measured throughput is far below the roofline, the gap tells you exactly which resource is the bottleneck and by how much.

### Amdahl's Law

Bounds the speedup from optimizing a fraction of execution time. Prevents wasting iterations on a component that contributes < 5% of total time.

```
Speedup = 1 / ((1 - f) + f / s)

f = fraction of time in the optimized section
s = speedup of that section
```

### Little's Law (Latency Hiding)

Relates concurrency, throughput, and latency. Directly applicable to GPU occupancy analysis and memory pipeline depth.

```
Concurrency = Throughput × Latency

Required warps = Memory Latency (cycles) × Throughput (warps/cycle)
```

If occupancy is too low to hide memory latency, this formula tells you exactly how many more concurrent warps are needed.

### Occupancy Analysis

Occupancy is determined by three independent resource constraints — compute the binding constraint:

```
Occupancy = min(
    max_warps_per_SM,
    floor(max_registers_per_SM / registers_per_thread / warp_size),
    floor(max_smem_per_SM / smem_per_block) × warps_per_block
)  /  max_warps_per_SM
```

This is not a heuristic — it's a closed-form calculation from `ptxas -v` output and hardware specs. The binding resource (registers, shared memory, or block count) tells you which knob to turn.

### Memory Bandwidth Utilization

Measures how efficiently the kernel uses the memory subsystem.

```
Bandwidth Utilization = Actual Bytes Transferred / (Elapsed Time × Peak Bandwidth)
Coalescing Efficiency = Useful Bytes / Total Bytes Requested by L1
```

Low utilization + high request volume = poor coalescing or redundant loads. High utilization + near-peak bandwidth = kernel is genuinely memory-bound, optimize the algorithm to reduce data movement.

### Instruction Mix Analysis

From SASS/PTX analysis, compute the ratio of useful compute vs. overhead:

```
Compute Efficiency = Useful Arithmetic Instructions / Total Instructions Issued
```

High address calculation overhead suggests the memory access pattern needs restructuring. High predicate/branch instructions suggest warp divergence.

---

## Architecture Essentials

### Modules

| Module                   | Type    | Role                                                        |
|--------------------------|---------|-------------------------------------------------------------|
| Orchestrator             | Agent   | Global control loop, state machine, context mgmt            |
| Strategy Navigator       | Agent   | Exploit/explore decision, search direction, tabu management |
| Coding Agent             | Agent   | Kernel generation (local mutation or structural change)     |
| Cross-Candidate Analyzer | Agent   | Semantic diff, gene identification, recombination hints     |
| Compiler Service         | Service | Compile + correctness + static analysis (remote GPU pod)    |
| Benchmarker              | Service | Bench + rank + deep profile (remote GPU pod)                |
| Profile Interpreter      | Service | Rule-based bottleneck tagging from profiling metrics        |

### Data Flow is a DAG, Not a Pipeline

Every stage can short-circuit:

- Compile/correctness fail → skip benchmark, feed error directly back to Coding Agent
- Benchmark regression → discard candidate, don't profile
- Same bottleneck N rounds → mark direction exhausted, force explore

### Exploit vs. Explore

- **Exploit**: small delta on current best kernel (param tuning, local rewrite). 3-8 candidates.
- **Explore**: structural change (algorithm swap, primitive upgrade, recombination). 2-3 candidates.

Mode selection: deterministic gates for clear signals, LLM reasoning for ambiguous cases.
