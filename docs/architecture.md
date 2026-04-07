# Kerlever System Architecture

## Overview

An agent-driven CUDA kernel optimization loop: **generate → compile → benchmark → profile → analyze → navigate → generate**.

```
Agents (LLM)                           Services (Deterministic, Remote GPU Pod)
=========================              ============================================
  Orchestrator                           Compiler Service
  Strategy Navigator                       (compile + correctness + static analysis)
  Coding Agent                           Benchmarker
  Cross-Candidate Analyzer                 (bench + rank + deep profile)
                                         Profile Interpreter
```

---

## System DAG

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Problem Spec / Oracle                                                              │
│  op semantics, shape/dtype, target GPU arch, baseline perf, objective, tolerance    │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                                       │
│  global control, task state machine, context mgmt, round summary, termination check │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Strategy Navigator                                                                 │
│  deterministic: tabu list, plateau detection                                        │
│  LLM reasoning: direction choice, tradeoff analysis, exploit/explore decision       │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │  mode + direction + constraints
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Coding Agent                                                                       │
│  EXPLOIT: param tune, local rewrite, pattern apply                                  │
│  EXPLORE: algo change, recombination, primitive upgrade                              │
│  output: N candidates, each with intent tag                                         │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Remote GPU Pod ─ ─ ─
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────┬───────────────────┐
│  Compiler Service                                               │                   │
│  nvcc compile, register/smem/spill stats, ptx+sass+cubin       │   FAIL ──────────────► Coding Agent
│  correctness: ref compare, multi-shape, tolerance, sanitizer    │   (compile err /  │   (short-circuit)
│  static analysis: resource usage, occupancy estimate            │    correctness)   │
└──────────────────────────────┬──────────────────────────────────┴───────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────────┬───────────────────┐
│  Benchmarker                                                    │                   │
│  fast bench: warmup, stat runs, p50/p95, clock & noise ctrl     │   REGRESSION ───────► discard
│  candidate ranking: filter regression, select top-K             │   (< threshold)   │   (log to workdir)
│  deep profiling: ncu (occupancy, throughput, cache, stalls)     │                   │
│                  nsys (timeline, overlap, memcpy)                │                   │
└──────────────────────────────┬──────────────────────────────────┴───────────────────┘
                               │ top-K results
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Profile Interpreter (rule-based)                                                   │
│  metrics → bottleneck tags → optimization direction map                              │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┼─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Cross-Candidate Analyzer (LLM)                                                     │
│  semantic diff between candidates, identify winning "genes", recomb suggestions     │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │
                               ▼
                        Orchestrator (next round)
```

---

## Early-Exit Paths

| #  | Trigger Point   | Condition                                    | Destination                     |
|----|-----------------|----------------------------------------------|---------------------------------|
| 1  | Compiler        | compile fail or correctness fail             | Coding Agent (w/ error context) |
| 2  | Benchmarker     | perf < threshold% of current best            | Discard, log to workdir         |
| 3  | Profile Interp. | same bottleneck tag for N consecutive rounds | Mark direction EXHAUSTED        |
| 4  | Orchestrator    | target met                                   | Terminate, return best kernel   |

---

## Module Reference

### Agents (LLM-powered)

| Module                   | Responsibility                                                                         |
|--------------------------|----------------------------------------------------------------------------------------|
| Orchestrator             | Global control, task state machine, context mgmt, termination, round summaries         |
| Strategy Navigator       | Search tree management, exploit/explore mode selection, direction recommendation       |
| Coding Agent             | Kernel generation in exploit (local mutation) or explore (structural change) mode      |
| Cross-Candidate Analyzer | Semantic diff between candidates, identify winning code "genes", suggest recombination |

### Services (Deterministic, Remote GPU Pod)

| Module              | Responsibility                                                                                    |
|---------------------|---------------------------------------------------------------------------------------------------|
| Compiler Service    | nvcc compile + rich metadata + correctness validation (ref compare, sanitizer) + static analysis   |
| Benchmarker         | Fast bench (warmup, stats) + candidate ranking (filter + top-K) + deep profiling (ncu, nsys)      |
| Profile Interpreter | Rule-based bottleneck tagging, maps metrics to optimization directions                            |
