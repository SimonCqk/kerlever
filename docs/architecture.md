# Kerlever System Architecture

## Overview

An agent-driven CUDA kernel optimization loop: **baseline bootstrap -> generate -> compile/verify -> benchmark -> profile -> analyze -> navigate -> generate**.

```text
Agents (LLM)                           Services (Deterministic, Remote GPU Pod)
=========================              ============================================
  Orchestrator                           Compiler Service
  Strategy Navigator                       (compile + correctness + static analysis)
  Coding Agent                           Benchmarker
  Cross-Candidate Analyzer                 (fast bench + rank + deep profile)
                                         Profile Interpreter
```

The system keeps the agent graph small and pushes deterministic work into services. Agents handle search policy, code synthesis, semantic comparison, and ambiguous tradeoffs. Services handle compilation, correctness, benchmark ranking, profiling, and rule-based metric interpretation.

---

## System DAG

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Problem Spec / Oracle                                                              │
│  op semantics, shape/dtype, target GPU arch, reference kernel, objective, tolerance │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Baseline Bootstrap                                                                 │
│  compile reference, validate correctness, benchmark objective shapes, profile seed  │
│  output: measured baseline artifact and initial incumbent                           │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                                       │◄─────────────┐
│  global control, task state machine, context mgmt, round summary, termination check │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │                                                             │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Strategy Navigator                                                                 │              │
│  deterministic: tabu list, plateau detection, exhaustion gates                      │              │
│  LLM reasoning: direction choice, tradeoff analysis, exploit/explore decision       │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │  mode + direction + constraints                             │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Coding Agent                                                                       │◄─────┐       │
│  EXPLOIT: param tune, local rewrite, pattern apply                                  │      │       │
│  EXPLORE: algo change, recombination, primitive upgrade                             │      │       │
│  output: N candidates, each with intent tag                                         │      │       │
└──────────────────────────────────────┬──────────────────────────────────────────────┘      │       │
                                       │                                                     │       │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Remote GPU Pod ─ ─ ─             │       │
                                       │                                                     │       │
                                       ▼                                                     │       │
┌─────────────────────────────────────────────────────────────────────────────────────┐      │       │
│  Compiler Service                                                                   │      │       │
│  nvcc compile, register/smem/spill stats, ptx+sass+cubin                            │ FAIL─┘       │
│  correctness: ref compare, multi-shape, tolerance, sanitizer                        │              │
│  static analysis: resource usage, occupancy estimate                                │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │ PASS                                                        │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Benchmarker                                                                        │              │
│  fast bench: warmup, stat runs, p50/p95, clock & noise ctrl                         │ REGRESSION   │
│  candidate ranking: objective score, regression guard, select top-K                 │ ──► discard  │
│  deep profiling: ncu (occupancy, throughput, cache, stalls)                         │ (< guard)    │
│                  nsys (timeline, overlap, memcpy)                                   │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │ top-K results                                               │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Profile Interpreter (rule-based)                                                   │              │
│  metrics → bottleneck tags with evidence → optimization direction map               │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │                                                             │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              │
                                       │                                                             │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Cross-Candidate Analyzer (LLM)                                                     │──────────────┘
│  semantic diff between candidates, identify winning "genes", recomb suggestions     │   (next round)
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Flow Summary

- Baseline bootstrap runs once before round 0; it must compile, verify, benchmark, and profile the reference kernel before seeding the initial incumbent.
- The Orchestrator owns round state and termination checks, then asks the Strategy Navigator for mode, direction, and constraints.
- The Coding Agent emits intent-tagged candidates; compile or correctness failure loops directly back with error context.
- Benchmarking scores candidates against the multi-shape objective, applies the regression guard, and selects top-K for deep profiling.
- The Profile Interpreter turns profiling metrics into bottleneck tags with supporting evidence and optimization-direction hints.
- The Cross-Candidate Analyzer summarizes semantic differences between passing candidates, then returns updated search context to the Orchestrator for the next round.

---

## Module Reference

### Agents (LLM-powered)

| Module | Responsibility |
|---|---|
| Orchestrator | Global control, task state machine, context management, measured baseline intake, round summaries, termination checks |
| Strategy Navigator | Search policy, exploit/explore mode selection, direction and constraint recommendation, tabu/plateau/exhaustion handling |
| Coding Agent | Intent-tagged kernel generation for exploit mutations or explore-mode structural changes, recombination, and primitive upgrades |
| Cross-Candidate Analyzer | Semantic diff between passing candidates, winning code "gene" identification, recombination suggestions, next-round context |

### Services (Deterministic, Remote GPU Pod)

| Module | Responsibility |
|---|---|
| Compiler Service | nvcc compile, static resource metadata, correctness validation across required shapes, failure context |
| Benchmarker | Fast benchmark, multi-shape objective scoring, regression guard, top-K selection, deep profiling |
| Profile Interpreter | Rule-based bottleneck tagging from profiling metrics, supporting evidence, optimization-direction mapping |
