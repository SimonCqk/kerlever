# Kerlever System Architecture

## Overview

An agent-driven CUDA kernel optimization loop: **baseline bootstrap -> generate -> compile/verify -> benchmark -> profile -> analyze -> navigate -> generate**. Kerlever is a **stateful search system**, not a stateless prompt loop: each round consumes durable measured context, updates the current incumbent, and carries search memory forward across interruption, migration, and resume.

```text
Agents (LLM)                           Services (Deterministic, Remote GPU Pod)
=========================              ============================================
  Orchestrator                           Compiler Service
  Strategy Navigator                       (compile + correctness + static analysis)
  Coding Agent                           Benchmarker
  Cross-Candidate Analyzer                 (fast bench + rank + deep profile)
                                         Profile Interpreter
```

The system keeps the agent graph small and pushes deterministic work into services. Agents handle lifecycle control, search policy, code synthesis, semantic comparison, and ambiguous tradeoffs over structured evidence. Services remain the measurement truth: compilation, correctness, benchmark ranking, profiling, and rule-based metric interpretation.

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
│  output: measured baseline artifact, seeded incumbent, initial search-memory record │
└──────────────────────────────────────┬──────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                                       │◄─────────────┐
│  durable state owner: lifecycle, persistence/resume, incumbent/baseline, stop checks│             │
│  mutates and persists search-memory records; supplies structured context each round  │             │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │                                                             │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Strategy Navigator                                                                 │              │
│  deterministic: derive policy signals, contextual avoid/exhaustion, plateau checks  │              │
│  LLM reasoning: exploit/explore, direction choice, tradeoff analysis over evidence   │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │  mode + direction + strategy constraints + hints           │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Coding Agent                                                                       │◄─────┐       │
│  candidate-generation executor for exploit tuning, local rewrites, and exploration  │      │       │
│  output: N candidates with structured intent, parent refs, and generation metadata  │      │       │
│                                                                                     │      │       │
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
│  metrics → bottleneck tags with evidence → policy-relevant direction hints          │              │
└──────────────────────────────────────┬──────────────────────────────────────────────┘              │
                                       │                                                             │
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─              │
                                       │                                                             │
                                       ▼                                                             │
┌─────────────────────────────────────────────────────────────────────────────────────┐              │
│  Cross-Candidate Analyzer (LLM)                                                     │──────────────┘
│  semantic deltas, reusable genes, recombination hints, next-round context signals   │   (next round)
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Flow Summary

- Baseline bootstrap runs once before round 0; it must compile, verify, benchmark, and profile the reference kernel before seeding the initial incumbent and initial search-memory record.
- The Orchestrator owns durable lifecycle state: baseline/incumbent context, round/task control, persistence/resume/migration, search-memory record mutation, and stop or budget checks.
- The Strategy Navigator consumes the Orchestrator's durable context snapshot and returns policy only: exploit vs. explore, direction, strategy constraints, search hints, and contextual avoid or revisit signals.
- The Coding Agent executes candidate generation against that policy; compile or correctness failure short-circuits directly back with measured failure context.
- Benchmarking scores candidates against the multi-shape objective, discards regressions before deeper work, and only profiles top-K survivors.
- The Profile Interpreter and other deterministic services are fact producers: they turn measurements into bottleneck evidence and policy-relevant hints without owning search policy.
- The Cross-Candidate Analyzer summarizes semantic deltas, reusable genes, and recombination hints; the Orchestrator persists those signals into the next durable search-memory snapshot.

## State and Boundary Principles

- **Kerlever is stateful.** Optimization decisions are made against accumulated measured evidence, not isolated prompts.
- **Durable search memory is first-class system state.** It must survive interruption, migration, and resume, so ownership belongs to the Orchestrator rather than to any single policy decision.
- **The primary durable unit is an attempt/search-memory record.** Each record captures context, action, and measured outcome for a strategy trial. A search tree may exist as an optional derived view, but it is not the canonical durable state.
- **Navigator owns policy semantics, not persistence.** It interprets durable search memory into next-step policy, including contextual tabu, plateau, exhaustion, revisit, and recombination decisions.
- **Services produce measurement truth.** Compiler Service, Benchmarker, and Profile Interpreter generate deterministic facts; LLM agents reason over those structured facts.

---

## Module Reference

### Agents (LLM-powered)

| Module | Responsibility |
|---|---|
| Orchestrator | Durable state owner and lifecycle controller: baseline bootstrap intake, incumbent management, round/task control, persistence/resume, search-memory mutation, and stop or budget checks |
| Strategy Navigator | Stateful search-policy engine over Orchestrator-provided durable context: exploit/explore selection, direction choice, contextual tabu/plateau/exhaustion logic, strategy constraints, and search hints |
| Coding Agent | Candidate-generation executor that turns strategy directives into exploit mutations, structural exploration, and recombination attempts |
| Cross-Candidate Analyzer | Semantic diff between passing candidates, reusable "gene" identification, recombination hints, and next-round context signals for the Orchestrator |

### Services (Deterministic, Remote GPU Pod)

| Module | Responsibility |
|---|---|
| Compiler Service | nvcc compile, static resource metadata, correctness validation across required shapes, failure context |
| Benchmarker | Fast benchmark, multi-shape objective scoring, regression guard, top-K selection, deep profiling |
| Profile Interpreter | Rule-based bottleneck tagging from profiling metrics, supporting evidence, and policy-relevant direction mapping |
