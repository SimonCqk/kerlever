# Kerlever Bitter Lessons

This document captures the bitter lessons from the initial architecture review of the Kerlever skeleton.

The point is not to criticize the V1 scaffold. The point is to prevent the system from drifting into a loop that looks agentic and sophisticated while remaining scientifically weak.

---

## 1. Baseline Must Be Measured, Not Declared

`baseline_perf_us` and `reference_kernel` in the spec are not enough.

If the system does not first compile, verify, benchmark, and profile the reference kernel on the target GPU, then it does not actually know:

- what the baseline latency is on this hardware,
- whether the reference kernel is correct on the target shapes,
- what the baseline bottleneck is,
- what kernel the first exploit round should mutate.

**Lesson:** the optimization loop cannot start from an empty "best kernel" state. It must start from a measured baseline artifact.

**Rule:** no round 0 search begins until the reference kernel has been promoted into a first-class baseline record.

---

## 2. Decision Signals Need Unit-Consistent Math

A threshold expressed as a percentage cannot be compared against a value stored in microseconds.

If plateau detection, regression detection, or near-target logic mixes:

- absolute latency deltas,
- relative performance gains,
- ratios to target,

then the search controller will make decisions that are deterministic but wrong.

**Lesson:** every decision signal must declare its unit and denominator.

**Rule:** store both absolute and relative improvements explicitly, and use the relative form for thresholded strategy gates.

---

## 3. Search Constraints Must Match Search State

If the invariant says tabu applies to `(base_kernel_hash, direction)`, then the state must store that pair.

Blocking a direction globally because it appeared recently is too coarse. In kernel optimization, the same direction can be exhausted on one parent kernel and still be productive on a different parent.

**Lesson:** a search constraint is only as correct as the state used to enforce it.

**Rule:** do not encode pairwise search rules into a flat string list.

---

## 4. Bottleneck Tags Are Conclusions, Not Evidence

Tags like `memory_bandwidth` or `occupancy` are useful summaries, but they are not sufficient evidence.

For rigorous operator optimization, the system must preserve:

- the raw or normalized profiling counters,
- the derived quantitative analysis,
- the rule path that produced the bottleneck label.

Otherwise the loop cannot answer the most important question: "what measured resource was underutilized, by how much, and why?"

**Lesson:** tags help navigation, but metrics justify decisions.

**Rule:** never let profile interpretation erase the measurements that produced it.

---

## 5. The Objective Must Reflect the Workload, Not a Single Scalar

A kernel is not "better" because it wins on one latency number if the workload actually spans:

- multiple shapes,
- different shape frequencies,
- different tolerance requirements,
- different regression sensitivities.

A single `best_latency_us` scalar is too weak for multi-shape operator optimization and will bias the loop toward overfitting.

**Lesson:** the search objective must model the workload distribution, not just one benchmark point.

**Rule:** keep per-shape results and optimize an explicit aggregate objective.

---

## What These Lessons Change

These lessons imply the following architectural stance:

1. Baseline seeding is part of the system, not pre-work done by the user.
2. Strategy signals must be derived from typed, unit-aware measurements.
3. Search memory must preserve lineage and attempt context.
4. Profile interpretation must preserve evidence, not only labels.
5. "Best kernel" must be defined against a workload objective, not a single scalar latency.
