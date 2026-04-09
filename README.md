# Kerlever

Kerlever is an agent-driven CUDA kernel optimization loop.

The goal is simple: take a kernel plus a target workload, run real measurement on GPU hardware, use that feedback to decide what to try next, and keep iterating until the target is met or the search budget is exhausted.

```text
baseline bootstrap -> generate -> compile -> benchmark -> profile -> analyze -> navigate -> generate
```

## What This Project Is

Kerlever is trying to make CUDA kernel optimization more like a disciplined search loop and less like prompt roulette.

The system separates:

- LLM work, for ambiguous decisions like structural exploration and code synthesis
- deterministic work, for compile checks, correctness validation, benchmarking, profiling, filtering, and threshold-based control

That split matters. Kernel optimization should be driven by measurements, not vibes.

## How It Works

At a high level:

1. Start from a reference kernel and bootstrap a measured baseline on the target GPU.
2. Ask the Strategy Navigator whether to exploit the current best kernel or explore a new direction.
3. Ask the Coding Agent to generate candidate kernels.
4. Compile and validate them.
5. Benchmark passing candidates, discard regressions, and deeply profile the most promising ones.
6. Convert profiling metrics into bottleneck assessments.
7. Feed the results back into the next round.

The important loop is:

- exploit when the data says local improvement is still available,
- explore when the current direction is exhausted or a structural change is justified,
- terminate when the target objective is reached.

## Design Principles

The project is built around a few constraints:

- First principles over pattern matching. Every optimization direction should trace back to a measurable bottleneck.
- Structured data over prose. Agents should exchange typed records, not hand-wavy summaries.
- Deterministic where possible, LLM where necessary. Threshold checks, ranking, tabu logic, and bottleneck rules should live in code.
- Grounded in facts. "Measured 1.2ms -> 0.9ms" is useful. "Should be faster" is not.

## Current Status

This repo is still an early skeleton.

What exists today:

- the core control-plane modules,
- typed protocols between agents and services,
- spec and validation infrastructure,
- strategy and code-generation scaffolding,
- tests around the main loop and module contracts.

What is still incomplete:

- real GPU pipeline integration for compile, benchmark, and profiling,
- real cross-candidate analysis,
- full baseline seeding and stronger data contracts through the runtime path.

In other words, the architecture is there, but part of the execution layer is still stubbed.

## Why The Architecture Looks Like This

Kerlever uses a small number of modules with clear roles:

- Orchestrator, for sequencing and state management
- Strategy Navigator, for search policy
- Coding Agent, for kernel generation
- deterministic GPU services, for validation and measurement

The intent is to keep the system inspectable. If a decision was made, you should be able to see what metric justified it.

The architecture review and its corrections live here:

- [docs/architecture.md](docs/architecture.md)
- [docs/bitter-lessons.md](docs/bitter-lessons.md)

## Example Problem Spec

Optimization targets are defined in YAML:

```yaml
op_name: matmul
op_semantics: "C[M,N] = A[M,K] @ B[K,N]"
shapes:
  - [4096, 4096, 4096]
dtype: float16
target_gpu: A100
baseline_perf_us: 5.0
target_perf_us: 1.0
tolerance: 0.05
max_rounds: 20
reference_kernel: |
  __global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K) {
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < M && col < N) {
          half sum = 0;
          for (int k = 0; k < K; k++) {
              sum += A[row * K + k] * B[k * N + col];
          }
          C[row * N + col] = sum;
      }
  }
```

This spec defines:

- what operation to optimize,
- what shapes and dtype matter,
- which GPU is the target,
- what baseline and target performance mean,
- what reference kernel the search starts from.

## Development

- Python >= 3.12
- `uv` for environment management
- `pytest` for tests
- `mypy --strict` and `ruff` for quality gates

At the time of writing, the test suite passes locally against the current skeleton and stubs.

## Installation

> _TODO_

## Usage

> _TODO_

## License

TBD
