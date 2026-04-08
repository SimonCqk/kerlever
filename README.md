# Kerlever

Agent-driven CUDA kernel optimization loop.

Generate kernel candidates → compile → benchmark → profile → analyze → navigate the search space → generate again. Repeat until the target latency is met or the budget is exhausted.

```
Problem Spec
    │
    ▼
Orchestrator ◄───────────────────────────────────────────────────┐
    │                                                            │
    ▼                                                            │
Strategy Navigator                                               │
    │  mode + direction + constraints                            │
    ▼                                                            │
Coding Agent  ◄── compile fail (retry w/ error)                  │
    │                                                            │
    ├─── Remote GPU Pod ───────────────────────────────────┐     │
    │   Compiler → Benchmarker → Profile Interpreter       │     │
    └──────────────────────────────────────────────────────┘     │
                                                                 │
Cross-Candidate Analyzer ────────────────────────────────────────┘
```

## Design Philosophy

**Few agents, strong deterministic layers.** LLM reasoning is reserved for genuinely ambiguous decisions — tradeoff analysis, creative structural changes, cross-candidate semantic diffs. Everything else (gate checks, tabu filtering, threshold comparisons, trend computation, bottleneck tagging) is deterministic code.

Key principles from [AGENTS.md](AGENTS.md):

- **First principles over pattern matching** — every optimization traces back to a quantifiable bottleneck
- **Structured data, not natural language** — agents communicate through typed Pydantic models, not prose
- **No over-engineering** — build the simplest thing that closes the loop
- **Grounded in facts** — "measured 1.2ms → 0.9ms on A100, p50 over 100 runs", not "should be faster"

## Architecture

4 LLM agents + 3 deterministic services, connected as a DAG with early-exit paths:

| Module | Type | Status | Role |
|---|---|---|---|
| **Orchestrator** | Agent | Implemented | Global control loop, state machine, round management, termination |
| **Strategy Navigator** | Agent | Implemented | Exploit/explore decision, search direction, tabu, UCB1 fallback |
| **Coding Agent** | Agent | Implemented | CUDA kernel generation with 6-layer optimization playbook |
| **Cross-Candidate Analyzer** | Agent | Stub | Semantic diff, winning gene identification, recombination hints |
| **Compiler Service** | Service | Stub | nvcc compile + correctness validation + static analysis |
| **Benchmarker** | Service | Stub | Latency measurement + candidate ranking + deep profiling (ncu/nsys) |
| **Profile Interpreter** | Service | Stub | Rule-based bottleneck tagging from profiling metrics |

### Exploit vs. Explore

The optimization loop operates in two modes, selected by the Strategy Navigator:

- **Exploit** — small delta on the current best kernel (param tuning, local rewrite, pattern apply). 3–8 candidates per round.
- **Explore** — structural change (algorithm swap, de novo generation, recombination). 2–3 candidates per round.

Mode selection uses deterministic gates for clear signals (cold start → explore, plateau → explore, near target → exploit) and LLM reasoning for ambiguous cases. When the LLM fails, UCB1 provides a deterministic fallback.

### Early-Exit Paths

Not every candidate goes through the full pipeline:

| Trigger | Condition | Action |
|---|---|---|
| Compiler | Compile or correctness fail | Skip benchmark, feed error back to Coding Agent |
| Benchmarker | Latency regresses beyond threshold | Discard candidate |
| Profile Interpreter | Same bottleneck N consecutive rounds | Mark direction exhausted, force explore |
| Orchestrator | Target latency met | Terminate, return best kernel |

## Project Structure

```
src/kerlever/
├── __init__.py
├── __main__.py              # CLI entry point
├── types.py                 # Shared Pydantic models and enums
├── protocols.py             # Protocol interfaces for all services
├── stubs.py                 # Stub implementations for testing
├── orchestrator.py          # Main optimization loop
├── state.py                 # Atomic state persistence
├── llm_client.py            # LLMClientProtocol + Anthropic implementation
├── problem_spec.py          # ProblemSpec YAML loader
├── spec_builder/            # Problem spec validation & interactive builder
│   ├── deterministic.py     #   6-category structural validation
│   ├── llm_judge.py         #   5-dimension LLM semantic judge
│   ├── resolver.py          #   Reference kernel resolution (inline/file/URL)
│   └── interactive.py       #   Conversational spec collection
├── navigator/               # Strategy Navigator (5-phase decision engine)
│   ├── signals.py           #   Derived signal computation (pure, deterministic)
│   ├── gates.py             #   5 hard gates (first-match-wins priority)
│   ├── llm_reasoning.py     #   LLM-based direction reasoning
│   ├── ucb1.py              #   UCB1 deterministic fallback
│   ├── assembly.py          #   Directive assembly (7-step)
│   └── config.py            #   11 tunable parameters
└── coding_agent/            # CUDA kernel code generator
    ├── hardware.py          #   GPU constraint table (V100/A100/H100/T4/L40/RTX4090)
    ├── playbook.py          #   6-layer CUDA optimization playbook
    ├── prompt_builder.py    #   System + user prompt construction (5 sub-modes)
    ├── code_validator.py    #   7 regex-level CUDA code checks
    ├── generator.py         #   LLM generation + parse + retry + skip
    └── config.py            #   Generation parameters

tests/
├── test_orchestrator.py     # Orchestrator round loop + state machine
├── test_state.py            # Atomic persistence
├── test_problem_spec.py     # YAML loading + validation
├── test_spec_builder/       # Spec validation pipeline
├── test_navigator/          # Strategy Navigator (signals, gates, UCB1, assembly)
└── test_coding_agent/       # Coding Agent (hardware, playbook, validator, generator)

docs/
├── architecture.md          # System DAG and module reference
├── strategy-navigator.md    # Navigator design (5 phases, gates, UCB1 formula)
├── orchestrator/spec.md     # Orchestrator formal specification
├── spec-builder/spec.md     # Spec Builder formal specification
├── navigator/spec.md        # Navigator formal specification
└── coding-agent/spec.md     # Coding Agent formal specification

examples/
└── matmul_spec.yaml         # Example: 4096×4096 FP16 matmul on A100
```

## Coding Agent Internals

The Coding Agent deserves special mention — it's the only module that directly produces CUDA code.

**6-Layer Optimization Playbook** (`playbook.py`):

| Layer | Focus | Typical Gain |
|---|---|---|
| 1 | Block/Grid configuration | 10–50% |
| 2 | Memory access (coalescing, tiling, vectorized loads, async copy) | 10–30% |
| 3 | Compute (mixed precision, tensor cores, loop unrolling) | 5–15% |
| 4 | Advanced (thread coarsening, kernel fusion, persistent kernels) | 5–20% |
| 5 | Architecture-specific (Ampere cp.async, Hopper TMA/clusters/FP8) | 5–15% |
| 6 | Kernel-specific algorithms (Flash Attention, Winograd, warp shuffle reduction) | varies |

**GPU Hardware Table** (`hardware.py`): Hardcoded specs for V100, A100, H100, T4, L40, RTX4090. Unknown GPUs get conservative defaults (48KB smem, sm_70). The table informs prompt construction — unsupported features (e.g., TMA on Ampere) are omitted to prevent the LLM from generating invalid code.

**Code Validator** (`code_validator.py`): 7 regex-level checks run on every generated kernel, no short-circuit:

1. `__global__` function exists (error)
2. `__launch_bounds__` present (warning)
3. `__restrict__` on pointer params (warning)
4. Bracket/brace balance (error)
5. No host-only APIs — malloc, printf, cudaMalloc, etc. (error)
6. Kernel signature dtype matches ProblemSpec (error)
7. Non-empty kernel body (error)

Errors → reject + retry with feedback. Warnings → pass through for information.

## Stub Modules

The following modules are currently stubbed (`stubs.py`) with Protocol-conforming implementations for testing:

- **GPU Pipeline** (Compiler + Benchmarker + Profile Interpreter) — requires remote GPU pod infrastructure
- **Cross-Candidate Analyzer** — LLM-powered semantic diff between candidates

The stubs simulate realistic behavior: the GPU pipeline returns random latency variations with progressive improvement and occasional compile failures; the analyzer returns empty analysis. All 243 tests pass against these stubs.

## Installation

> _TODO_

## Usage

> _TODO_

## Problem Spec Format

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

| Field | Description |
|---|---|
| `op_name` | Operation identifier (matmul, attention, reduction, etc.) |
| `op_semantics` | Mathematical definition of the operation |
| `shapes` | Input tensor dimensions to optimize for |
| `dtype` | Data type (float16, float32, bfloat16, etc.) |
| `target_gpu` | Target GPU architecture (A100, H100, V100, etc.) |
| `baseline_perf_us` | Reference kernel latency in microseconds |
| `target_perf_us` | Target latency to achieve |
| `tolerance` | Acceptable margin above target (0.05 = 5%) |
| `max_rounds` | Maximum optimization rounds before termination |
| `reference_kernel` | CUDA source of the baseline kernel |

## Tech Stack

- **Python ≥ 3.12** with asyncio
- **Pydantic** for all data models (validation + serialization)
- **Anthropic Claude SDK** for LLM calls (wrapped in `LLMClientProtocol` for testability)
- **uv** as package manager
- **pytest + pytest-asyncio** for testing (243 tests, 100% pass)
- **mypy --strict** + **ruff** for type checking and linting

## License

TBD
