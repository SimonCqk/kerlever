# Coding Agent Module Specification

## §1 Overview

The Coding Agent is the sole CUDA code generator in the Kerlever optimization loop. It receives a strategy directive from the Strategy Navigator (specifying mode, direction, sub-mode, and constraints) along with the problem specification and optionally the current best kernel source, then produces a list of kernel candidates for GPU evaluation.

The Coding Agent combines three knowledge sources to generate high-quality CUDA code:
- **LLM code generation** via `LLMClientProtocol` — the creative engine that produces CUDA kernel implementations.
- **6-layer CUDA optimization playbook** — a structured knowledge base of optimization techniques organized by bottleneck type and hardware generation, injected into the LLM system prompt.
- **GPU hardware constraint table** — hardcoded specifications for target GPU architectures (shared memory, registers, tensor core support, async copy capabilities), ensuring generated code respects physical hardware limits.

The generation flow has four steps: (1) resolve context from the directive, GPU constraint table, and playbook, (2) build system and user prompts tailored to the requested sub-mode, (3) invoke N parallel LLM calls and parse CUDA code from responses, (4) validate extracted code and assemble `KernelCandidate` objects with unique hashes and lineage metadata.

**Scope:** Full implementation of the generation pipeline, playbook, hardware table, prompt construction, code validation, and retry logic. The module satisfies `CodingAgentProtocol` from `kerlever.protocols`. Tests use stub LLM clients.

**Non-Goals:**
- Real CUDA compilation (the GPU Pipeline's responsibility)
- AST-level code analysis (regex-level validation only)
- Kernel parameter auto-tuning (that is PARAM_SEARCH sub-mode combined with GPU Pipeline evaluation across rounds)
- Multi-model LLM support (single `LLMClientProtocol` client)
- Code caching or deduplication (the Orchestrator manages candidate hashes)

---

## §2 Requirements

### Functional Requirements

**REQ-CA-001: DE_NOVO Kernel Generation** [traces SC-1]
When the directive specifies EXPLORE mode with DE_NOVO sub-mode, the Coding Agent must generate N kernel candidates from scratch based on the operation semantics, shapes, dtype, and optimization direction. Each candidate must contain a syntactically plausible `__global__` kernel function. The generated code must not reference or depend on any prior kernel source.

**REQ-CA-002: LOCAL_REWRITE Mutation** [traces SC-2]
When the directive specifies EXPLOIT mode with LOCAL_REWRITE sub-mode, the Coding Agent must produce variants of the current best kernel source. Each variant must apply a targeted modification in the specified optimization direction while preserving the kernel's overall structure and correctness intent. The current best source must be present in the prompt context.

**REQ-CA-003: Playbook Relevance** [traces SC-3]
When building the LLM prompt, the Coding Agent must query the optimization playbook using the directive's direction and mode to retrieve relevant optimization techniques. If the direction does not match any specific playbook layer, the playbook must return universal techniques (block/grid configuration) as a fallback. The retrieved techniques must be included in the LLM system prompt.

**REQ-CA-004: GPU Constraint Accuracy** [traces SC-4]
The GPU hardware constraint table must provide correct specifications for at least A100 and H100 architectures. The constraint data must include shared memory capacity, register limits, maximum threads per block, memory bandwidth, and feature support flags (async copy, TMA, FP8, tensor core types). These constraints must be included in the LLM system prompt to prevent generation of code that violates hardware limits.

**REQ-CA-005: Code Validation** [traces SC-5]
Every generated CUDA code block must pass through a code validator before being accepted as a candidate. The validator must detect: absence of a `__global__` function (error), unbalanced brackets or braces (error), presence of host-only APIs (error), empty kernel body (error), missing `__launch_bounds__` (warning), and missing `__restrict__` on pointer parameters (warning). Code that triggers any error-severity issue must be rejected.

**REQ-CA-006: LLM Failure Retry and Skip** [traces SC-6]
When an LLM call fails to produce a valid CUDA code block (no parseable code, or code fails validation), the Coding Agent must retry that candidate once with error feedback appended to the prompt. If the retry also fails, the candidate slot is skipped. The Coding Agent must never stall or crash due to LLM unavailability or malformed output.

**REQ-CA-007: Candidate Identity and Lineage** [traces SC-7]
Each returned `KernelCandidate` must have a unique `code_hash` (SHA-256 of the source code, truncated), an `intent: CandidateIntent` describing the optimization intent (direction, mode, sub_mode, and rationale), and `parent_hashes` set to `[directive.base_kernel_hash]` for single-parent modes, `directive.parent_candidates` for RECOMBINATION, or `[]` for DE_NOVO. No two candidates in the same generation call may share the same `code_hash`.

**REQ-CA-008: PARAM_SEARCH Variant Generation** [traces SC-2]
When the directive specifies EXPLOIT mode with PARAM_SEARCH sub-mode, the Coding Agent must generate variants of the current best kernel with specific parameter values drawn from the directive's `search_range`. Each candidate should explore a different point in the parameter space.

**REQ-CA-009: PATTERN_APPLY Application** [traces SC-3]
When the directive specifies EXPLOIT mode with PATTERN_APPLY sub-mode, the Coding Agent must apply a specific optimization pattern from the playbook to the current best kernel. The pattern is identified by the directive's direction field.

**REQ-CA-010: RECOMBINATION Merge** [traces SC-2, SC-10]
When the directive specifies EXPLORE mode with RECOMBINATION sub-mode, the Coding Agent must merge code sections from multiple parent candidates as specified by the directive's `gene_map`. Parent source code must come from `directive.parent_sources` when present. The Coding Agent must not invent missing parent source bodies; if a requested parent source is absent, it must degrade gracefully by using available parent sources and explicitly noting the missing source in the prompt context.

**REQ-CA-011: GPU Constraint Fallback for Unknown Hardware** [traces SC-4]
When the target GPU specified in `ProblemSpec` is not present in the hardware constraint table, the Coding Agent must use conservative default values (48KB shared memory per block, 255 max registers per thread, 1024 max threads per block, no async copy, no TMA, no FP8). Generation must not fail due to an unrecognized GPU target.

### Quality Gates

**QG-CA-001: Type Safety** [traces SC-9]
All source code must pass `mypy --strict` with no errors.

**QG-CA-002: Lint** [traces SC-9]
All source code must pass `ruff check` with no errors.

**QG-CA-003: Existing Test Compatibility** [traces SC-8]
All 142 existing tests must continue to pass after the Coding Agent module is added. No existing module behavior may be altered.

---

## §3 Scenarios

### Generation Scenarios

**SCN-CA-001-01: DE_NOVO generates N kernels with __global__ functions**
- GIVEN: a ProblemSpec for a matmul operation on A100 with dtype float16
- AND: a directive with mode = EXPLORE, sub_mode = DE_NOVO, num_candidates = 3
- AND: no current best source (None)
- WHEN: the Coding Agent generates candidates
- THEN: 3 KernelCandidate objects are returned
- AND: each candidate's source_code contains a `__global__` function
- AND: each candidate has a CandidateIntent with mode = EXPLORE and sub_mode = DE_NOVO
- AND: each candidate has parent_hashes = []
- AND: all 3 code_hash values are distinct

**SCN-CA-001-02: DE_NOVO with LLM producing fewer valid candidates**
- GIVEN: a directive with num_candidates = 3
- AND: the LLM produces valid code for 2 of 3 calls, and 1 call fails both attempts
- WHEN: the Coding Agent generates candidates
- THEN: 2 KernelCandidate objects are returned (the failed slot is skipped)
- AND: both returned candidates pass code validation

**SCN-CA-002-01: LOCAL_REWRITE mutates current best**
- GIVEN: a directive with mode = EXPLOIT, sub_mode = LOCAL_REWRITE, direction = "reduce_register_pressure"
- AND: current_best_source is a valid CUDA kernel
- AND: num_candidates = 5
- WHEN: the Coding Agent generates candidates
- THEN: up to 5 KernelCandidate objects are returned
- AND: each candidate has parent_hashes = [directive.base_kernel_hash]
- AND: each candidate has mode = EXPLOIT and sub_mode = LOCAL_REWRITE
- AND: each candidate's intent.direction references the optimization direction
- AND: each candidate's intent.mode = EXPLOIT and intent.sub_mode = LOCAL_REWRITE

**SCN-CA-002-02: LOCAL_REWRITE without current best source**
- GIVEN: a directive with mode = EXPLOIT, sub_mode = LOCAL_REWRITE
- AND: current_best_source is None (unexpected but possible)
- WHEN: the Coding Agent generates candidates
- THEN: generation falls back to DE_NOVO behavior (generates from scratch)
- AND: the candidates are still returned with mode = EXPLOIT (reflecting the original directive)

**SCN-CA-002-03: PARAM_SEARCH generates parameter variants**
- GIVEN: a directive with mode = EXPLOIT, sub_mode = PARAM_SEARCH
- AND: search_range = {"block_size": [128, 256, 512], "tile_size": [16, 32]}
- AND: current_best_source is a valid CUDA kernel
- AND: num_candidates = 3
- WHEN: the Coding Agent generates candidates
- THEN: up to 3 KernelCandidate objects are returned
- AND: each candidate's prompt included specific parameter values from the search_range
- AND: each candidate has sub_mode = PARAM_SEARCH

**SCN-CA-002-04: PATTERN_APPLY applies playbook technique**
- GIVEN: a directive with mode = EXPLOIT, sub_mode = PATTERN_APPLY, direction = "shared_memory_tiling"
- AND: the playbook contains a technique entry for shared_memory_tiling
- AND: current_best_source is a valid CUDA kernel
- WHEN: the Coding Agent generates candidates
- THEN: the system prompt includes the specific technique details from the playbook
- AND: returned candidates have sub_mode = PATTERN_APPLY

**SCN-CA-002-05: RECOMBINATION merges parent candidates**
- GIVEN: a directive with mode = EXPLORE, sub_mode = RECOMBINATION
- AND: parent_candidates = ["hash_A", "hash_B"]
- AND: parent_sources maps both hashes to CUDA source code
- AND: gene_map = {"memory_access": "hash_A", "compute_loop": "hash_B"}
- WHEN: the Coding Agent generates candidates
- THEN: the user prompt includes both parent source bodies and the gene_map instructions
- AND: returned candidates have sub_mode = RECOMBINATION

**SCN-CA-002-06: RECOMBINATION degrades gracefully when a parent source is missing**
- GIVEN: a directive with mode = EXPLORE, sub_mode = RECOMBINATION
- AND: parent_candidates = ["hash_A", "hash_missing"]
- AND: parent_sources contains only "hash_A"
- WHEN: the Coding Agent builds the recombination prompt
- THEN: the prompt includes the source for "hash_A"
- AND: the prompt explicitly states that "hash_missing" source is unavailable
- AND: the prompt does not include placeholder or fabricated code for the missing parent
- AND: generation still attempts to produce candidates from available context

### Playbook Scenarios

**SCN-CA-003-01: Playbook returns relevant techniques for memory direction**
- GIVEN: the directive direction is "reduce_memory_bandwidth" and mode is EXPLOIT
- WHEN: the playbook is queried
- THEN: the returned techniques include entries from Layer 2 (memory access optimization)
- AND: the techniques include coalesced access, shared memory tiling, vectorized loads

**SCN-CA-003-02: Playbook returns architecture-specific techniques for Hopper**
- GIVEN: the target GPU is H100 (sm_90)
- AND: the directive direction relates to memory optimization
- WHEN: the playbook is queried
- THEN: the returned techniques include Layer 5 Hopper-specific entries (TMA, clusters, distributed shared memory, FP8)

**SCN-CA-003-03: Playbook returns universal fallback for unknown direction**
- GIVEN: the directive direction is "some_unknown_optimization_tag"
- WHEN: the playbook is queried
- THEN: the returned techniques include Layer 1 entries (block/grid configuration)
- AND: the result is non-empty (never returns zero techniques)

**SCN-CA-003-04: Playbook returns kernel-specific algorithms when applicable**
- GIVEN: the ProblemSpec op_name is "matmul"
- AND: the directive direction relates to compute optimization
- WHEN: the playbook is queried
- THEN: the returned techniques include Layer 6 matmul-specific entries (hierarchical tiling, Goto's algorithm)

### GPU Constraint Scenarios

**SCN-CA-004-01: A100 constraint lookup returns correct specs**
- GIVEN: ProblemSpec target_gpu = "A100"
- WHEN: the GPU constraint table is queried
- THEN: the returned spec has arch = "sm_80"
- AND: smem_per_sm_kb = 164
- AND: supports_cp_async = True
- AND: supports_tma = False
- AND: supports_fp8 = False

**SCN-CA-004-02: H100 constraint lookup returns correct specs**
- GIVEN: ProblemSpec target_gpu = "H100"
- WHEN: the GPU constraint table is queried
- THEN: the returned spec has arch = "sm_90"
- AND: smem_per_sm_kb = 228
- AND: supports_tma = True
- AND: supports_fp8 = True

**SCN-CA-004-03: Unknown GPU returns conservative defaults**
- GIVEN: ProblemSpec target_gpu = "RTX_9090" (not in the table)
- WHEN: the GPU constraint table is queried
- THEN: a default spec is returned with conservative values
- AND: smem_per_sm_kb = 48
- AND: max_registers_per_thread = 255
- AND: max_threads_per_block = 1024
- AND: supports_cp_async = False
- AND: supports_tma = False
- AND: supports_fp8 = False

### Code Validation Scenarios

**SCN-CA-005-01: Missing __global__ is rejected**
- GIVEN: a code block containing a function without the `__global__` qualifier
- WHEN: the code validator runs
- THEN: the result contains an error-severity issue referencing missing `__global__`
- AND: the code is rejected (not accepted as a candidate)

**SCN-CA-005-02: Unbalanced braces are rejected**
- GIVEN: a code block with mismatched `{` and `}` counts
- WHEN: the code validator runs
- THEN: the result contains an error-severity issue referencing unbalanced braces
- AND: the code is rejected

**SCN-CA-005-03: Host-only API usage is rejected**
- GIVEN: a code block containing `malloc(`, `printf(` (outside a debug guard), or `std::` usage
- WHEN: the code validator runs
- THEN: the result contains an error-severity issue referencing host-only API
- AND: the code is rejected

**SCN-CA-005-04: Empty kernel body is rejected**
- GIVEN: a code block with a `__global__` function whose body contains only comments or whitespace
- WHEN: the code validator runs
- THEN: the result contains an error-severity issue referencing empty kernel body
- AND: the code is rejected

**SCN-CA-005-05: Missing __launch_bounds__ produces warning**
- GIVEN: a code block with a valid `__global__` function but no `__launch_bounds__` annotation
- WHEN: the code validator runs
- THEN: the result contains a warning-severity issue referencing missing `__launch_bounds__`
- AND: the code is NOT rejected (warnings do not block acceptance)

**SCN-CA-005-06: Missing __restrict__ on pointers produces warning**
- GIVEN: a code block with pointer parameters that lack the `__restrict__` qualifier
- WHEN: the code validator runs
- THEN: the result contains a warning-severity issue referencing missing `__restrict__`
- AND: the code is NOT rejected

**SCN-CA-005-07: Kernel signature dtype mismatch is rejected**
- GIVEN: a ProblemSpec with dtype = "float16"
- AND: a code block whose kernel function signature uses `float*` parameters instead of `half*`
- WHEN: the code validator runs with the ProblemSpec context
- THEN: the result contains an error-severity issue referencing dtype mismatch
- AND: the code is rejected

**SCN-CA-005-08: Valid code passes all checks**
- GIVEN: a well-formed CUDA kernel with `__global__`, `__launch_bounds__`, `__restrict__` on all pointers, balanced braces, no host-only APIs, a non-empty body, and correct dtype
- WHEN: the code validator runs
- THEN: no error-severity issues are produced
- AND: the code is accepted

### LLM Failure and Retry Scenarios

**SCN-CA-006-01: LLM returns no cuda code block**
- GIVEN: the LLM response contains text but no ```cuda code block
- WHEN: the response is parsed
- THEN: the parser attempts to extract a `__global__` function from the raw text as a fallback
- AND: if extraction fails, this counts as a generation failure and triggers retry

**SCN-CA-006-02: First failure triggers retry with error feedback**
- GIVEN: the first LLM call for a candidate produces code that fails validation (e.g., missing `__global__`)
- WHEN: the validation failure is detected
- THEN: the Coding Agent retries with the specific validation error appended to the prompt
- AND: the retry prompt includes the original context plus "Your previous attempt failed because: {error_details}"

**SCN-CA-006-03: Second failure skips the candidate**
- GIVEN: both the first attempt and the retry for a candidate fail validation
- WHEN: the second failure is detected
- THEN: the candidate slot is skipped (None for that slot)
- AND: no exception is raised
- AND: the other candidates' generation is not affected

**SCN-CA-006-04: LLM exception does not crash generation**
- GIVEN: the LLM client raises an exception (network timeout, API error)
- WHEN: the exception occurs during a candidate's generation
- THEN: the exception is caught and treated as a generation failure
- AND: retry logic applies (one retry, then skip)
- AND: other concurrent candidate generations continue unaffected

**SCN-CA-006-05: All candidates fail returns empty list**
- GIVEN: a directive with num_candidates = 3
- AND: all 3 candidate generation attempts fail both tries
- WHEN: generation completes
- THEN: an empty list is returned
- AND: no exception is raised

### Candidate Assembly Scenarios

**SCN-CA-007-01: code_hash is SHA-256 of source_code**
- GIVEN: a validated CUDA source code string
- WHEN: the KernelCandidate is assembled
- THEN: code_hash equals the first 16 hex characters of SHA-256 of the source code encoded as UTF-8

**SCN-CA-007-02: intent reflects directive direction and mode**
- GIVEN: a directive with direction = "reduce_register_pressure", mode = EXPLOIT, and sub_mode = LOCAL_REWRITE
- WHEN: the KernelCandidate is assembled
- THEN: intent is a CandidateIntent with direction = "reduce_register_pressure", mode = EXPLOIT, sub_mode = LOCAL_REWRITE, and rationale = None

**SCN-CA-007-03: Duplicate hash within a generation is deduplicated**
- GIVEN: the LLM produces identical code for two different candidate slots
- WHEN: candidate assembly runs
- THEN: only one candidate with that hash is included in the returned list
- AND: the duplicate is silently dropped

---

## §4 Invariants

**INV-CA-001: Every returned candidate contains a __global__ function**
The Coding Agent must never return a KernelCandidate whose source_code lacks a `__global__` function definition. This is the minimum requirement for CUDA kernel code that can be compiled.
*Enforcement:* The code validator checks for `__global__` presence before any code is accepted. Validation runs on every candidate before it is added to the return list. There is no code path that bypasses the validator.

**INV-CA-002: LLM failures never propagate as unhandled exceptions**
No LLM call failure (network error, malformed response, validation failure, timeout) may propagate as an unhandled exception from `generate()`. The worst-case result is an empty candidate list.
*Enforcement:* Each per-candidate generation task wraps the LLM call and validation in exception handling. Exceptions trigger retry, and on second failure, the candidate slot produces None. The outer `generate()` method filters None values and returns the surviving list.

**INV-CA-003: code_hash is deterministic and collision-resistant**
The same source_code must always produce the same code_hash, and distinct source_code must produce distinct hashes with overwhelming probability. The hash function is SHA-256 truncated to 16 hex characters.
*Enforcement:* code_hash is computed as `hashlib.sha256(source_code.encode()).hexdigest()[:16]` with no salting or randomness. The 16-character hex prefix provides 64 bits of collision resistance (birthday bound ~4 billion candidates before expected collision).

**INV-CA-004: Playbook query never returns an empty result**
For any combination of direction and mode, the playbook must return at least the universal optimization techniques (Layer 1: block/grid configuration). An empty playbook result would leave the LLM without structured optimization knowledge.
*Enforcement:* The playbook query function always includes Layer 1 techniques in its return value. If no direction-specific layers match, Layer 1 alone is returned.

**INV-CA-005: GPU constraint lookup never fails**
For any target GPU string (including unrecognized ones), the hardware constraint table must return a valid specification. A lookup failure would prevent prompt construction.
*Enforcement:* The lookup function returns the matching GPU spec if found, or a conservative default spec if not. The default spec uses the lowest-common-denominator values that are safe for all NVIDIA GPUs (48KB shared memory, 255 registers, 1024 threads, no advanced features).

**INV-CA-006: Candidate intent accurately reflects the directive**
Every returned KernelCandidate must have its `intent` field set to a `CandidateIntent` whose `direction`, `mode`, and `sub_mode` match the directive. These fields are metadata for downstream consumption and must accurately reflect the generation strategy.
*Enforcement:* The `CandidateIntent` is constructed from the directive's fields during candidate assembly, not inferred or computed. The assembly step is the single point where KernelCandidate objects are constructed.

**INV-CA-007: Recombination prompts never fabricate parent source**
When a recombination directive references a parent hash whose source is absent from `directive.parent_sources`, the Coding Agent must represent that source as unavailable rather than using placeholder code or inventing a body.
*Enforcement:* Prompt construction iterates over `directive.parent_candidates`, includes source only for hashes present in `directive.parent_sources`, and records missing hashes in a separate unavailable list.

---

## §5 Interfaces

### Protocol Interface (consumed by Orchestrator)

The Coding Agent satisfies `CodingAgentProtocol`:

```
generate(
    problem_spec: ProblemSpec,
    directive: StrategyDirective,
    current_best_source: str | None,
) -> list[KernelCandidate]
```

- `problem_spec`: Defines the target operation (op_name, op_semantics, shape_cases, dtype), target hardware (target_gpu), performance objective, and the reference kernel. The Coding Agent reads target_gpu for hardware constraint lookup, op_semantics/dtype for prompt construction, and shape_cases (each a ShapeCase with shape_id, dims, weight, correctness_tolerance, and profile flag) for shape information in prompts.
- `directive`: The strategy directive from the Navigator and Orchestrator hydration step. The Coding Agent reads mode, direction, sub_mode, num_candidates, base_kernel_hash, tabu, search_range, parent_candidates, gene_map, hard_constraints, and optional parent_sources for recombination prompts. `parent_sources` is carried on the directive; it does not add a new `generate()` parameter.
- `current_best_source`: The CUDA source code of the current best-performing kernel. Required for EXPLOIT sub-modes (LOCAL_REWRITE, PARAM_SEARCH, PATTERN_APPLY). None on the first round (DE_NOVO) or when no candidate has passed evaluation yet.

### LLM Client Protocol (dependency)

```
LLMClientProtocol:
    complete(system_prompt: str, user_prompt: str) -> str
```

The Coding Agent depends on an injected LLM client for code generation. The client is called once per candidate (N calls for N candidates). All calls within a generation round run concurrently.

### CodingAgentConfig

```
CodingAgentConfig:
    max_code_length: int = 4096        # max characters in generated code block
    max_retries: int = 1               # retries per candidate on failure
    temperature_base: float = 0.7      # base LLM temperature (if supported)
    temperature_spread: float = 0.1    # variation between candidates
```

All fields have defaults. None config at construction uses all defaults.

### GPUSpec (internal type, returned by hardware table)

```
GPUSpec:
    arch: str                           # e.g., "sm_80", "sm_90"
    smem_per_sm_kb: int                 # total shared memory per SM
    max_smem_per_block_kb: int          # max shared memory per block
    registers_per_sm: int               # total registers per SM
    max_registers_per_thread: int       # max registers per thread
    max_warps_per_sm: int               # max concurrent warps per SM
    max_threads_per_block: int          # max threads per block
    hbm_bandwidth_tbps: float           # HBM bandwidth in TB/s
    l2_cache_mb: int                    # L2 cache size in MB
    supports_cp_async: bool             # Ampere+ cp.async support
    supports_tma: bool                  # Hopper+ TMA support
    supports_fp8: bool                  # Hopper+ FP8 support
    tensor_core_types: list[str]        # supported TC data types
```

### Code Validation Types (internal)

```
CodeIssueSeverity: "error" | "warning"

CodeIssue:
    severity: CodeIssueSeverity
    message: str
    check_name: str                     # e.g., "global_function", "bracket_balance"
```

The validator returns a list of `CodeIssue`. Any issue with severity "error" means the code is rejected.

### Playbook Structure (internal)

```
PlaybookTechnique:
    name: str                           # e.g., "coalesced_access"
    layer: int                          # 1-6
    applicable_when: str                # condition description
    expected_gain: str                  # e.g., "10-30%"
    template: str                       # code template or example
    caveats: str                        # gotchas, constraints

PlaybookLayer:
    layer_number: int
    name: str                           # e.g., "Memory Access Optimization"
    techniques: list[PlaybookTechnique]
```

---

## §6 Behavioral Specification

### 6.1 GPU Constraint Table

The GPU constraint table is a hardcoded mapping from GPU name strings to GPUSpec objects. It provides the Coding Agent with physical hardware limits that the generated kernel must respect.

**Supported GPUs:**

| GPU | arch | smem_per_sm_kb | max_smem_per_block_kb | registers_per_sm | max_regs_per_thread | max_warps_per_sm | max_threads_per_block | hbm_bw_tbps | l2_mb | cp.async | TMA | FP8 | TC types |
|-----|------|---------|---------|---------|-----|-----|------|------|-----|------|------|------|------|
| V100 | sm_70 | 96 | 96 | 65536 | 255 | 64 | 1024 | 0.9 | 6 | No | No | No | fp16 |
| A100 | sm_80 | 164 | 163 | 65536 | 255 | 64 | 1024 | 2.0 | 40 | Yes | No | No | fp16, bf16, tf32, fp64, int8, int4 |
| H100 | sm_90 | 228 | 227 | 65536 | 255 | 64 | 1024 | 3.35 | 50 | Yes | Yes | Yes | fp16, bf16, tf32, fp64, int8, fp8 |

**Lookup logic:**
1. Normalize the input GPU name: strip whitespace, convert to uppercase.
2. Look up the normalized name in the GPU_SPECS dictionary.
3. If found, return the corresponding GPUSpec.
4. If not found, return a conservative default GPUSpec:
   - arch = "sm_70" (oldest supported)
   - smem_per_sm_kb = 48
   - max_smem_per_block_kb = 48
   - registers_per_sm = 65536
   - max_registers_per_thread = 255
   - max_warps_per_sm = 64
   - max_threads_per_block = 1024
   - hbm_bandwidth_tbps = 0.9
   - l2_cache_mb = 6
   - supports_cp_async = False
   - supports_tma = False
   - supports_fp8 = False
   - tensor_core_types = ["fp16"]

**Formatting for prompt inclusion:**
The GPUSpec is formatted into a human-readable summary for the system prompt. The summary includes the architecture, key resource limits, and feature support flags. Advanced features that are not supported are omitted (e.g., if supports_tma is False, TMA is not mentioned) to avoid confusing the LLM into generating code that uses unsupported features.

### 6.2 Playbook

The playbook is a structured knowledge base organized into six layers of CUDA optimization techniques. Each layer targets a different class of performance bottleneck. Techniques within a layer are ordered by generality and expected impact.

**Layer 1: Block/Grid Configuration** (universal, expected gain: 10-50%)

Applies to all kernels regardless of bottleneck type. These are the foundational launch parameters that affect occupancy, wave quantization, and basic parallelism.

| Technique | Applicable When | Template Summary | Caveats |
|-----------|----------------|------------------|---------|
| block_size_tuning | Always | Try 128, 256, 512 threads per block; must be a multiple of 32 (warp size) | Larger blocks reduce occupancy if register pressure is high |
| grid_sizing | Always | Grid dim = ceiling division of problem size by block size; account for wave quantization (grid should be a multiple of SM count) | Over-provisioning grid wastes scheduling overhead |
| launch_bounds_declaration | Always | `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` on every kernel | Without this, the compiler may spill registers aggressively |

**Layer 2: Memory Access Optimization** (memory-bound kernels, expected gain: 10-30%)

Applies when profiling indicates memory bandwidth underutilization, high L1/L2 miss rates, or uncoalesced access patterns.

| Technique | Applicable When | Template Summary | Caveats |
|-----------|----------------|------------------|---------|
| coalesced_access | Strided or scattered global memory access | Consecutive threads access consecutive addresses; transpose access patterns where needed | May require data layout changes |
| shared_memory_tiling | Repeated global memory reads of same data | Load tile into `__shared__` memory, `__syncthreads()`, compute from shared | Shared memory capacity is limited per block (check GPUSpec) |
| vectorized_loads | Aligned, contiguous memory access | Use `float4`/`int4` for 128-bit transactions per thread | Requires 16-byte alignment; not applicable to scattered access |
| async_copy | Bulk data movement from global to shared (Ampere+) | `cp.async` with `__pipeline_memop_async` / `memcpy_async` | Only on GPUs with supports_cp_async = True |
| bank_conflict_avoidance | Shared memory access with stride that is a multiple of 32 | Pad shared memory declaration (+1 column) or use swizzled indexing | Extra shared memory usage from padding |

**Layer 3: Compute Optimization** (compute-bound kernels, expected gain: 5-15%)

Applies when profiling indicates compute throughput is the bottleneck (high arithmetic intensity but below peak FLOPS).

| Technique | Applicable When | Template Summary | Caveats |
|-----------|----------------|------------------|---------|
| mixed_precision | Accumulation allows reduced precision for intermediate results | FP16/BF16 for computation, FP32 for accumulation | Requires careful handling of numerical stability |
| tensor_core_utilization | Matrix operations with compatible shapes (multiples of 16) | `wmma` / `mma.sync` intrinsics; ensure tile dimensions align with TC requirements | Shape alignment is mandatory; not all operations map to TC |
| loop_unrolling | Inner loops with known trip count | `#pragma unroll` or `#pragma unroll N` | Over-unrolling increases register pressure and code size |
| fma_utilization | Multiply-add sequences | Use `fma()` or let compiler generate FMA by structuring `a * b + c` patterns | Compiler usually handles this, but manual can help with complex expressions |

**Layer 4: Advanced Techniques** (expected gain: 5-20%)

Applies when basic optimizations are saturated and the kernel needs structural changes for further improvement.

| Technique | Applicable When | Template Summary | Caveats |
|-----------|----------------|------------------|---------|
| thread_coarsening | Each thread processes only one element; occupancy is sufficient | Each thread processes K elements in a loop; reduces thread scheduling overhead | Too much coarsening reduces parallelism |
| kernel_fusion | Multiple sequential kernels with global memory round-trips between them | Merge kernels into one; intermediate results stay in registers or shared memory | Increases kernel complexity and register pressure |
| persistent_kernels | Grid-level load imbalance or many small tasks | Launch exactly SM_count blocks; each block loops over work items from a global queue | Requires careful synchronization for the work queue |
| split_k | Parallelism limited along M/N dimensions in matmul-like operations | Partition the K dimension across blocks; reduce partial results | Requires a second reduction kernel or atomic operations |

**Layer 5: Architecture-Specific Optimizations** (expected gain: 5-15%)

Applies only to specific GPU architectures. These techniques are included in the prompt only when the target GPU supports them.

| Technique | Architecture | Applicable When | Template Summary | Caveats |
|-----------|-------------|----------------|------------------|---------|
| ampere_cp_async | sm_80+ (A100+) | Bulk global-to-shared data movement | `cp.async.cg.shared.global` with pipeline stages | Requires supports_cp_async = True |
| ampere_l2_persistence | sm_80+ | Working set fits in L2 | `cudaAccessPolicyWindow` to pin data in L2 cache | Only effective when data reuse fits L2 size |
| hopper_tma | sm_90+ (H100+) | Tensor data movement | TMA descriptors for bulk async copy with hardware address generation | Requires supports_tma = True |
| hopper_clusters | sm_90+ | Cross-SM shared memory access | Thread block clusters with distributed shared memory | Requires sm_90+; changes programming model |
| hopper_fp8 | sm_90+ | Inference or training with FP8 tolerance | FP8 tensor core operations | Requires supports_fp8 = True; numerical precision tradeoff |

**Layer 6: Kernel-Specific Algorithms**

Applies when the operation type matches a known algorithmic pattern. These provide high-level algorithmic guidance specific to the ProblemSpec's operation.

| Algorithm | Operation | Key Ideas |
|-----------|-----------|-----------|
| matmul (Goto's algorithm) | matmul, gemm, batched_matmul | Hierarchical tiling: thread-block tile → warp tile → thread tile; M-N-K loop ordering; register-level accumulation |
| attention (online softmax) | attention, flash_attention | Online softmax (Flash Attention): compute softmax statistics in a single pass; never materialize the full attention matrix; tile along sequence dimension |
| reduction (warp shuffle) | sum, mean, max, min, norm | Warp-level shuffle reduction (`__shfl_down_sync`); block-level tree reduction; grid-level atomic or multi-pass |
| normalization (Welford) | layernorm, batchnorm, rmsnorm | Welford's online algorithm for numerically stable single-pass mean/variance; fused normalization+activation |
| convolution | conv2d, depthwise_conv | Implicit GEMM (no explicit im2col), Winograd for small filter sizes |

**Playbook query logic:**
1. Start with Layer 1 (always included — universal techniques).
2. Match the directive's direction against known bottleneck categories:
   - Directions mentioning "memory", "bandwidth", "coalescing", "cache" → include Layer 2.
   - Directions mentioning "compute", "throughput", "flops", "arithmetic" → include Layer 3.
   - Directions mentioning "fusion", "coarsening", "persistent", "structural" → include Layer 4.
3. Check the target GPU's architecture for Layer 5 inclusion:
   - If supports_cp_async: include Ampere techniques.
   - If supports_tma: include Hopper techniques.
4. Match ProblemSpec.op_name against Layer 6 entries:
   - If op_name contains a known operation keyword → include the corresponding algorithm entry.
5. If no direction-specific layers matched in step 2, include Layer 1 only (universal fallback).
6. Return the union of all matched layers.

### 6.3 Prompt Construction

Prompt construction builds a system prompt (static per generation call) and a user prompt (varies per candidate and sub-mode). The system prompt establishes the LLM's role and provides structured knowledge; the user prompt specifies the concrete task.

**System prompt structure:**

The system prompt has four sections, assembled in order:

1. **Role declaration:** Instructs the LLM that it is a CUDA kernel optimization expert. Establishes that the output must be a single, complete `__global__` kernel function in a ```cuda code block with no host code.

2. **Code standards (mandatory rules):**
   - `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` must be declared on every kernel.
   - All pointer parameters must use `__restrict__` and `const` where the parameter is read-only.
   - Block size must be a multiple of 32 (warp size).
   - All global memory accesses must be bounds-checked.
   - Grid size computation must use ceiling division: `(N + BLOCK_SIZE - 1) / BLOCK_SIZE`.

3. **Target GPU constraints:** The GPUSpec formatted summary (from §6.1). Includes architecture name, shared memory limits, register limits, thread limits, bandwidth, and supported features. Features not supported by the target GPU are explicitly noted as unavailable.

4. **Optimization playbook (relevant layers):** The playbook techniques retrieved by the query (from §6.2). Each technique is formatted with its name, when to apply it, a code template or example, and caveats. Only relevant layers are included to avoid diluting the prompt with irrelevant information.

5. **Output format instruction:** "Return exactly one ```cuda code block containing a complete kernel function. Do not include host code, main functions, or kernel launch syntax."

**User prompt structure by sub-mode:**

*EXPLOIT / LOCAL_REWRITE:*
- Optimization direction: `{directive.direction}`
- Current best kernel: the full source code of `current_best_source`
- Task: "Apply a targeted local rewrite to the above kernel to improve {directive.direction}."
- Constraints: `{directive.hard_constraints}` if present
- Tabu: "Do not replicate these previously attempted approaches: {directive.tabu}"

*EXPLOIT / PARAM_SEARCH:*
- Optimization direction: `{directive.direction}`
- Current best kernel: `{current_best_source}`
- Parameter search range: `{directive.search_range}`
- Task: "Generate a variant of the above kernel with the following parameter values: {specific_params}."
  - For each candidate, a different point in the search range is selected. If there are more candidates than distinct points, the remaining candidates use interpolated or boundary values.
- Constraints: `{directive.hard_constraints}` if present

*EXPLOIT / PATTERN_APPLY:*
- Optimization direction: `{directive.direction}`
- Current best kernel: `{current_best_source}`
- Pattern to apply: the specific playbook technique entry matching the direction
- Task: "Apply the {technique_name} optimization pattern to the above kernel."
- Constraints: `{directive.hard_constraints}` if present

*EXPLORE / DE_NOVO:*
- Target operation: `{problem_spec.op_semantics}`
- Input shapes: `{[(sc.shape_id, sc.dims, sc.weight) for sc in problem_spec.shape_cases]}`, dtype: `{problem_spec.dtype}`
- Optimization direction: `{directive.direction}`
- Task: "Implement a high-performance {problem_spec.op_name} kernel from scratch."
- Reference kernel (if available): `{problem_spec.reference_kernel}` as a behavioral reference (not to be copied)

*EXPLORE / RECOMBINATION:*
- Parent candidates: `{directive.parent_candidates}`
- Available parent sources: full source code for each parent hash present in `{directive.parent_sources}`
- Missing parent sources: parent hashes requested by `parent_candidates` but absent from `parent_sources`; these are explicitly listed as unavailable and must not be fabricated
- Gene map: `{directive.gene_map}` — which semantic code sections to take from which parent
- Task: "Combine the specified code sections from the available parent kernels into a single kernel. If a mapped parent source is unavailable, preserve the intent using available context and do not invent that parent's code."
- Constraints: `{directive.hard_constraints}` if present

**Candidate variation:**
To encourage diversity among candidates in the same generation call, each candidate's user prompt is varied slightly. The variation mechanism appends a candidate index and a variation hint (e.g., "Explore a different algorithmic approach than your other outputs" or "Prioritize minimizing register usage in this variant"). This is a soft hint — the LLM may or may not follow it.

### 6.4 Code Validation

The code validator performs seven regex-level checks on extracted CUDA code. Each check produces zero or one `CodeIssue`. The checks run in order, and all checks always run (no short-circuit) so that the full issue list is available for retry feedback.

**Check 1: __global__ function exists** (severity: error)
- Pattern: search for `__global__\s+void\s+\w+\s*\(` or `__global__\s+\w+\s+\w+\s*\(` in the code.
- Pass: at least one match found.
- Fail: no match. Issue message: "No __global__ function found in generated code."

**Check 2: __launch_bounds__ present** (severity: warning)
- Pattern: search for `__launch_bounds__\s*\(` in the code.
- Pass: at least one match found.
- Fail: no match. Issue message: "No __launch_bounds__ annotation found. This may cause suboptimal register allocation."

**Check 3: __restrict__ on pointer parameters** (severity: warning)
- Pattern: find pointer parameters in the kernel signature (matching `\w+\s*\*` within the `__global__` function's parameter list), then check if each has `__restrict__` preceding the `*`.
- Pass: all pointer parameters have `__restrict__`.
- Fail: at least one pointer lacks `__restrict__`. Issue message: "Pointer parameter(s) missing __restrict__ qualifier."

**Check 4: Bracket and brace balance** (severity: error)
- Count `{` and `}` in the entire code. Count `(` and `)`. Count `[` and `]`.
- Pass: all three pairs have equal counts.
- Fail: any pair is imbalanced. Issue message: "Unbalanced brackets: {details}."

**Check 5: No host-only API** (severity: error)
- Pattern: search for known host-only APIs: `\bmalloc\s*\(`, `\bfree\s*\(`, `\bprintf\s*\(`, `\bstd::`, `\bcudaMalloc`, `\bcudaMemcpy`, `\bcudaFree`.
- Pass: no matches found.
- Fail: at least one match. Issue message: "Host-only API detected: {matched_api}. Kernel code must not contain host-side calls."

**Check 6: Kernel signature dtype match** (severity: error)
- Map ProblemSpec dtype to expected CUDA pointer types: "float16" → "half", "float32" → "float", "float64" → "double", "bfloat16" → "nv_bfloat16", "int8" → "int8_t" or "char", "int32" → "int".
- Extract parameter types from the `__global__` function signature.
- Pass: at least one pointer parameter uses the expected type (or the code uses a `using` / `typedef` alias that is reasonable).
- Fail: no pointer parameter matches the expected dtype. Issue message: "Kernel signature dtype mismatch: expected {expected_type}* parameters for dtype={dtype}."
- Note: this check uses heuristic matching and may produce false negatives for complex type aliases. It errs on the side of permissiveness — only clear mismatches are flagged.

**Check 7: Non-empty kernel body** (severity: error)
- Extract the body of the `__global__` function (content between the opening `{` after the signature and the matching closing `}`).
- Strip comments (both `//` line comments and `/* */` block comments) and whitespace.
- Pass: the stripped body is non-empty (contains at least one statement).
- Fail: the body is empty or contains only comments. Issue message: "Kernel body is empty or contains only comments."

**Validation result:**
The validator returns a list of all CodeIssue objects found across all seven checks. The caller checks if any issue has severity "error" — if so, the code is rejected. If only warnings exist, the code is accepted (with warnings logged for diagnostic purposes).

### 6.5 Generation Flow

The generation flow describes how a single candidate is generated, from LLM call through validation to KernelCandidate assembly. This flow runs once per candidate, with N instances running concurrently.

**Step 1: LLM call**
- Send the system prompt and user prompt (constructed per §6.3) to the LLM client via `complete()`.
- The system prompt is shared across all candidates in a generation call. The user prompt is per-candidate (with variation hints).
- The LLM returns a text response.

**Step 2: Response parsing**
- Search the response for a ```cuda code block. Extract the content between the opening ``` and closing ``` markers.
- If no ```cuda block is found, search for a ```c or ``` (generic) code block as a secondary attempt.
- If no code block is found at all, attempt to extract a `__global__` function from the raw response text by searching for `__global__` and extracting from there to the end of the matching brace scope.
- If extraction fails entirely, this is a parse failure. Proceed to retry (Step 4).

**Step 3: Validation**
- Run the extracted code through the code validator (§6.4) with the ProblemSpec for dtype context.
- If any error-severity issue is found, this is a validation failure. Collect the error messages for retry feedback. Proceed to Step 4.
- If only warnings or no issues, proceed to Step 5.

**Step 4: Retry**
- If this is the first failure for this candidate:
  - Construct a retry prompt. The retry user prompt appends the validation errors or parse failure description to the original user prompt: "Your previous attempt failed: {error_details}. Please fix these issues and try again."
  - Send the retry prompt to the LLM client.
  - Parse the retry response (same as Step 2).
  - Validate the retry result (same as Step 3).
  - If the retry succeeds, proceed to Step 5.
  - If the retry also fails, this candidate slot produces None (skipped).
- If this is already the retry attempt: candidate slot produces None (skipped).

**Step 5: Candidate assembly**
- Compute `code_hash = hashlib.sha256(source_code.encode()).hexdigest()[:16]`.
- Construct `intent = CandidateIntent(direction=directive.direction, mode=directive.mode, sub_mode=effective_sub_mode, rationale=None)`, where `effective_sub_mode` is the directive's sub_mode (which may differ from the original if fallback to DE_NOVO occurred).
- Set `parent_hashes`:
  - For DE_NOVO: `parent_hashes = []`
  - For single-parent modes (LOCAL_REWRITE, PARAM_SEARCH, PATTERN_APPLY): `parent_hashes = [directive.base_kernel_hash]`
  - For RECOMBINATION: `parent_hashes = directive.parent_candidates`
- Return the constructed KernelCandidate.

### 6.6 CodingAgent Class

**Construction:**
The CodingAgent is constructed with:
- `llm_client: LLMClientProtocol` — the LLM client for code generation. Required (not optional, unlike the Navigator's LLM client). Without an LLM client, the Coding Agent cannot generate code.
- `config: CodingAgentConfig | None` — configuration parameters. If None, defaults are used.

The constructor stores these and initializes the playbook and hardware table (both are static data, not computed per call).

**generate() flow:**

1. **Resolve context:**
   - Look up the GPU spec for `problem_spec.target_gpu` from the hardware table (§6.1).
   - Query the playbook for relevant techniques based on the directive's direction, mode, and problem_spec.op_name (§6.2).
   - If directive sub_mode is EXPLOIT/* and current_best_source is None, log a warning and fall back to DE_NOVO-style generation (generate from op_semantics instead of mutating existing code).
   - If directive sub_mode is RECOMBINATION, collect available parent sources from `directive.parent_sources` and record any requested parent hashes that are missing.

2. **Build system prompt:**
   - Construct the system prompt from the GPU spec and playbook results (§6.3). This prompt is shared across all candidates in this call.

3. **Generate candidates concurrently:**
   - Create N tasks (one per `directive.num_candidates`) using `asyncio.TaskGroup`.
   - Each task calls the per-candidate generation flow (§6.5) with a unique candidate index for variation.
   - `TaskGroup` ensures all tasks complete (or fail individually) before proceeding.
   - If a task raises an unexpected exception (beyond what the per-candidate error handling covers), `TaskGroup` propagation is caught at the outer level.

4. **Collect and deduplicate results:**
   - Gather results from all N tasks. Filter out None values (failed candidates).
   - Deduplicate by code_hash: if two candidates produced identical source code, keep only the first.
   - Return the final list of KernelCandidate objects.

**Error handling summary:**
- LLM call exception → caught per-candidate, triggers retry, then skip (INV-CA-002).
- Response parse failure → treated as LLM failure, same retry/skip logic.
- Validation failure → retry with error feedback, then skip.
- All candidates fail → return empty list (SCN-CA-006-05).
- Hardware table miss → conservative defaults (INV-CA-005).
- Playbook miss → Layer 1 fallback (INV-CA-004).
- Missing current_best_source on EXPLOIT → fall back to DE_NOVO-style prompt.
- Missing parent source on RECOMBINATION → include only available source bodies, list missing hashes explicitly, and continue without fabricated placeholders (INV-CA-007).

---

## §7 Production Path Trace

This traces two representative paths through the Coding Agent to show how the generation flow behaves in practice.

### Path A: EXPLOIT / LOCAL_REWRITE — Reduce Register Pressure on A100

**Trigger:** The Orchestrator calls `generate()` with a matmul ProblemSpec targeting A100, an EXPLOIT/LOCAL_REWRITE directive with direction "reduce_register_pressure" and num_candidates = 5, and the current best kernel source code.

1. **Resolve context:** The GPU constraint table returns the A100 spec (sm_80, 164KB smem, cp.async supported, no TMA). The playbook is queried with direction "reduce_register_pressure" — this matches Layer 1 (universal: launch_bounds) and Layer 3 (compute: loop_unrolling affects register pressure). Since the target is A100, Layer 5 Ampere techniques are included. Since op_name is "matmul", Layer 6 matmul algorithm is included.

2. **Build system prompt:** The system prompt is assembled: CUDA expert role, code standards (launch_bounds mandatory, restrict, bounds checks), A100 constraints (164KB smem, cp.async available, no TMA/FP8), and the matched playbook layers (Layers 1, 3, 5-Ampere, 6-matmul). The output format instruction closes the prompt.

3. **Build user prompts:** For each of the 5 candidates, a LOCAL_REWRITE user prompt is constructed. The current best kernel source is included. The direction "reduce_register_pressure" is stated. Each candidate gets a variation hint (e.g., candidate 0: "Focus on reducing loop-carried dependencies", candidate 1: "Try increasing thread coarsening to reduce per-thread register count").

4. **Generate concurrently:** Five LLM calls are dispatched via `asyncio.TaskGroup`. Suppose:
   - Candidates 0, 1, 3 return valid ```cuda blocks that pass all validation checks.
   - Candidate 2 returns code missing `__launch_bounds__` — this produces a warning (not an error), so the code is accepted.
   - Candidate 4 returns code with unbalanced braces — this is an error. Retry is triggered with the error message appended. The retry produces balanced code that passes validation.

5. **Collect results:** All 5 candidates produce valid KernelCandidate objects. Each has: code_hash = SHA-256 prefix of its source, intent = CandidateIntent(direction="reduce_register_pressure", mode=EXPLOIT, sub_mode=LOCAL_REWRITE, rationale=None), parent_hashes = [directive.base_kernel_hash]. All 5 hashes are distinct (the LLM produced different code for each). The list of 5 candidates is returned to the Orchestrator.

### Path B: EXPLORE / DE_NOVO — Initial Kernel Generation on H100

**Trigger:** Round 0 cold start. The Orchestrator calls `generate()` with a reduction ProblemSpec targeting H100, an EXPLORE/DE_NOVO directive with direction "initial_exploration" and num_candidates = 3, and current_best_source = None.

1. **Resolve context:** The GPU constraint table returns the H100 spec (sm_90, 228KB smem, cp.async, TMA, FP8 all supported). The playbook is queried with direction "initial_exploration" — this does not match any specific bottleneck keyword, so only Layer 1 (universal) is returned. However, since op_name is "reduction", Layer 6 reduction algorithm (warp shuffle, block-level tree reduction) is also included. Since H100 supports all advanced features, Layer 5 Hopper techniques are included.

2. **Build system prompt:** CUDA expert role, code standards, H100 constraints (228KB smem, TMA available, FP8 available, full TC suite), playbook Layers 1, 5-Hopper, and 6-reduction.

3. **Build user prompts:** DE_NOVO prompts are constructed. Each includes the operation semantics (e.g., "sum reduction over the last dimension of a tensor"), shape_cases (with their IDs, dims, and weights), dtype, and the reference kernel as behavioral reference. Variation hints encourage structural diversity (e.g., "Use warp shuffle primitives", "Use shared memory tree reduction", "Try a persistent kernel approach").

4. **Generate concurrently:** Three LLM calls are dispatched.
   - Candidate 0 returns a valid warp-shuffle reduction kernel. Passes all checks.
   - Candidate 1 returns code that uses `std::vector` (host-only API). Error detected. Retry is triggered with "Host-only API detected: std::. Kernel code must not contain host-side calls." The retry returns clean kernel code. Passes validation.
   - Candidate 2's LLM call times out (network error). Exception is caught. Retry is attempted. The retry also times out. Candidate 2 is skipped.

5. **Collect results:** 2 valid candidates are returned. Each has parent_hashes = [] (DE_NOVO), intent = CandidateIntent(direction="initial_exploration", mode=EXPLORE, sub_mode=DE_NOVO, rationale=None). The Orchestrator receives 2 candidates instead of the requested 3, but this is a valid outcome — the system continues with what was produced.

### Path C: EXPLORE / RECOMBINATION — Structured Parent Sources

**Trigger:** The Orchestrator calls `generate()` with an EXPLORE/RECOMBINATION directive produced from structured cross-candidate analysis. The directive has parent_candidates = ["hash_A", "hash_B"], gene_map = {"memory_access": "hash_A", "compute_loop": "hash_B"}, and parent_sources hydrated for both hashes.

1. **Resolve context:** The Coding Agent reads the parent candidate hashes, verifies that `parent_sources` contains source bodies for both, and queries the playbook using the recombination direction and target operation.

2. **Build prompt:** The recombination user prompt includes the source for hash_A, the source for hash_B, the structured gene_map, and any hard constraints. The prompt asks the LLM to combine mapped semantic sections from available parents.

3. **Handle missing sources if needed:** If hash_B were absent from `parent_sources`, the prompt would list hash_B as unavailable and would not include placeholder code. Generation would continue with hash_A and the gene_map caveat.

4. **Generate and validate:** Candidate generation follows the normal parse, validation, retry, and assembly flow. Returned candidates have parent_hashes set from `directive.parent_candidates`, preserving lineage even if some source bodies were unavailable during prompt construction.

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | LLM exception crashes entire generation batch | Not wrapping individual candidate LLM calls in exception handling; letting `asyncio.TaskGroup` propagate the first failure | A single LLM timeout kills all N candidates, returning zero instead of N-1 | Each per-candidate task has its own try/except. Exceptions are caught, logged, and trigger retry/skip for that candidate only. TaskGroup sees successful completion for all tasks (INV-CA-002). |
| 2 | Prompt exceeds LLM context window | Stuffing the full playbook (all 6 layers) and the full current_best_source into every prompt without checking total length | The LLM truncates the prompt, losing the user task instruction at the end, producing garbage output | Only relevant playbook layers are included (§6.2 query logic). Prompt length is bounded by config. If the prompt exceeds the budget, playbook content is truncated before the task instruction. |
| 3 | Identical candidates waste evaluation budget | All N candidates receive the exact same prompt and the LLM produces identical code for all | N duplicate candidates go to GPU pipeline, consuming compilation and benchmark resources for zero information gain | Candidate variation hints (§6.3) encourage diversity. Post-generation deduplication by code_hash removes exact duplicates (§6.6 step 4). |
| 4 | Code validator too strict rejects valid CUDA | Overly rigid regex patterns (e.g., requiring exact signature format) reject kernels with valid but unconventional syntax | Good kernels are discarded; the Coding Agent returns fewer candidates than possible | Regex patterns are designed to be permissive for positive checks (presence of `__global__`) and strict only for negative checks (host-only APIs). Dtype matching uses heuristic, not exact pattern matching (§6.4 Check 6). |
| 5 | Code validator too lenient accepts broken CUDA | Not checking bracket balance or empty kernel body | Broken code reaches the GPU Pipeline, fails compilation, wastes a compilation round | All seven checks run on every candidate. Error-severity checks (bracket balance, empty body, host-only API) reject code before it leaves the Coding Agent (§6.4). |
| 6 | Missing current_best_source on EXPLOIT silently generates nonsense | EXPLOIT sub-mode prompt expects current_best_source but receives None; the prompt has a placeholder gap | The LLM generates code without context, producing random kernels labeled as "mutations" of nothing | When current_best_source is None during an EXPLOIT sub-mode, the Coding Agent falls back to DE_NOVO-style generation with a logged warning (§6.6 step 1). |
| 7 | GPU constraint table missing target GPU produces wrong features in prompt | Unknown GPU name falls through without a default, causing a KeyError or including no constraint info in the prompt | Generated code uses features the GPU does not support (e.g., TMA on V100), leading to compile failures | The hardware table returns conservative defaults for unknown GPUs (INV-CA-005). Conservative defaults disable all advanced features, so generated code only uses universally supported constructs. |
| 8 | Retry prompt without error context fails the same way | Retrying by simply re-sending the same prompt (no feedback about what went wrong) | The LLM makes the same mistake twice, wasting the retry opportunity | The retry prompt appends the specific validation errors from the first attempt (§6.5 Step 4). The error context guides the LLM to fix the specific issue. |
| 9 | Fake recombination parent | Prompt builder inserts placeholder code or asks the LLM to infer a missing parent source | The returned kernel is not a real recombination of measured parents and lineage becomes misleading | Use `directive.parent_sources` only; list missing parent hashes explicitly and never fabricate source (INV-CA-007). |

---

## §9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|-------------|-----------|
| SC-1: DE_NOVO generates N kernels each with `__global__` function | REQ-CA-001 | SCN-CA-001-01, SCN-CA-001-02 |
| SC-2: EXPLOIT/LOCAL_REWRITE mutates current_best and RECOMBINATION uses parent sources | REQ-CA-002, REQ-CA-008, REQ-CA-010 | SCN-CA-002-01, SCN-CA-002-02, SCN-CA-002-03, SCN-CA-002-05, SCN-CA-002-06 |
| SC-3: Playbook returns relevant techniques for direction | REQ-CA-003, REQ-CA-009 | SCN-CA-003-01, SCN-CA-003-02, SCN-CA-003-03, SCN-CA-003-04, SCN-CA-002-04 |
| SC-4: GPU constraint table correct for A100/H100 | REQ-CA-004, REQ-CA-011 | SCN-CA-004-01, SCN-CA-004-02, SCN-CA-004-03 |
| SC-5: Code validator catches missing `__global__`, unbalanced brackets | REQ-CA-005 | SCN-CA-005-01, SCN-CA-005-02, SCN-CA-005-03, SCN-CA-005-04, SCN-CA-005-05, SCN-CA-005-06, SCN-CA-005-07, SCN-CA-005-08 |
| SC-6: LLM failure -> retry -> skip candidate | REQ-CA-006 | SCN-CA-006-01, SCN-CA-006-02, SCN-CA-006-03, SCN-CA-006-04, SCN-CA-006-05 |
| SC-7: Each KernelCandidate has unique hash, correct intent (CandidateIntent), parent lineage (parent_hashes) | REQ-CA-007 | SCN-CA-007-01, SCN-CA-007-02, SCN-CA-007-03 |
| SC-8: Existing 142 tests pass | QG-CA-003 | (verified by running existing test suite after module addition) |
| SC-9: mypy --strict and ruff check pass | QG-CA-001, QG-CA-002 | (verified by CI tooling) |
| SC-10: Recombination prompts receive hydrated parent sources and degrade gracefully when missing | REQ-CA-010 | SCN-CA-002-05, SCN-CA-002-06 |
