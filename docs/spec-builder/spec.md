# Spec Builder Module Specification

## S1 Overview

The Spec Builder is the entry point of the Kerlever optimization loop. Before the Orchestrator can run, it needs a validated `ProblemSpec`. The Spec Builder provides two modes for producing one:

- **Batch validation** (`--validate`): Takes a YAML file, runs a two-stage validation pipeline (deterministic checks followed by LLM semantic judgment), and returns a structured pass/fail result.
- **Interactive collection** (`--interactive`): Conversationally extracts fields from the user via an LLM, assembles a `ProblemSpec`, runs the full validation pipeline, and outputs YAML.

Both modes produce the same output: a validated `ProblemSpec` that the Orchestrator consumes. The Spec Builder is a standalone CLI tool (`python -m kerlever.spec_builder`) with no runtime dependency on the Orchestrator.

### Two-Stage Validation Pipeline

**Stage 1 (deterministic):** Structural and semantic checks that require no LLM. These are fast, reproducible, and always run. They cover Pydantic schema validation, reference kernel resolution and CUDA surface checks, shape cases validation, dtype recognition, objective and metric sanity, and target GPU recognition.

**Stage 2 (LLM judge):** A Claude model evaluates the spec for semantic consistency, specificity, feasibility, completeness, and reference kernel quality. Returns structured JSON with five dimensions, each scored pass/warn/fail with a reason. Stage 2 is skippable via `--no-llm`.

---

## S2 Requirements

### Functional Requirements

**REQ-SB-001: Batch Validation** [traces SC-1]
The `--validate` command must accept a YAML file path containing the new ProblemSpec structure (with `shape_cases`, `objective`, and `target_metric_value` fields), run the two-stage validation pipeline, and return a structured result. The result is `is_valid=True` when all deterministic checks pass and no LLM dimension scores `fail`. When any check fails, the result is `is_valid=False` with a list of structured errors identifying which checks failed and why.

**REQ-SB-002: Reference Kernel Resolution** [traces SC-2]
The `reference_kernel` field in the YAML input supports three forms: an inline CUDA source string, a `file:///path` URI pointing to a local `.cu` file, and an `https://...` URL pointing to a remote source. The resolver must detect the form, fetch the content, and replace the field value with the resolved CUDA source code. After resolution, the source must contain at least one `__global__` or `__device__` function signature.

**REQ-SB-003: Deterministic-Only Mode** [traces SC-3]
When `--no-llm` is passed, the validation pipeline must skip Stage 2 entirely. No API key is required. The result reflects only Stage 1 deterministic checks. This mode must work without any LLM client configured.

**REQ-SB-004: LLM Judge Structured Output** [traces SC-4]
The LLM judge must return a structured JSON response parsed into a list of `ValidationIssue` objects. Each issue has a dimension (one of: consistency, specificity, feasibility, completeness, kernel_quality), a severity (pass, warn, fail), and a reason string. If the LLM response cannot be parsed as valid JSON on the first attempt, the judge retries once. If the second attempt also fails to parse, the judge returns a single `ValidationIssue` with dimension="parse_error", severity=fail, and a reason describing the parse failure.

**REQ-SB-005: Interactive Collection** [traces SC-5]
The `--interactive` command starts a conversational session. An LLM extracts `ProblemSpec` fields from the user's natural language description. After each user message, the LLM identifies which fields have been provided and which are still missing, then asks targeted follow-up questions for missing fields. Once all required fields are collected, the system runs the full validation pipeline (Stage 1 + Stage 2 unless `--no-llm`). If validation passes, the final `ProblemSpec` is serialized to YAML and printed to stdout. If validation fails, the errors are shown and the user can correct fields.

**REQ-SB-006: Type Safety and Lint** [traces SC-6]
All source code must pass `mypy --strict` and `ruff check` with no errors.

### Quality Gates

**QG-SB-001: Type Safety** [traces SC-6]
All source code passes `mypy --strict` with zero errors.

**QG-SB-002: Lint** [traces SC-6]
All source code passes `ruff check` with zero errors.

---

## S3 Scenarios

**SCN-SB-001-01: Valid YAML passes batch validation**
- GIVEN: a YAML file with all required `ProblemSpec` fields populated correctly
- AND: the `reference_kernel` contains inline CUDA source with a `__global__` function
- AND: `shape_cases` is a non-empty list with unique `shape_id` values, positive dims, and positive weights
- AND: at least one ShapeCase has `profile: true`
- AND: dtype is a recognized CUDA type
- AND: target_gpu is a known GPU architecture
- AND: `objective.primary_metric` is a recognized metric and `objective.aggregation` is a recognized aggregation
- AND: `target_metric_value` > 0
- AND: max_rounds >= 1
- WHEN: `--validate spec.yaml` is run
- THEN: the result has `is_valid=True` and an empty error list

**SCN-SB-001-02: Invalid YAML fails with structured errors**
- GIVEN: a YAML file where `dtype` is "imaginary_type" and `objective.primary_metric` is "invalid_metric"
- WHEN: `--validate spec.yaml` is run
- THEN: the result has `is_valid=False`
- AND: the error list contains one entry identifying the unrecognized dtype
- AND: the error list contains one entry identifying the invalid objective metric

**SCN-SB-001-03: Duplicate shape_id fails validation**
- GIVEN: a YAML file where `shape_cases` contains two entries both with `shape_id: "square_4k"`
- WHEN: `--validate spec.yaml` is run
- THEN: the result has `is_valid=False`
- AND: the error list contains one entry identifying the duplicate shape_id

**SCN-SB-001-04: No profile shape emits warning**
- GIVEN: a YAML file where all ShapeCase entries have `profile: false` (or omit the field, defaulting to false)
- AND: all other fields are valid
- WHEN: `--validate spec.yaml` is run
- THEN: the result has `is_valid=True` (warnings do not cause failure)
- AND: the issue list contains one entry with dimension="shape_cases" and severity=warn indicating no shape is marked for profiling

**SCN-SB-001-05: Invalid objective fields fail validation**
- GIVEN: a YAML file where `target_metric_value` is -1.0
- AND: `objective.regression_guard_pct` is -5.0
- WHEN: `--validate spec.yaml` is run
- THEN: the result has `is_valid=False`
- AND: the error list contains one entry identifying the non-positive target_metric_value
- AND: the error list contains one entry identifying the negative regression_guard_pct

**SCN-SB-002-01: File URI reference kernel resolution**
- GIVEN: a YAML file where `reference_kernel` is `file:///tmp/kernel.cu`
- AND: `/tmp/kernel.cu` exists and contains a `__global__ void foo()` function
- WHEN: the resolver processes the reference_kernel field
- THEN: the field value is replaced with the file contents

**SCN-SB-002-02: HTTPS URL reference kernel resolution**
- GIVEN: a YAML file where `reference_kernel` is `https://example.com/kernel.cu`
- AND: the URL returns CUDA source with a `__device__` function
- WHEN: the resolver processes the reference_kernel field
- THEN: the field value is replaced with the fetched content

**SCN-SB-002-03: Reference kernel missing CUDA markers**
- GIVEN: a resolved reference kernel that contains only plain C code (no `__global__` or `__device__`)
- WHEN: the resolver validates the resolved content
- THEN: resolution fails with an error stating the source is not valid CUDA

**SCN-SB-002-04: File URI points to nonexistent file**
- GIVEN: a YAML file where `reference_kernel` is `file:///nonexistent/path.cu`
- WHEN: the resolver processes the reference_kernel field
- THEN: resolution fails with an error identifying the missing file

**SCN-SB-002-05: HTTPS URL fetch fails**
- GIVEN: a YAML file where `reference_kernel` is `https://example.com/404.cu`
- AND: the URL returns HTTP 404
- WHEN: the resolver processes the reference_kernel field
- THEN: resolution fails with an error identifying the fetch failure and HTTP status

**SCN-SB-003-01: Deterministic-only mode skips LLM**
- GIVEN: no `ANTHROPIC_API_KEY` environment variable is set
- AND: a valid YAML spec file
- WHEN: `--validate --no-llm spec.yaml` is run
- THEN: validation completes successfully using only deterministic checks
- AND: no LLM calls are attempted

**SCN-SB-004-01: LLM judge returns well-formed JSON**
- GIVEN: a stub LLM client that returns valid JSON with five dimensions
- WHEN: the LLM judge is invoked
- THEN: the response is parsed into five `ValidationIssue` objects
- AND: each has a recognized dimension, severity, and non-empty reason

**SCN-SB-004-02: LLM judge retries on parse failure then succeeds**
- GIVEN: a stub LLM client that returns malformed JSON on the first call and valid JSON on the second call
- WHEN: the LLM judge is invoked
- THEN: the judge retries once
- AND: the second response is parsed successfully into `ValidationIssue` objects

**SCN-SB-004-03: LLM judge degrades after two parse failures**
- GIVEN: a stub LLM client that returns malformed JSON on both calls
- WHEN: the LLM judge is invoked
- THEN: the judge returns a single `ValidationIssue` with dimension="parse_error" and severity=fail

**SCN-SB-005-01: Interactive collection completes full flow**
- GIVEN: a stub LLM client for field extraction and a mock stdin providing user answers
- WHEN: `--interactive` is run
- AND: the user provides all required fields across multiple messages
- THEN: the system assembles a complete `ProblemSpec`
- AND: runs the full validation pipeline
- AND: prints the validated `ProblemSpec` as YAML to stdout

**SCN-SB-005-02: Interactive collection handles partial input**
- GIVEN: a stub LLM client and a mock stdin
- WHEN: the user's first message provides only `op_name` and `op_semantics`
- THEN: the system identifies the remaining missing fields (including `shape_cases`, `objective`, `target_metric_value`, `dtype`, `target_gpu`, `max_rounds`, `reference_kernel`)
- AND: asks follow-up questions for those fields specifically

---

## S4 Invariants

**INV-SB-001: Reference kernel is always resolved before validation**
The reference_kernel field must be resolved (URI fetched, content validated for CUDA markers) before any deterministic or LLM validation runs on the full spec. If resolution fails, validation reports the resolution error and does not proceed to further checks.
*Enforcement:* The validation pipeline calls the resolver as its first step. Deterministic checks and LLM judge receive the spec only after successful resolution.

**INV-SB-002: Deterministic checks are pure and reproducible**
Stage 1 deterministic checks must produce identical results for identical inputs. They must not call external services, use randomness, or depend on mutable state.
*Enforcement:* Deterministic check functions accept only a `ProblemSpec` (after resolution) and return a list of `ValidationIssue` objects. They have no side effects and no injected dependencies.

**INV-SB-003: LLM judge failures never crash the pipeline**
An LLM call failure (network error, malformed response, timeout) must never cause the validation pipeline to raise an unhandled exception. LLM failures degrade to a structured error result.
*Enforcement:* The LLM judge wraps all LLM calls in try/except. Parse failures trigger one retry. Any remaining failure produces a `ValidationIssue` with severity=fail rather than propagating the exception.

**INV-SB-004: Interactive mode always validates before output**
In interactive mode, a ProblemSpec is never serialized to YAML output without passing through the full validation pipeline. Even if the LLM extraction step claims all fields are complete, the pipeline must run.
*Enforcement:* The interactive flow calls the same validation entry point used by batch mode before writing any YAML output.

---

## S5 Interfaces

### CLI Arguments

```
python -m kerlever.spec_builder --validate <path.yaml> [--no-llm]
python -m kerlever.spec_builder --interactive [--no-llm]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--validate` | flag + positional path | mutually exclusive with `--interactive` | Run batch validation on a YAML file |
| `--interactive` | flag | mutually exclusive with `--validate` | Start interactive field collection |
| `--no-llm` | flag | optional | Skip Stage 2 LLM judge; no API key needed |

Exit codes: 0 on success (valid spec or interactive completion), 1 on validation failure, 2 on usage error.

### Validation Types

```
ValidationSeverity: Literal["pass", "warn", "fail"]

ValidationDimension: Literal[
    "schema", "reference_kernel", "shape_cases", "dtype",
    "objective", "target_gpu",
    "consistency", "specificity", "feasibility",
    "completeness", "kernel_quality",
    "parse_error"
]

ValidationIssue:
    dimension: ValidationDimension
    severity: ValidationSeverity
    message: str

ValidationResult:
    is_valid: bool
    issues: list[ValidationIssue]
```

`is_valid` is True when no issue has `severity="fail"`. Issues with `severity="warn"` do not cause failure.

Dimensions "schema" through "target_gpu" are deterministic (Stage 1). The deterministic dimensions are: "schema", "reference_kernel", "shape_cases", "dtype", "objective", "target_gpu". Dimensions "consistency" through "kernel_quality" are LLM judge (Stage 2). "parse_error" is used only for LLM response parse failures.

### LLMClientProtocol

```
class LLMClientProtocol(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the text response."""
        ...
```

The LLM client is injected into both the LLM judge and the interactive collector. The production implementation uses the Anthropic Python SDK with model `claude-sonnet`. The protocol enables stub/mock injection for testing.

### YAML Input Format

```yaml
op_name: matmul
op_semantics: "C[M,N] = A[M,K] @ B[K,N]"
dtype: float16
target_gpu: A100
shape_cases:
  - shape_id: "4k_square"
    dims: [4096, 4096, 4096]
    weight: 1.0
    profile: true
  - shape_id: "tall_skinny"
    dims: [8192, 128, 4096]
    weight: 0.5
    correctness_tolerance: 0.01
objective:
  primary_metric: weighted_p50_us
  aggregation: weighted_mean
  regression_guard_pct: 0.0
target_metric_value: 1.0
max_rounds: 20
reference_kernel: |
  __global__ void matmul(const half* A, const half* B, half* C, int M, int N, int K) {
      // naive reference implementation
  }
```

The `reference_kernel` field can alternatively be:
```yaml
reference_kernel: "file:///path/to/kernel.cu"
```
or:
```yaml
reference_kernel: "https://example.com/kernel.cu"
```

### YAML Output Format (Interactive Mode)

Same schema as the input format. The `reference_kernel` field contains the resolved inline CUDA source, never a URI. All `shape_cases` entries are preserved as-is. The `objective` block is included in the output.

---

## S6 Behavioral Specification

### 6.1 Reference Kernel Resolution

The resolver determines how to interpret the `reference_kernel` field value and produces resolved CUDA source code.

**Form detection:** The resolver inspects the raw string value:
- If it starts with `file:///`, treat as a local file path URI.
- If it starts with `https://`, treat as a remote URL.
- Otherwise, treat as inline CUDA source (no fetch needed).

**Local file resolution (`file:///`):**
1. Strip the `file://` prefix to obtain the filesystem path.
2. Verify the file exists. If not, return an error with the missing path.
3. Read the file contents as UTF-8.

**Remote URL resolution (`https://`):**
1. Fetch the URL content. Use a timeout of 30 seconds.
2. If the HTTP response status is not 2xx, return an error with the URL and status code.
3. Decode the response body as UTF-8.

**CUDA surface check (all forms):**
After obtaining the source content (inline, from file, or from URL), verify it contains at least one occurrence of `__global__` or `__device__`. This is a substring search, not a full parse. If neither marker is found, return an error stating the content does not appear to be valid CUDA source.

**On success:** Return the resolved source string. The validation pipeline replaces the `reference_kernel` field value with this resolved content before proceeding to further checks.

### 6.2 Deterministic Checks (Stage 1)

Stage 1 runs six categories of checks on the `ProblemSpec` after reference kernel resolution. Each check produces zero or more `ValidationIssue` entries. All checks run unconditionally (no short-circuit between categories).

**Check 1 -- Schema validation:**
The YAML content must parse into a valid `ProblemSpec` via Pydantic. Missing required fields, wrong types, or extra unknown fields produce issues with dimension="schema" and severity=fail.

**Check 2 -- Reference kernel resolution:**
Covered by the resolver (section 6.1). If resolution failed, a single issue with dimension="reference_kernel" and severity=fail is added. If resolution succeeded but the source is suspiciously short (fewer than 20 non-whitespace characters), an issue with severity=warn is added.

**Check 3 -- Shape cases validation:**
- `shape_cases` must be a non-empty list.
- Each ShapeCase must have a non-empty `shape_id` string.
- All `shape_id` values must be unique across the list. Duplicate shape_ids produce an issue with severity=fail.
- Each ShapeCase.`dims` must be a non-empty list of positive integers.
- No dimension in `dims` may exceed 2^31 - 1 (maximum 32-bit int, the common CUDA kernel argument limit).
- Each ShapeCase.`weight` must be greater than 0.
- If `correctness_tolerance` is provided, it must be in the range (0, 1) exclusive.
- At least one ShapeCase should have `profile: true`. If no ShapeCase has `profile: true`, produce an issue with severity=warn (the system can still run but deep profiling will have no designated shapes).
- Violations produce issues with dimension="shape_cases" and severity=fail (except the missing profile flag which is severity=warn).

**Check 4 -- Dtype recognition:**
`dtype` must be one of a recognized set: `float16`, `float32`, `float64`, `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. An unrecognized dtype produces an issue with dimension="dtype" and severity=fail.

**Check 5 -- Objective and metric validation:**
- `target_metric_value` must be greater than 0.
- `objective.primary_metric` must be one of: `"weighted_p50_us"`, `"weighted_p95_us"`, `"worst_case_p50_us"`.
- `objective.aggregation` must be one of: `"weighted_mean"`, `"max"`.
- `objective.regression_guard_pct` must be greater than or equal to 0.
- `max_rounds` must be a positive integer, at least 1.
- Violations produce issues with dimension="objective" and severity=fail.

**Check 6 -- Target GPU recognition:**
`target_gpu` must be one of a recognized set of GPU architectures. The recognized set includes at least: `A100`, `H100`, `A10`, `L4`, `L40`, `T4`, `V100`, `A6000`, `RTX3090`, `RTX4090`. The comparison is case-insensitive. An unrecognized GPU produces an issue with dimension="target_gpu" and severity=warn (warn, not fail, because the set may not be exhaustive and an unrecognized GPU is not necessarily wrong).

### 6.3 LLM Judge (Stage 2)

Stage 2 invokes the LLM to evaluate five semantic dimensions that cannot be checked deterministically. It is skipped entirely when `--no-llm` is active.

**Prompt construction:**
The system prompt instructs the LLM to act as a CUDA kernel optimization specification reviewer. The user prompt contains the full `ProblemSpec` serialized as YAML (with the resolved reference kernel). The prompt requests the LLM to evaluate five dimensions and return a JSON array.

**Expected response format:**
```json
[
  {"dimension": "consistency", "severity": "pass", "reason": "..."},
  {"dimension": "specificity", "severity": "pass", "reason": "..."},
  {"dimension": "feasibility", "severity": "warn", "reason": "..."},
  {"dimension": "completeness", "severity": "pass", "reason": "..."},
  {"dimension": "kernel_quality", "severity": "pass", "reason": "..."}
]
```

**Dimension definitions:**
- **consistency**: Do the fields form a coherent specification? (e.g., do the shape_cases dims match the op_semantics, does the reference kernel signature match the dtype and shapes, is the objective metric consistent with the shape_cases weights?)
- **specificity**: Is the specification precise enough for an optimization agent to act on? (e.g., are op_semantics unambiguous, is the target_metric_value concrete, do shape_cases cover representative workload points?)
- **feasibility**: Is the performance target achievable given the hardware and operation? (Use roofline reasoning: is the target_metric_value within the theoretical peak for the target GPU and operation type given the shape_cases?)
- **completeness**: Are all fields populated with meaningful content, or are any fields placeholder stubs? Do shape_cases cover enough of the workload surface?
- **kernel_quality**: Is the reference kernel a reasonable starting point? (Does it implement the stated operation, is it syntactically plausible CUDA, would it compile?)

**Response parsing:**
1. Extract JSON from the LLM response. If the response contains markdown code fences, strip them. Attempt `json.loads()` on the result.
2. Validate the parsed array: must contain exactly 5 objects, each with `dimension`, `severity`, and `reason` fields. The dimension values must be the five expected dimensions. Severity must be one of `pass`, `warn`, `fail`.
3. Convert each object to a `ValidationIssue`.

**Retry on parse failure:**
If step 1 or step 2 fails (JSON decode error, missing fields, wrong dimensions, etc.), retry the LLM call exactly once with the same prompts. If the second attempt also fails to parse, produce a single `ValidationIssue` with dimension="parse_error", severity=fail, and a reason describing what went wrong.

**LLM error handling:**
If the LLM call itself raises an exception (network error, auth failure, rate limit), treat it the same as a parse failure for retry purposes. After two failures, degrade to the parse_error issue.

### 6.4 Validation Pipeline Orchestration

The validation pipeline combines Stage 1 and Stage 2 into a single flow.

1. **Resolve** the reference kernel (section 6.1). If resolution fails, return `ValidationResult(is_valid=False, issues=[resolution_error])` immediately.
2. **Run deterministic checks** (section 6.2) on the spec with the resolved reference kernel. Collect all issues.
3. **If `--no-llm` is not active**, run the LLM judge (section 6.3). Collect additional issues.
4. **Merge** all issues from steps 2 and 3 into a single list.
5. **Compute `is_valid`**: True if no issue has severity=fail. Warn-only results are valid with warnings.
6. Return `ValidationResult(is_valid=is_valid, issues=all_issues)`.

If deterministic checks produce any severity=fail issues, the LLM judge still runs (when enabled). This is intentional: the LLM may catch additional problems worth reporting.

### 6.5 Interactive Mode

Interactive mode uses an LLM to extract ProblemSpec fields from conversational user input.

**Session loop:**
1. Print a welcome message explaining what information is needed.
2. Read user input from stdin.
3. Send the conversation history to the LLM with a system prompt that instructs it to: (a) extract any ProblemSpec fields from the user's message, (b) identify which fields are still missing, (c) generate a response asking for the missing fields.
4. The LLM response is parsed to extract two parts: (a) a structured JSON object of extracted field values, and (b) a natural language follow-up question for the user.
5. The extracted field values are accumulated into a partial ProblemSpec.
6. The follow-up question is printed to stdout.
7. If all required fields are now present, proceed to step 8. Otherwise, go to step 2.
8. Construct a full `ProblemSpec` from the accumulated fields.
9. Run the full validation pipeline (section 6.4).
10. If validation passes, serialize the `ProblemSpec` to YAML and print to stdout. Exit with code 0.
11. If validation fails, print the validation errors and ask the user if they want to correct any fields. If yes, go to step 2 with the current field values retained. If no, exit with code 1.

**Field extraction prompt:**
The system prompt provides the ProblemSpec schema (field names, types, descriptions) and instructs the LLM to return a JSON block with any fields it can extract from the user's message, plus a natural language follow-up. The schema presented to the LLM includes the new structured fields: `shape_cases` (list of ShapeCase with shape_id, dims, weight, correctness_tolerance, profile), `objective` (PerformanceObjective with primary_metric, aggregation, regression_guard_pct), and `target_metric_value`. The expected format:

```json
{"extracted": {"op_name": "matmul", "dtype": "float16", "shape_cases": [{"shape_id": "4k_square", "dims": [4096, 4096, 4096], "weight": 1.0, "profile": true}]}, "follow_up": "What performance objective and target metric do you want?"}
```

Fields already collected in previous turns are included in the prompt context so the LLM does not re-ask for them.

### 6.6 CLI Entry Points

**`__main__.py`:**
The module entry point parses CLI arguments, instantiates the LLM client (if needed), and dispatches to the appropriate mode.

1. Parse arguments: `--validate <path>`, `--interactive`, `--no-llm`.
2. If neither `--validate` nor `--interactive` is provided, print usage and exit with code 2.
3. If both are provided, print error and exit with code 2.
4. **Validate mode:**
   a. Load the YAML file. If the file does not exist or is not valid YAML, print error and exit with code 1.
   b. Create the LLM client (unless `--no-llm`). The production client reads `ANTHROPIC_API_KEY` from environment. If `--no-llm` is not set and no key is available, print error and exit with code 1.
   c. Run the validation pipeline.
   d. Print the `ValidationResult` as structured JSON to stdout.
   e. Exit with code 0 if valid, code 1 if invalid.
5. **Interactive mode:**
   a. Create the LLM client (for field extraction; required even with `--no-llm` since extraction uses LLM). If `--no-llm` is set, the LLM client is still needed for extraction but the validation pipeline skips Stage 2.
   b. Run the interactive session loop (section 6.5).
   c. Exit with the code determined by the session outcome.

**Note on interactive + --no-llm:** Interactive mode always requires an LLM client for field extraction. The `--no-llm` flag only suppresses Stage 2 (LLM judge) of the validation pipeline, not the interactive extraction LLM.

---

## S7 Production Path Trace

### 7.1 Batch Validation Path

**Trigger:** User runs `python -m kerlever.spec_builder --validate spec.yaml`.

1. **Argument parsing.** The CLI parses the command line. It identifies `--validate` mode with the path `spec.yaml`. It checks whether `--no-llm` is set.

2. **YAML loading.** The CLI reads `spec.yaml` and parses it with `yaml.safe_load()`. If the file is missing or unparseable, the user sees an error message and exit code 1.

3. **Pydantic construction.** The loaded dict is passed to `ProblemSpec.model_validate()`. If required fields are missing or types are wrong, this raises a `ValidationError`. The pipeline catches this and converts it to one or more `ValidationIssue` entries with dimension="schema".

4. **Reference kernel resolution.** The resolver examines the `reference_kernel` value. For inline source, it runs the CUDA surface check directly. For `file:///` or `https://` URIs, it fetches the content first, then checks for CUDA markers. If resolution fails, validation returns immediately with the resolution error.

5. **Deterministic checks.** Six check categories run on the spec with the resolved kernel: schema, reference kernel, shape cases, dtype, objective and metric, and target GPU. Each produces zero or more issues. All six run regardless of earlier failures.

6. **LLM judge (if enabled).** The spec is serialized to YAML and sent to the LLM with a reviewer prompt. The response is parsed into five `ValidationIssue` objects. Parse failures trigger one retry, then degrade to an error issue.

7. **Result assembly.** All issues from steps 3-6 are collected. `is_valid` is computed (no fail-severity issues). The `ValidationResult` is serialized as JSON and printed to stdout.

8. **Exit.** Code 0 if valid, code 1 if invalid.

### 7.2 Interactive Collection Path

**Trigger:** User runs `python -m kerlever.spec_builder --interactive`.

1. **Argument parsing.** The CLI parses `--interactive` mode and creates an LLM client.

2. **Welcome.** The system prints a message explaining what information it needs to define a kernel optimization problem.

3. **Conversation loop.** The user types a natural language description. The LLM extracts field values and generates a follow-up question. Extracted fields accumulate. The system repeats until all required fields are present.

4. **Spec assembly.** The accumulated fields are used to construct a `ProblemSpec`.

5. **Full validation.** The same two-stage pipeline from the batch path runs on the assembled spec. If `--no-llm` was set, only Stage 1 runs.

6. **Output.** If valid, the `ProblemSpec` is serialized to YAML and printed. If invalid, errors are shown and the user can correct fields or exit.

7. **Exit.** Code 0 on successful output, code 1 if the user exits without producing a valid spec.

---

## S8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | Unresolved URI in validated spec | Skipping reference kernel resolution or running it after other checks | The Orchestrator receives a `file:///` or `https://` string instead of CUDA source code; downstream compilation fails | INV-SB-001: resolver always runs first; spec fields are replaced with resolved content before any other check |
| 2 | LLM failure crashes validation | Not wrapping LLM calls in try/except; letting network errors or JSON parse errors propagate | `--validate` exits with an unhandled traceback instead of a structured error | INV-SB-003: all LLM interactions are wrapped; parse failures degrade to a structured fail issue after one retry |
| 3 | Interactive output without validation | Skipping the validation pipeline in interactive mode after field collection appears complete | User receives a YAML file that fails when the Orchestrator tries to use it | INV-SB-004: interactive mode always calls the full validation pipeline before emitting YAML |
| 4 | Non-deterministic stage 1 | Calling the LLM or using random/time-dependent logic in deterministic checks | Same input produces different results across runs; tests become flaky | INV-SB-002: deterministic checks are pure functions with no side effects |
| 5 | Silent reference kernel check skip | Treating resolution errors as warnings instead of fails, allowing specs with no valid CUDA source to pass | The Orchestrator receives a spec with a reference_kernel that cannot compile; the entire optimization loop fails on round 0 | Resolution errors are always severity=fail; the pipeline short-circuits on resolution failure |

---

## S9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|-------------|-----------|
| SC-1: `--validate` returns is_valid=True for valid YAML, structured ERROR for invalid | REQ-SB-001 | SCN-SB-001-01, SCN-SB-001-02, SCN-SB-001-03, SCN-SB-001-04, SCN-SB-001-05 |
| SC-2: reference_kernel inline/file/URL all resolve correctly | REQ-SB-002 | SCN-SB-002-01, SCN-SB-002-02, SCN-SB-002-03, SCN-SB-002-04, SCN-SB-002-05 |
| SC-3: `--no-llm` works without API key (deterministic only) | REQ-SB-003 | SCN-SB-003-01 |
| SC-4: LLM judge stub returns structured JSON, parsed to ValidationIssue | REQ-SB-004 | SCN-SB-004-01, SCN-SB-004-02, SCN-SB-004-03 |
| SC-5: `--interactive` with stub LLM + mock input completes full flow to YAML output | REQ-SB-005 | SCN-SB-005-01, SCN-SB-005-02 |
| SC-6: mypy --strict and ruff check pass | REQ-SB-006, QG-SB-001, QG-SB-002 | (quality gate, verified by CI) |
