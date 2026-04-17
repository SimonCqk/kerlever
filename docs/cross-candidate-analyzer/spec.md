# Cross-Candidate Analyzer Module Specification

## §1 Overview

The Cross-Candidate Analyzer is the semantic comparison agent in the Kerlever optimization loop. It receives a round-local set of correctness-passing, benchmarked kernel candidates and extracts structured facts about what changed, which candidate traits appear reusable, which combinations are worth trying, and which measured regressions should become avoid patterns.

The Analyzer is not a ranking service and does not own search policy. Ranking, incumbent updates, tabu handling, and exploit/explore selection remain owned by the Benchmarker, Orchestrator, and Strategy Navigator. The Analyzer operates only on structured evidence already produced by deterministic services: candidate source, lineage metadata, benchmark objective scores, static analysis, profiling metrics, and bottleneck assessments. LLM reasoning is optional and may synthesize semantic deltas or genes, but every output must trace to measured evidence.

The Analyzer is greenfield for this task. It must satisfy the existing `CrossCandidateAnalyzerProtocol.analyze(top_k_results, problem_spec) -> CrossCandidateAnalysis` signature while extending shared output types backward-compatibly. Existing consumers that read only `insights`, `winning_genes`, and `recombination_suggestions` must continue to work.

**Scope:** The module specification covers the shared contracts and behavior needed for a future implementation: deterministic feature extraction, optional JSON-only LLM semantic synthesis, validation/retry/fallback, regression-as-negative-evidence handling, structured recombination hints, and integration contracts with Orchestrator, Navigator, and Coding Agent.

**Non-Goals:**
- No CUDA AST parser, clang integration, SASS parser, or tree-sitter dependency.
- No real GPU measurement, benchmarking, profiling, or Profile Interpreter implementation.
- No incumbent update, candidate ranking, tabu mutation, or exploit/explore policy decision.
- No analysis of compile failures, correctness failures, or infrastructure errors as semantic evidence.
- No broad protocol rewrite for `CrossCandidateAnalyzerProtocol` or `CodingAgentProtocol`.

---

## §2 Requirements

### Success Criteria

**SC-CCA-001: Implementation-ready enduring spec**
The Cross-Candidate Analyzer behavior, contracts, edge cases, and integration boundaries are specified deeply enough that Coding can implement without guessing.

**SC-CCA-002: Backward-compatible structured output**
Analyzer output is JSON-serializable and extends `CrossCandidateAnalysis` without breaking legacy `insights`, `winning_genes`, or `recombination_suggestions` consumers.

**SC-CCA-003: Regressions are negative evidence only**
Benchmark-regressed candidates can produce avoid patterns and anti-genes but never become incumbents and are not treated as positive recombination parents by default.

**SC-CCA-004: Structured recombination reaches Navigator**
Navigator can construct recombination directives from `recombination_hints.parent_candidates` and `gene_map`, with legacy fallback when structured hints are absent.

**SC-CCA-005: Coding Agent receives real parent sources**
Recombination directives support Orchestrator-hydrated `StrategyDirective.parent_sources` so Coding Agent prompts contain actual selected parent source bodies when available.

**SC-CCA-006: Incumbent and failed-candidate safety**
Compile failures, correctness failures, and infrastructure errors remain excluded from analysis; incumbent updates still ignore regressions.

**SC-CCA-007: Quality gates pass**
Future implementation must pass `ruff check .`, `mypy .`, and `pytest`.

### Functional Requirements

**REQ-CCA-001: Stable protocol with richer shared types** [traces SC-CCA-002]
The Analyzer must keep the existing protocol signature stable and return `CrossCandidateAnalysis`. Rich fields must be optional or default-empty so old constructors, stubs, and consumers remain valid.

**REQ-CCA-002: Eligible analysis input** [traces SC-CCA-003, SC-CCA-006]
The Analyzer may analyze only candidates whose evaluation reached benchmarking and whose outcomes are one of `IMPROVED`, `BASELINE_MATCH`, or `REGRESSION`. Candidates with `COMPILE_FAIL`, `CORRECTNESS_FAIL`, or `ERROR` must be excluded before analysis and rejected defensively if present.

**REQ-CCA-003: Deterministic evidence extraction** [traces SC-CCA-001, SC-CCA-002]
For each eligible candidate, the Analyzer must deterministically extract structured evidence from candidate metadata, source text feature tags, objective scores, per-shape benchmark results, static analysis, and profile data when available.

**REQ-CCA-004: Optional LLM semantic synthesis** [traces SC-CCA-001, SC-CCA-002]
When an LLM client is configured, the Analyzer may ask it to synthesize semantic deltas, reusable genes, recombination hints, and avoid patterns from bounded structured evidence. The LLM output must be JSON only, parsed into typed models, validated, retried once on malformed or invalid output, and replaced by deterministic fallback if still invalid.

**REQ-CCA-005: Measured-evidence requirement** [traces SC-CCA-003, SC-CCA-006]
Every `CandidateGene`, `RecombinationHint`, and `AvoidPattern` must reference at least one candidate hash and at least one measured evidence source: objective score, relative objective ratio, per-shape latency, static resource metric, profiling metric, or bottleneck assessment evidence. Source-code pattern detection alone is insufficient for a positive or negative claim.

**REQ-CCA-006: Regression-only negative evidence** [traces SC-CCA-003, SC-CCA-006]
Candidates with outcome `REGRESSION` may contribute `AvoidPattern` entries and negative context in `insights`. They must not be emitted as positive `CandidateGene.source_candidate_hash` entries or selected in `RecombinationHint.parent_candidates` unless a future spec explicitly marks a subcomponent safe using positive measured evidence from another candidate.

**REQ-CCA-007: Structured recombination hints** [traces SC-CCA-004]
When at least two non-regressing candidates provide complementary measured strengths, the Analyzer should emit `RecombinationHint` records with explicit parent candidate hashes, a `gene_map` from semantic section names to parent hashes, confidence, risk flags, and evidence links.

**REQ-CCA-008: Legacy field bridging** [traces SC-CCA-002, SC-CCA-004]
The Analyzer must populate legacy `insights`, `winning_genes`, and `recombination_suggestions` from the same validated rich evidence so older Navigator code still receives useful context. Legacy strings must not contain claims that are absent from structured fields.

**REQ-CCA-009: Boundary preservation** [traces SC-CCA-001, SC-CCA-006]
The Analyzer must not choose the next strategy directive, mutate `OptimizationState`, update incumbents, rank candidates for replacement, alter tabu entries, or reinterpret Profile Interpreter bottleneck tags. It summarizes evidence for downstream policy modules.

**REQ-CCA-010: Parent-source contract propagation** [traces SC-CCA-005]
The shared `StrategyDirective` contract must support optional `parent_sources: dict[str, str] | None`, hydrated by Orchestrator after Navigator selects recombination parents. Analyzer output must use stable candidate hashes so this hydration can resolve sources from the incumbent or persisted `kernels/<hash>.cu` files.

### Quality Gates

**QG-CCA-001: Type Safety** [traces SC-CCA-007]
Future implementation must pass `mypy .` with no errors under the repository configuration.

**QG-CCA-002: Lint** [traces SC-CCA-007]
Future implementation must pass `ruff check .` with no errors.

**QG-CCA-003: Test Coverage** [traces SC-CCA-007]
Future implementation must include tests for deterministic extraction, regression-to-avoid-pattern handling, LLM parse/validation/retry/fallback, rich-to-legacy bridging, and integration contract compatibility.

---

## §3 Scenarios

**SCN-CCA-001-01: Existing protocol remains usable**
- GIVEN: a caller invokes `analyze(top_k_results, problem_spec)` with eligible candidates
- WHEN: analysis completes
- THEN: the result is a `CrossCandidateAnalysis`
- AND: the legacy fields `insights`, `winning_genes`, and `recombination_suggestions` are present
- AND: rich fields are present as default-empty lists when no rich evidence is produced

**SCN-CCA-002-01: Improved and baseline-match candidates enter analysis**
- GIVEN: a round has candidates with outcomes `IMPROVED` and `BASELINE_MATCH`
- AND: each candidate has a benchmark bundle
- WHEN: the Analyzer receives the filtered analysis input
- THEN: both candidates are eligible for semantic delta and positive gene extraction

**SCN-CCA-002-02: Benchmarked regression enters as negative evidence**
- GIVEN: a candidate has outcome `REGRESSION`
- AND: the candidate has a benchmark bundle with objective and per-shape results
- WHEN: the Analyzer compares it against non-regressing candidates
- THEN: the regressing candidate may produce an `AvoidPattern`
- AND: the regressing candidate is not emitted as a positive recombination parent

**SCN-CCA-002-03: Failed candidates are rejected defensively**
- GIVEN: a candidate with outcome `COMPILE_FAIL`, `CORRECTNESS_FAIL`, or `ERROR` appears in the input unexpectedly
- WHEN: the Analyzer validates inputs
- THEN: the candidate is ignored for semantic analysis
- AND: no gene, recombination hint, or avoid pattern references that candidate hash

**SCN-CCA-003-01: Deterministic extraction captures measured facts**
- GIVEN: an eligible candidate with source code, parent hashes, objective score, per-shape benchmark results, static analysis, and profile metrics
- WHEN: deterministic extraction runs
- THEN: the extracted evidence includes candidate hash, parent hashes, mode, sub-mode, direction, objective value, relative objective ratios, shape latencies, static resource metrics, bottleneck tags, and simple source feature tags

**SCN-CCA-004-01: Valid LLM JSON is accepted**
- GIVEN: deterministic evidence for at least two eligible candidates
- AND: the LLM returns valid JSON matching the rich analysis schema
- AND: every rich entry has required measured evidence references
- WHEN: validation runs
- THEN: the parsed semantic deltas, genes, recombination hints, and avoid patterns are included in `CrossCandidateAnalysis`

**SCN-CCA-004-02: Malformed LLM JSON retries once then falls back**
- GIVEN: the first LLM response is malformed JSON
- WHEN: the Analyzer detects the parse failure
- THEN: it retries once with the validation error and JSON-only instruction
- AND: if the retry also fails, it returns deterministic fallback analysis without raising an exception

**SCN-CCA-005-01: Source-only claim is rejected**
- GIVEN: the LLM proposes a gene because a candidate source contains `__shared__`
- AND: the proposal references no objective, static-analysis, benchmark, profile, or bottleneck evidence
- WHEN: validation runs
- THEN: the gene is rejected
- AND: it does not appear in legacy `winning_genes`

**SCN-CCA-006-01: Regression produces avoid pattern**
- GIVEN: candidate A improves objective score and candidate B regresses objective score
- AND: candidate B has a source feature tag `extra_syncs`
- AND: candidate B has measured latency worse than incumbent across one or more shape cases
- WHEN: analysis runs
- THEN: the output may include an `AvoidPattern` for unnecessary synchronization linked to candidate B and the measured latency regression

**SCN-CCA-007-01: Complementary winners produce recombination hint**
- GIVEN: candidate A improves memory-related profile metrics on shape `large`
- AND: candidate B improves compute-related objective results on shape `small`
- AND: both candidates have outcomes `IMPROVED` or `BASELINE_MATCH`
- WHEN: analysis identifies complementary measured strengths
- THEN: it emits a `RecombinationHint` with `parent_candidates = [A, B]`
- AND: `gene_map` maps semantic sections such as `memory_access` and `compute_loop` to the corresponding parent hashes

**SCN-CCA-008-01: Legacy fields mirror rich fields**
- GIVEN: rich `CandidateGene` and `RecombinationHint` entries are produced
- WHEN: the final analysis object is assembled
- THEN: `winning_genes` contains concise legacy descriptions of the validated candidate genes
- AND: `recombination_suggestions` contains concise legacy descriptions of the validated recombination hints

**SCN-CCA-010-01: Parent-source hydration can resolve analyzer-selected hashes**
- GIVEN: a recombination hint references parent hashes that correspond to persisted candidate source files
- WHEN: Navigator chooses the hint and Orchestrator hydrates a recombination directive
- THEN: `StrategyDirective.parent_sources` can be populated with a mapping from each parent hash to its source code

---

## §4 Invariants

**INV-CCA-001: Every rich claim is evidence-linked**
Every semantic delta, candidate gene, recombination hint, and avoid pattern must identify candidate hashes and evidence references sufficient to audit the claim against measured evaluation data.
*Enforcement:* Rich model validation rejects entries missing candidate hash links or measured evidence references. Legacy fields are derived only after rich validation.

**INV-CCA-002: Failed candidates never influence analysis**
Compile failures, correctness failures, and infrastructure errors must not influence positive or negative semantic analysis because they lack trustworthy benchmark evidence.
*Enforcement:* Input validation filters allowed outcomes to `IMPROVED`, `BASELINE_MATCH`, and `REGRESSION`; all other outcomes are skipped before feature extraction.

**INV-CCA-003: Regressions are negative-only by default**
A regressing candidate must not be selected as a positive gene source or recombination parent in this version.
*Enforcement:* Candidate gene and recombination hint validators reject entries whose parent or source candidate has outcome `REGRESSION`. Regression evidence is routed only to avoid patterns and negative insights.

**INV-CCA-004: LLM failure cannot stall the loop**
LLM unavailability, malformed output, schema mismatch, or unsupported claims must not prevent the optimization round from completing.
*Enforcement:* The LLM call is optional, wrapped in exception handling, retried once on invalid output, and replaced by deterministic fallback on failure.

**INV-CCA-005: Analyzer output remains JSON-serializable**
All analysis outputs must be Pydantic/JSON-compatible primitives and lists/dicts. No source file handles, live objects, NumPy arrays, or non-serializable model clients may appear in `CrossCandidateAnalysis`.
*Enforcement:* Output is constructed from shared BaseModel types and validated via model serialization before returning.

**INV-CCA-006: Analyzer does not own policy decisions**
The Analyzer must not mutate optimization state or decide the next `StrategyDirective`.
*Enforcement:* The protocol accepts only round-local candidate/evaluation pairs and `ProblemSpec`, and returns only `CrossCandidateAnalysis`; there is no state mutation parameter or directive return path.

---

## §5 Interfaces

### Protocol Interface

The Analyzer satisfies the existing shared protocol:

```python
class CrossCandidateAnalyzerProtocol(Protocol):
    async def analyze(
        self,
        top_k_results: list[tuple[KernelCandidate, EvaluationResult]],
        problem_spec: ProblemSpec,
    ) -> CrossCandidateAnalysis:
        ...
```

Parameter semantics:
- `top_k_results`: Round-local candidate/evaluation pairs selected by Orchestrator for semantic analysis. Despite the historical name, this set may include benchmarked regressions as negative evidence. Every pair must contain a `KernelCandidate` and its corresponding `EvaluationResult`.
- `problem_spec`: Operation semantics, workload shapes, dtype, target GPU, performance objective, and target value. Used for evidence normalization and for interpreting shape weights; not used to run benchmarks.

### Shared Type Extensions

All new types are shared data contracts owned by `kerlever/types.py` and must be JSON-serializable Pydantic models.

**SemanticDelta**:
- `candidate_hash: str` — candidate this delta describes.
- `parent_hashes: list[str]` — candidate lineage from `KernelCandidate.parent_hashes`.
- `outcome: CandidateOutcome` — evaluation outcome for safety checks.
- `summary: str` — concise semantic change description.
- `changed_features: list[str]` — deterministic source/evidence feature tags such as `shared_memory`, `vectorized_load`, `warp_shuffle`, `tensor_core`, `unroll`, `extra_sync`, `cp_async`, `launch_bounds`, or `indexing_change`.
- `evidence_refs: list[str]` — references to measured evidence fields, e.g. `benchmark.objective_score.relative_to_incumbent`, `profile.metrics.dram_throughput_pct_of_peak`, or `static_analysis.registers_per_thread`.
- `confidence: Literal["low", "medium", "high"]`.

**CandidateGene**:
- `gene_id: str` — stable identifier unique within the analysis result.
- `source_candidate_hash: str` — non-regressing candidate that exhibits the gene.
- `gene_type: str` — semantic section tag such as `memory_access`, `tiling`, `compute_loop`, `launch_config`, `synchronization`, `instruction_mix`, `resource_usage`, or `algorithmic_structure`.
- `description: str` — what appears reusable.
- `evidence: dict[str, float]` — measured numeric evidence supporting the gene.
- `affected_shape_ids: list[str]` — workload shapes where evidence applies.
- `risk_flags: list[str] = []` — known caveats, e.g. `shape_specific`, `register_pressure`, `smem_pressure`, `noise_sensitive`.
- `confidence: Literal["low", "medium", "high"]`.

**RecombinationHint**:
- `hint_id: str` — stable identifier unique within the analysis result.
- `parent_candidates: list[str]` — two or more non-regressing parent candidate hashes.
- `gene_map: dict[str, str]` — maps semantic section names to parent candidate hashes.
- `expected_benefit: str` — concise evidence-grounded rationale, not a promise of speedup.
- `evidence_candidate_hashes: list[str]` — candidates whose measured data support the hint.
- `required_constraints: list[str] = []` — constraints Coding Agent should preserve, e.g. `keep_tile_size_32`.
- `risk_flags: list[str] = []` — caveats that Navigator/Coding should see.
- `confidence: Literal["low", "medium", "high"]`.

**AvoidPattern**:
- `pattern_id: str` — stable identifier unique within the analysis result.
- `source_candidate_hash: str` — candidate whose measured regression or weakness exposed the pattern.
- `pattern: str` — pattern to avoid, expressed as a structured tag or short phrase.
- `reason: str` — evidence-grounded rationale.
- `evidence: dict[str, float]` — measured numeric evidence supporting avoidance.
- `affected_shape_ids: list[str]` — shapes where the negative effect appears.
- `scope: str = "candidate_local"` — how broadly to apply the avoid pattern; default is local to similar candidates.
- `confidence: Literal["low", "medium", "high"]`.

**CrossCandidateAnalysis** extends the existing model:
- Existing required legacy fields remain:
  - `insights: list[str]`
  - `winning_genes: list[str]`
  - `recombination_suggestions: list[str]`
- New backward-compatible fields default to empty lists:
  - `semantic_deltas: list[SemanticDelta] = []`
  - `candidate_genes: list[CandidateGene] = []`
  - `recombination_hints: list[RecombinationHint] = []`
  - `avoid_patterns: list[AvoidPattern] = []`

**StrategyDirective** extends the existing model:
- `parent_sources: dict[str, str] | None = None` — optional mapping from parent candidate hash to full source code. Navigator selects parent hashes; Orchestrator hydrates this field before calling Coding Agent; Coding Agent consumes it when constructing recombination prompts.

### Analyzer Dependencies

**LLMClientProtocol (optional):**

```python
complete(system_prompt: str, user_prompt: str) -> str
```

If no LLM client is provided, Analyzer behavior is deterministic-only and still returns a valid `CrossCandidateAnalysis`.

### Input Evidence Semantics

For a candidate to be eligible, its `EvaluationResult` must satisfy:
- `outcome in {IMPROVED, BASELINE_MATCH, REGRESSION}`.
- `benchmark is not None`.
- `compile_status == SUCCESS`.
- `correctness is not None and correctness.passed is True`.

Optional evidence may include:
- `static_analysis`: registers, shared memory, spills, occupancy estimate.
- `profile.metrics`: occupancy, DRAM/SM throughput, stalls, cache, tensor core utilization, arithmetic intensity.
- `profile.assessment`: bottleneck tags, primary tag, evidence, and rule trace.

---

## §6 Behavioral Specification

### 6.1 Input Validation and Eligibility

The Analyzer first normalizes the input list into eligible and ignored candidate sets.

Eligible candidates are those with benchmarked outcomes: `IMPROVED`, `BASELINE_MATCH`, and `REGRESSION`. This reflects the distinction between measurement failures and measured negative evidence. Regressions have benchmark data and can teach the system what not to repeat; compile failures, correctness failures, and infrastructure errors do not have trustworthy performance semantics and must be ignored.

Validation rules:
1. Candidate hash in `KernelCandidate.code_hash` must match `EvaluationResult.candidate_hash`; mismatches are ignored as invalid pairs.
2. Evaluation outcome must be one of the eligible outcomes.
3. Benchmark bundle must be present.
4. Compile status must be `SUCCESS` and correctness must be present and passed.
5. Duplicate candidate hashes are deduplicated by keeping the first valid pair.

If fewer than two eligible candidates remain, the Analyzer still returns a valid fallback `CrossCandidateAnalysis`, but it may contain only basic insights and avoid patterns. Orchestrator normally gates invocation at two or more benchmarked candidates; this defensive behavior prevents crashes in tests and stubs.

### 6.2 Deterministic Feature Extraction

For each eligible candidate, deterministic extraction produces a normalized evidence record. This record is the sole input to optional LLM synthesis.

Extracted fields:
- Candidate identity: `candidate_hash`, `parent_hashes`, `intent.direction`, `intent.mode`, `intent.sub_mode`.
- Outcome class: `IMPROVED`, `BASELINE_MATCH`, or `REGRESSION`.
- Objective evidence: aggregate objective value, relative-to-baseline ratio, relative-to-incumbent ratio, and metric name.
- Shape evidence: per-shape p50/p95 latency, stdev, run count, and shape weight from `ProblemSpec`.
- Static resource evidence: registers/thread, shared memory/block, spills, and occupancy estimate when present.
- Profile evidence: raw profile metrics and `BottleneckAssessment` tags/evidence/rule trace when present.
- Source feature tags from simple textual scans: shared memory use, vectorized types, warp intrinsics, tensor core fragments/MMA, unroll pragmas, synchronization, async copy, launch bounds, restrict qualifiers, and indexing expressions.

Deterministic extraction does not infer that a feature caused a performance change. It records co-occurrence with measured facts. Causal language is reserved for validated synthesis and must remain evidence-qualified.

### 6.3 Deterministic Baseline Analysis

Even without an LLM, the Analyzer returns useful structured output:

1. Create one `SemanticDelta` per eligible candidate summarizing source feature tags and measured outcome.
2. Create `AvoidPattern` entries for regressing candidates when a negative source/static/profile feature co-occurs with worse objective or shape latency.
3. Create conservative `CandidateGene` entries for non-regressing candidates only when a measured improvement or baseline match is linked to a concrete source/profile/static feature.
4. Create `RecombinationHint` entries only when at least two non-regressing candidates have complementary genes in different gene types.
5. Populate legacy fields from the validated rich entries.

The deterministic path should prefer under-claiming to over-claiming. If evidence is weak or ambiguous, record a low-confidence insight rather than a gene or recombination hint.

### 6.4 Optional LLM Semantic Synthesis

When an LLM client is configured, synthesis runs after deterministic extraction.

**System prompt requirements:**
- The LLM must act as a CUDA semantic comparison assistant, not a strategy policy owner.
- It must use only supplied structured evidence and bounded source snippets/diffs.
- It must not claim speedups without measured evidence.
- It must treat regressions as negative evidence only.
- It must return exactly one JSON object matching the requested schema.

**User prompt content, in priority order:**
1. ProblemSpec summary: operation semantics, dtype, target GPU, shape IDs/dims/weights, objective.
2. Eligible candidate evidence records with outcome and measured fields.
3. Bounded source feature summaries and small snippets/diffs if available.
4. Existing deterministic genes and avoid patterns.
5. Explicit instruction that compile/correctness/error outcomes are absent by design and must not be discussed.

**Expected response shape:**

```json
{
  "semantic_deltas": [],
  "candidate_genes": [],
  "recombination_hints": [],
  "avoid_patterns": [],
  "insights": []
}
```

The response may omit entries when evidence is insufficient, but it must not introduce fields outside the schema.

### 6.5 Parse, Validation, Retry, and Fallback

The Analyzer validates LLM output in four layers:

1. **JSON parse:** response must parse as a single JSON object with no surrounding prose.
2. **Schema validation:** fields must conform to shared rich models.
3. **Evidence validation:** each rich claim must reference known candidate hashes and measured evidence fields.
4. **Safety validation:** regression candidates may appear only in `SemanticDelta` and `AvoidPattern`, not in positive `CandidateGene` or `RecombinationHint.parent_candidates`.

On first failure, the Analyzer retries once. The retry prompt includes the validation error and repeats the JSON-only instruction. On second failure, or on any LLM exception when no retry remains, the Analyzer discards the LLM output and returns deterministic fallback. Invalid LLM entries are not partially accepted unless they pass all validations independently and do not depend on invalid entries.

### 6.6 Rich-to-Legacy Bridging

After rich output is finalized, legacy fields are populated from validated entries:

- `insights`: concise measured summaries of candidate deltas, avoid patterns, and high-confidence observations.
- `winning_genes`: human-readable descriptions derived from `candidate_genes` only.
- `recombination_suggestions`: human-readable descriptions derived from `recombination_hints` only.

If no rich entries exist, legacy fields may contain conservative deterministic summaries such as “No evidence-backed reusable genes identified.” They must never include source-only speculation or unvalidated LLM claims.

### 6.7 Recombination Safety

A recombination hint is valid only when:
- It names at least two parent candidates.
- Every parent candidate is known and non-regressing.
- Every `gene_map` value is one of the parent candidate hashes.
- Each mapped gene type corresponds to a validated `CandidateGene` or measured semantic delta.
- Risk flags are propagated when evidence is shape-specific, resource-sensitive, or low-confidence.

The Analyzer may rank hints internally for output ordering, with highest-confidence and strongest evidence first, but this ordering is advisory. Navigator remains responsible for deciding whether to use a hint.

### 6.8 Avoid Pattern Safety

Avoid patterns are local by default. A single regressing candidate is not enough to globally ban a technique across all future rounds. The default `scope` is `candidate_local`, and broader scopes require repeated measured evidence across candidate hashes or shapes. Navigator may include avoid patterns in LLM context, but it should not treat them as hard constraints unless the pattern is repeated and high-confidence.

### 6.9 Output Assembly

The final `CrossCandidateAnalysis` is assembled in this order:

1. Validated semantic deltas.
2. Validated candidate genes.
3. Validated recombination hints.
4. Validated avoid patterns.
5. Legacy field strings derived from the above.

The output is serialized once before return to verify JSON compatibility. Serialization failure is treated as an internal validation failure and falls back to an empty but valid analysis object.

---

## §7 Production Path Trace

This trace follows the first thing a user will try after implementation: run an optimization round that produces multiple benchmarked candidates and expect the next round to receive recombination context.

**Trigger:** Orchestrator completes candidate evaluation for a round and has at least two correctness-passing candidates whose evaluation reached benchmarking.

1. **Orchestrator filters analysis input.** It selects candidates with outcomes `IMPROVED`, `BASELINE_MATCH`, and `REGRESSION`. Compile failures, correctness failures, and infrastructure errors are excluded. Regressions remain eligible only because they have benchmark measurements that can produce negative evidence.

2. **Analyzer validates candidate/evaluation pairs.** Candidate hashes are checked, benchmark presence is confirmed, correctness-passing status is confirmed, and duplicates or invalid pairs are ignored.

3. **Deterministic extraction runs.** For each eligible candidate, Analyzer records lineage, intent, outcome, objective score, shape latencies, static resource usage, profile metrics, bottleneck tags, and source feature tags.

4. **Optional LLM synthesis runs.** If configured, Analyzer sends structured evidence and bounded snippets to the LLM. The LLM returns JSON. Analyzer parses, validates, retries once if needed, and falls back to deterministic analysis if validation fails.

5. **Rich analysis is finalized.** Non-regressing candidates can produce semantic deltas, candidate genes, and recombination hints. Regressing candidates can produce semantic deltas and avoid patterns. Every claim is linked to measured evidence.

6. **Legacy fields are bridged.** Analyzer derives `insights`, `winning_genes`, and `recombination_suggestions` from validated rich fields so old consumers continue to receive useful strings.

7. **Orchestrator persists round state.** The `CrossCandidateAnalysis` is stored in the round state and becomes `cross_analysis` for the next Navigator invocation.

8. **Navigator consumes structured hints.** On a future explore/recombination decision, Navigator prefers `recombination_hints[0].parent_candidates` and `gene_map`. It includes `avoid_patterns` in LLM context and falls back to legacy fields if rich hints are absent.

9. **Orchestrator hydrates parent sources.** After Navigator returns a recombination directive, Orchestrator resolves selected parent hashes from the incumbent or `kernels/<hash>.cu` and sets `StrategyDirective.parent_sources` for hashes it can load.

10. **Coding Agent builds recombination prompt.** Coding Agent includes actual parent source bodies from `parent_sources` and the structured `gene_map`. If a requested parent source is missing, it degrades gracefully and does not invent source code for that parent.

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | LLM speculation becomes policy fact | Accepting prose or JSON claims that do not reference measured evidence | Navigator and Coding chase imaginary “winning” traits, wasting GPU rounds | Validate every rich claim against candidate hashes and measured evidence (INV-CCA-001) |
| 2 | Regression used as positive parent | Treating all benchmarked candidates equally after allowing regressions into analysis | Coding recombines a known bad kernel and amplifies its regression | Route `REGRESSION` only to avoid patterns; validators reject regression parents (INV-CCA-003) |
| 3 | Failed candidate contaminates semantic comparison | Passing compile/correctness/error outcomes into Analyzer | A non-running or wrong kernel is analyzed as if it had performance meaning | Orchestrator filters and Analyzer defensively rejects non-benchmarked outcomes (REQ-CCA-002) |
| 4 | Legacy and rich outputs disagree | Populating legacy strings independently from structured fields | Navigator receives contradictory recombination guidance depending on which field it reads | Derive legacy fields only from validated rich entries (REQ-CCA-008) |
| 5 | Missing parent source causes fake recombination | Parent hashes are selected but source bodies are not available to Coding Agent | Prompt contains placeholders or the LLM invents a second parent | Orchestrator hydrates `parent_sources`; Coding Agent degrades gracefully if any source is missing (REQ-CCA-010) |
| 6 | Over-generalized avoid pattern blocks useful optimization | One noisy regressing candidate becomes a global ban | Navigator avoids a technique that may work on other shapes or parent kernels | Avoid patterns default to local scope and include confidence/risk flags (§6.8) |
| 7 | Analyzer duplicates Navigator policy | Analyzer ranks next directions or chooses exploit/explore | Conflicting policy owners make the control loop unpredictable | Analyzer returns evidence only; StrategyDirective remains Navigator output (INV-CCA-006) |

---

## §9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|--------------|-----------|
| SC-CCA-001: Implementation-ready enduring spec | REQ-CCA-003, REQ-CCA-004, REQ-CCA-009 | SCN-CCA-003-01, SCN-CCA-004-01, SCN-CCA-004-02 |
| SC-CCA-002: Backward-compatible structured output | REQ-CCA-001, REQ-CCA-003, REQ-CCA-008 | SCN-CCA-001-01, SCN-CCA-008-01 |
| SC-CCA-003: Regressions are negative evidence only | REQ-CCA-002, REQ-CCA-005, REQ-CCA-006 | SCN-CCA-002-02, SCN-CCA-005-01, SCN-CCA-006-01 |
| SC-CCA-004: Structured recombination reaches Navigator | REQ-CCA-007, REQ-CCA-008 | SCN-CCA-007-01, SCN-CCA-008-01 |
| SC-CCA-005: Coding Agent receives real parent sources | REQ-CCA-010 | SCN-CCA-010-01 |
| SC-CCA-006: Incumbent and failed-candidate safety | REQ-CCA-002, REQ-CCA-005, REQ-CCA-006, REQ-CCA-009 | SCN-CCA-002-02, SCN-CCA-002-03, SCN-CCA-005-01 |
| SC-CCA-007: Quality gates pass | QG-CCA-001, QG-CCA-002, QG-CCA-003 | (verified by implementation CI) |
