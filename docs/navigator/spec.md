# Strategy Navigator Module Specification

## §1 Overview

The Strategy Navigator is the search-space controller of the Kerlever optimization loop. It receives the accumulated optimization state and decides **what to optimize next** (direction) and **how** (exploit vs. explore). It is the only module in the system that contains optimization intelligence — all other modules execute the directives it produces.

The Navigator combines two reasoning modes:
- **Deterministic gates** for situations with clear signals (cold start, plateau, near-target, exhausted directions). These are evaluated in strict priority order; the first match wins.
- **LLM reasoning** for ambiguous situations where the signals are mixed or a new bottleneck class appears. LLM failures degrade gracefully to a UCB1 deterministic fallback.

The module operates in five sequential phases: (1) compute derived signals from the optimization state, (2) check deterministic gates, (3) invoke LLM reasoning if no gate matched, (4) assemble the full strategy directive, (5) return the directive to the Orchestrator.

**Scope:** Full implementation of all five phases. The LLM path uses `LLMClientProtocol` (extracted to a shared location). Tests use stub LLM clients. The module satisfies `StrategyNavigatorProtocol` from `kerlever.protocols`.

**Non-Goals:**
- Search tree persistence (the Orchestrator's `OptimizationState` carries round history)
- LLM prompt optimization or few-shot examples
- Multi-model LLM support
- A/B testing of parallel strategies

---

## §2 Requirements

### Functional Requirements

**REQ-NAV-001: Cold Start Detection** [traces SC-1]
On the first round of optimization (no prior history), the Navigator must produce an EXPLORE directive with DE_NOVO sub-mode. This is the only valid action when there is no search history to reason about.

**REQ-NAV-002: Plateau Detection and Forced Exploration** [traces SC-2]
When the average performance improvement over the last N exploit rounds falls below a configurable threshold, and those rounds have been consecutive exploit rounds, the Navigator must produce an EXPLORE directive. This breaks out of local optima that exploit-mode refinement cannot escape.

**REQ-NAV-003: Near-Target Exploitation** [traces SC-3]
When the current best performance reaches a configurable percentage of the target, the Navigator must produce an EXPLOIT directive. Near the target, structural changes risk regression; only fine-grained refinement is safe.

**REQ-NAV-004: New Bottleneck Escalation to LLM** [traces SC-4]
When a bottleneck tag appears that has never been seen in any prior round's history, the Navigator must defer to LLM reasoning rather than making a deterministic decision. A new bottleneck class requires contextual judgment that deterministic gates cannot provide.

**REQ-NAV-005: Exhausted Direction Detection** [traces SC-5]
When the same bottleneck persists for K consecutive rounds (stable bottleneck) and the optimization direction addressing that bottleneck has been attempted M or more times, the Navigator must mark that direction as exhausted and produce an EXPLORE directive. Repeating the same approach to the same bottleneck wastes rounds.

**REQ-NAV-006: LLM Failure Graceful Degradation** [traces SC-6]
When LLM reasoning fails (malformed JSON, invalid direction, low confidence), the Navigator must retry once with a narrowed prompt. If the second attempt also fails, the Navigator must fall back to UCB1-based deterministic direction selection. The system must never stall due to LLM unavailability.

**REQ-NAV-007: UCB1 Direction Selection** [traces SC-7]
The UCB1 fallback must select the direction that maximizes the upper confidence bound: `avg_perf_gain(direction) + C * sqrt(ln(total_rounds) / visits(direction))`. This balances exploiting known-good directions against exploring under-visited ones. Directions with zero visits must be prioritized over all others.

**REQ-NAV-008: Complete Strategy Directive Output** [traces SC-8]
Every directive returned by the Navigator must populate all fields of the extended StrategyDirective: mode, direction, reason, sub_mode, base_kernel_hash, num_candidates, tabu, and (where applicable) parent_candidates, gene_map, search_range, and hard_constraints. The Coding Agent consumes these fields programmatically.

**REQ-NAV-009: Backward-Compatible StrategyDirective Extension** [traces SC-9]
The extended StrategyDirective fields (sub_mode, parent_candidates, gene_map, search_range, hard_constraints) must all be Optional with None defaults. Existing Orchestrator code that consumes StrategyDirective must continue to function without modification.

### Quality Gates

**QG-NAV-001: Type Safety** [traces SC-10]
All source code must pass `mypy --strict` with no errors.

**QG-NAV-002: Lint** [traces SC-10]
All source code must pass `ruff check` with no errors.

---

## §3 Scenarios

### Gate Scenarios

**SCN-NAV-001-01: Cold start produces EXPLORE de_novo**
- GIVEN: the optimization state has current_round = 0
- AND: no prior round summaries exist
- WHEN: the Navigator is asked for a decision
- THEN: the directive has mode = EXPLORE
- AND: the directive has sub_mode = DE_NOVO
- AND: the directive has direction indicating a broad initial search
- AND: the directive has reason referencing cold start

**SCN-NAV-002-01: Plateau after N exploit rounds forces EXPLORE**
- GIVEN: the last 3 rounds were all EXPLOIT mode
- AND: the average performance improvement across those rounds is below the plateau threshold (< 2%)
- WHEN: the Navigator is asked for a decision
- THEN: the directive has mode = EXPLORE
- AND: the directive has reason referencing plateau detection

**SCN-NAV-002-02: No plateau when rounds include EXPLORE**
- GIVEN: the last 3 rounds include at least one EXPLORE round
- AND: the average improvement is below the threshold
- WHEN: the Navigator is asked for a decision
- THEN: the plateau gate does NOT fire (consecutive exploit count < N)
- AND: the decision proceeds to subsequent gates or LLM reasoning

**SCN-NAV-003-01: Near target forces EXPLOIT**
- GIVEN: the current global best latency is within 95% of the target (i.e., best_latency <= target_perf_us / 0.95, equivalently the performance ratio global_best / target >= 0.95 when lower is better)
- AND: the optimization is not on the first round
- AND: no higher-priority gate matches
- WHEN: the Navigator is asked for a decision
- THEN: the directive has mode = EXPLOIT
- AND: the directive has reason referencing near-target refinement

**SCN-NAV-004-01: New bottleneck triggers LLM reasoning**
- GIVEN: the latest round's profiling produces a bottleneck tag that does not appear in any prior round's bottleneck history
- AND: no higher-priority gate matches (not cold start, not plateau, not near-target)
- WHEN: the Navigator is asked for a decision
- THEN: Phase 3 LLM reasoning is invoked (not a deterministic gate decision)

**SCN-NAV-005-01: Exhausted direction forces EXPLORE**
- GIVEN: the same bottleneck tag has appeared for K = 3 consecutive rounds (stable bottleneck)
- AND: the optimization direction addressing that bottleneck has been attempted M = 3 or more times
- AND: no higher-priority gate matches
- WHEN: the Navigator is asked for a decision
- THEN: the direction is marked as exhausted
- AND: the directive has mode = EXPLORE
- AND: the directive has reason referencing exhausted direction

**SCN-NAV-005-02: Stable bottleneck with fewer attempts proceeds to LLM**
- GIVEN: the same bottleneck tag has appeared for K consecutive rounds
- AND: the direction has been attempted fewer than M times
- WHEN: the Navigator is asked for a decision
- THEN: the exhausted gate does NOT fire
- AND: the decision proceeds to LLM reasoning

### LLM Reasoning Scenarios

**SCN-NAV-006-01: LLM returns valid decision**
- GIVEN: no deterministic gate matched (ambiguous signal)
- WHEN: the LLM receives the assembled context and returns a well-formed JSON response
- AND: the chosen direction is not in the tabu list
- AND: the chosen direction is not in the exhausted set
- AND: confidence is at or above the minimum threshold
- THEN: the LLM decision is accepted and used for directive assembly

**SCN-NAV-006-02: LLM first failure triggers retry**
- GIVEN: the LLM returns malformed JSON on the first attempt
- WHEN: the Navigator detects the parse failure
- THEN: the Navigator retries with a narrowed prompt (same context but explicit instruction to return only JSON)
- AND: if the retry succeeds with a valid response, that response is used

**SCN-NAV-006-03: LLM double failure triggers UCB1 fallback**
- GIVEN: the LLM returns invalid output on both the first and second attempts
- WHEN: both validation attempts fail
- THEN: the Navigator falls back to UCB1 deterministic direction selection
- AND: the directive has mode = EXPLOIT (UCB1 defaults to exploitation of best-known direction)
- AND: the directive has reason referencing LLM failure and UCB1 fallback

**SCN-NAV-006-04: LLM suggests tabu direction triggers retry**
- GIVEN: the LLM returns a well-formed response
- AND: the chosen direction is in the tabu list
- WHEN: validation detects the tabu violation
- THEN: this is treated as a validation failure and retry/fallback logic applies

**SCN-NAV-006-05: LLM returns low confidence triggers retry**
- GIVEN: the LLM returns a well-formed response with confidence = "low"
- AND: the minimum confidence threshold is "medium"
- WHEN: validation detects the confidence is below threshold
- THEN: this is treated as a validation failure and retry/fallback logic applies

### UCB1 Scenarios

**SCN-NAV-007-01: UCB1 selects least-explored direction**
- GIVEN: direction A has been visited 5 times with average gain 3%
- AND: direction B has been visited 1 time with average gain 5%
- AND: direction C has been visited 5 times with average gain 4%
- WHEN: UCB1 scores are computed with C = 1.414 and total_rounds = 11
- THEN: direction B has the highest UCB1 score (exploration bonus dominates)
- AND: direction B is selected

**SCN-NAV-007-02: UCB1 prioritizes unvisited directions**
- GIVEN: direction A has been visited 3 times
- AND: direction B has never been visited (visits = 0)
- WHEN: UCB1 scoring runs
- THEN: direction B receives an infinite (or effectively infinite) score
- AND: direction B is selected

**SCN-NAV-007-03: UCB1 with all directions unvisited**
- GIVEN: no directions have been visited (first time reaching UCB1)
- WHEN: UCB1 scoring runs
- THEN: one direction is selected (the first in the available set, or a deterministic selection among equal-scored candidates)

### Tabu and Assembly Scenarios

**SCN-NAV-008-01: Tabu filter on same base kernel**
- GIVEN: the directive direction is "reduce_register_pressure"
- AND: the tabu list contains an entry with the same (base_kernel_hash, "reduce_register_pressure") pair from a recent round
- WHEN: the tabu filter runs in Phase 4
- THEN: the direction is flagged as tabu for that specific base kernel
- AND: either a different direction is substituted or the base kernel is changed

**SCN-NAV-008-02: Same direction on different base kernel is allowed**
- GIVEN: the tabu list contains ("hash_A", "reduce_register_pressure")
- AND: the current base kernel hash is "hash_B"
- WHEN: the directive assembles with direction "reduce_register_pressure" on hash_B
- THEN: the tabu filter does NOT block this combination
- AND: the directive proceeds with the direction

**SCN-NAV-008-03: Exploit directive candidate count**
- GIVEN: the directive mode is EXPLOIT
- WHEN: the directive is assembled
- THEN: num_candidates is set to the configured exploit_candidates value (default 5)

**SCN-NAV-008-04: Explore directive candidate count**
- GIVEN: the directive mode is EXPLORE
- WHEN: the directive is assembled
- THEN: num_candidates is set to the configured explore_candidates value (default 3)

**SCN-NAV-008-05: Complete directive fields for EXPLOIT mode**
- GIVEN: the mode is EXPLOIT and the direction targets a specific bottleneck
- WHEN: the directive is assembled
- THEN: base_kernel_hash is set to the current global best hash
- AND: sub_mode is set to one of PARAM_SEARCH, LOCAL_REWRITE, or PATTERN_APPLY
- AND: tabu contains the current tabu list entries
- AND: all required fields are populated

**SCN-NAV-008-06: Complete directive fields for EXPLORE DE_NOVO mode**
- GIVEN: the mode is EXPLORE and the sub_mode is DE_NOVO
- WHEN: the directive is assembled
- THEN: base_kernel_hash is None (de novo starts from scratch)
- AND: parent_candidates is None
- AND: gene_map is None

**SCN-NAV-008-07: Complete directive fields for EXPLORE RECOMBINATION mode**
- GIVEN: the mode is EXPLORE and the sub_mode is RECOMBINATION
- AND: cross-candidate analysis identified winning genes and recombination suggestions
- WHEN: the directive is assembled
- THEN: parent_candidates contains at least two kernel hashes
- AND: gene_map describes which code sections to take from which parent

---

## §4 Invariants

**INV-NAV-001: Gate evaluation order is fixed and first-match-wins**
The five deterministic gates must always be evaluated in the same priority order: (1) cold start, (2) plateau, (3) near target, (4) new bottleneck, (5) exhausted direction. The first gate whose condition is satisfied produces the result; subsequent gates are not evaluated. If no gate matches, the decision proceeds to LLM reasoning.
*Enforcement:* Gates are implemented as an ordered sequence of condition checks with early return on the first match. The sequence is defined once and cannot be reordered at runtime.

**INV-NAV-002: Signal computation is deterministic and side-effect-free**
Given the same OptimizationState and configuration, signal computation must always produce the same DerivedSignals output. Signal computation must not modify the OptimizationState or any other external state.
*Enforcement:* Signal computation is implemented as a pure function that reads from its inputs and returns a new value object. It takes no mutable references.

**INV-NAV-003: LLM failures never stall the system**
If the LLM is unavailable, returns unparseable output, or returns an invalid decision after two attempts, the Navigator must still produce a valid StrategyDirective via the UCB1 fallback path. No code path exists where an LLM failure propagates as an unhandled exception to the Orchestrator.
*Enforcement:* The LLM reasoning call is wrapped in exception handling. Any exception on the second attempt triggers the UCB1 fallback unconditionally. The UCB1 path is fully deterministic and cannot itself fail (it operates on in-memory direction statistics).

**INV-NAV-004: Exhausted directions are never re-selected**
Once a direction is marked as exhausted (stable bottleneck + M attempts exceeded), it must not be selected by the LLM path (validated during LLM output checking) or by UCB1 (excluded from the candidate set). Exhausted status persists for the remainder of the optimization run.
*Enforcement:* The LLM validation step checks `direction not in exhausted_set`. UCB1 scoring excludes directions in the exhausted set before computing scores. The exhausted set is computed from the optimization state and is append-only within a run.

**INV-NAV-005: Tabu filtering matches on (base_kernel_hash, intent_tag) pairs**
Tabu filtering does not block a direction globally. It blocks the specific combination of a direction applied to the same base kernel within the tabu window. The same direction applied to a different base kernel is permitted.
*Enforcement:* The tabu check compares the candidate (hash, tag) pair against entries in the tabu list. Only exact pair matches trigger tabu blocking.

**INV-NAV-006: Every StrategyDirective has a valid mode and non-empty reason**
The Navigator must never return a directive with an unset mode or an empty reason string. The reason must explain the decision path taken (gate name, LLM reasoning summary, or UCB1 fallback).
*Enforcement:* The directive assembly function validates that mode is set and reason is non-empty before returning. This is the single exit path for all directives.

---

## §5 Interfaces

### Protocol Interface (consumed by Orchestrator)

The Navigator satisfies `StrategyNavigatorProtocol`:

```
decide(
    problem_spec: ProblemSpec,
    optimization_state: OptimizationState,
    round_summary: RoundSummary | None,
    cross_analysis: CrossCandidateAnalysis | None,
) -> StrategyDirective
```

- `problem_spec`: Defines the target operation, hardware, performance baseline/target, and tolerance. The Navigator reads `target_perf_us` and `baseline_perf_us` for near-target gate computation.
- `optimization_state`: Full accumulated state — rounds history, tabu list, bottleneck history, global best, current round counter. This is the Navigator's primary input for signal computation.
- `round_summary`: Compressed summary of the most recent round. None on the first round. Used for quick access to the latest round's mode, direction, and improvement delta.
- `cross_analysis`: Output from cross-candidate analysis of the previous round. None on the first round or when analysis was skipped. Used by LLM reasoning context and for populating recombination fields.

### LLM Client Protocol (dependency)

```
LLMClientProtocol:
    complete(system_prompt: str, user_prompt: str) -> str
```

The Navigator depends on an injected LLM client. If None is provided at construction, the LLM reasoning path is skipped entirely and UCB1 is used as the primary fallback for ambiguous cases. This enables fully deterministic operation in tests.

### Extended StrategyDirective (output type)

The existing StrategyDirective is extended with five new Optional fields:

```
StrategyDirective:
    mode: Mode                                    # EXPLOIT or EXPLORE
    direction: str                                # optimization target tag
    reason: str                                   # decision rationale
    base_kernel_hash: str | None                  # kernel to mutate (None for de novo)
    num_candidates: int                           # how many candidates to generate
    tabu: list[str]                               # current tabu list for Coding Agent
    sub_mode: SubMode | None = None               # finer strategy classification
    parent_candidates: list[str] | None = None    # kernel hashes for recombination
    gene_map: dict[str, str] | None = None        # code section mapping for recombination
    search_range: dict[str, list[float]] | None = None  # parameter bounds for param search
    hard_constraints: list[str] | None = None     # constraints for Coding Agent to respect
```

All new fields default to None, preserving backward compatibility with existing Orchestrator code.

### Navigator Configuration

```
NavigatorConfig:
    plateau_threshold: float = 0.02          # minimum avg improvement to not be plateau
    plateau_rounds: int = 3                  # (N) consecutive exploit rounds for plateau
    stable_rounds: int = 3                   # (K) rounds with same bottleneck for stability
    max_direction_attempts: int = 3          # (M) attempts before marking exhausted
    tabu_window: int = 5                     # (W) rounds to keep entries in tabu
    target_threshold: float = 0.95           # perf ratio triggering exploit-only
    llm_context_budget: int = 2048           # max tokens for LLM context
    llm_confidence_min: str = "medium"       # minimum LLM confidence to accept
    ucb1_c: float = 1.414                    # UCB1 exploration coefficient
    exploit_candidates: int = 5              # candidates in exploit mode
    explore_candidates: int = 3              # candidates in explore mode
```

### Internal Types (Navigator-private)

**DerivedSignals** — output of Phase 1 signal computation:
- `avg_delta: float` — mean performance improvement over last N rounds (0.0 when no history)
- `is_plateau: bool` — avg_delta < plateau_threshold
- `is_regress: bool` — avg_delta < 0
- `stable_bottleneck: str | None` — the bottleneck tag if it has been the same for K consecutive rounds, else None
- `new_bottleneck: str | None` — the latest bottleneck tag if it was never seen before, else None
- `consecutive_exploit_rounds: int` — count of most recent consecutive rounds with mode = EXPLOIT
- `direction_attempt_counts: dict[str, int]` — mapping from direction tag to number of times it has been attempted
- `exhausted_directions: set[str]` — directions that meet the exhaustion criteria

**GateResult** — output of a matched gate in Phase 2:
- `mode: Mode` — EXPLOIT or EXPLORE
- `direction: str` — optimization target
- `reason: str` — explanation (e.g., "cold start", "plateau after 3 exploit rounds")
- `sub_mode: SubMode | None` — optional sub-mode (e.g., DE_NOVO for cold start)

**LLMDecision** — parsed output of LLM reasoning in Phase 3:
- `mode: Mode` — EXPLOIT or EXPLORE
- `direction: str` — optimization target
- `sub_mode: SubMode | None` — DE_NOVO, RECOMBINATION, or None
- `reasoning: str` — 1-2 sentence justification
- `confidence: str` — "high", "medium", or "low"

**DirectionStats** — per-direction performance tracking for UCB1:
- `direction: str` — direction tag
- `visits: int` — number of rounds that used this direction
- `total_perf_gain: float` — sum of improvement_over_prev_best for rounds using this direction
- `avg_perf_gain: float` — total_perf_gain / visits (0.0 when visits = 0)

---

## §6 Behavioral Specification

### 6.1 Phase 1: Signal Computation

Phase 1 is a deterministic, side-effect-free computation that derives decision signals from the accumulated optimization state. It runs on every invocation of `decide()`.

**avg_delta (improvement trend):**
- Collect the `improvement_over_prev_best` values from the most recent `plateau_rounds` (N) rounds in the rounds list.
- If a round has `improvement_over_prev_best = None` (no improvement that round), treat it as 0.0 for the purpose of the average.
- `avg_delta = sum(deltas) / len(deltas)`.
- If the rounds list is empty (round 0), avg_delta = 0.0.

**is_plateau:**
- `is_plateau = avg_delta < plateau_threshold` AND `consecutive_exploit_rounds >= plateau_rounds`.
- Both conditions are required: a low average alone does not trigger plateau if there have been EXPLORE rounds interspersed (which reset the consecutive exploit count).

**is_regress:**
- `is_regress = avg_delta < 0`.
- This signal is informational — it is included in the LLM context to help the LLM reason about whether to change direction. It does not directly trigger any gate.

**stable_bottleneck:**
- Examine the last `stable_rounds` (K) entries in `bottleneck_history`.
- Each entry is a list of bottleneck tags for that round.
- If all K entries share at least one common bottleneck tag (intersection of the K lists is non-empty), that tag is the stable bottleneck.
- If multiple tags are common, take the one that appears most frequently across all K lists.
- If fewer than K rounds of history exist, stable_bottleneck = None.
- If any of the K lists is empty (no profiling data that round), stable_bottleneck = None.

**new_bottleneck:**
- Take the bottleneck tags from the most recent round's history.
- For each tag, check whether it appears in any prior round's bottleneck history.
- If a tag is found that has never appeared before, new_bottleneck = that tag.
- If multiple tags are new, take the first one encountered.
- If the latest round has no bottleneck tags, or all tags have been seen before, new_bottleneck = None.
- On round 0 (no history at all), new_bottleneck = None.

**consecutive_exploit_rounds:**
- Count backwards from the most recent round in the rounds list.
- Increment the count for each consecutive round with mode = EXPLOIT.
- Stop counting at the first EXPLORE round or when the beginning of history is reached.
- On round 0, the count is 0.

**direction_attempt_counts:**
- Build a mapping from direction string to the number of rounds that used that direction.
- Source: iterate over all round summaries in the optimization state, counting occurrences of each `direction` value.

**exhausted_directions:**
- For each direction where `direction_attempt_counts[direction] >= max_direction_attempts`, check whether the direction corresponds to a stable bottleneck (i.e., the direction was targeting a bottleneck that has been stable for K rounds).
- A direction is exhausted if both conditions are met: the bottleneck it targets has been stable AND the direction has been attempted M or more times.
- In practice: if `stable_bottleneck` is not None and `direction_attempt_counts[stable_bottleneck] >= max_direction_attempts`, then `stable_bottleneck` is in the exhausted set.
- Additionally, any direction previously marked exhausted remains exhausted (the set is cumulative within the run).

### 6.2 Phase 2: Gate Logic

Phase 2 evaluates five deterministic gates in strict priority order. The first gate whose condition is true produces a GateResult and exits Phase 2. If no gate matches, Phase 2 returns None and the decision proceeds to Phase 3.

**Gate 1: Cold Start**
- Condition: `optimization_state.current_round == 0`
- Output: GateResult(mode=EXPLORE, direction="initial_exploration", reason="Cold start: no optimization history, performing broad initial search", sub_mode=DE_NOVO)
- Rationale: With no history, there is nothing to exploit. Broad exploration establishes the search tree root.

**Gate 2: Plateau**
- Condition: `signals.is_plateau == True` (which implies avg_delta < threshold AND consecutive_exploit_rounds >= N)
- Output: GateResult(mode=EXPLORE, direction=<bottleneck tag from latest round or generic "structural_change">, reason="Plateau detected: avg improvement {avg_delta:.1%} over {N} consecutive exploit rounds below {threshold:.1%} threshold", sub_mode=None)
- Rationale: Local search space is depleted. The system must try a structural change to find a new basin.

**Gate 3: Near Target**
- Condition: `optimization_state.global_best_latency_us is not None` AND `optimization_state.global_best_latency_us <= problem_spec.target_perf_us / target_threshold`
- Explanation: This checks whether the best latency is within the "near target" zone. Since lower latency is better, and `target_threshold` = 0.95, the condition `best_latency <= target / 0.95` means the best is within 95% of the target performance. For example, if target = 1.0us and threshold = 0.95, then any best latency <= 1.053us triggers this gate.
- Output: GateResult(mode=EXPLOIT, direction=<current bottleneck tag or "fine_tune">, reason="Near target: current best is within {pct:.1%} of target, fine-tuning only", sub_mode=PARAM_SEARCH)
- Rationale: Structural changes risk regression when the kernel is nearly optimal. Only fine-grained parameter tuning is safe.

**Gate 4: New Bottleneck**
- Condition: `signals.new_bottleneck is not None`
- Output: None (does not produce a GateResult). Instead, this gate causes Phase 2 to return None with an annotation that Phase 3 should include the new bottleneck context. The LLM needs to assess whether the new bottleneck is relevant and how to address it.
- Rationale: A bottleneck tag never seen before requires contextual judgment. The deterministic gates cannot reason about novel bottleneck classes.

**Gate 5: Exhausted Direction**
- Condition: `signals.stable_bottleneck is not None` AND `signals.stable_bottleneck in signals.exhausted_directions`
- Output: GateResult(mode=EXPLORE, direction="structural_change", reason="Direction exhausted: {direction} attempted {count} times against stable bottleneck {tag}, trying new approach", sub_mode=None)
- Rationale: Repeating the same direction against the same stable bottleneck is wasteful. The system must try a fundamentally different approach.

**No Match:**
- If all five gates are checked without producing a result, Phase 2 returns None and the decision proceeds to Phase 3 LLM reasoning.

### 6.3 Phase 3: LLM Reasoning

Phase 3 is invoked only when no deterministic gate produced a decision. It uses LLM reasoning to handle ambiguous signals. If no LLM client was injected, this phase is skipped entirely and proceeds to UCB1 fallback.

**System Prompt Content:**
The system prompt instructs the LLM to act as an optimization strategy advisor for CUDA kernel optimization. It specifies:
- The LLM must decide between exploit (refine current approach) and explore (structural change)
- If explore: choose between de_novo (fresh start) and recombination (combine winning traits from multiple candidates)
- The LLM must provide a confidence level for its decision
- The output must be a single JSON object with exactly five fields

The system prompt is static (not regenerated per call).

**User Prompt Construction:**
The user prompt is assembled from the optimization state, capped at `llm_context_budget` tokens. It includes, in priority order (if budget permits):
1. Current bottleneck tags from the most recent round
2. Performance trend: avg_delta, is_plateau, is_regress
3. Top-3 candidates by latency with their performance numbers
4. Last round's mode, direction, and improvement delta
5. List of exhausted directions
6. Cross-candidate analysis insights and recombination suggestions (if available)
7. Search tree summary (direction history with visit counts)

Items are included in the above priority order. If the budget is exceeded, lower-priority items are truncated or omitted.

**Expected LLM Response Format:**
```json
{
    "mode": "exploit" | "explore",
    "direction": "<optimization_target_tag>",
    "sub_mode": "de_novo" | "recombination" | null,
    "reasoning": "<1-2 sentence justification>",
    "confidence": "high" | "medium" | "low"
}
```

**Validation Rules (all must pass):**
1. The response is valid JSON containing exactly the five expected fields.
2. `mode` is one of "exploit" or "explore" (case-insensitive).
3. `direction` is a non-empty string.
4. `direction` is not in the tabu list for the current base kernel.
5. `direction` is not in the exhausted directions set.
6. `sub_mode`, if non-null, is one of "de_novo" or "recombination" (for explore mode) or "param_search", "local_rewrite", "pattern_apply" (for exploit mode).
7. `confidence` is one of "high", "medium", "low".
8. The confidence level meets or exceeds the configured minimum. Ordering: high > medium > low.

**Retry Logic:**
- On first validation failure: retry with a narrowed prompt. The narrowed prompt appends the specific validation error (e.g., "Your previous response had direction 'X' which is in the tabu list. Choose a different direction.") and re-emphasizes the JSON format requirement.
- On second validation failure: abandon LLM path entirely. Proceed to UCB1 fallback.

**UCB1 Fallback Trigger:**
If the LLM path fails twice (parse error, validation error, or any exception), the Navigator falls back to UCB1 direction selection. The fallback produces an EXPLOIT mode directive with the direction selected by UCB1. The reason field explicitly notes the LLM failure and fallback.

### 6.4 UCB1 Fallback

UCB1 (Upper Confidence Bound 1) selects a direction by balancing exploitation of historically good directions against exploration of under-tried ones.

**Formula:**
```
UCB1(direction) = avg_perf_gain(direction) + C * sqrt(ln(total_rounds) / visits(direction))
```

Where:
- `avg_perf_gain(direction)` = mean of `improvement_over_prev_best` across all rounds that used this direction (treating None as 0.0)
- `C` = `ucb1_c` configuration parameter (default 1.414 = sqrt(2))
- `total_rounds` = number of completed rounds (optimization_state.current_round)
- `visits(direction)` = number of rounds that used this direction

**Direction Statistics:**
Direction statistics are computed from the optimization state's rounds list. For each unique direction in the round summaries, compute:
- `visits`: count of rounds with that direction
- `total_perf_gain`: sum of `improvement_over_prev_best` (None treated as 0.0)
- `avg_perf_gain`: total_perf_gain / visits

**Edge Cases:**

*visits = 0 for a direction:* If a direction has never been visited, its UCB1 score is positive infinity. This guarantees that unvisited directions are always selected before visited ones. Among multiple unvisited directions, select the first one encountered (deterministic ordering).

*total_rounds = 0:* This should not occur because UCB1 is only invoked from Phase 3, which requires at least one prior round of history (round 0 is handled by Gate 1). If it does occur defensively, treat `ln(0)` as 0 and select based on avg_perf_gain only.

*All directions exhausted:* If the exhausted set covers all known directions, UCB1 must still produce an answer. In this case, select the direction with the highest avg_perf_gain from the exhausted set (the "least bad" option) and flag the situation in the reason.

**Available Directions:**
The set of candidate directions for UCB1 is derived from two sources:
1. All directions that have appeared in the round history
2. The current bottleneck tags (which may suggest new directions)

Exhausted directions are excluded from the candidate set (unless all are exhausted, per the edge case above).

**Output:**
UCB1 returns a direction string. The caller wraps it into an LLMDecision with mode=EXPLOIT, the selected direction, sub_mode=None, reasoning describing the UCB1 selection, and confidence="medium".

### 6.5 Phase 4: Direction Assembly

Phase 4 takes the decision from Phase 2 (gate) or Phase 3 (LLM/UCB1) and assembles a complete StrategyDirective.

**Input:** Either a GateResult (from Phase 2) or an LLMDecision (from Phase 3), plus the full optimization state and configuration.

**Step 1: Extract core fields**
- `mode`: from the gate result or LLM decision
- `direction`: from the gate result or LLM decision
- `reason`: from the gate result or LLM decision reasoning
- `sub_mode`: from the gate result or LLM decision; may be None

**Step 2: Apply tabu filter**
- Check whether the (base_kernel_hash, direction) pair exists in the tabu list within the tabu window (last W rounds).
- The tabu list in OptimizationState stores intent_tags from recent rounds. The filter checks whether the proposed direction, when combined with the current base kernel hash, was recently tried.
- If the direction is tabu for the current base kernel: attempt to substitute a different direction from the available set (using UCB1 scoring among non-tabu alternatives). If no alternative is available, proceed with the tabu direction and note it in the reason.

**Step 3: Determine candidate count**
- If mode is EXPLOIT: num_candidates = `exploit_candidates` (default 5)
- If mode is EXPLORE: num_candidates = `explore_candidates` (default 3)

**Step 4: Set base kernel hash**
- If mode is EXPLOIT: base_kernel_hash = optimization_state.global_best_hash
- If mode is EXPLORE and sub_mode is DE_NOVO: base_kernel_hash = None
- If mode is EXPLORE and sub_mode is RECOMBINATION: base_kernel_hash = None (parents are specified separately)

**Step 5: Populate mode-specific fields**

For EXPLOIT mode:
- `sub_mode`: If not already set, infer from direction. Directions targeting parameter tuning (e.g., launch bounds, tile size) use PARAM_SEARCH. Directions targeting code structure use LOCAL_REWRITE. Directions applying known patterns use PATTERN_APPLY. Default to LOCAL_REWRITE if ambiguous.
- `search_range`: If sub_mode is PARAM_SEARCH, populate with parameter bounds relevant to the direction (e.g., {"launch_bounds": [128, 256], "tile_size": [16, 32, 64]}). Derived from the problem spec and hardware constraints.
- `hard_constraints`: Populate from hardware limits (e.g., "smem <= 48KB" for devices with 48KB shared memory per SM, "registers <= 255").

For EXPLORE DE_NOVO mode:
- `parent_candidates`: None
- `gene_map`: None
- `search_range`: None
- `hard_constraints`: Same as EXPLOIT (hardware limits still apply)

For EXPLORE RECOMBINATION mode:
- `parent_candidates`: List of kernel hashes to recombine. Sourced from cross-candidate analysis's winning_genes or from the top-performing candidates in recent history.
- `gene_map`: Maps code section names to parent hashes, indicating which sections to take from which parent. Sourced from cross-candidate analysis's recombination_suggestions.
- `search_range`: None
- `hard_constraints`: Same as EXPLOIT

**Step 6: Assemble tabu list for output**
- The `tabu` field on the output StrategyDirective contains the intent_tags from the optimization state's tabu list, windowed to the last `tabu_window` rounds.

**Step 7: Construct and return StrategyDirective**
All fields are populated. The directive is returned to the caller (the main `decide()` method), which returns it to the Orchestrator.

### 6.6 StrategyNavigator Class

**Construction:**
The StrategyNavigator is constructed with:
- `llm_client: LLMClientProtocol | None` — the LLM client for Phase 3 reasoning. If None, Phase 3 skips LLM and goes directly to UCB1.
- `config: NavigatorConfig | None` — configuration parameters. If None, defaults are used.

The constructor stores these and initializes an empty exhausted-directions set.

**decide() Flow:**
```
1. Phase 1: Compute derived signals from optimization_state
2. Phase 2: Check gates (signals, state, config)
   - If a gate matched → skip to step 4 with GateResult
3. Phase 3: LLM reasoning (if llm_client is not None)
   - Try LLM → validate → on success, go to step 4 with LLMDecision
   - On failure, retry once → on success, go to step 4
   - On second failure, UCB1 fallback → go to step 4
   - If llm_client is None, UCB1 directly → go to step 4
4. Phase 4: Assemble directive from gate/LLM/UCB1 result
5. Phase 5: Return StrategyDirective
```

**Error Handling:**
- Signal computation errors (e.g., unexpected data shapes): Log warning, use safe defaults (avg_delta=0.0, all booleans False, counts empty). This should not happen with well-formed OptimizationState but defensive coding prevents crashes.
- Gate evaluation errors: Should not occur (pure conditionals on computed signals). If they do, log and proceed to Phase 3 as if no gate matched.
- LLM errors: Caught by the retry/fallback mechanism (REQ-NAV-006).
- Assembly errors: Should not occur. If they do, return a minimal safe directive: EXPLORE mode, DE_NOVO, generic direction, with the error noted in reason.

---

## §7 Production Path Trace

This traces a multi-round optimization sequence to show how the Navigator behaves across different situations.

**Round 0 — Cold Start**

Trigger: The Orchestrator calls `decide()` with an empty optimization state (current_round=0, no rounds, no bottleneck history).

1. Phase 1 computes signals: avg_delta=0.0, is_plateau=False, is_regress=False, stable_bottleneck=None, new_bottleneck=None, consecutive_exploit_rounds=0, direction_attempt_counts={}, exhausted_directions={}.
2. Phase 2 checks gates: Gate 1 fires (current_round==0). Returns EXPLORE / DE_NOVO / "cold start".
3. Phase 3 is skipped (gate matched).
4. Phase 4 assembles directive: mode=EXPLORE, sub_mode=DE_NOVO, direction="initial_exploration", base_kernel_hash=None, num_candidates=3, tabu=[].
5. The directive is returned to the Orchestrator. The Coding Agent generates 3 structurally diverse kernel candidates from scratch.

**Rounds 1-3 — Exploit Improving**

Trigger: Round 1 begins. The Orchestrator passes state with one round of history, global_best set, bottleneck_history populated.

1. Phase 1: avg_delta is positive (improvement found in round 0), is_plateau=False, consecutive_exploit_rounds=0 (round 0 was EXPLORE).
2. Phase 2: Gate 1 skipped (round > 0). Gate 2 skipped (not plateau, consecutive exploit is 0). Gate 3: check near-target — best latency is far from target, so skipped. Gate 4: check new bottleneck — bottleneck tags are new (first round had some), but since round 0 was EXPLORE with de novo, the history context varies. If a new bottleneck is detected, proceed to Phase 3. Otherwise, no gate matches → Phase 3.
3. Phase 3: LLM receives context about the single round of history, bottleneck tags, and performance numbers. LLM returns {mode: "exploit", direction: "reduce_memory_bandwidth", confidence: "high"}. Validation passes.
4. Phase 4 assembles directive: mode=EXPLOIT, direction="reduce_memory_bandwidth", base_kernel_hash=<global_best_hash>, num_candidates=5, sub_mode=LOCAL_REWRITE.
5. Rounds 2 and 3 follow similarly, with the LLM or gates guiding exploitation of the current best kernel.

**Round 4 — Plateau Detected**

Trigger: Rounds 1-3 were all EXPLOIT mode with small improvements (avg_delta < 2%).

1. Phase 1: avg_delta=0.015 (1.5%, below 2% threshold), consecutive_exploit_rounds=3 (rounds 1-3 were exploit). is_plateau=True.
2. Phase 2: Gate 1 skipped (round > 0). Gate 2 fires: is_plateau AND consecutive_exploit_rounds >= 3. Returns EXPLORE / "plateau detected".
3. Phase 3 skipped.
4. Phase 4: mode=EXPLORE, direction="structural_change", num_candidates=3.
5. The system breaks out of the local optimum with a structural change.

**Round 7 — Exhausted Direction**

Trigger: Rounds 5-7 all had the same stable bottleneck "low_occupancy_regs" and used the direction "reduce_register_pressure" 3 times.

1. Phase 1: stable_bottleneck="low_occupancy_regs" (same tag for K=3 rounds), direction_attempt_counts["reduce_register_pressure"]=3, exhausted_directions={"reduce_register_pressure"}.
2. Phase 2: Gate 1 skipped. Gate 2 skipped (not plateau — some rounds were EXPLORE). Gate 3 skipped (not near target). Gate 4 skipped (bottleneck is stable, not new). Gate 5 fires: stable_bottleneck is not None AND it is in exhausted_directions. Returns EXPLORE / "direction exhausted".
3. Phase 3 skipped.
4. Phase 4: mode=EXPLORE, direction="structural_change", num_candidates=3.

**Round 10 — LLM Failure with UCB1 Fallback**

Trigger: No deterministic gate matches. LLM is called but returns malformed JSON twice.

1. Phase 1 computes signals normally.
2. Phase 2: no gate matches.
3. Phase 3: First LLM call returns invalid JSON → retry with narrowed prompt → second call also returns invalid JSON → UCB1 fallback. UCB1 computes scores for all non-exhausted directions. Direction "optimize_shared_memory" has the highest UCB1 score (low visit count, moderate gain). Selected.
4. Phase 4: mode=EXPLOIT, direction="optimize_shared_memory", reason="LLM failed after 2 attempts; UCB1 fallback selected least-explored direction", num_candidates=5.

**Round 15 — Near Target**

Trigger: Global best latency has reached within 95% of target.

1. Phase 1 computes signals normally.
2. Phase 2: Gate 1 skipped. Gate 2 skipped (no plateau). Gate 3 fires: global_best_latency <= target_perf_us / 0.95. Returns EXPLOIT / "near target".
3. Phase 3 skipped.
4. Phase 4: mode=EXPLOIT, sub_mode=PARAM_SEARCH, direction="fine_tune", base_kernel_hash=<global_best>, num_candidates=5.
5. The system performs only safe parameter tuning to close the remaining gap.

---

## §8 Shortcut Risks

| # | Risk | Shortcut that causes it | Consequence | Mitigation |
|---|------|------------------------|-------------|------------|
| 1 | Division by zero in signal computation | Computing avg_delta without checking for empty rounds list | Crash on round 0 or when all improvements are None | Return 0.0 for avg_delta when rounds list is empty or all deltas are None. Guard every division with a length check. |
| 2 | Gate priority inversion | Evaluating Gate 5 (exhausted) before Gate 3 (near target) | A near-target kernel is forced into EXPLORE and regresses instead of fine-tuning to completion | Gates are evaluated in strict priority order via an ordered sequence with early return. The order is tested explicitly in unit tests. |
| 3 | LLM exception propagation | Not wrapping LLM call in try/except | An LLM timeout or network error crashes the Navigator and the entire optimization loop | All LLM calls are wrapped in exception handling. Any exception increments the failure counter and triggers retry or UCB1 fallback (INV-NAV-003). |
| 4 | Tabu filter too strict | Checking only the direction tag without the base kernel hash | A direction that was tabu on one kernel is blocked even when applied to a different, potentially better base kernel | Tabu matches on (base_kernel_hash, intent_tag) pairs, not on the tag alone (INV-NAV-005). |
| 5 | UCB1 starvation of known-good directions | Setting UCB1 C coefficient too high | The exploration bonus overwhelms exploitation, causing the system to keep trying bad directions instead of refining good ones | C defaults to sqrt(2) per standard UCB1 theory. The value is configurable for tuning. Direction stats include avg_perf_gain to maintain exploitation pressure. |
| 6 | Exhausted direction silently re-selected by LLM | Not validating the LLM's chosen direction against the exhausted set | The system wastes rounds re-trying a direction that has already proven ineffective | LLM validation checks direction against the exhausted set (validation rule 5 in §6.3). Exhausted directions are also excluded from UCB1 candidates. |
| 7 | StrategyDirective extension breaks Orchestrator | Adding required (non-Optional) fields to StrategyDirective | Existing Orchestrator code and stubs that construct StrategyDirective without the new fields fail at runtime | All new fields are Optional with None defaults (REQ-NAV-009). Existing code continues to work unchanged. |
| 8 | Stale bottleneck history producing false stable_bottleneck | Not handling rounds with empty bottleneck tags (all candidates failed) | A round with no profiling data is treated as continuing the previous bottleneck streak | Rounds with empty bottleneck tag lists break the stability streak. stable_bottleneck requires K consecutive non-empty rounds with the same tag. |

---

## §9 Traceability Matrix

| Success Criteria | Requirements | Scenarios |
|-----------------|-------------|-----------|
| SC-1: Cold start returns EXPLORE de_novo | REQ-NAV-001 | SCN-NAV-001-01 |
| SC-2: Plateau detection forces EXPLORE | REQ-NAV-002 | SCN-NAV-002-01, SCN-NAV-002-02 |
| SC-3: Near target forces EXPLOIT | REQ-NAV-003 | SCN-NAV-003-01 |
| SC-4: New bottleneck triggers LLM reasoning | REQ-NAV-004 | SCN-NAV-004-01 |
| SC-5: Exhausted direction forces EXPLORE | REQ-NAV-005 | SCN-NAV-005-01, SCN-NAV-005-02 |
| SC-6: LLM failure degrades to retry then UCB1 | REQ-NAV-006 | SCN-NAV-006-01, SCN-NAV-006-02, SCN-NAV-006-03, SCN-NAV-006-04, SCN-NAV-006-05 |
| SC-7: UCB1 selects least-explored direction | REQ-NAV-007 | SCN-NAV-007-01, SCN-NAV-007-02, SCN-NAV-007-03 |
| SC-8: StrategyDirective has complete fields | REQ-NAV-008 | SCN-NAV-008-03, SCN-NAV-008-04, SCN-NAV-008-05, SCN-NAV-008-06, SCN-NAV-008-07 |
| SC-9: Orchestrator tests pass after extension | REQ-NAV-009 | (verified by running existing test suite after StrategyDirective extension) |
| SC-10: mypy --strict and ruff check pass | QG-NAV-001, QG-NAV-002 | (verified by CI tooling) |
