# Strategy Navigator

The Strategy Navigator is the search-space controller of the optimization loop. It decides **what to optimize next** and **how** (exploit vs. explore), combining deterministic rules for clear signals with LLM reasoning for ambiguous tradeoffs. It is a **stateful search-policy engine** over Orchestrator-provided durable context, not the owner of persistent search state.

---

## Inputs

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│ Orchestrator Search  │  │ Profile Interpreter  │  │ Cross-Candidate      │  │ Problem Spec       │
│ Context Snapshot     │  │ Output               │  │ Analysis             │  │                    │
│ durable search       │  │ bottleneck tags      │  │ semantic deltas      │  │ target perf        │
│ memory + incumbent   │  │ evidence + allowed   │  │ reusable genes       │  │ tolerance          │
│ baseline + attempts  │  │ direction map        │  │ recombination hints  │  │ baseline ref       │
└─────────┬────────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └─────────┬──────────┘
          │                          │                          │                        │
          └─────────────┬────────────┴──────────────┬───────────┘                        │
                        │                           │                                    │
                        ▼                           ▼                                    │
               ┌───────────────────────────────────────────────────────────────────────────┘
               │
               ▼
      ═════════════════════════════════════════════════════════════════════
       Phase 1 — Search Context Interpretation / Derived Signal Computation
      ═════════════════════════════════════════════════════════════════════
```

The Navigator consumes an Orchestrator-provided durable snapshot. Persistence, resume, baseline/incumbent ownership, and stop or budget checks remain outside this module.

---

## Phase 1 — Search Context Interpretation / Derived Signal Computation (Deterministic)

Read the latest durable search snapshot and compute policy signals without mutating persistent state.

```
    ┌─────────────────────────────────────────────────────────────┐
    │            Read Orchestrator Search Context Snapshot        │
    │                                                             │
    │  1. Read baseline/incumbent and parent-kernel references    │
    │  2. Read recent attempt outcomes vs parent and incumbent    │
    │  3. Read bottleneck/evidence history from attempt records   │
    │  4. Read contextual avoid/exhausted records for this context│
    │  5. Optionally read a derived search-tree view if useful    │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                Compute Improvement Trend                    │
    │                                                             │
    │  delta_perf   = [d1, d2, ..., dN]   (recent attempts)      │
    │  avg_delta    = mean(delta_perf)                            │
    │  is_plateau   = avg_delta < PLATEAU_THRESHOLD               │
    │  is_regress   = avg_delta < 0                               │
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │      Compute Bottleneck Stability + Contextual Exhaustion   │
    │                                                             │
    │  last_K_tags          = recent bottleneck tags              │
    │  stable_bottleneck    = len(unique(last_K_tags)) == 1       │
    │  new_bottleneck       = latest_tag not in prior history     │
    │  contextual_exhausted = same context + same direction failed│
    └───────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                         [ enter Phase 2 ]
```

### Derived Signals Summary

| Signal | Type | Definition | Used In |
|--------|------|------------|---------|
| `avg_delta` | float | Mean improvement over the recent attempt window | Plateau check |
| `is_plateau` | bool | `avg_delta < PLATEAU_THRESHOLD` | Gate 2 |
| `is_regress` | bool | `avg_delta < 0` | Phase 3 context |
| `stable_bottleneck` | bool | Same bottleneck tag for K consecutive attempts in comparable context | Gate 5 |
| `new_bottleneck` | bool | Latest bottleneck class not seen in prior search memory for this run | Gate 4 |
| `contextual_exhausted` | bool | Search memory shows the same direction already exhausted for the current base/goal context | Gate 5 |

---

## Phase 2 — Hard Gate Checks (Deterministic)

Five policy gates are evaluated in priority order. **First match wins** — the remaining gates are skipped. Termination, budget, and stop-condition checks belong to the Orchestrator before the Navigator is invoked.

```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  Gate 1: baseline-seeded first policy decision ?                       │
    │    YES ──────────────────────────────────────► Phase 3 (LLM, first     │
    │    NO ──▼                                      move from measured      │
    │                                                 baseline)              │
    │                                                                        │
    │  Gate 2: is_plateau && exploit_rounds >= N ?                           │
    │    YES ──────────────────────────────────────► EXPLORE (forced)        │
    │    NO ──▼                                      reason: plateau          │
    │                                                                        │
    │  Gate 3: current_best >= 95% of target ?                               │
    │    YES ──────────────────────────────────────► EXPLOIT (forced)        │
    │    NO ──▼                                      reason: near target      │
    │                                                                        │
    │  Gate 4: new bottleneck class detected ?                               │
    │    YES ──────────────────────────────────────► Phase 3 (LLM)           │
    │    NO ──▼                                      needs evaluation         │
    │                                                                        │
    │  Gate 5: stable_bottleneck && contextual_exhausted ?                   │
    │    YES ──────────────────────────────────────► EXPLORE (forced)        │
    │    NO ──▼                                      reason: shift strategy   │
    │                                                                        │
    │  No gate matched                                                       │
    │    ──────────────────────────────────────────► Phase 3 (LLM)           │
    │                                                 ambiguous signal        │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
```

### Gate Design Rationale

| Gate | Rationale |
|------|-----------|
| 1 | Round 0 is the first policy decision **after** measured baseline bootstrap. It is often explore-biased, but not treated as a stateless cold-start invariant. |
| 2 | Plateau after N exploit rounds means the current local neighborhood is no longer yielding meaningful gains. |
| 3 | Near target, fine-grained refinement is usually safer than structural disruption. |
| 4 | A new bottleneck class (for example `tensor_core_not_triggered`) needs contextual judgment rather than a fixed rule. |
| 5 | Repeating the same direction in the same context after evidence of exhaustion should force a policy shift, not a naive retry. |

---

## Phase 3 — LLM Reasoning (Ambiguous Cases Only)

Entered when no deterministic gate fires a clear decision, when round 0 needs a baseline-seeded first move, or when a new bottleneck class needs contextual evaluation.

```
    ┌───────────────────────────────────────────────────────────┐
    │              Assemble LLM Context                          │
    │              (budget: <= 2K tokens)                        │
    │                                                           │
    │  Include:                                                 │
    │    - current bottleneck tags and evidence                 │
    │    - incumbent/baseline gap and recent perf deltas        │
    │    - recent attempt-ledger summary                        │
    │    - contextual avoid/exhausted set for this base context │
    │    - optional derived search-tree summary (if useful)     │
    │    - cross-candidate gene/recombination hints             │
    │                                                           │
    │  Exclude:                                                 │
    │    - raw profiling counters                               │
    │    - full kernel source code                              │
    │    - unbounded historical dump                            │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │              LLM Structured Decision                       │
    │                                                           │
    │  Q1: exploit or explore?                                  │
    │  Q2: which bottleneck or objective shift matters most?    │
    │  Q3: if explore — de novo or recombination?               │
    │  Q4: what intent / sub-mode best fits this context?       │
    │  Q5: confidence level? (high / medium / low)              │
    │                                                           │
    │  Output format:                                           │
    │  {                                                        │
    │    "mode": "exploit" | "explore",                         │
    │    "direction": "<optimization_target>",                  │
    │    "sub_mode": "de_novo" | "recombination" | null,        │
    │    "reasoning": "<1-2 sentence justification>",           │
    │    "confidence": "high" | "medium" | "low"                │
    │  }                                                        │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │             Validate LLM Output                            │
    │                                                           │
    │  Check all of:                                            │
    │    [ ] chosen direction not in contextual avoid set       │
    │    [ ] chosen direction not already exhausted here        │
    │    [ ] output JSON is well-formed                         │
    │    [ ] confidence above minimum threshold                 │
    └──────────────────────────┬────────────────────────────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                PASS │              FAIL │
                     │                   │
                     │         ┌─────────▼─────────┐
                     │         │   first failure?   │
                     │         └────┬──────────┬────┘
                     │              │          │
                     │         YES  │     NO   │
                     │              │  (2nd)   │
                     │              ▼          ▼
                     │     retry LLM w/   ┌──────────────────┐
                     │     narrowed       │ Deterministic    │
                     │     prompt         │ Fallback:        │
                     │         │          │ pick least-      │
                     │         │          │ explored dir     │
                     │         │          │ via UCB1 score   │
                     │         │          └────────┬─────────┘
                     │         │                   │
                     ▼         ▼                   │
              ┌──────────────────────┐             │
              │                      │◄────────────┘
              │    Phase 4           │
              │                      │
              └──────────────────────┘
```

### Why UCB1 as Deterministic Fallback?

When LLM fails validation twice, the system must not stall. UCB1 (Upper Confidence Bound) balances exploitation of known-good directions against exploration of under-visited ones:

```
UCB1(direction) = avg_perf_gain(direction) + C * sqrt(ln(total_rounds) / visits(direction))
```

This guarantees progress even without LLM reasoning — the system degrades gracefully.

---

## Phase 4 — Direction Assembly

All paths from Phase 2 (deterministic gates) and Phase 3 (LLM/fallback) converge here.

```
    Inputs from Phase 2 / Phase 3:
      - EXPLORE (forced)        -- from plateau or contextual exhaustion
      - EXPLOIT (forced)        -- from near-target policy
      - Phase 3 first-move bias -- from Gate 1 baseline-seeded decision
      - LLM decision            -- from Phase 3
      - UCB1 fallback           -- from Phase 3 fallback
              │
              ▼
    ┌───────────────────────────────────────────────────────────┐
    │      Merge Mode + Direction + Reason / Evidence Summary   │
    │                                                           │
    │  mode:      exploit | explore                             │
    │  direction: specific optimization target                  │
    │  reason:    why this choice fits measured context         │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │    Apply Contextual Avoid / Exhaustion Filter             │
    │                                                           │
    │  Derive avoid-set from prior attempts in the same         │
    │  base/parent context and remove already-exhausted moves.  │
    │                                                           │
    │  Note: avoid is contextual, not a global direction ban.   │
    │  Same direction on a different base kernel may be valid.  │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
                    ┌──────────┴──────────┐
                    │   mode = ?          │
                    └──┬──────────────┬───┘
                       │              │
              EXPLOIT  │              │  EXPLORE
                       │              │
                       ▼              ▼
              ┌────────────┐  ┌───────┴────────┐
              │            │  │  sub-mode = ?  │
              │            │  └───┬────────┬───┘
              │            │      │        │
              ▼        de novo    │   recombination
                           │      │        │
    ┌──────────────────┐   ▼      ▼        ▼
    │ EXPLOIT          │  ┌──────────┐  ┌────────────────────┐
    │                  │  │ DE NOVO  │  │ RECOMBINATION      │
    │ base_kernel_ref: │  │          │  │                    │
    │   incumbent or   │  │ start:   │  │ parent_refs:       │
    │   chosen parent  │  │   problem│  │   [hash_A, hash_B] │
    │                  │  │   spec + │  │                    │
    │ intent/sub_mode: │  │ baseline │  │ recombination_     │
    │   param_tune /   │  │          │  │ hints: reusable    │
    │   local_rewrite  │  │ strategy │  │ genes + semantic   │
    │                  │  │ constraint│  │ deltas to combine  │
    │ direction focus: │  │ avoid    │  │                    │
    │   bottleneck or  │  │ exhausted│  │ strategy_          │
    │   objective gap  │  │ families │  │ constraints: keep  │
    │                  │  │          │  │ proven wins        │
    │ search_hints:    │  │ search_  │  │                    │
    │   bounded params │  │ hints:   │  │ num_candidates:    │
    │ num_candidates:  │  │ new path │  │   2-3              │
    │   3-8            │  │ num: 2-3 │  │                    │
    └────────┬─────────┘  └────┬─────┘  └────────┬───────────┘
             │                 │                  │
             └────────┬────────┴──────────────────┘
                      │
                      ▼
               [ Phase 5 ]
```

### Why More Candidates in Exploit?

Exploit mutations are **small deltas** — cheap to generate and cheap to evaluate (most pass correctness, fast bench is quick). Explore candidates are **structural changes** — expensive to generate (LLM thinks harder) and expensive to evaluate (more likely to fail compile/correctness). Budget accordingly.

---

## Phase 5 — Output to Coding Agent

```
    ┌───────────────────────────────────────────────────────────┐
    │                  Strategy Directive                        │
    │                                                           │
    │  {                                                        │
    │    "mode":                "exploit" | "explore",        │
    │    "direction":           "reduce_register_pressure",   │
    │    "reason_summary":      "regs-bound incumbent stalled",│
    │    "evidence_summary":    { ... },                      │
    │    "intent_sub_mode":     "local_rewrite" | null,      │
    │    "base_kernel_ref":     "a3f7c2..." | null,          │
    │    "parent_kernel_refs":  ["hash_A", "hash_B"] | null,│
    │    "strategy_constraints": [ ... ],                     │
    │    "search_hints":        { ... },                      │
    │    "contextual_avoid_set": [ ... ],                     │
    │    "recombination_hints": { ... } | null,               │
    │    "num_candidates":      5                             │
    │  }                                                        │
    └──────────────────────────┬────────────────────────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 │                            │
                 ▼                            ▼
    ┌──────────────────────┐    ┌────────────────────────────┐
    │    Coding Agent      │    │       Orchestrator         │
    │    (generate         │    │ persist decision + mutate  │
    │     candidates)      │    │ durable search-memory      │
    └──────────────────────┘    └────────────────────────────┘
```

The directive is a **structured JSON**, not natural language. The Coding Agent parses it programmatically to select its generation strategy. The Orchestrator is responsible for persisting the decision and updating durable search memory; the Navigator does not append directly to global state.

---

## Full Phase Flow Summary

```
    INPUTS ──► Phase 1: Search Context Interpretation ──► Phase 2: Hard Gates
                (deterministic)                              (deterministic)
                                                                 │
                                                     ┌───────────┼───────────────┐
                                                     │           │               │
                                               gate matched   no match     first-move /
                                               (clear policy) (ambiguous)  new-context
                                                     │           │               │
                                                     │           ▼               ▼
                                                     │     Phase 3: LLM      Phase 3: LLM
                                                     │     Reasoning         (baseline-seeded)
                                                     │           │
                                                     │     ┌─────┴──────┐
                                                     │     │            │
                                                     │  validated    fallback
                                                     │     │         (UCB1)
                                                     │     │            │
                                                     ▼     ▼            ▼
                                              Phase 4: Direction Assembly
                                              (merge + contextual avoid + detail)
                                                             │
                                                             ▼
                                              Phase 5: Output Directive
                                              (to Coding Agent; Orchestrator persists)
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PLATEAU_THRESHOLD` | 2% | Minimum average improvement before local refinement is treated as stalled |
| `PLATEAU_ROUNDS` (N) | 3 | Consecutive exploit rounds below threshold before policy forces exploration |
| `STABLE_ROUNDS` (K) | 3 | Comparable-context rounds with the same bottleneck before stability is treated as real |
| `MAX_DIR_ATTEMPTS` (M) | 3 | Similar-context attempts on the same direction before treating it as exhausted |
| `TABU_WINDOW` (W) | 5 | Search-memory lookback window for deriving contextual avoid signals |
| `TARGET_THRESHOLD` | 95% | Near-target heuristic that biases policy toward exploit rather than structural change |
| `LLM_CONTEXT_BUDGET` | 2048 | Max tokens for the structured policy context sent to the LLM |
| `LLM_CONFIDENCE_MIN` | medium | Minimum confidence to accept an LLM decision without deterministic fallback |
| `UCB1_C` | 1.414 | Exploration coefficient for the deterministic fallback policy |
| `EXPLOIT_CANDIDATES` | 3-8 | Candidate count range for exploit-mode generation |
| `EXPLORE_CANDIDATES` | 2-3 | Candidate count range for explore-mode generation |
