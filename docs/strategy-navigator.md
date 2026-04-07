# Strategy Navigator

The Strategy Navigator is the search-space controller of the optimization loop. It decides **what to optimize next** and **how** (exploit vs. explore), combining deterministic rules for clear signals with LLM reasoning for ambiguous tradeoffs.

---

## Inputs

```
┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐
│ Experiment Registry │  │ Profile Interpreter  │  │ Cross-Candidate      │  │ Problem Spec       │
│                    │  │ Output               │  │ Analysis             │  │                    │
│ latest round       │  │ bottleneck tags      │  │ winning genes        │  │ target perf        │
│ search tree state  │  │ allowed opt          │  │ recombination        │  │ tolerance          │
│ all history        │  │ direction map        │  │ suggestions          │  │ baseline ref       │
└────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘  └─────────┬──────────┘
         │                         │                          │                        │
         └─────────────┬───────────┴──────────────┬───────────┘                        │
                       │                          │                                    │
                       ▼                          ▼                                    │
              ┌──────────────────────────────────────────────────────────────────────────┘
              │
              ▼
      ════════════════════════════════════════════════
       Phase 1 — State Ingestion
      ════════════════════════════════════════════════
```

---

## Phase 1 — State Ingestion (Deterministic)

Ingest the latest round results and compute three derived signals that drive all downstream decisions.

```
    ┌───────────────────────────────────────────────────────────┐
    │                   Ingest Latest Round                     │
    │                                                           │
    │  1. Update search tree: attach new candidates to parent   │
    │  2. Record perf delta vs parent kernel                    │
    │  3. Record perf delta vs global best                      │
    │  4. Append bottleneck tags to tag history                 │
    │  5. Update tabu list: add (code_hash, intent_tag) pairs   │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │              Compute Improvement Trend                     │
    │                                                           │
    │  delta_perf  = [d1, d2, ..., dN]   (last N rounds)       │
    │  avg_delta   = mean(delta_perf)                           │
    │  is_plateau  = avg_delta < PLATEAU_THRESHOLD              │
    │  is_regress  = avg_delta < 0                              │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │             Bottleneck Stability Check                     │
    │                                                           │
    │  last_K_tags      = bottleneck tags from last K rounds    │
    │  stable_bottleneck = len(unique(last_K_tags)) == 1        │
    │  new_bottleneck    = latest_tag not in all prior history  │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
                        [ enter Phase 2 ]
```

### Derived Signals Summary

| Signal              | Type   | Definition                                  | Used In         |
|---------------------|--------|---------------------------------------------|-----------------|
| `avg_delta`         | float  | Mean perf improvement over last N rounds    | Plateau check   |
| `is_plateau`        | bool   | `avg_delta < PLATEAU_THRESHOLD`             | Gate 4          |
| `is_regress`        | bool   | `avg_delta < 0`                             | LLM context     |
| `stable_bottleneck` | bool   | Same bottleneck tag for K consecutive rounds | Gate 7          |
| `new_bottleneck`    | bool   | Tag never seen in history                   | Gate 6          |

---

## Phase 2 — Hard Gate Checks (Deterministic)

Seven gates evaluated in priority order. **First match wins** — the remaining gates are skipped.

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Gate 1: iteration == 0 ?                                           │
    │    YES ──────────────────────────────────────► EXPLORE (de novo)    │
    │    NO ──▼                                      reason: cold start   │
    │                                                                     │
    │  Gate 2: current_best >= target ?                                   │
    │    YES ──────────────────────────────────────► signal DONE          │
    │    NO ──▼                                      to Orchestrator      │
    │                                                                     │
    │  Gate 3: remaining_budget <= 0 ?                                    │
    │    YES ──────────────────────────────────────► signal BUDGET_OUT    │
    │    NO ──▼                                      to Orchestrator      │
    │                                                                     │
    │  Gate 4: is_plateau && exploit_rounds >= N ?                        │
    │    YES ──────────────────────────────────────► EXPLORE (forced)     │
    │    NO ──▼                                      reason: plateau      │
    │                                                                     │
    │  Gate 5: current_best >= 95% of target ?                            │
    │    YES ──────────────────────────────────────► EXPLOIT (forced)     │
    │    NO ──▼                                      reason: near target  │
    │                                                                     │
    │  Gate 6: new_bottleneck class detected ?                            │
    │    YES ──────────────────────────────────────► Phase 3 (LLM)       │
    │    NO ──▼                                      needs evaluation     │
    │                                                                     │
    │  Gate 7: stable_bottleneck && direction tried >= M times ?          │
    │    YES ──────────────────────────────────────► mark EXHAUSTED       │
    │    NO ──▼                                      then EXPLORE forced  │
    │                                                                     │
    │  No gate matched                                                    │
    │    ──────────────────────────────────────────► Phase 3 (LLM)       │
    │                                                 ambiguous signal    │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

### Gate Design Rationale

| Gate | Rationale                                                                                                 |
|------|-----------------------------------------------------------------------------------------------------------|
| 1    | First iteration has no history — must do broad exploration to establish search tree root                   |
| 2-3  | Termination conditions checked early to avoid wasting compute                                             |
| 4    | Plateau after N exploit rounds means local search space is depleted — must try structural change           |
| 5    | Near target — don't risk regression with structural change, fine-tune only                                 |
| 6    | New bottleneck class (e.g., first time seeing `tensor_core_not_triggered`) needs LLM to assess relevance  |
| 7    | Same bottleneck after M attempts on same direction — that direction is exhausted, don't retry             |

---

## Phase 3 — LLM Reasoning (Ambiguous Cases Only)

Entered when no deterministic gate fires a clear decision, or when a new bottleneck class needs evaluation.

```
    ┌───────────────────────────────────────────────────────────┐
    │              Assemble LLM Context                          │
    │              (budget: <= 2K tokens)                        │
    │                                                           │
    │  Include:                                                 │
    │    - current bottleneck tags (this round)                 │
    │    - search tree summary (depth <= 3 levels)              │
    │    - top-3 candidates with perf numbers                   │
    │    - last round's delta + intent tags                     │
    │    - exhausted directions list                            │
    │    - cross-candidate diff insights (if available)         │
    │                                                           │
    │  Exclude:                                                 │
    │    - raw profiling counters                               │
    │    - full kernel source code                              │
    │    - complete search history                              │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │              LLM Structured Decision                       │
    │                                                           │
    │  Q1: exploit or explore?                                  │
    │  Q2: if multiple bottlenecks — which to prioritize? why?  │
    │  Q3: if explore — de novo or recombination?               │
    │  Q4: confidence level? (high / medium / low)              │
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
    │    [ ] chosen direction not in tabu list                  │
    │    [ ] chosen direction not in exhausted set              │
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
      - EXPLORE (de novo)       -- from Gate 1, Gate 4, Gate 7
      - EXPLOIT (forced)        -- from Gate 5
      - EXPLORE (forced)        -- from plateau or exhausted
      - LLM decision            -- from Phase 3
      - UCB1 fallback           -- from Phase 3 fallback
              │
              ▼
    ┌───────────────────────────────────────────────────────────┐
    │           Merge Mode + Direction + Reason                  │
    │                                                           │
    │  mode:      exploit | explore                             │
    │  direction: specific optimization target                  │
    │  reason:    why this choice (for decision log)            │
    └──────────────────────────┬────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────┐
    │           Apply Tabu Filter                                │
    │                                                           │
    │  Remove directions where (code_hash, intent_tag) pair     │
    │  was already tried in recent W rounds.                    │
    │                                                           │
    │  Note: tabu matches on (hash + tag) pair, not tag alone.  │
    │  Same direction on a different base kernel is allowed.     │
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
              ┌────────────┐  ┌───────┴───────┐
              │            │  │  sub-mode = ?  │
              │            │  └───┬────────┬───┘
              │            │      │        │
              ▼        de novo    │   recombination
                           │      │        │
    ┌──────────────────┐   ▼      ▼        ▼
    │ EXPLOIT          │  ┌──────────┐  ┌──────────────────┐
    │                  │  │ DE NOVO  │  │ RECOMBINATION    │
    │ base_kernel:     │  │          │  │                  │
    │   current best   │  │ start:   │  │ parents:         │
    │   hash           │  │   problem│  │   [hash_A,       │
    │                  │  │   spec   │  │    hash_B]        │
    │ mutation_type:   │  │          │  │                  │
    │   param_search   │  │ strategy │  │ gene_from_A:     │
    │   | local_rewrite│  │   hint:  │  │   <code section> │
    │   | pattern_apply│  │   algo   │  │ gene_from_B:     │
    │                  │  │   approach│  │   <code section> │
    │ target:          │  │          │  │                  │
    │   bottleneck tag │  │ avoid:   │  │ hypothesis:      │
    │                  │  │   exhaust│  │   why this combo │
    │ search_range:    │  │   -ed    │  │                  │
    │   param bounds   │  │   struct │  │ num_candidates:  │
    │                  │  │   -ures  │  │   2-3            │
    │ num_candidates:  │  │          │  │                  │
    │   3-8            │  │ num:2-3  │  │                  │
    └────────┬─────────┘  └────┬─────┘  └────────┬─────────┘
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
    │    "mode":              "exploit" | "explore",            │
    │    "direction":         "reduce_register_pressure",       │
    │    "reason":            "low_occupancy_regs 2 rounds",    │
    │    "base_kernel":       "a3f7c2..." | null,               │
    │    "mutation_type":     "local_rewrite" | null,           │
    │    "parent_candidates": ["hash_A", "hash_B"] | null,     │
    │    "gene_map":          { ... } | null,                   │
    │    "search_range":      { "launch_bounds": [128,256] },   │
    │    "num_candidates":    5,                                │
    │    "tabu":              ["increase_tile_size", ...],      │
    │    "hard_constraints":  ["smem <= 48KB", ...]            │
    │  }                                                        │
    └──────────────────────────┬────────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │    Coding Agent      │   │    Decision Log      │
    │    (generate         │   │    (append to        │
    │     candidates)      │   │     Experiment       │
    │                      │   │     Registry)        │
    └──────────────────────┘   └──────────────────────┘
```

The directive is a **structured JSON**, not natural language. The Coding Agent parses it programmatically to select its generation strategy. The same directive is logged to the Experiment Registry for audit and future plateau/tabu analysis.

---

## Full Phase Flow Summary

```
    INPUTS ──► Phase 1: State Ingestion ──► Phase 2: Hard Gates
                (deterministic)               (deterministic)
                                                    │
                                        ┌───────────┼───────────────┐
                                        │           │               │
                                  gate matched   no match      termination
                                  (clear signal)  (ambiguous)   (Gate 2/3)
                                        │           │               │
                                        │           ▼               ▼
                                        │     Phase 3: LLM      signal to
                                        │     Reasoning          Orchestrator
                                        │           │
                                        │     ┌─────┴──────┐
                                        │     │            │
                                        │  validated    fallback
                                        │     │         (UCB1)
                                        │     │            │
                                        ▼     ▼            ▼
                                   Phase 4: Direction Assembly
                                   (merge + tabu filter + detail)
                                              │
                                              ▼
                                   Phase 5: Output Directive
                                   (to Coding Agent + Decision Log)
```

---

## Configuration Parameters

| Parameter             | Default | Description                                                         |
|-----------------------|---------|---------------------------------------------------------------------|
| `PLATEAU_THRESHOLD`   | 2%      | Minimum avg improvement to not be considered plateau                |
| `PLATEAU_ROUNDS` (N)  | 3       | Consecutive exploit rounds below threshold before forcing explore   |
| `STABLE_ROUNDS` (K)   | 3       | Rounds with same bottleneck tag to trigger stability check          |
| `MAX_DIR_ATTEMPTS` (M)| 3       | Max attempts on a direction before marking exhausted                |
| `TABU_WINDOW` (W)     | 5       | Rounds to keep a (hash, tag) pair in tabu list                      |
| `TARGET_THRESHOLD`    | 95%     | Perf ratio to target that triggers exploit-only mode                |
| `LLM_CONTEXT_BUDGET`  | 2048    | Max tokens for LLM reasoning context                                |
| `LLM_CONFIDENCE_MIN`  | medium  | Minimum confidence to accept LLM decision without fallback         |
| `UCB1_C`              | 1.414   | Exploration coefficient for UCB1 fallback                           |
| `EXPLOIT_CANDIDATES`  | 3-8     | Number of candidates in exploit mode                                |
| `EXPLORE_CANDIDATES`  | 2-3     | Number of candidates in explore mode                                |
