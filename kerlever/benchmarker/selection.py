"""Benchmarker — Phase 6 pre-work: profile-target selection.

Implements top-K ∪ top-M with dedup, limited to candidates that passed
the incumbent-comparison gate (spec §6.6). Shift-potential is scored from
pre-profile signals only (INV-BENCH-007).

Spec: docs/benchmarker/spec.md §6.6
Design: docs/benchmarker/design.md §2.1 selection.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kerlever.benchmarker.types import (
    IncumbentComparison,
    ObjectiveScore,
    StaticAnalysis,
)


@dataclass(frozen=True)
class ShiftPotentialHints:
    """Pre-profile signals used by shift-potential scoring (spec §6.6)."""

    intent_direction: str | None = None
    intent_sub_mode: str | None = None
    static_analysis: StaticAnalysis | None = None
    incumbent_static_analysis: StaticAnalysis | None = None
    effective_bandwidth_gbps: float | None = None
    incumbent_effective_bandwidth_gbps: float | None = None
    achieved_flops: float | None = None
    incumbent_achieved_flops: float | None = None
    arithmetic_intensity: float | None = None
    incumbent_arithmetic_intensity: float | None = None
    novelty_score: float = 0.0


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with the fields needed for profile selection."""

    candidate_hash: str
    incumbent_comparison: IncumbentComparison
    objective_score: ObjectiveScore
    candidate_cv_pct: float | None = None
    shift_hints: ShiftPotentialHints = field(default_factory=ShiftPotentialHints)


_PROFILE_ELIGIBLE: frozenset[IncumbentComparison] = frozenset(
    {IncumbentComparison.IMPROVED, IncumbentComparison.STATISTICAL_TIE}
)


def top_k_by_score(candidates: list[ScoredCandidate], k: int) -> list[ScoredCandidate]:
    """Return the ``k`` lowest-scoring candidates (lower-is-better).

    Ties are broken by (CV ascending, candidate_hash ascending) so the result
    is deterministic across runs.

    Implements: REQ-BENCH-016
    """
    if k <= 0:
        return []
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.objective_score.value,
            c.candidate_cv_pct if c.candidate_cv_pct is not None else float("inf"),
            c.candidate_hash,
        ),
    )
    return ranked[:k]


def _abs_delta_int(a: int | None, b: int | None) -> int:
    """Absolute integer delta; 0 when either input is missing."""
    if a is None or b is None:
        return 0
    return abs(a - b)


def _abs_delta_float(a: float | None, b: float | None) -> float:
    """Absolute float delta; 0.0 when either input is missing."""
    if a is None or b is None:
        return 0.0
    return abs(a - b)


def shift_potential_score(
    candidate: ScoredCandidate,
    incumbent: ScoredCandidate,
    hints: ShiftPotentialHints,
) -> float:
    """Score a candidate's pre-profile bottleneck-shift potential.

    Combines four pre-profile signals per spec §6.6:

    * intent direction / sub-mode novelty — flat bonus when present;
    * static-analysis delta (registers, smem, spills, occupancy);
    * fast-benchmark throughput shape (bandwidth, arithmetic intensity);
    * novelty vs. incumbent.

    NCU counters from this batch are never consulted — they don't exist at
    selection time (INV-BENCH-007).

    Implements: REQ-BENCH-016
    Invariant: INV-BENCH-007
    """
    score = 0.0
    if hints.intent_direction:
        score += 0.5
    if hints.intent_sub_mode:
        score += 0.25
    if hints.static_analysis and hints.incumbent_static_analysis:
        sa = hints.static_analysis
        inc_sa = hints.incumbent_static_analysis
        score += min(
            4.0,
            _abs_delta_int(sa.registers_per_thread, inc_sa.registers_per_thread) / 8.0,
        )
        score += min(
            4.0,
            _abs_delta_int(sa.smem_bytes_per_block, inc_sa.smem_bytes_per_block)
            / 4096.0,
        )
        score += min(
            2.0,
            _abs_delta_int(sa.spill_stores, inc_sa.spill_stores) / 4.0
            + _abs_delta_int(sa.spill_loads, inc_sa.spill_loads) / 4.0,
        )
        score += min(
            2.0,
            _abs_delta_float(sa.occupancy_estimate_pct, inc_sa.occupancy_estimate_pct)
            / 10.0,
        )
    score += min(
        2.0,
        _abs_delta_float(
            hints.effective_bandwidth_gbps,
            hints.incumbent_effective_bandwidth_gbps,
        )
        / 50.0,
    )
    score += min(
        2.0,
        _abs_delta_float(hints.achieved_flops, hints.incumbent_achieved_flops)
        / 1.0e11,
    )
    score += min(
        2.0,
        _abs_delta_float(
            hints.arithmetic_intensity, hints.incumbent_arithmetic_intensity
        )
        / 4.0,
    )
    score += max(0.0, hints.novelty_score)
    # incumbent reference is unused in V1 but kept for future novelty paths.
    _ = incumbent, candidate
    return score


def top_m_by_shift_potential(
    candidates: list[ScoredCandidate],
    incumbent: ScoredCandidate,
    m: int,
    hints_per_candidate: dict[str, ShiftPotentialHints],
) -> list[ScoredCandidate]:
    """Return the ``m`` candidates with the largest shift-potential.

    Candidates without hints receive a default neutral score (0.0); ties
    break by ``objective_score.value`` ascending, then ``candidate_hash``.

    Implements: REQ-BENCH-016
    Invariant: INV-BENCH-007
    """
    if m <= 0:
        return []
    scored: list[tuple[float, ScoredCandidate]] = []
    for cand in candidates:
        hints = hints_per_candidate.get(
            cand.candidate_hash, ShiftPotentialHints()
        )
        sp = shift_potential_score(cand, incumbent, hints)
        scored.append((sp, cand))
    scored.sort(
        key=lambda kv: (
            -kv[0],
            kv[1].objective_score.value,
            kv[1].candidate_hash,
        ),
    )
    return [c for _sp, c in scored[:m]]


def build_profile_set(
    scoreable: list[ScoredCandidate],
    k: int,
    m: int,
    incumbent: ScoredCandidate,
    include_incumbent: bool,
    hints_per_candidate: dict[str, ShiftPotentialHints],
) -> list[ScoredCandidate]:
    """Assemble the Phase 6 profile set: top-K ∪ top-M with dedup.

    Only ``IMPROVED`` and ``STATISTICAL_TIE`` candidates are eligible
    (spec §6.6 and §8 Shortcut Risk #11). Regressed and unstable candidates
    are excluded from ``top_k_profiled``.

    Dedup preserves first-insertion order (top-K before top-M) so two runs
    with the same inputs produce the same profile order.

    Args:
        scoreable: Candidates considered for profiling.
        k: ``top_k_profile`` from the request.
        m: ``top_m_profile_shift_potential`` from the request.
        incumbent: Incumbent anchor candidate for shift-potential delta.
        include_incumbent: Whether to append the incumbent for side-by-side.
        hints_per_candidate: Pre-profile signals per candidate.

    Returns:
        Deduplicated ordered list of candidates selected for profiling.

    Implements: REQ-BENCH-016
    """
    eligible = [c for c in scoreable if c.incumbent_comparison in _PROFILE_ELIGIBLE]
    top_k = top_k_by_score(eligible, k)
    seen: set[str] = {c.candidate_hash for c in top_k}
    top_m_pool = [c for c in eligible if c.candidate_hash not in seen]
    top_m = top_m_by_shift_potential(top_m_pool, incumbent, m, hints_per_candidate)
    ordered: list[ScoredCandidate] = list(top_k)
    for c in top_m:
        if c.candidate_hash in seen:
            continue
        ordered.append(c)
        seen.add(c.candidate_hash)
    if include_incumbent and incumbent.candidate_hash not in seen:
        ordered.append(incumbent)
    return ordered


__all__ = [
    "ScoredCandidate",
    "ShiftPotentialHints",
    "build_profile_set",
    "shift_potential_score",
    "top_k_by_score",
    "top_m_by_shift_potential",
]
