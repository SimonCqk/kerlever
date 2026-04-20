"""Benchmarker — pure numeric statistics.

No GPU, no I/O, no asyncio. Every function in this module is a deterministic
function of its inputs. The noise-margin formula is spec §6.5 pseudocode
translated to Python.

Spec: docs/benchmarker/spec.md §6.4, §6.5
Design: docs/benchmarker/design.md §2.1 stats.py
"""

from __future__ import annotations

import math


def p50(samples: list[float]) -> float:
    """Return the median (50th percentile) of ``samples``.

    Linear interpolation between the two central values for even sample counts.

    Args:
        samples: Non-empty list of latency samples (microseconds).

    Returns:
        The median latency in microseconds.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("p50 requires at least one sample")
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    mid = n // 2
    if n % 2 == 1:
        return sorted_samples[mid]
    return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0


def p95(samples: list[float], min_required: int) -> float | None:
    """Return the 95th percentile when ``len(samples) >= min_required``.

    Gate per REQ-BENCH-011: when the sample count is below ``min_required``,
    returns ``None`` rather than fabricating a value from too-few samples.

    Args:
        samples: Latency samples (microseconds).
        min_required: Minimum sample count; below this the result is ``None``.

    Returns:
        The p95 latency in microseconds, or ``None`` when the gate fails.

    Implements: REQ-BENCH-011
    """
    if len(samples) < min_required or not samples:
        return None
    sorted_samples = sorted(samples)
    rank = 0.95 * (len(sorted_samples) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_samples[int(rank)]
    weight = rank - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight


def mean(samples: list[float]) -> float:
    """Return the arithmetic mean of ``samples``.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("mean requires at least one sample")
    return sum(samples) / len(samples)


def stdev(samples: list[float]) -> float:
    """Return the sample standard deviation (Bessel-corrected).

    Returns ``0.0`` when there are fewer than two samples.
    """
    n = len(samples)
    if n < 2:
        return 0.0
    mu = mean(samples)
    variance = sum((x - mu) ** 2 for x in samples) / (n - 1)
    return math.sqrt(variance)


def cv_pct(mean_value: float, stdev_value: float) -> float | None:
    """Coefficient of variation, expressed as a percentage.

    Returns ``None`` when ``mean_value`` is not strictly positive. Caller
    decides how to fold ``None`` into the noise margin (spec §6.5).
    """
    if mean_value <= 0.0:
        return None
    return (stdev_value / mean_value) * 100.0


def anchor_drift_pct(pre: float, post: float) -> float:
    """Per-shape anchor drift fraction (absolute).

    ``anchor_drift_pct = |post - pre| / pre`` (spec §6.5). Returns ``0.0`` when
    ``pre`` is not positive — the caller's noise-margin formula will fall back
    to the noise floor.

    Args:
        pre: Pre-block anchor objective value.
        post: Post-block anchor objective value.

    Implements: REQ-BENCH-013
    """
    if pre <= 0.0:
        return 0.0
    return abs(post - pre) / pre


def aggregate_noise_margin(
    candidate_cv_pct: float | None,
    anchor_cv_pct: float | None,
    anchor_drift_fraction: float,
    floor: float,
) -> float:
    """Compute ``noise_margin_pct`` per spec §6.5.

    ``noise_margin_pct = max(floor, candidate_cv/100, anchor_cv/100, drift)``.

    CV inputs are in **percent** (e.g., 1.2 for 1.2%); the formula divides
    them by 100 to put them on the same fractional scale as the drift and
    the floor. A ``None`` CV (too few samples to compute) contributes 0.0.

    Example test table (traces spec §6.5):

    * floor=0.01, cand_cv=1.2, anch_cv=1.0, drift=0.04 →
      max(0.01, 0.012, 0.010, 0.04) = 0.04
    * floor=0.01, cand_cv=None, anch_cv=None, drift=0.0 → 0.01 (floor)
    * floor=0.01, cand_cv=10.0, anch_cv=1.0, drift=0.02 →
      0.10 (candidate CV dominates)

    Args:
        candidate_cv_pct: Candidate CV (percent). May be ``None``.
        anchor_cv_pct: Anchor CV (percent). May be ``None``.
        anchor_drift_fraction: Anchor drift as a fraction (e.g., 0.04 = 4%).
        floor: Noise floor fraction (e.g., 0.01).

    Returns:
        The effective noise margin fraction.

    Implements: REQ-BENCH-015
    Invariant: INV-BENCH-006 (sole decision-path input for improvement/regression)
    """
    candidate_frac = (candidate_cv_pct or 0.0) / 100.0
    anchor_frac = (anchor_cv_pct or 0.0) / 100.0
    return max(floor, candidate_frac, anchor_frac, anchor_drift_fraction)


def ratio(value: float, reference: float) -> float | None:
    """Safe ratio helper; returns ``None`` when ``reference`` is non-positive."""
    if reference <= 0.0:
        return None
    return value / reference


__all__ = [
    "aggregate_noise_margin",
    "anchor_drift_pct",
    "cv_pct",
    "mean",
    "p50",
    "p95",
    "ratio",
    "stdev",
]
