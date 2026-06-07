"""Aggregate statistics, stdlib-only so the math is unit-testable in isolation.

The headline numbers here answer the question that makes an automated grader
usable in a real course: *does it agree with human faculty?* Quadratic-weighted
kappa and Pearson correlation against human gold scores are how you earn the
right to put an AI grade in front of a student.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def mae(a: Sequence[float], b: Sequence[float]) -> float:
    """Mean absolute error between two score series (e.g. AI vs human)."""
    if not a:
        return 0.0
    return mean([abs(x - y) for x, y in zip(a, b)])


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 if either series has no variance."""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)


def quadratic_weighted_kappa(
    rater_a: Sequence[int], rater_b: Sequence[int], min_rating: int = 0, max_rating: int = 4
) -> float:
    """Cohen's quadratic-weighted kappa — the standard agreement statistic for
    ordinal grades. 1.0 is perfect agreement, 0.0 is chance, negative is worse
    than chance. Quadratic weights penalize being off by 2 points four times as
    much as being off by 1, which is the right shape for rubric scores.
    """
    n_ratings = max_rating - min_rating + 1
    if not rater_a:
        return 0.0

    def _hist(ratings):
        h = [0] * n_ratings
        for r in ratings:
            h[int(r) - min_rating] += 1
        return h

    observed = [[0] * n_ratings for _ in range(n_ratings)]
    for a, b in zip(rater_a, rater_b):
        observed[int(a) - min_rating][int(b) - min_rating] += 1

    hist_a, hist_b = _hist(rater_a), _hist(rater_b)
    n = len(rater_a)

    num = den = 0.0
    for i in range(n_ratings):
        for j in range(n_ratings):
            w = ((i - j) ** 2) / ((n_ratings - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / n
            num += w * observed[i][j]
            den += w * expected
    if den == 0:
        return 1.0  # no expected disagreement -> perfect by convention
    return 1.0 - num / den


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate — well-behaved at small n and at
    the p=0/p=1 boundaries where the normal approximation collapses."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
