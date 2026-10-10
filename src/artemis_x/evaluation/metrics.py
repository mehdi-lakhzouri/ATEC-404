"""Small, dependency-free detection metrics used in the reports and notebooks."""

from __future__ import annotations

import math


def precision_recall_f(tp: int, fp: int, fn: int, beta: float = 1.0) -> dict[str, float]:
    """Precision, recall and F-beta from counts (0 when undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return {"precision": p, "recall": r, f"f{beta:g}": f}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (e.g. recall within a size bin).

    Assumes independent trials; boxes in one image are correlated, so the true uncertainty is larger."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def false_positives_per_image(fp: int, images: int) -> float:
    return fp / images if images else 0.0
