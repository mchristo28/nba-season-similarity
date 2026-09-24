"""Absolute closeness on a fixed, interpretable 0–100 scale.

Distances are weighted means of group RMS differences in standard deviations.
100 is identical on measured features; 84 is a 0.5-SD difference; 50 is 1 SD;
6 is 2 SD. This is neither a percentile nor a probability of identical play.
"""

import numpy as np


def similarity_score(distance: float) -> float:
    if np.isnan(distance) or distance < 0:
        raise ValueError("Distance must be nonnegative")
    return float(100 * np.exp2(-(float(distance) ** 2)))


def combine_distances(distances: dict[str, float], weights: dict[str, float]) -> float:
    available = [
        (d, weights.get(g, 0))
        for g, d in distances.items()
        if np.isfinite(d) and weights.get(g, 0) > 0
    ]
    total = sum(w for _, w in available)
    return sum(d * w for d, w in available) / total if total else float("inf")


def validate_weights(weights: dict[str, float], groups) -> dict[str, float]:
    unknown = set(weights) - set(groups)
    if unknown:
        raise ValueError(f"Unknown dimensions: {sorted(unknown)}")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if not any(v > 0 for v in weights.values()):
        raise ValueError("Enable at least one matching dimension")
    return dict(weights)
