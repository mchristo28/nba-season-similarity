"""Absolute closeness on a fixed, interpretable 0–100 scale.

Profile distances are a joint weighted RMS of standardized differences.
100 is identical on measured features; 84 is a 0.5-SD difference; 50 is 1 SD;
6 is 2 SD. This is neither a percentile nor a probability of identical play.
"""

import numpy as np

MODEL_VERSION = "3.2"


def difference_band(distance: float) -> str:
    """Shared visual thresholds in model units, not statistical significance tests."""
    if not np.isfinite(distance) or distance < 0:
        return "unavailable"
    if distance < 0.5:
        return "close"
    if distance < 1.0:
        return "noticeable"
    return "large"


# Closeness bands are anchored to the reference pool: the share of all random pairs of
# rotation-player seasons that are farther apart than this match.
BANDS = (
    (0.99, "very_close", "VERY CLOSE MATCH"),
    (0.95, "close", "CLOSE MATCH"),
    (0.80, "moderate", "MODERATE MATCH"),
    (0.0, "loose", "LOOSE MATCH"),
)


def closeness_band(percentile: float) -> tuple[str, str]:
    """Return (key, label) for the fraction of random pairs this match beats."""
    for floor, key, label in BANDS:
        if percentile >= floor:
            return key, label
    return BANDS[-1][1:]


def similarity_score(distance: float) -> float:
    if np.isnan(distance) or distance < 0:
        raise ValueError("Distance must be nonnegative")
    return float(100 * np.exp2(-(float(distance) ** 2)))


def combine_distances(
    distances: dict[str, float], weights: dict[str, float], *, joint: bool = False
) -> float:
    available = [
        (d, weights.get(g, 0))
        for g, d in distances.items()
        if np.isfinite(d) and weights.get(g, 0) > 0
    ]
    total = sum(w for _, w in available)
    if not total:
        return float("inf")
    if joint:
        return float(np.sqrt(sum(d * d * w for d, w in available) / total))
    return sum(d * w for d, w in available) / total


def validate_weights(weights: dict[str, float], groups) -> dict[str, float]:
    unknown = set(weights) - set(groups)
    if unknown:
        raise ValueError(f"Unknown dimensions: {sorted(unknown)}")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if not any(v > 0 for v in weights.values()):
        raise ValueError("Enable at least one matching dimension")
    return dict(weights)
