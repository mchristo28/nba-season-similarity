"""Distance metrics for player similarity."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class DistanceMetrics:
    """Various distance/similarity metrics for comparing players.

    Supports:
    - Euclidean distance
    - Cosine similarity
    - Custom weighted metrics
    """

    def __init__(self):
        pass

    def euclidean(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors."""
        return float(np.linalg.norm(vec1 - vec2))

    def cosine(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0, 0])

    def weighted_euclidean(
        self, vec1: np.ndarray, vec2: np.ndarray, weights: np.ndarray
    ) -> float:
        """Weighted Euclidean distance.

        Args:
            vec1, vec2: Feature vectors
            weights: Weight for each dimension
        """
        return float(np.sqrt(np.sum(weights * (vec1 - vec2) ** 2)))

    def pairwise_distances(
        self, vectors: np.ndarray, metric: str = "euclidean"
    ) -> np.ndarray:
        """Compute pairwise distances for a matrix of vectors.

        Args:
            vectors: (n_players, n_features) matrix
            metric: 'euclidean' or 'cosine'
        """
        if metric == "euclidean":
            return euclidean_distances(vectors)
        elif metric == "cosine":
            return 1 - cosine_similarity(vectors)
        else:
            raise ValueError(f"Unknown metric: {metric}")
