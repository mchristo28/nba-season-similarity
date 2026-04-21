"""KNN and similarity matching for finding similar players."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class NeighborEngine:
    """Find nearest neighbors (most similar players).

    Uses sklearn NearestNeighbors for efficient similarity search.
    Supports configurable feature weights for different similarity dimensions.
    """

    # Default feature columns for similarity matching
    DEFAULT_FEATURES = [
        # Volume stats (per season averages)
        "PTS", "AST", "REB", "STL", "BLK",
        # Efficiency
        "fg_pct", "fg3_pct", "ft_pct", "ts_pct",
        # Composition (role on team)
        "pts_share", "ast_share", "reb_share", "min_share",
    ]

    # Feature groups for toggleable dimensions
    FEATURE_GROUPS = {
        "scoring": ["PTS", "pts_share", "ts_pct", "fg_pct"],
        "playmaking": ["AST", "ast_share"],
        "rebounding": ["REB", "reb_share"],
        "defense": ["STL", "BLK", "stl_share", "blk_share"],
        "shooting": ["fg_pct", "fg3_pct", "ft_pct", "fg3a_share"],
        "usage": ["pts_share", "ast_share", "reb_share", "min_share"],
    }

    def __init__(self, metric: str = "euclidean", n_neighbors: int = 10):
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.nn_model: NearestNeighbors | None = None
        self.player_ids: list[int] = []
        self.player_names: dict[int, str] = {}
        self.feature_matrix: np.ndarray | None = None
        self.feature_columns: list[str] = []
        self.scaler: StandardScaler | None = None

    def fit(self, feature_matrix: np.ndarray, player_ids: list[int]):
        """Build the nearest neighbors index.

        Args:
            feature_matrix: (n_players, n_features) array
            player_ids: List of player IDs corresponding to rows
        """
        self.feature_matrix = feature_matrix
        self.player_ids = list(player_ids)
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(player_ids)),
            metric=self.metric,
        )
        self.nn_model.fit(feature_matrix)

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        player_id_col: str = "PLAYER_ID",
        player_name_col: str = "PLAYER_NAME",
        scale: bool = True,
    ):
        """Fit from a DataFrame with player features.

        Args:
            df: DataFrame with player features
            feature_columns: Columns to use as features (default: DEFAULT_FEATURES)
            player_id_col: Column containing player IDs
            player_name_col: Column containing player names
            scale: Whether to standardize features (recommended for mixed scales)
        """
        if feature_columns is None:
            feature_columns = [c for c in self.DEFAULT_FEATURES if c in df.columns]

        self.feature_columns = feature_columns

        # Extract features
        features = df[feature_columns].copy()

        # Handle missing values
        features = features.fillna(0)

        # Scale features
        if scale:
            self.scaler = StandardScaler()
            feature_matrix = self.scaler.fit_transform(features)
        else:
            feature_matrix = features.values

        # Store player info
        player_ids = df[player_id_col].tolist()
        if player_name_col in df.columns:
            self.player_names = dict(zip(df[player_id_col], df[player_name_col]))

        self.fit(feature_matrix, player_ids)

    def find_similar(
        self, player_id: int, n: int | None = None
    ) -> list[tuple[int, str, float]]:
        """Find N most similar players to a given player.

        Args:
            player_id: ID of query player
            n: Number of neighbors (default: self.n_neighbors)

        Returns:
            List of (player_id, player_name, distance) tuples
        """
        if self.nn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if player_id not in self.player_ids:
            raise ValueError(f"Player ID {player_id} not in index")

        n = n or self.n_neighbors
        idx = self.player_ids.index(player_id)
        query_vector = self.feature_matrix[idx].reshape(1, -1)

        # Get n+1 neighbors (includes self)
        distances, indices = self.nn_model.kneighbors(
            query_vector, n_neighbors=min(n + 1, len(self.player_ids))
        )

        results = []
        for dist, i in zip(distances[0], indices[0]):
            pid = self.player_ids[i]
            if pid != player_id:  # Exclude self
                name = self.player_names.get(pid, "Unknown")
                results.append((pid, name, float(dist)))

        return results[:n]

    def find_similar_by_name(
        self, player_name: str, n: int | None = None
    ) -> list[tuple[int, str, float]]:
        """Find similar players by name (partial match).

        Args:
            player_name: Player name to search for
            n: Number of neighbors

        Returns:
            List of (player_id, player_name, distance) tuples
        """
        name_lower = player_name.lower()
        for pid, name in self.player_names.items():
            if name_lower in name.lower():
                return self.find_similar(pid, n)
        raise ValueError(f"Player '{player_name}' not found")

    def find_similar_to_vector(
        self, feature_vector: np.ndarray, n: int | None = None
    ) -> list[tuple[int, str, float]]:
        """Find players similar to an arbitrary feature vector.

        Useful for "what if" queries or composite player searches.

        Args:
            feature_vector: Feature vector (same dimensions as training data)
            n: Number of neighbors

        Returns:
            List of (player_id, player_name, distance) tuples
        """
        if self.nn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n = n or self.n_neighbors

        # Scale if we used scaling during fit
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        else:
            feature_vector = feature_vector.reshape(1, -1)

        distances, indices = self.nn_model.kneighbors(
            feature_vector, n_neighbors=min(n, len(self.player_ids))
        )

        results = []
        for dist, i in zip(distances[0], indices[0]):
            pid = self.player_ids[i]
            name = self.player_names.get(pid, "Unknown")
            results.append((pid, name, float(dist)))

        return results

    def get_player_vector(self, player_id: int) -> np.ndarray:
        """Get the feature vector for a player."""
        if player_id not in self.player_ids:
            raise ValueError(f"Player ID {player_id} not in index")
        idx = self.player_ids.index(player_id)
        return self.feature_matrix[idx]

    def compare_players(self, player_id_1: int, player_id_2: int) -> dict:
        """Compare two players and return similarity details.

        Args:
            player_id_1: First player ID
            player_id_2: Second player ID

        Returns:
            Dict with distance and feature-by-feature comparison
        """
        vec1 = self.get_player_vector(player_id_1)
        vec2 = self.get_player_vector(player_id_2)

        if self.metric == "cosine":
            # Cosine distance
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            distance = 1 - (dot / (norm1 * norm2)) if norm1 * norm2 > 0 else 1.0
        else:
            distance = np.linalg.norm(vec1 - vec2)

        # Feature-by-feature differences
        feature_diffs = {}
        for i, col in enumerate(self.feature_columns):
            feature_diffs[col] = {
                "player_1": float(vec1[i]),
                "player_2": float(vec2[i]),
                "diff": float(vec1[i] - vec2[i]),
            }

        return {
            "player_1": {
                "id": player_id_1,
                "name": self.player_names.get(player_id_1, "Unknown"),
            },
            "player_2": {
                "id": player_id_2,
                "name": self.player_names.get(player_id_2, "Unknown"),
            },
            "distance": float(distance),
            "similarity": float(1 - distance) if self.metric == "cosine" else None,
            "features": feature_diffs,
        }

    def save(self, path: str | Path):
        """Save the fitted model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "feature_matrix": self.feature_matrix,
            "player_ids": self.player_ids,
            "player_names": self.player_names,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "metric": self.metric,
            "n_neighbors": self.n_neighbors,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "NeighborEngine":
        """Load a fitted model from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        engine = cls(metric=data["metric"], n_neighbors=data["n_neighbors"])
        engine.feature_matrix = data["feature_matrix"]
        engine.player_ids = data["player_ids"]
        engine.player_names = data["player_names"]
        engine.feature_columns = data["feature_columns"]
        engine.scaler = data["scaler"]

        # Rebuild the NN model
        engine.nn_model = NearestNeighbors(
            n_neighbors=min(engine.n_neighbors + 1, len(engine.player_ids)),
            metric=engine.metric,
        )
        engine.nn_model.fit(engine.feature_matrix)

        return engine


def build_engine_from_features(
    features_path: str = "data/features/player_features.parquet",
    feature_columns: list[str] | None = None,
) -> NeighborEngine:
    """Convenience function to build engine from saved features.

    Args:
        features_path: Path to player features parquet file
        feature_columns: Optional list of feature columns to use

    Returns:
        Fitted NeighborEngine
    """
    df = pd.read_parquet(features_path)
    engine = NeighborEngine()
    engine.fit_from_dataframe(df, feature_columns=feature_columns)
    return engine
