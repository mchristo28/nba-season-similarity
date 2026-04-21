"""Career trajectory comparison - year 1 vs year 1, year 2 vs year 2, etc.

Compares players by their career development patterns rather than
aggregate stats. Useful for "who developed like Player X?" queries.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrajectoryMatcher:
    """Match players by career trajectory patterns.

    Compares year-by-year development:
    - Year 1 vs Year 1 comparison
    - Year 2 vs Year 2 comparison
    - etc.
    - Aggregates into single similarity score
    """

    # Features to use for similarity (per-game stats + composition)
    FEATURE_COLS = [
        "PTS", "AST", "REB", "STL", "BLK", "MIN",
        "fg_pct", "fg3_pct", "ft_pct", "ts_pct",
        "pts_share", "ast_share", "reb_share", "min_share",
    ]

    def __init__(self):
        self.career_features: pd.DataFrame | None = None
        self.player_names: dict[int, str] = {}
        self.scaler = StandardScaler()
        # player_id -> {career_year: scaled_vector}
        self._features_by_year: dict[int, dict[int, np.ndarray]] = {}
        # player_id -> {age: scaled_vector}
        self._features_by_age: dict[int, dict[int, np.ndarray]] = {}
        # player_id -> set of career years they have
        self._player_years: dict[int, set[int]] = {}
        # player_id -> set of ages they have
        self._player_ages: dict[int, set[int]] = {}

    def fit(self, career_features: pd.DataFrame):
        """Fit the matcher with career year features.

        Args:
            career_features: DataFrame with PLAYER_ID, CAREER_YEAR, AGE, and stat columns
        """
        self.career_features = career_features.copy()

        # Store player names
        self.player_names = dict(
            zip(career_features["PLAYER_ID"], career_features["PLAYER_NAME"])
        )

        # Get feature columns that exist
        feature_cols = [c for c in self.FEATURE_COLS if c in career_features.columns]

        # Fit scaler on all data
        all_features = career_features[feature_cols].fillna(0)
        self.scaler.fit(all_features)

        # Build scaled feature vectors for each player, indexed by both career year and age
        self._features_by_year = {}
        self._features_by_age = {}
        self._player_years = {}
        self._player_ages = {}

        for player_id in career_features["PLAYER_ID"].unique():
            player_data = career_features[
                career_features["PLAYER_ID"] == player_id
            ].sort_values("CAREER_YEAR")

            self._features_by_year[player_id] = {}
            self._features_by_age[player_id] = {}
            self._player_years[player_id] = set()
            self._player_ages[player_id] = set()

            for _, row in player_data.iterrows():
                career_year = int(row["CAREER_YEAR"])
                age = int(row["AGE"]) if pd.notna(row.get("AGE")) else None

                features = row[feature_cols].fillna(0).values.reshape(1, -1)
                scaled = self.scaler.transform(features)[0]

                # Index by career year
                self._features_by_year[player_id][career_year] = scaled
                self._player_years[player_id].add(career_year)

                # Index by age (if available)
                if age is not None:
                    self._features_by_age[player_id][age] = scaled
                    self._player_ages[player_id].add(age)

    def get_player_career_length(self, player_id: int) -> int:
        """Get number of seasons for a player in our data."""
        if player_id not in self._features_by_year:
            return 0
        return len(self._features_by_year[player_id])

    def get_player_years(self, player_id: int) -> set[int]:
        """Get the set of career years we have for a player."""
        return self._player_years.get(player_id, set())

    def get_player_ages(self, player_id: int) -> set[int]:
        """Get the set of ages we have for a player."""
        return self._player_ages.get(player_id, set())

    def compute_trajectory_distance(
        self,
        player_id_1: int,
        player_id_2: int,
        compare_by: str = "year",
        max_value: int | None = None,
    ) -> tuple[float, int]:
        """Compute trajectory distance between two players.

        Compares year-by-year (or age-by-age) for overlapping values and averages the distances.

        Args:
            player_id_1: First player
            player_id_2: Second player
            compare_by: "year" for career year comparison, "age" for age comparison
            max_value: Maximum value to compare (e.g., 5 = only compare up to Year 5 or Age 25)

        Returns:
            Tuple of (average Euclidean distance, number of periods compared)
        """
        if compare_by == "age":
            features = self._features_by_age
            keys1 = self._player_ages.get(player_id_1, set())
            keys2 = self._player_ages.get(player_id_2, set())
        else:
            features = self._features_by_year
            keys1 = self._player_years.get(player_id_1, set())
            keys2 = self._player_years.get(player_id_2, set())

        if player_id_1 not in features or player_id_2 not in features:
            return float("inf"), 0

        # Find overlapping keys (career years or ages)
        common_keys = keys1 & keys2

        if max_value:
            common_keys = {k for k in common_keys if k <= max_value}

        if not common_keys:
            return float("inf"), 0

        # Compute distance for each common key and average
        total_distance = 0.0
        for key in common_keys:
            vec1 = features[player_id_1][key]
            vec2 = features[player_id_2][key]
            dist = np.linalg.norm(vec1 - vec2)
            total_distance += dist

        return total_distance / len(common_keys), len(common_keys)

    def find_similar_trajectories(
        self,
        player_id: int,
        n: int = 10,
        min_overlap: int = 2,
        compare_by: str = "year",
        require_full_coverage: bool = True,
    ) -> list[tuple[int, str, float, int]]:
        """Find players with similar career trajectories.

        Args:
            player_id: Query player ID
            n: Number of similar players to return
            min_overlap: Minimum overlapping periods (years or ages) required for comparison
            compare_by: "year" for career year comparison, "age" for age comparison
            require_full_coverage: If True, only compare with players who have ALL
                                   the same periods as the query player

        Returns:
            List of (player_id, name, distance, periods_compared) tuples
        """
        if compare_by == "age":
            features = self._features_by_age
            query_keys = self._player_ages.get(player_id, set())
        else:
            features = self._features_by_year
            query_keys = self._player_years.get(player_id, set())

        if player_id not in features:
            raise ValueError(f"Player ID {player_id} not found")

        results = []
        for other_id in features:
            if other_id == player_id:
                continue

            # Get the other player's keys
            if compare_by == "age":
                other_keys = self._player_ages.get(other_id, set())
            else:
                other_keys = self._player_years.get(other_id, set())

            # If require_full_coverage, the other player must have ALL query player's periods
            if require_full_coverage:
                if not query_keys.issubset(other_keys):
                    continue

            # Compute distance - returns (distance, periods_compared)
            distance, periods_compared = self.compute_trajectory_distance(
                player_id, other_id, compare_by=compare_by
            )

            # Skip if not enough overlapping periods
            if periods_compared < min_overlap:
                continue

            name = self.player_names.get(other_id, "Unknown")
            results.append((other_id, name, distance, periods_compared))

        # Sort by distance
        results.sort(key=lambda x: x[2])

        return results[:n]

    def find_similar_by_name(
        self,
        player_name: str,
        n: int = 10,
    ) -> list[tuple[int, str, float, int]]:
        """Find similar trajectories by player name."""
        name_lower = player_name.lower()
        for pid, name in self.player_names.items():
            if name_lower in name.lower():
                return self.find_similar_trajectories(pid, n)
        raise ValueError(f"Player '{player_name}' not found")

    def get_year_by_year_comparison(
        self,
        player_id_1: int,
        player_id_2: int,
    ) -> pd.DataFrame:
        """Get detailed year-by-year comparison between two players.

        Returns:
            DataFrame with stats for each career year side-by-side
        """
        if self.career_features is None:
            raise ValueError("Matcher not fitted")

        p1_data = self.career_features[
            self.career_features["PLAYER_ID"] == player_id_1
        ].sort_values("CAREER_YEAR")

        p2_data = self.career_features[
            self.career_features["PLAYER_ID"] == player_id_2
        ].sort_values("CAREER_YEAR")

        # Get player names
        name1 = self.player_names.get(player_id_1, "Player 1")
        name2 = self.player_names.get(player_id_2, "Player 2")

        # Build comparison for overlapping years
        max_years = min(len(p1_data), len(p2_data))

        comparison_data = []
        for year in range(1, max_years + 1):
            p1_year = p1_data[p1_data["CAREER_YEAR"] == year].iloc[0]
            p2_year = p2_data[p2_data["CAREER_YEAR"] == year].iloc[0]

            comparison_data.append({
                "Career Year": year,
                f"{name1} PTS": f"{p1_year['PTS']:.1f}",
                f"{name2} PTS": f"{p2_year['PTS']:.1f}",
                f"{name1} AST": f"{p1_year['AST']:.1f}",
                f"{name2} AST": f"{p2_year['AST']:.1f}",
                f"{name1} REB": f"{p1_year['REB']:.1f}",
                f"{name2} REB": f"{p2_year['REB']:.1f}",
            })

        return pd.DataFrame(comparison_data)

    def get_player_trajectory_df(self, player_id: int) -> pd.DataFrame:
        """Get a player's trajectory as a formatted DataFrame."""
        if self.career_features is None:
            raise ValueError("Matcher not fitted")

        data = self.career_features[
            self.career_features["PLAYER_ID"] == player_id
        ].sort_values("CAREER_YEAR")

        return data

    def save(self, path: str | Path):
        """Save the fitted matcher."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "career_features": self.career_features,
                "player_names": self.player_names,
                "scaler": self.scaler,
                "_features_by_year": self._features_by_year,
                "_features_by_age": self._features_by_age,
                "_player_years": self._player_years,
                "_player_ages": self._player_ages,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryMatcher":
        """Load a fitted matcher."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        matcher = cls()
        matcher.career_features = data["career_features"]
        matcher.player_names = data["player_names"]
        matcher.scaler = data["scaler"]
        matcher._features_by_year = data.get("_features_by_year", {})
        matcher._features_by_age = data.get("_features_by_age", {})
        matcher._player_years = data.get("_player_years", {})
        matcher._player_ages = data.get("_player_ages", {})

        return matcher


def build_trajectory_matcher(
    career_features_path: str = "data/features/career_year_features.parquet",
) -> TrajectoryMatcher:
    """Build a trajectory matcher from saved career features."""
    df = pd.read_parquet(career_features_path)
    matcher = TrajectoryMatcher()
    matcher.fit(df)
    return matcher
