"""Weighted similarity matching with configurable feature groups.

Allows users to emphasize different aspects of player comparison:
- Physical attributes (height, weight)
- Scoring volume and efficiency
- Shot profile (where they shoot from)
- Playmaking
- Rebounding
- Defense
- Overall impact

Each group is standardized separately, then distances are computed
per-group and combined with configurable weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class WeightedMatcher:
    """Match players using weighted feature groups."""

    # Default feature groups
    FEATURE_GROUPS = {
        "physical": {
            "features": ["height_inches", "weight"],
            "default_weight": 0.5,  # Lower weight - physical is less about "similarity" of play
            "description": "Height and weight",
        },
        "scoring_volume": {
            "features": ["PTS", "FGA", "FG3A", "FTA", "MIN", "pts_share", "fga_share", "min_share"],
            "default_weight": 1.0,
            "description": "Points, shots, minutes",
        },
        "scoring_efficiency": {
            "features": ["ts_pct", "efg_pct", "fg_pct", "fg3_pct", "ft_pct"],
            "default_weight": 1.0,
            "description": "Shooting percentages",
        },
        "shot_profile": {
            "features": [
                "pct_fga_restricted", "pct_fga_paint", "pct_fga_midrange",
                "pct_fga_corner3", "pct_fga_above_break3",
            ],
            "default_weight": 1.0,
            "description": "Where they shoot from",
        },
        "shot_creation": {
            "features": [
                # Assisted vs unassisted
                "PCT_UAST_2PM", "PCT_UAST_3PM", "PCT_UAST_FGM",
                # Pull-up vs catch-shoot
                "PULL_UP_FGA", "PULL_UP_FG_PCT", "PULL_UP_FG3_PCT",
                "CATCH_SHOOT_FGA", "CATCH_SHOOT_FG_PCT", "CATCH_SHOOT_FG3_PCT",
            ],
            "default_weight": 1.0,
            "description": "Self-created vs assisted shots, pull-up vs catch-shoot",
        },
        "drives": {
            "features": [
                "DRIVES", "DRIVE_FGA", "DRIVE_FG_PCT",
                "DRIVE_PTS", "DRIVE_AST", "DRIVE_TOV",
            ],
            "default_weight": 1.0,
            "description": "Driving ability and efficiency",
        },
        "playmaking": {
            "features": ["AST", "TOV", "ast_share", "tov_share", "e_ast_ratio", "e_tov_pct",
                        "PASSES_MADE", "POTENTIAL_AST", "AST_PTS_CREATED"],
            "default_weight": 1.0,
            "description": "Assists, passing, and turnovers",
        },
        "rebounding": {
            "features": ["REB", "OREB", "DREB", "reb_share", "oreb_share", "dreb_share", "e_oreb_pct", "e_dreb_pct"],
            "default_weight": 1.0,
            "description": "Rebounds",
        },
        "defense": {
            "features": [
                "STL", "BLK", "stl_share", "blk_share",
                "contested_shots", "contested_shots_2pt", "contested_shots_3pt",
                "deflections", "charges_drawn",
                "def_loose_balls_recovered", "pct_box_outs_def",
            ],
            "default_weight": 1.0,
            "description": "Steals, blocks, contests, hustle",
        },
        "usage": {
            "features": ["e_usg_pct"],
            "default_weight": 1.0,
            "description": "Usage rate",
        },
        "ball_handling": {
            "features": [
                "TOUCHES", "TIME_OF_POSS", "AVG_DRIB_PER_TOUCH",
                "FRONT_CT_TOUCHES", "ELBOW_TOUCHES", "POST_TOUCHES", "PAINT_TOUCHES",
            ],
            "default_weight": 1.0,
            "description": "Ball handling, touches, and possession time",
        },
    }

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize with optional custom weights.

        Args:
            weights: Dict mapping group name to weight (0-2 typically).
                     If None, uses default weights.
        """
        self.weights = weights or {g: self.FEATURE_GROUPS[g]["default_weight"] for g in self.FEATURE_GROUPS}
        self.career_features: pd.DataFrame | None = None
        self.player_names: dict[int, str] = {}
        self.scalers: dict[str, StandardScaler] = {}

        # player_id -> {year: {group: scaled_vector}}
        self._features_by_year: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        # player_id -> {age: {group: scaled_vector}}
        self._features_by_age: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        # player_id -> set of years/ages
        self._player_years: dict[int, set[int]] = {}
        self._player_ages: dict[int, set[int]] = {}
        # player_id -> {year: PPG} and {age: PPG} for per-period filtering
        self._pts_by_year: dict[int, dict[int, float]] = {}
        self._pts_by_age: dict[int, dict[int, float]] = {}

    def fit(self, career_features: pd.DataFrame):
        """Fit the matcher with career features.

        Args:
            career_features: DataFrame with PLAYER_ID, CAREER_YEAR, AGE, and stat columns
        """
        self.career_features = career_features.copy()

        # Store player names
        self.player_names = dict(
            zip(career_features["PLAYER_ID"], career_features["PLAYER_NAME"])
        )

        # Fit a scaler for each feature group
        self.scalers = {}
        for group_name, group_info in self.FEATURE_GROUPS.items():
            feature_cols = [c for c in group_info["features"] if c in career_features.columns]
            if not feature_cols:
                continue

            scaler = StandardScaler()
            data = career_features[feature_cols].fillna(0)
            scaler.fit(data)
            self.scalers[group_name] = {
                "scaler": scaler,
                "columns": feature_cols,
            }

        # Build feature vectors for each player
        self._features_by_year = {}
        self._features_by_age = {}
        self._player_years = {}
        self._player_ages = {}
        self._pts_by_year = {}
        self._pts_by_age = {}

        for player_id in career_features["PLAYER_ID"].unique():
            player_data = career_features[
                career_features["PLAYER_ID"] == player_id
            ].sort_values("CAREER_YEAR")

            self._features_by_year[player_id] = {}
            self._features_by_age[player_id] = {}
            self._player_years[player_id] = set()
            self._player_ages[player_id] = set()
            self._pts_by_year[player_id] = {}
            self._pts_by_age[player_id] = {}

            for _, row in player_data.iterrows():
                career_year = int(row["CAREER_YEAR"])
                age = int(row["AGE"]) if pd.notna(row.get("AGE")) else None
                pts = float(row["PTS"]) if pd.notna(row.get("PTS")) else 0.0

                # Build scaled feature vector for each group
                group_vectors = {}
                for group_name, scaler_info in self.scalers.items():
                    feature_vals = np.nan_to_num(row[scaler_info["columns"]].to_numpy(dtype=float)).reshape(1, -1)
                    scaled = scaler_info["scaler"].transform(feature_vals)[0]
                    group_vectors[group_name] = scaled

                # Store by career year
                self._features_by_year[player_id][career_year] = group_vectors
                self._player_years[player_id].add(career_year)
                self._pts_by_year[player_id][career_year] = pts

                # Store by age
                if age is not None:
                    self._features_by_age[player_id][age] = group_vectors
                    self._player_ages[player_id].add(age)
                    self._pts_by_age[player_id][age] = pts

        print(f"Fitted matcher with {len(self._features_by_year)} players, {len(self.scalers)} feature groups")

    def set_weights(self, weights: dict[str, float]):
        """Update the feature group weights."""
        self.weights = weights

    def get_player_years(self, player_id: int) -> set[int]:
        """Get the set of career years for a player."""
        return self._player_years.get(player_id, set())

    def get_player_ages(self, player_id: int) -> set[int]:
        """Get the set of ages for a player."""
        return self._player_ages.get(player_id, set())

    def compute_distance(
        self,
        player_id_1: int,
        player_id_2: int,
        compare_by: str = "year",
        weights: dict[str, float] | None = None,
    ) -> tuple[float, int, dict[str, float]]:
        """Compute weighted distance between two players.

        Args:
            player_id_1: First player
            player_id_2: Second player
            compare_by: "year" or "age"
            weights: Optional custom weights (overrides self.weights)

        Returns:
            Tuple of (overall_distance, periods_compared, group_distances)
        """
        weights = weights or self.weights

        if compare_by == "age":
            features = self._features_by_age
            keys1 = self._player_ages.get(player_id_1, set())
            keys2 = self._player_ages.get(player_id_2, set())
        else:
            features = self._features_by_year
            keys1 = self._player_years.get(player_id_1, set())
            keys2 = self._player_years.get(player_id_2, set())

        if player_id_1 not in features or player_id_2 not in features:
            return float("inf"), 0, {}

        # Find overlapping periods
        common_keys = keys1 & keys2
        if not common_keys:
            return float("inf"), 0, {}

        # Compute distance for each group across all common periods
        group_distances = {g: 0.0 for g in self.scalers.keys()}
        group_counts = {g: 0 for g in self.scalers.keys()}

        for key in common_keys:
            vectors1 = features[player_id_1][key]
            vectors2 = features[player_id_2][key]

            for group_name in self.scalers.keys():
                if group_name in vectors1 and group_name in vectors2:
                    dist = np.linalg.norm(vectors1[group_name] - vectors2[group_name])
                    group_distances[group_name] += dist
                    group_counts[group_name] += 1

        # Average each group's distance
        for group_name in group_distances:
            if group_counts[group_name] > 0:
                group_distances[group_name] /= group_counts[group_name]

        # Compute weighted overall distance
        total_weight = sum(weights.get(g, 0) for g in group_distances if group_distances[g] > 0)
        if total_weight == 0:
            return float("inf"), 0, group_distances

        overall_distance = sum(
            weights.get(g, 0) * group_distances[g]
            for g in group_distances
        ) / total_weight

        return overall_distance, len(common_keys), group_distances

    def find_similar(
        self,
        player_id: int,
        n: int = 10,
        compare_by: str = "year",
        min_overlap: int = 2,
        require_full_coverage: bool = True,
        weights: dict[str, float] | None = None,
        max_ppg_diff_pct: float = 0.25,
    ) -> list[tuple[int, str, float, int, dict[str, float]]]:
        """Find similar players.

        Args:
            player_id: Query player ID
            n: Number of results
            compare_by: "year" or "age"
            min_overlap: Minimum overlapping periods
            require_full_coverage: If True, require other player to have all query player's periods
            weights: Optional custom weights
            max_ppg_diff_pct: Maximum PPG difference as percentage (0.30 = 30%)
                              Players with avg PPG differing by more than this are excluded

        Returns:
            List of (player_id, name, distance, periods_compared, group_distances)
        """
        if compare_by == "age":
            features = self._features_by_age
            query_keys = self._player_ages.get(player_id, set())
        else:
            features = self._features_by_year
            query_keys = self._player_years.get(player_id, set())

        if player_id not in features:
            raise ValueError(f"Player ID {player_id} not found")

        # Get query player's PPG by period for filtering
        if compare_by == "age":
            query_pts = self._pts_by_age.get(player_id, {})
        else:
            query_pts = self._pts_by_year.get(player_id, {})

        results = []
        for other_id in features:
            if other_id == player_id:
                continue

            # Filter by production tier - check PPG at each overlapping period
            if compare_by == "age":
                other_pts = self._pts_by_age.get(other_id, {})
                other_keys = self._player_ages.get(other_id, set())
            else:
                other_pts = self._pts_by_year.get(other_id, {})
                other_keys = self._player_years.get(other_id, set())

            # PPG filter disabled - let trajectory similarity handle it
            # common_keys checked below for coverage

            # Check coverage if required
            if require_full_coverage:
                if compare_by == "age":
                    other_keys = self._player_ages.get(other_id, set())
                else:
                    other_keys = self._player_years.get(other_id, set())

                if not query_keys.issubset(other_keys):
                    continue

            # Compute distance
            distance, periods, group_dists = self.compute_distance(
                player_id, other_id, compare_by=compare_by, weights=weights
            )

            if periods < min_overlap:
                continue

            name = self.player_names.get(other_id, "Unknown")
            results.append((other_id, name, distance, periods, group_dists))

        # Sort by distance
        results.sort(key=lambda x: x[2])
        return results[:n]

    def find_similar_season(
        self,
        player_id: int,
        season_key: int,
        n: int = 10,
        compare_by: str = "year",
        weights: dict[str, float] | None = None,
    ) -> list[tuple[int, str, int, float, dict[str, float]]]:
        """Find similar individual seasons across all players.

        This is a single-season comparison - finds the most similar
        player-seasons regardless of when they occurred.

        Args:
            player_id: Query player ID
            season_key: Career year or age to match
            n: Number of results
            compare_by: "year" or "age"
            weights: Optional custom weights

        Returns:
            List of (player_id, name, their_season_key, distance, group_distances)
        """
        weights = weights or self.weights

        if compare_by == "age":
            features = self._features_by_age
        else:
            features = self._features_by_year

        if player_id not in features or season_key not in features[player_id]:
            raise ValueError(f"Player {player_id} season {season_key} not found")

        query_vectors = features[player_id][season_key]

        results = []
        for other_id, other_seasons in features.items():
            for other_key, other_vectors in other_seasons.items():
                # Skip the exact same player-season
                if other_id == player_id and other_key == season_key:
                    continue

                # Compute distance for this single season
                group_distances = {}
                for group_name in self.scalers.keys():
                    if group_name in query_vectors and group_name in other_vectors:
                        dist = np.linalg.norm(query_vectors[group_name] - other_vectors[group_name])
                        group_distances[group_name] = dist

                # Compute weighted overall distance
                total_weight = sum(weights.get(g, 0) for g in group_distances if group_distances[g] > 0)
                if total_weight == 0:
                    continue

                overall_distance = sum(
                    weights.get(g, 0) * group_distances[g]
                    for g in group_distances
                ) / total_weight

                name = self.player_names.get(other_id, "Unknown")
                results.append((other_id, name, other_key, overall_distance, group_distances))

        # Sort by distance
        results.sort(key=lambda x: x[3])
        return results[:n]

    def get_season_info(self, player_id: int, season_key: int, compare_by: str = "year") -> dict | None:
        """Get season info for display."""
        if self.career_features is None:
            return None

        if compare_by == "age":
            key_col = "AGE"
        else:
            key_col = "CAREER_YEAR"

        row = self.career_features[
            (self.career_features["PLAYER_ID"] == player_id) &
            (self.career_features[key_col] == season_key)
        ]

        if row.empty:
            return None

        row = row.iloc[0]
        return {
            "player_name": row["PLAYER_NAME"],
            "season": row["SEASON"],
            "career_year": int(row["CAREER_YEAR"]),
            "age": int(row["AGE"]) if pd.notna(row.get("AGE")) else None,
            "pts": row["PTS"],
            "ast": row["AST"],
            "reb": row["REB"],
            "gp": int(row["GP"]),
        }

    def save(self, path: str | Path):
        """Save the fitted matcher."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "career_features": self.career_features,
                "player_names": self.player_names,
                "scalers": self.scalers,
                "weights": self.weights,
                "_features_by_year": self._features_by_year,
                "_features_by_age": self._features_by_age,
                "_player_years": self._player_years,
                "_player_ages": self._player_ages,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "WeightedMatcher":
        """Load a fitted matcher."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        matcher = cls(weights=data["weights"])
        matcher.career_features = data["career_features"]
        matcher.player_names = data["player_names"]
        matcher.scalers = data["scalers"]
        matcher._features_by_year = data["_features_by_year"]
        matcher._features_by_age = data["_features_by_age"]
        matcher._player_years = data["_player_years"]
        matcher._player_ages = data["_player_ages"]

        return matcher
