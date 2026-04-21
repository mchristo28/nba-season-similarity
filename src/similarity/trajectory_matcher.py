"""Career trajectory matching using hybrid year-for-year + DTW approach.

This module compares players using multiple sophisticated techniques:
1. Year-for-year matching: Compares career stages directly (Year 1 vs Year 1)
2. DTW shape matching: Finds similar curve shapes even with different timing
3. Production-weighted: Factors in absolute production level
4. Multi-stat comparison: PTS, AST, REB, efficiency, usage, minutes
"""

import numpy as np
import pandas as pd
from dtw import dtw
from sklearn.preprocessing import MinMaxScaler


class TrajectoryMatcher:
    """Match players using hybrid year-for-year + DTW trajectory comparison."""

    # Stats organized by category (mirrors WeightedMatcher)
    STAT_GROUPS = {
        "physical": {
            "stats": ["height_inches", "weight"],
            "weight": 1.5,  # Important for matching similar player types
        },
        "scoring_volume": {
            "stats": ["PTS", "FGA", "FG3A", "FTA", "MIN"],
            "weight": 1.5,  # Scoring volume important
        },
        "scoring_efficiency": {
            "stats": ["ts_pct", "efg_pct", "fg_pct", "fg3_pct", "ft_pct"],
            "weight": 1.0,
        },
        "shot_profile": {
            "stats": ["pct_fga_restricted", "pct_fga_paint", "pct_fga_midrange",
                      "pct_fga_corner3", "pct_fga_above_break3"],
            "weight": 1.0,
        },
        "playmaking": {
            "stats": ["AST", "TOV"],
            "weight": 1.2,
        },
        "rebounding": {
            "stats": ["REB", "OREB", "DREB"],
            "weight": 1.0,
        },
        "defense": {
            "stats": ["STL", "BLK", "deflections"],
            "weight": 0.8,  # Often has missing data
        },
        "overall_impact": {
            "stats": ["e_off_rating", "e_def_rating", "e_usg_pct"],
            "weight": 1.0,
        },
    }

    # Flatten for easy access
    TRAJECTORY_STATS = []
    STAT_WEIGHTS = {}
    for group_name, group_info in STAT_GROUPS.items():
        for stat in group_info["stats"]:
            TRAJECTORY_STATS.append(stat)
            STAT_WEIGHTS[stat] = group_info["weight"]

    def __init__(self):
        self.player_trajectories: dict[int, dict] = {}
        self.player_names: dict[int, str] = {}
        self.league_averages: dict[str, float] = {}
        self._fitted = False

    def fit(self, career_features: pd.DataFrame):
        """Build trajectory vectors for each player.

        Args:
            career_features: DataFrame with PLAYER_ID, CAREER_YEAR, and stat columns
        """
        self.player_names = dict(
            zip(career_features["PLAYER_ID"], career_features["PLAYER_NAME"])
        )

        # Compute league averages for each stat (for normalization)
        for stat in self.TRAJECTORY_STATS:
            if stat in career_features.columns:
                self.league_averages[stat] = career_features[stat].mean()

        # Build trajectory for each player
        self.player_trajectories = {}

        for player_id in career_features["PLAYER_ID"].unique():
            player_data = career_features[
                career_features["PLAYER_ID"] == player_id
            ].sort_values("CAREER_YEAR")

            if len(player_data) < 2:
                continue

            available_stats = [s for s in self.TRAJECTORY_STATS if s in player_data.columns]
            if not available_stats:
                continue

            # Build trajectory indexed by career year
            trajectory_by_year = {}
            for _, row in player_data.iterrows():
                year = int(row["CAREER_YEAR"])
                stats = [row.get(s, 0) or 0 for s in available_stats]
                trajectory_by_year[year] = np.array(stats)

            # Build raw trajectory matrix for DTW (seasons x stats)
            raw_trajectory = player_data[available_stats].fillna(0).values

            # Normalize for DTW shape comparison (0-1 per stat)
            scaler = MinMaxScaler()
            if raw_trajectory.max() > raw_trajectory.min():
                normalized_trajectory = scaler.fit_transform(raw_trajectory)
            else:
                normalized_trajectory = raw_trajectory

            # Compute summary stats
            pts_values = player_data["PTS"].values if "PTS" in player_data.columns else []
            peak_ppg = float(max(pts_values)) if len(pts_values) > 0 else 0.0
            avg_ppg = float(np.mean(pts_values)) if len(pts_values) > 0 else 0.0

            # Physical attributes (use first row, doesn't change)
            first_row = player_data.iloc[0]
            height = float(first_row.get("height_inches", 0) or 0)
            weight = float(first_row.get("weight", 0) or 0)

            self.player_trajectories[player_id] = {
                # For year-for-year comparison
                "by_year": trajectory_by_year,
                "years": sorted(trajectory_by_year.keys()),
                # For DTW comparison
                "raw": raw_trajectory,
                "normalized": normalized_trajectory,
                # Summary stats
                "n_seasons": len(player_data),
                "peak_ppg": peak_ppg,
                "avg_ppg": avg_ppg,
                "height": height,
                "weight": weight,
                "stats": available_stats,
                "seasons": player_data["SEASON"].tolist() if "SEASON" in player_data.columns else [],
                # For visualization
                "pts_by_year": {int(row["CAREER_YEAR"]): row["PTS"] for _, row in player_data.iterrows()},
                "ast_by_year": {int(row["CAREER_YEAR"]): row.get("AST", 0) for _, row in player_data.iterrows()},
                "reb_by_year": {int(row["CAREER_YEAR"]): row.get("REB", 0) for _, row in player_data.iterrows()},
                "min_by_year": {int(row["CAREER_YEAR"]): row.get("MIN", 0) for _, row in player_data.iterrows()},
                "ts_by_year": {int(row["CAREER_YEAR"]): row.get("ts_pct", 0) or 0 for _, row in player_data.iterrows()},
                "usg_by_year": {int(row["CAREER_YEAR"]): row.get("e_usg_pct", 0) or 0 for _, row in player_data.iterrows()},
            }

        self._fitted = True
        print(f"Fitted trajectory matcher with {len(self.player_trajectories)} players")

    def _compute_year_for_year_distance(
        self,
        traj1: dict,
        traj2: dict,
        max_years: int,
    ) -> float:
        """Compute year-for-year distance (Year 1 vs Year 1, etc.)."""
        years_to_compare = [
            y for y in range(1, max_years + 1)
            if y in traj1["by_year"] and y in traj2["by_year"]
        ]

        if len(years_to_compare) < 2:
            return float("inf")

        stat_weights = np.array([self.STAT_WEIGHTS.get(s, 1.0) for s in traj1["stats"]])
        stat_weights = stat_weights / stat_weights.sum()

        league_avgs = np.array([self.league_averages.get(s, 1) for s in traj1["stats"]])
        league_avgs = np.where(league_avgs > 0, league_avgs, 1)

        year_distances = []
        for year in years_to_compare:
            vec1 = traj1["by_year"][year] / league_avgs
            vec2 = traj2["by_year"][year] / league_avgs
            diff = (vec1 - vec2) ** 2
            year_distances.append(np.sqrt(np.sum(diff * stat_weights)))

        return float(np.mean(year_distances))

    def _compute_dtw_distance(
        self,
        traj1: dict,
        traj2: dict,
        max_years: int,
    ) -> float:
        """Compute DTW distance on normalized trajectories for shape matching."""
        # Use only the first max_years of each trajectory
        norm1 = traj1["normalized"][:max_years]
        norm2 = traj2["normalized"][:max_years]

        if len(norm1) < 2 or len(norm2) < 2:
            return float("inf")

        try:
            alignment = dtw(norm1, norm2, keep_internals=True)
            return float(alignment.normalizedDistance)
        except Exception:
            return float("inf")

    def compute_trajectory_distance(
        self,
        player_id_1: int,
        player_id_2: int,
        max_years: int | None = None,
    ) -> tuple[float, dict]:
        """Compute hybrid trajectory distance between two players.

        Combines:
        - Year-for-year comparison (50%): Same career stage similarity
        - DTW shape matching (30%): Overall trajectory shape
        - Production similarity (20%): Peak PPG match

        Args:
            player_id_1: Query player ID
            player_id_2: Comparison player ID
            max_years: Compare only first N years (default: query player's career)

        Returns:
            Tuple of (combined_distance, info_dict)
        """
        if player_id_1 not in self.player_trajectories:
            return float("inf"), {}
        if player_id_2 not in self.player_trajectories:
            return float("inf"), {}

        traj1 = self.player_trajectories[player_id_1]
        traj2 = self.player_trajectories[player_id_2]

        if max_years is None:
            max_years = traj1["n_seasons"]

        # Component 1: Year-for-year distance
        yfy_distance = self._compute_year_for_year_distance(traj1, traj2, max_years)
        if yfy_distance == float("inf"):
            return float("inf"), {"reason": "Not enough overlapping years"}

        # Component 2: DTW shape distance
        dtw_distance = self._compute_dtw_distance(traj1, traj2, max_years)
        if dtw_distance == float("inf"):
            dtw_distance = yfy_distance  # Fallback to year-for-year

        # Component 3: Production similarity (peak PPG)
        peak_diff = abs(traj1["peak_ppg"] - traj2["peak_ppg"])
        production_distance = peak_diff / max(traj1["peak_ppg"], 1)

        # Component 4: Physical similarity (height)
        # Height difference in inches, normalized (3 inches = 0.5 penalty)
        height_diff = abs(traj1["height"] - traj2["height"])
        physical_distance = height_diff / 6.0  # 6 inch diff = 1.0 penalty

        # Weighted combination
        # 40% year-for-year (most important - same career stage)
        # 25% DTW shape (captures overall trajectory pattern)
        # 15% production level (ensures similar caliber players)
        # 20% physical similarity (ensures similar body types)
        combined = (
            0.40 * yfy_distance +
            0.25 * dtw_distance +
            0.15 * production_distance +
            0.20 * physical_distance
        )

        info = {
            "years_compared": min(max_years, traj2["n_seasons"]),
            "year_for_year_dist": yfy_distance,
            "dtw_dist": dtw_distance,
            "production_dist": production_distance,
            "physical_dist": physical_distance,
            "combined_dist": combined,
            "p1_peak": traj1["peak_ppg"],
            "p2_peak": traj2["peak_ppg"],
            "p1_height": traj1["height"],
            "p2_height": traj2["height"],
        }

        return combined, info

    def find_similar_trajectories(
        self,
        player_id: int,
        n: int = 10,
        min_seasons: int = None,
        min_peak_ppg: float = 5.0,
        peak_ppg_range: float = 0.5,
    ) -> list[tuple[int, str, float, dict]]:
        """Find players with similar career trajectories.

        Uses hybrid approach: year-for-year + DTW + production matching.

        Args:
            player_id: Query player ID
            n: Number of results
            min_seasons: Min seasons required (default: same as query)
            min_peak_ppg: Minimum peak PPG filter
            peak_ppg_range: Only include within this % of query's peak (0.5 = 50%)

        Returns:
            List of (player_id, name, distance, info) sorted by distance
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        if player_id not in self.player_trajectories:
            raise ValueError(f"Player ID {player_id} not found")

        query_traj = self.player_trajectories[player_id]
        query_seasons = query_traj["n_seasons"]
        query_peak = query_traj["peak_ppg"]

        if min_seasons is None:
            min_seasons = query_seasons

        # Peak PPG bounds
        min_peak = query_peak * (1 - peak_ppg_range)
        max_peak = query_peak * (1 + peak_ppg_range)

        results = []
        for other_id, other_traj in self.player_trajectories.items():
            if other_id == player_id:
                continue

            # Filters
            if other_traj["n_seasons"] < min_seasons:
                continue
            if other_traj["peak_ppg"] < min_peak_ppg:
                continue
            if not (min_peak <= other_traj["peak_ppg"] <= max_peak):
                continue

            # Compute hybrid distance
            distance, info = self.compute_trajectory_distance(
                player_id, other_id, max_years=query_seasons
            )

            if distance < float("inf"):
                name = self.player_names.get(other_id, "Unknown")
                info["n_seasons"] = other_traj["n_seasons"]
                info["peak_ppg"] = other_traj["peak_ppg"]
                results.append((other_id, name, distance, info))

        results.sort(key=lambda x: x[2])
        return results[:n]

    def get_player_trajectory(self, player_id: int) -> dict | None:
        """Get trajectory data for a player."""
        return self.player_trajectories.get(player_id)

    def visualize_comparison(
        self,
        player_id_1: int,
        player_id_2: int,
        stat: str = "PTS",
    ) -> dict:
        """Get data for visualizing trajectory comparison."""
        traj1 = self.player_trajectories.get(player_id_1)
        traj2 = self.player_trajectories.get(player_id_2)

        if not traj1 or not traj2:
            return {}

        stat_key_map = {
            "PTS": "pts_by_year",
            "AST": "ast_by_year",
            "REB": "reb_by_year",
            "MIN": "min_by_year",
            "TS%": "ts_by_year",
            "USG%": "usg_by_year",
        }
        stat_key = stat_key_map.get(stat, "pts_by_year")

        p1_data = traj1.get(stat_key, traj1.get("pts_by_year", {}))
        p2_data = traj2.get(stat_key, traj2.get("pts_by_year", {}))

        p1_years = sorted(p1_data.keys())
        p2_years = sorted(p2_data.keys())

        p1_values = [p1_data[y] for y in p1_years]
        p2_values = [p2_data[y] for y in p2_years]

        def normalize(vals):
            if not vals:
                return []
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return [0.5] * len(vals)
            return [(v - min_v) / (max_v - min_v) for v in vals]

        return {
            "player1": {
                "name": self.player_names.get(player_id_1),
                "years": p1_years,
                "values": p1_values,
                "normalized": normalize(p1_values),
            },
            "player2": {
                "name": self.player_names.get(player_id_2),
                "years": p2_years,
                "values": p2_values,
                "normalized": normalize(p2_values),
            },
            "stat": stat,
        }
