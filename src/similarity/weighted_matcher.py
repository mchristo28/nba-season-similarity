"""Vectorized, missing-aware season and career matching."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.registry import FEATURE_GROUPS
from src.features.schema import validate_season_features

from .profiles import get_profile
from .scoring import combine_distances, validate_weights


class WeightedMatcher:
    """Compare group RMS differences in standardized units.

    Missing observations remain missing. Coverage is the weighted fraction of
    requested features observed for both seasons. Zero distances retain weight.
    """

    FEATURE_GROUPS = FEATURE_GROUPS

    def __init__(self, weights: dict[str, float] | None = None, *, profile: str | None = None):
        self.profile = profile
        self.profile_spec = get_profile(profile) if profile else None
        if self.profile_spec:
            self.FEATURE_GROUPS = self.profile_spec["groups"]
        defaults = {k: v["default_weight"] for k, v in self.FEATURE_GROUPS.items()}
        self.weights = validate_weights(
            defaults if weights is None else weights, self.FEATURE_GROUPS
        )
        self.career_features = None
        self.player_names = {}
        self.scalers = {}
        self._matrices = {}
        self._indices = {"year": {}, "age": {}}

    def fit(self, career_features: pd.DataFrame):
        validate_season_features(career_features)
        self.career_features = (
            career_features.sort_values(["PLAYER_ID", "CAREER_YEAR"]).reset_index(drop=True).copy()
        )
        df = self.career_features
        self.player_names = dict(zip(df.PLAYER_ID, df.PLAYER_NAME))
        self.scalers = {}
        self._matrices = {}
        reference = df
        if self.profile_spec:
            required = {c for spec in self.FEATURE_GROUPS.values() for c in spec["features"]}
            if not required <= set(df):
                raise ValueError(
                    "Comparison data is outdated; rebuild exact totals and comparison features"
                )
            reference = df[
                (df.GP >= 20)
                & (df.MIN >= 15)
                & (df.SEASON.str[:4].astype(int) >= self.profile_spec["first_year"])
            ]
            if len(reference) < 2:
                raise ValueError(
                    "Not enough rotation-player seasons to fit this comparison profile"
                )
        for group, spec in self.FEATURE_GROUPS.items():
            columns = [c for c in spec["features"] if c in df and df[c].notna().any()]
            if not columns:
                continue
            scaler = StandardScaler()
            scaler.fit(reference[columns].to_numpy(dtype=float))
            self._matrices[group] = scaler.transform(df[columns].to_numpy(dtype=float))
            self.scalers[group] = {"scaler": scaler, "columns": columns}
        self._indices = {"year": {}, "age": {}}
        for i, row in df.iterrows():
            pid = int(row.PLAYER_ID)
            self._indices["year"].setdefault(pid, {})[int(row.CAREER_YEAR)] = i
            if pd.notna(row.AGE):
                # If the published age repeats, consistently use the later season.
                self._indices["age"].setdefault(pid, {})[int(row.AGE)] = i
        return self

    def set_weights(self, weights: dict[str, float]):
        self.weights = validate_weights(weights, self.FEATURE_GROUPS)

    def _weights(self, weights):
        return validate_weights(self.weights if weights is None else weights, self.FEATURE_GROUPS)

    def _index(self, compare_by):
        if self.career_features is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if compare_by not in self._indices:
            raise ValueError("compare_by must be 'year' or 'age'")
        return self._indices[compare_by]

    def get_player_years(self, player_id: int) -> set[int]:
        return set(self._index("year").get(player_id, {}))

    def get_player_ages(self, player_id: int) -> set[int]:
        return set(self._index("age").get(player_id, {}))

    def _distances(self, query: int, candidates: np.ndarray, weights: dict):
        group_distances = {}
        coverage = np.zeros(len(candidates))
        numerator = np.zeros(len(candidates))
        denominator = np.zeros(len(candidates))
        complete = np.ones(len(candidates), dtype=bool)
        for group, matrix in self._matrices.items():
            diff = matrix[candidates] - matrix[query]
            valid = np.isfinite(diff)
            count = valid.sum(axis=1)
            squared = np.where(valid, diff * diff, 0).sum(axis=1)
            distance = np.sqrt(
                np.divide(squared, count, out=np.full(len(count), np.nan), where=count > 0)
            )
            group_distances[group] = distance
            weight = weights.get(group, 0)
            if self.profile and weight > 0:
                # Every candidate must share the same query-observed evidence for this ranking.
                complete &= count == np.isfinite(matrix[query]).sum()
            numerator += np.where(count > 0, distance, 0) * weight
            denominator += (count > 0) * weight
            # Use registry length so absent columns don't masquerade as full coverage.
            coverage += weight * count / len(self.FEATURE_GROUPS[group]["features"])
        overall = np.divide(
            numerator, denominator, out=np.full(len(candidates), np.inf), where=denominator > 0
        )
        coverage /= sum(weights.values())
        overall[~complete] = np.inf
        return overall, group_distances, coverage

    def season_coverage(
        self, player_id, season_key, other_id, other_key, compare_by="year", weights=None
    ):
        index = self._index(compare_by)
        _, _, coverage = self._distances(
            index[player_id][season_key],
            np.array([index[other_id][other_key]]),
            self._weights(weights),
        )
        return float(coverage[0])

    def find_similar_season(
        self,
        player_id: int,
        season_key: int,
        n: int = 10,
        compare_by: str = "year",
        weights=None,
        *,
        min_games: int = 0,
        min_minutes: float = 0,
        min_coverage: float = 0.5,
        exclude_same: bool = False,
        season_start: int | None = None,
        season_end: int | None = None,
        max_age_difference: int | None = None,
    ):
        index = self._index(compare_by)
        weights = self._weights(weights)
        if player_id not in index or season_key not in index[player_id]:
            raise ValueError(f"Player {player_id} season {season_key} not found")
        if n < 0 or min_games < 0 or min_minutes < 0 or not 0 <= min_coverage <= 1:
            raise ValueError("Invalid search limits")
        query = index[player_id][season_key]
        if max_age_difference is not None and max_age_difference < 0:
            raise ValueError("Age difference must be nonnegative")
        if self.profile_spec:
            first_year = self.profile_spec["first_year"]
            if int(self.career_features.iloc[query].SEASON[:4]) < first_year:
                raise ValueError(
                    f"This comparison profile starts in {first_year}; choose historical coverage"
                )
            season_start = max(first_year, season_start or first_year)
        entries = [
            (pid, key, row)
            for pid, seasons in index.items()
            for key, row in seasons.items()
            if row != query and (not exclude_same or pid != player_id)
        ]
        if not entries or n == 0:
            return []
        candidates = np.array([row for _, _, row in entries])
        df = self.career_features.iloc[candidates]
        mask = (df.GP.to_numpy() >= min_games) & (df.MIN.to_numpy() >= min_minutes)
        years = df.SEASON.str[:4].astype(int).to_numpy()
        if season_start is not None:
            mask &= years >= season_start
        if season_end is not None:
            mask &= years <= season_end
        if max_age_difference is not None:
            mask &= (
                (df.AGE - self.career_features.iloc[query].AGE)
                .abs()
                .le(max_age_difference)
                .to_numpy()
            )
        distances, groups, coverage = self._distances(query, candidates, weights)
        eligible = np.flatnonzero(mask & np.isfinite(distances) & (coverage >= min_coverage))
        ranked = eligible[np.argsort(distances[eligible], kind="stable")[:n]]
        return [
            (
                entries[i][0],
                self.player_names[entries[i][0]],
                entries[i][1],
                float(distances[i]),
                {g: float(v[i]) for g, v in groups.items() if np.isfinite(v[i])},
            )
            for i in ranked
        ]

    def compute_distance(self, player_id_1, player_id_2, compare_by="year", weights=None):
        index = self._index(compare_by)
        weights = self._weights(weights)
        first, second = index.get(player_id_1, {}), index.get(player_id_2, {})
        common = first.keys() & second.keys()
        if not common:
            return float("inf"), 0, {}
        groups = {}
        periods = 0
        for key in common:
            if self.profile_spec and any(
                int(self.career_features.iloc[i].SEASON[:4]) < self.profile_spec["first_year"]
                for i in [first[key], second[key]]
            ):
                continue
            distances, per_group, _ = self._distances(first[key], np.array([second[key]]), weights)
            if not np.isfinite(distances[0]):
                continue
            periods += 1
            for group, values in per_group.items():
                if np.isfinite(values[0]):
                    groups.setdefault(group, []).append(float(values[0]))
        averaged = {g: float(np.mean(v)) for g, v in groups.items()}
        return combine_distances(averaged, weights), periods, averaged

    def find_similar(
        self,
        player_id,
        n=10,
        compare_by="year",
        min_overlap=2,
        require_full_coverage=True,
        weights=None,
        max_ppg_diff_pct=None,
    ):
        index = self._index(compare_by)
        if player_id not in index:
            raise ValueError(f"Player ID {player_id} not found")
        weights = self._weights(weights)
        query = index[player_id]
        results = []
        for other, periods in index.items():
            if other == player_id or (require_full_coverage and not query.keys() <= periods.keys()):
                continue
            common = query.keys() & periods.keys()
            if max_ppg_diff_pct is not None:
                if max_ppg_diff_pct < 0:
                    raise ValueError("PPG tolerance must be nonnegative")
                pts = self.career_features.PTS
                if any(
                    abs(pts[query[k]] - pts[periods[k]])
                    > max(abs(pts[query[k]]), 1) * max_ppg_diff_pct
                    for k in common
                ):
                    continue
            distance, count, groups = self.compute_distance(player_id, other, compare_by, weights)
            if count >= min_overlap and np.isfinite(distance):
                results.append((other, self.player_names[other], distance, count, groups))
        return sorted(results, key=lambda r: r[2])[:n]

    def get_season_info(self, player_id, season_key, compare_by="year"):
        index = self._index(compare_by)
        row_index = index.get(player_id, {}).get(season_key)
        if row_index is None:
            return None
        row = self.career_features.iloc[row_index]
        return {
            "player_name": row.PLAYER_NAME,
            "season": row.SEASON,
            "career_year": int(row.CAREER_YEAR),
            "age": int(row.AGE) if pd.notna(row.AGE) else None,
            "pts": row.PTS,
            "ast": row.AST,
            "reb": row.REB,
            "gp": int(row.GP),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "career_features": self.career_features,
                    "weights": self.weights,
                    "profile": self.profile,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path):
        """Load trusted local model files only; refit to rebuild all indexes."""
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        return cls(data["weights"], profile=data.get("profile")).fit(data["career_features"])
