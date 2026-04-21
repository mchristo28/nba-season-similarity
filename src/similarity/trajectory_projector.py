"""Career Trajectory Projector.

Given a player's career so far, find historical players who were in a similar
position at the same career stage, and show the distribution of outcomes.

Key insight: We're not asking "who's similar?" but "what could this player become?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class OutcomeTier(Enum):
    """Career outcome tiers."""
    STAR = "Star"  # All-Star, All-NBA, or sustained 20+ PPG
    QUALITY_STARTER = "Quality Starter"  # Long career, solid production
    ROLE_PLAYER = "Role Player"  # Rotation player, limited peak
    DID_NOT_ESTABLISH = "Did Not Establish"  # Short career or very low production


@dataclass
class PlayerOutcome:
    """A player's career outcome."""
    player_id: int
    player_name: str
    tier: OutcomeTier
    seasons_played: int
    peak_ppg: float
    career_ppg: float
    all_star_appearances: int
    # What they looked like at each career stage
    stats_by_year: dict  # year -> {stat: value}


@dataclass
class ProjectionResult:
    """Projection result for a player."""
    query_player_id: int
    query_player_name: str
    career_year: int

    # Outcome distribution
    star_pct: float
    starter_pct: float
    role_player_pct: float
    did_not_establish_pct: float

    # Exemplar players for each tier
    star_examples: list[PlayerOutcome]
    starter_examples: list[PlayerOutcome]
    role_player_examples: list[PlayerOutcome]
    did_not_establish_examples: list[PlayerOutcome]

    # All comparable players used
    comparable_players: list[tuple[PlayerOutcome, float]]  # (outcome, similarity_score)


class TrajectoryProjector:
    """Project career outcomes based on historical comparisons."""

    # Stats to compare at each career stage (full profile)
    COMPARISON_STATS = [
        # Scoring volume
        "PTS", "FGA", "FG3A", "FTA", "MIN",
        # Scoring efficiency
        "ts_pct", "efg_pct", "fg_pct", "fg3_pct", "ft_pct",
        # Shot profile
        "pct_fga_restricted", "pct_fga_paint", "pct_fga_midrange",
        "pct_fga_corner3", "pct_fga_above_break3",
        # Playmaking
        "AST", "TOV",
        # Rebounding
        "REB", "OREB", "DREB",
        # Defense
        "STL", "BLK",
        # Overall impact
        "e_off_rating", "e_def_rating", "e_usg_pct",
    ]

    # Physical for archetype matching
    PHYSICAL_STATS = ["height_inches", "weight"]

    def __init__(self):
        self.player_outcomes: dict[int, PlayerOutcome] = {}
        self.player_names: dict[int, str] = {}
        self.league_averages: dict[str, float] = {}
        self._fitted = False

    def _classify_outcome(
        self,
        seasons_played: int,
        peak_ppg: float,
        career_ppg: float,
        all_star_apps: int = 0,
        all_nba: int = 0,
    ) -> OutcomeTier:
        """Classify a player's career outcome into a tier."""

        # Star tier: All-Star, All-NBA, or elite production
        if all_star_apps > 0 or all_nba > 0:
            return OutcomeTier.STAR
        if peak_ppg >= 20 and seasons_played >= 5:
            return OutcomeTier.STAR
        if peak_ppg >= 18 and seasons_played >= 8:
            return OutcomeTier.STAR

        # Quality Starter: Long career with solid production
        if seasons_played >= 8 and peak_ppg >= 12:
            return OutcomeTier.QUALITY_STARTER
        if seasons_played >= 6 and peak_ppg >= 14:
            return OutcomeTier.QUALITY_STARTER
        if seasons_played >= 10 and peak_ppg >= 10:
            return OutcomeTier.QUALITY_STARTER

        # Role Player: Established rotation player
        if seasons_played >= 4 and peak_ppg >= 6:
            return OutcomeTier.ROLE_PLAYER
        if seasons_played >= 6 and career_ppg >= 5:
            return OutcomeTier.ROLE_PLAYER

        # Did Not Establish
        return OutcomeTier.DID_NOT_ESTABLISH

    def fit(self, career_features: pd.DataFrame, awards_data: dict[int, dict] = None):
        """Fit the projector with historical career data.

        Args:
            career_features: DataFrame with PLAYER_ID, CAREER_YEAR, and stat columns
            awards_data: Optional dict of player_id -> {all_star: count, all_nba: count}
        """
        awards_data = awards_data or {}

        self.player_names = dict(
            zip(career_features["PLAYER_ID"], career_features["PLAYER_NAME"])
        )

        # Compute league averages for normalization
        for stat in self.COMPARISON_STATS:
            if stat in career_features.columns:
                self.league_averages[stat] = career_features[stat].mean()

        # Build outcome for each player
        self.player_outcomes = {}

        for player_id in career_features["PLAYER_ID"].unique():
            player_data = career_features[
                career_features["PLAYER_ID"] == player_id
            ].sort_values("CAREER_YEAR")

            if len(player_data) < 1:
                continue

            player_name = player_data["PLAYER_NAME"].iloc[0]

            # Career stats
            seasons_played = len(player_data)
            pts_values = player_data["PTS"].values
            peak_ppg = float(max(pts_values)) if len(pts_values) > 0 else 0
            career_ppg = float(np.mean(pts_values)) if len(pts_values) > 0 else 0

            # Awards (if provided)
            player_awards = awards_data.get(player_id, {})
            all_star_apps = player_awards.get("all_star", 0)
            all_nba = player_awards.get("all_nba", 0)

            # Classify outcome
            tier = self._classify_outcome(
                seasons_played, peak_ppg, career_ppg,
                all_star_apps, all_nba
            )

            # Build stats by year
            stats_by_year = {}
            for _, row in player_data.iterrows():
                year = int(row["CAREER_YEAR"])
                stats_by_year[year] = {
                    stat: float(row.get(stat, 0) or 0)
                    for stat in self.COMPARISON_STATS + self.PHYSICAL_STATS
                    if stat in row.index
                }

            # Physical attributes (constant)
            first_row = player_data.iloc[0]
            height = float(first_row.get("height_inches", 0) or 0)

            self.player_outcomes[player_id] = PlayerOutcome(
                player_id=player_id,
                player_name=player_name,
                tier=tier,
                seasons_played=seasons_played,
                peak_ppg=peak_ppg,
                career_ppg=career_ppg,
                all_star_appearances=all_star_apps,
                stats_by_year=stats_by_year,
            )

        self._fitted = True

        # Print tier distribution
        tier_counts = {}
        for outcome in self.player_outcomes.values():
            tier_counts[outcome.tier.value] = tier_counts.get(outcome.tier.value, 0) + 1
        print(f"Fitted trajectory projector with {len(self.player_outcomes)} players:")
        for tier, count in sorted(tier_counts.items()):
            print(f"  {tier}: {count}")

    def _compute_similarity(
        self,
        query_stats: dict[int, dict],  # year -> stats
        other_stats: dict[int, dict],
        max_year: int,
        query_height: float,
        other_height: float,
    ) -> float:
        """Compute similarity between two players at a career stage.

        Compares stats year-by-year up to max_year, with more weight on recent years.
        """
        total_distance = 0.0
        total_weight = 0.0

        for year in range(1, max_year + 1):
            if year not in query_stats or year not in other_stats:
                continue

            q_stats = query_stats[year]
            o_stats = other_stats[year]

            # Weight recent years more heavily
            # Year N gets weight 1.0, year N-1 gets 0.8, etc.
            year_weight = 0.6 + 0.4 * (year / max_year)

            # Compute stat differences
            year_distance = 0.0
            stat_count = 0

            for stat in self.COMPARISON_STATS:
                if stat not in q_stats or stat not in o_stats:
                    continue

                q_val = q_stats[stat]
                o_val = o_stats[stat]

                # Normalize by league average
                league_avg = self.league_averages.get(stat, 1)
                if league_avg > 0:
                    q_norm = q_val / league_avg
                    o_norm = o_val / league_avg
                else:
                    q_norm, o_norm = q_val, o_val

                year_distance += (q_norm - o_norm) ** 2
                stat_count += 1

            if stat_count > 0:
                year_distance = np.sqrt(year_distance / stat_count)
                total_distance += year_distance * year_weight
                total_weight += year_weight

        if total_weight == 0:
            return float("inf")

        avg_distance = total_distance / total_weight

        # Add height penalty (significant for archetype matching)
        height_diff = abs(query_height - other_height)
        height_penalty = height_diff / 4.0  # 4 inches diff = 1.0 penalty

        return avg_distance + 0.3 * height_penalty

    def project(
        self,
        player_id: int,
        n_comparables: int = 50,
        n_examples_per_tier: int = 3,
    ) -> ProjectionResult | None:
        """Project a player's career outcome based on historical comparisons.

        Args:
            player_id: Player to project
            n_comparables: Number of comparable players to use
            n_examples_per_tier: Number of example players to show per tier

        Returns:
            ProjectionResult with outcome distribution and examples
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")

        if player_id not in self.player_outcomes:
            return None

        query = self.player_outcomes[player_id]
        query_years = query.seasons_played
        query_height = query.stats_by_year.get(1, {}).get("height_inches", 0)

        # Find comparable players (those with MORE seasons, so we know their outcome)
        comparables = []

        for other_id, other in self.player_outcomes.items():
            if other_id == player_id:
                continue

            # Must have played longer (so we know their outcome)
            if other.seasons_played <= query_years:
                continue

            # Must have data for the years we're comparing
            if not all(y in other.stats_by_year for y in range(1, query_years + 1)):
                continue

            other_height = other.stats_by_year.get(1, {}).get("height_inches", 0)

            # Compute similarity at the same career stage
            similarity = self._compute_similarity(
                query.stats_by_year,
                other.stats_by_year,
                query_years,
                query_height,
                other_height,
            )

            if similarity < float("inf"):
                comparables.append((other, similarity))

        # Sort by similarity (lower is better)
        comparables.sort(key=lambda x: x[1])
        comparables = comparables[:n_comparables]

        if not comparables:
            return None

        # Calculate outcome distribution (weighted by similarity)
        tier_weights = {tier: 0.0 for tier in OutcomeTier}
        total_weight = 0.0

        for other, similarity in comparables:
            # Convert distance to weight (closer = higher weight)
            weight = 1.0 / (1.0 + similarity)
            tier_weights[other.tier] += weight
            total_weight += weight

        # Normalize to percentages
        if total_weight > 0:
            for tier in tier_weights:
                tier_weights[tier] /= total_weight

        # Get example players for each tier (closest matches in each tier)
        tier_examples = {tier: [] for tier in OutcomeTier}
        for other, similarity in comparables:
            if len(tier_examples[other.tier]) < n_examples_per_tier:
                tier_examples[other.tier].append(other)

        return ProjectionResult(
            query_player_id=player_id,
            query_player_name=query.player_name,
            career_year=query_years,
            star_pct=tier_weights[OutcomeTier.STAR] * 100,
            starter_pct=tier_weights[OutcomeTier.QUALITY_STARTER] * 100,
            role_player_pct=tier_weights[OutcomeTier.ROLE_PLAYER] * 100,
            did_not_establish_pct=tier_weights[OutcomeTier.DID_NOT_ESTABLISH] * 100,
            star_examples=tier_examples[OutcomeTier.STAR],
            starter_examples=tier_examples[OutcomeTier.QUALITY_STARTER],
            role_player_examples=tier_examples[OutcomeTier.ROLE_PLAYER],
            did_not_establish_examples=tier_examples[OutcomeTier.DID_NOT_ESTABLISH],
            comparable_players=comparables,
        )

    def get_player_outcome(self, player_id: int) -> PlayerOutcome | None:
        """Get outcome data for a player."""
        return self.player_outcomes.get(player_id)
