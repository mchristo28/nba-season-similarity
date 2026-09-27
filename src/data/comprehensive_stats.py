"""Comprehensive NBA stats pipeline.

Pulls and merges data from multiple NBA API endpoints:
1. Basic box score stats (LeagueDashPlayerStats)
2. Shot location data (LeagueDashPlayerShotLocations)
3. Hustle/defense stats (LeagueHustleStatsPlayer)
4. Advanced estimated metrics (PlayerEstimatedMetrics)

Organizes into feature groups for weighted similarity matching.
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    CommonAllPlayers,
    LeagueDashPlayerBioStats,
    LeagueDashPlayerShotLocations,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueHustleStatsPlayer,
    PlayerEstimatedMetrics,
)

from src.data.data_loader import DataLoader
from src.data.merge_stats import merge_player_measurements
from src.features.registry import FEATURE_GROUPS
from src.features.transforms import add_career_year, efficiency_stats, team_shares

# Rate limiting
API_DELAY = 0.6


class ComprehensiveStatsPipeline:
    """Pull comprehensive stats from multiple NBA API endpoints."""

    FEATURE_GROUPS = FEATURE_GROUPS

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"

        for d in [self.raw_dir, self.processed_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def fetch_basic_stats(self, season: str) -> pd.DataFrame:
        """Fetch basic box score stats."""
        print(f"  Fetching basic stats for {season}...")
        time.sleep(API_DELAY)

        try:
            stats = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
            )
            df = stats.get_data_frames()[0]
            df["SEASON"] = season
            return df
        except Exception as e:
            print(f"    Error fetching basic stats: {e}")
            return pd.DataFrame()

    def fetch_shot_locations(self, season: str) -> pd.DataFrame:
        """Fetch shot location data."""
        print(f"  Fetching shot locations for {season}...")
        time.sleep(API_DELAY)

        try:
            shots = LeagueDashPlayerShotLocations(
                season=season,
                per_mode_detailed="PerGame",
            )
            df = shots.get_data_frames()[0]

            # Flatten multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    f"{zone}_{stat}".lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", "_")
                    if zone
                    else stat
                    for zone, stat in df.columns
                ]

            df["SEASON"] = season
            return df
        except Exception as e:
            print(f"    Error fetching shot locations: {e}")
            return pd.DataFrame()

    def fetch_hustle_stats(self, season: str) -> pd.DataFrame:
        """Fetch hustle/defense stats."""
        print(f"  Fetching hustle stats for {season}...")
        time.sleep(API_DELAY)

        try:
            hustle = LeagueHustleStatsPlayer(
                season=season,
                per_mode_time="PerGame",
            )
            df = hustle.get_data_frames()[0]
            df["SEASON"] = season

            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"    Error fetching hustle stats: {e}")
            return pd.DataFrame()

    def fetch_advanced_metrics(self, season: str) -> pd.DataFrame:
        """Fetch estimated advanced metrics."""
        print(f"  Fetching advanced metrics for {season}...")
        time.sleep(API_DELAY)

        try:
            metrics = PlayerEstimatedMetrics(season=season)
            df = metrics.get_data_frames()[0]
            df["SEASON"] = season

            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"    Error fetching advanced metrics: {e}")
            return pd.DataFrame()

    def fetch_bio_stats(self, season: str) -> pd.DataFrame:
        """Fetch player bio/physical stats."""
        print(f"  Fetching bio stats for {season}...")
        time.sleep(API_DELAY)

        try:
            bio = LeagueDashPlayerBioStats(season=season)
            df = bio.get_data_frames()[0]
            df["SEASON"] = season

            # Rename height/weight columns
            if "PLAYER_HEIGHT_INCHES" in df.columns:
                df["height_inches"] = pd.to_numeric(df["PLAYER_HEIGHT_INCHES"], errors="coerce")
            if "PLAYER_WEIGHT" in df.columns:
                df["weight"] = pd.to_numeric(df["PLAYER_WEIGHT"], errors="coerce")

            # Keep useful columns
            cols_to_keep = [
                "PLAYER_ID",
                "SEASON",
                "height_inches",
                "weight",
                "COLLEGE",
                "COUNTRY",
                "DRAFT_YEAR",
                "DRAFT_ROUND",
                "DRAFT_NUMBER",
                "NET_RATING",
                "OREB_PCT",
                "DREB_PCT",
                "USG_PCT",
                "TS_PCT",
                "AST_PCT",
            ]
            df = df[[c for c in cols_to_keep if c in df.columns]]

            # Rename to lowercase
            df.columns = [c.lower() if c not in ["PLAYER_ID", "SEASON"] else c for c in df.columns]
            return df
        except Exception as e:
            print(f"    Error fetching bio stats: {e}")
            return pd.DataFrame()

    def fetch_player_info(self) -> pd.DataFrame:
        """Fetch player info including rookie year."""
        print("Fetching player info...")
        time.sleep(API_DELAY)

        players = CommonAllPlayers(is_only_current_season=0)
        df = players.get_data_frames()[0]
        return df[["PERSON_ID", "DISPLAY_FIRST_LAST", "FROM_YEAR", "TO_YEAR"]].rename(
            columns={
                "PERSON_ID": "player_id",
                "DISPLAY_FIRST_LAST": "player_name",
                "FROM_YEAR": "from_year",
                "TO_YEAR": "to_year",
            }
        )

    def fetch_season_data(self, season: str) -> pd.DataFrame:
        """Fetch and merge all data for a season."""
        print(f"\nFetching data for {season}...")

        # Fetch all data sources
        basic = self.fetch_basic_stats(season)
        shots = self.fetch_shot_locations(season)
        hustle = self.fetch_hustle_stats(season)
        advanced = self.fetch_advanced_metrics(season)
        bio = self.fetch_bio_stats(season)

        if basic.empty:
            print(f"  No basic stats for {season}, skipping")
            return pd.DataFrame()

        # Tracking uses explicit season totals; build_features converts them once.
        loader = DataLoader(str(self.data_dir), refresh=True)
        tracking = loader.get_tracking_stats(season) if int(season[:4]) >= 2013 else pd.DataFrame()
        scoring = loader.get_scoring_stats(season)

        # Start with basic stats
        df = basic.copy()

        for extra in (shots, hustle, advanced, bio, tracking, scoring):
            df = merge_player_measurements(df, extra)

        print(f"  Merged {len(df)} players with {len(df.columns)} columns")
        return df

    def compute_derived_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived statistics."""
        result = efficiency_stats(df)

        # Shot distribution percentages
        total_fga_col = "FGA"
        if total_fga_col in result.columns:
            total_fga = result[total_fga_col].where(result[total_fga_col] > 0)

            # Map shot location columns to our naming
            zone_mappings = {
                "restricted_area_fga": "pct_fga_restricted",
                "in_the_paint_non_ra_fga": "pct_fga_paint",
                "mid_range_fga": "pct_fga_midrange",
                "above_the_break_3_fga": "pct_fga_above_break3",
            }

            for src_col, dest_col in zone_mappings.items():
                if src_col in result.columns:
                    result[dest_col] = result[src_col] / total_fga

            # Corner 3 = left + right
            if "left_corner_3_fga" in result.columns and "right_corner_3_fga" in result.columns:
                corner3_fga = result["left_corner_3_fga"] + result["right_corner_3_fga"]
                result["pct_fga_corner3"] = corner3_fga / total_fga

            # Shot zone FG%
            fg_pct_mappings = {
                "restricted_area_fg_pct": "fg_pct_restricted",
                "in_the_paint_non_ra_fg_pct": "fg_pct_paint",
                "mid_range_fg_pct": "fg_pct_midrange",
                "above_the_break_3_fg_pct": "fg_pct_above_break3",
            }
            for src_col, dest_col in fg_pct_mappings.items():
                if src_col in result.columns:
                    attempts_col = src_col.removesuffix("_fg_pct") + "_fga"
                    result[dest_col] = result[src_col].where(result[attempts_col] > 0)

            # Corner 3 FG% (weighted average)
            if all(
                c in result.columns
                for c in [
                    "left_corner_3_fgm",
                    "left_corner_3_fga",
                    "right_corner_3_fgm",
                    "right_corner_3_fga",
                ]
            ):
                corner_fgm = result["left_corner_3_fgm"] + result["right_corner_3_fgm"]
                corner_fga = result["left_corner_3_fga"] + result["right_corner_3_fga"]
                result["fg_pct_corner3"] = corner_fgm / corner_fga.where(corner_fga > 0)

        return result

    def fetch_team_stats(self, season: str) -> pd.DataFrame:
        """Actual team totals, cached separately from player aggregates."""
        path = self.raw_dir / f"team_totals_{season}.parquet"
        time.sleep(API_DELAY)
        frame = LeagueDashTeamStats(season=season, per_mode_detailed="Totals").get_data_frames()[0]
        frame["SEASON"] = season
        frame.to_parquet(path, index=False)
        return frame

    def compute_team_shares(self, df: pd.DataFrame, teams=None) -> pd.DataFrame:
        if teams is None:
            teams = pd.concat(
                [self.fetch_team_stats(s) for s in df.SEASON.unique()], ignore_index=True
            )
        return team_shares(df, teams)

    def add_career_year(self, df: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
        years = pd.to_numeric(player_info.set_index("player_id").from_year, errors="coerce")
        return add_career_year(df, years.dropna().to_dict())

    def pull_all_seasons(
        self,
        start_season: str = "2013-14",  # Tracking data starts here
        end_season: str = "2025-26",
    ) -> pd.DataFrame:
        """Pull comprehensive stats for all seasons."""

        # Generate season list
        start_year = int(start_season[:4])
        end_year = int(end_season[:4])
        seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(start_year, end_year + 1)]

        print(f"Pulling data for {len(seasons)} seasons: {seasons[0]} to {seasons[-1]}")

        # Fetch player info first
        player_info = self.fetch_player_info()

        all_data = []
        for season in seasons:
            try:
                season_data = self.fetch_season_data(season)
                if not season_data.empty:
                    all_data.append(season_data)
            except Exception as e:
                print(f"  Error with {season}: {e}")
                raise

        if len(all_data) != len(seasons):
            raise ValueError(
                "Some requested seasons could not be fetched; refusing partial rebuild"
            )

        if not all_data:
            print("No data fetched!")
            return pd.DataFrame()

        # Combine all seasons
        print("\nCombining all seasons...")
        df = pd.concat(all_data, ignore_index=True)

        # Compute derived stats
        print("Computing derived statistics...")
        df = self.compute_derived_stats(df)

        # Compute team shares
        print("Computing team shares...")
        df = self.compute_team_shares(df)

        # Add career year
        print("Adding career years...")
        df = self.add_career_year(df, player_info)

        print(
            f"\nFinal dataset: {len(df)} player-seasons, {df['PLAYER_ID'].nunique()} unique players"
        )
        print(f"Seasons: {sorted(df['SEASON'].unique())}")
        print(f"Columns: {len(df.columns)}")

        return df

    def save_data(self, df: pd.DataFrame, filename: str = "comprehensive_stats.parquet"):
        """Save the comprehensive stats."""
        path = self.processed_dir / filename
        df.to_parquet(path, index=False)
        print(f"Saved to {path}")
        return path


def pull_comprehensive_stats(
    start_season: str = "2013-14",
    end_season: str = "2025-26",
) -> pd.DataFrame:
    """Convenience function to pull all comprehensive stats."""
    pipeline = ComprehensiveStatsPipeline()
    df = pipeline.pull_all_seasons(start_season, end_season)
    if not df.empty:
        pipeline.save_data(df)
    return df


if __name__ == "__main__":
    pull_comprehensive_stats()
