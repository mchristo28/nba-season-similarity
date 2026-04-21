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
    LeagueDashPlayerStats,
    LeagueDashPlayerShotLocations,
    LeagueHustleStatsPlayer,
    PlayerEstimatedMetrics,
    LeagueDashPlayerBioStats,
    CommonAllPlayers,
)

# Rate limiting
API_DELAY = 0.6


class ComprehensiveStatsPipeline:
    """Pull comprehensive stats from multiple NBA API endpoints."""

    # Feature group definitions - what stats belong to each group
    FEATURE_GROUPS = {
        "physical": [
            "height_inches", "weight",
        ],
        "scoring_volume": [
            "PTS", "FGA", "FG3A", "FTA", "MIN",
            "pts_share", "fga_share", "min_share",
        ],
        "scoring_efficiency": [
            "ts_pct", "efg_pct", "fg_pct", "fg3_pct", "ft_pct",
        ],
        "shot_profile": [
            "pct_fga_restricted", "pct_fga_paint", "pct_fga_midrange",
            "pct_fga_corner3", "pct_fga_above_break3",
            "fg_pct_restricted", "fg_pct_paint", "fg_pct_midrange",
            "fg_pct_corner3", "fg_pct_above_break3",
        ],
        "playmaking": [
            "AST", "TOV", "ast_share", "tov_share",
            "ast_ratio", "tov_pct",
        ],
        "rebounding": [
            "REB", "OREB", "DREB", "reb_share", "oreb_share", "dreb_share",
            "oreb_pct", "dreb_pct",
        ],
        "defense": [
            "STL", "BLK", "stl_share", "blk_share",
            "contested_shots", "contested_shots_2pt", "contested_shots_3pt",
            "deflections", "charges_drawn", "loose_balls_recovered",
        ],
        "overall_impact": [
            "e_off_rating", "e_def_rating", "e_net_rating",
            "e_usg_pct", "e_pace",
        ],
    }

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
                    f"{zone}_{stat}".lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    if zone else stat
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
            cols_to_keep = ["PLAYER_ID", "SEASON", "height_inches", "weight",
                           "COLLEGE", "COUNTRY", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER",
                           "NET_RATING", "OREB_PCT", "DREB_PCT", "USG_PCT", "TS_PCT", "AST_PCT"]
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
            columns={"PERSON_ID": "player_id", "DISPLAY_FIRST_LAST": "player_name",
                     "FROM_YEAR": "from_year", "TO_YEAR": "to_year"}
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

        # Start with basic stats
        df = basic.copy()

        # Merge shot locations
        if not shots.empty:
            shot_cols = [c for c in shots.columns if c not in df.columns or c in ["PLAYER_ID", "player_id"]]
            id_col = "PLAYER_ID" if "PLAYER_ID" in shots.columns else "player_id"
            if id_col in shots.columns:
                df = df.merge(
                    shots[shot_cols],
                    left_on="PLAYER_ID",
                    right_on=id_col,
                    how="left"
                )

        # Merge hustle stats
        if not hustle.empty:
            hustle_cols = [c for c in hustle.columns if c not in [x.lower() for x in df.columns] or c == "player_id"]
            if "player_id" in hustle.columns:
                df = df.merge(
                    hustle[hustle_cols],
                    left_on="PLAYER_ID",
                    right_on="player_id",
                    how="left"
                )

        # Merge advanced metrics
        if not advanced.empty:
            adv_cols = [c for c in advanced.columns if c not in [x.lower() for x in df.columns] or c == "player_id"]
            if "player_id" in advanced.columns:
                df = df.merge(
                    advanced[adv_cols],
                    left_on="PLAYER_ID",
                    right_on="player_id",
                    how="left"
                )

        # Merge bio stats (height, weight, etc.)
        if not bio.empty:
            bio_cols = [c for c in bio.columns if c not in df.columns or c == "PLAYER_ID"]
            if "PLAYER_ID" in bio.columns:
                df = df.merge(
                    bio[bio_cols],
                    on="PLAYER_ID",
                    how="left"
                )

        print(f"  Merged {len(df)} players with {len(df.columns)} columns")
        return df

    def compute_derived_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived statistics."""
        result = df.copy()

        # Efficiency stats
        if "FGM" in result.columns and "FGA" in result.columns:
            result["fg_pct"] = result["FGM"] / result["FGA"].replace(0, 1)

        if "FG3M" in result.columns and "FG3A" in result.columns:
            result["fg3_pct"] = result["FG3M"] / result["FG3A"].replace(0, 1)

        if "FTM" in result.columns and "FTA" in result.columns:
            result["ft_pct"] = result["FTM"] / result["FTA"].replace(0, 1)

        # True shooting
        if all(c in result.columns for c in ["PTS", "FGA", "FTA"]):
            tsa = result["FGA"] + 0.44 * result["FTA"]
            result["ts_pct"] = result["PTS"] / (2 * tsa.replace(0, 1))

        # Effective FG%
        if all(c in result.columns for c in ["FGM", "FG3M", "FGA"]):
            result["efg_pct"] = (result["FGM"] + 0.5 * result["FG3M"]) / result["FGA"].replace(0, 1)

        # Shot distribution percentages
        total_fga_col = "FGA"
        if total_fga_col in result.columns:
            total_fga = result[total_fga_col].replace(0, 1)

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
                corner3_fga = result["left_corner_3_fga"].fillna(0) + result["right_corner_3_fga"].fillna(0)
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
                    result[dest_col] = result[src_col]

            # Corner 3 FG% (weighted average)
            if all(c in result.columns for c in ["left_corner_3_fgm", "left_corner_3_fga",
                                                   "right_corner_3_fgm", "right_corner_3_fga"]):
                corner_fgm = result["left_corner_3_fgm"].fillna(0) + result["right_corner_3_fgm"].fillna(0)
                corner_fga = result["left_corner_3_fga"].fillna(0) + result["right_corner_3_fga"].fillna(0)
                result["fg_pct_corner3"] = corner_fgm / corner_fga.replace(0, 1)

        return result

    def compute_team_shares(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute player's share of team stats."""
        result = df.copy()

        share_stats = ["PTS", "AST", "REB", "OREB", "DREB", "STL", "BLK", "TOV", "FGA", "FG3A", "FTA", "MIN"]

        for stat in share_stats:
            if stat in result.columns and "TEAM_ID" in result.columns:
                team_totals = result.groupby(["TEAM_ID", "SEASON"])[stat].transform("sum")
                result[f"{stat.lower()}_share"] = result[stat] / team_totals.replace(0, 1)

        return result

    def add_career_year(self, df: pd.DataFrame, player_info: pd.DataFrame) -> pd.DataFrame:
        """Add career year based on rookie season."""
        result = df.copy()

        # Create rookie year lookup - ensure values are integers
        rookie_years = {}
        for _, row in player_info.iterrows():
            pid = row["player_id"]
            from_year = row["from_year"]
            if pd.notna(from_year):
                try:
                    rookie_years[pid] = int(from_year)
                except (ValueError, TypeError):
                    pass

        # Extract season start year
        result["_season_year"] = result["SEASON"].str[:4].astype(int)

        # Calculate career year
        def calc_career_year(row):
            season_year = row["_season_year"]
            player_id = row["PLAYER_ID"]
            rookie_year = rookie_years.get(player_id, season_year)
            return season_year - rookie_year + 1

        result["CAREER_YEAR"] = result.apply(calc_career_year, axis=1)
        result.loc[result["CAREER_YEAR"] < 1, "CAREER_YEAR"] = 1

        result = result.drop(columns=["_season_year"])
        return result

    def pull_all_seasons(
        self,
        start_season: str = "2013-14",  # Tracking data starts here
        end_season: str = "2025-26",
    ) -> pd.DataFrame:
        """Pull comprehensive stats for all seasons."""

        # Generate season list
        start_year = int(start_season[:4])
        end_year = int(end_season[:4])
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

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
                continue

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

        print(f"\nFinal dataset: {len(df)} player-seasons, {df['PLAYER_ID'].nunique()} unique players")
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
