"""Orchestrates feature generation for all players."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.cache_manager import CacheManager
from src.data.data_loader import DataLoader

from .composition_stats import CompositionStatsCalculator


class FeaturePipeline:
    """Orchestrates feature generation for player similarity.

    Pipeline steps:
    1. Load league stats for each season
    2. Load team stats
    3. Compute composition stats
    4. Build feature vectors
    5. Store computed features
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache = CacheManager(cache_dir)
        self.loader = DataLoader(cache_dir)
        self.composition_calc = CompositionStatsCalculator()

    def load_team_stats_dict(self) -> dict[int, pd.DataFrame]:
        """Load all team stats into a dictionary."""
        teams = self.loader.client.get_all_teams()
        team_stats = {}
        for team in teams:
            key = f"team_{team['id']}_all_seasons"
            df = self.cache.get(key)
            if df is not None:
                team_stats[team["id"]] = df
        return team_stats

    def process_season(self, season: str, team_stats: dict[int, pd.DataFrame], include_tracking: bool = True) -> pd.DataFrame:
        """Process a single season's data with composition stats.

        Args:
            season: Season string like '2023-24'
            team_stats: Dict of team_id -> team stats DataFrame
            include_tracking: Whether to include tracking stats (drives, catch-shoot, etc.)

        Returns:
            DataFrame with player stats + composition stats
        """
        if include_tracking:
            # Try to get stats with tracking data
            league_stats = self.loader.get_league_stats_with_tracking(season)
        else:
            league_stats = self.loader.get_league_stats(season)

        if league_stats is None or league_stats.empty:
            # Fallback to regular stats if tracking fails
            league_stats = self.loader.get_league_stats(season)

        if league_stats is None:
            return pd.DataFrame()

        # Add composition stats
        result = self.composition_calc.calculate_for_season(
            league_stats, team_stats, season
        )
        result["SEASON"] = season
        return result

    def process_all_seasons(
        self,
        seasons: list[str] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Process all seasons and compute composition stats.

        Args:
            seasons: List of seasons to process, or None for all cached
            verbose: Print progress

        Returns:
            Combined DataFrame with all player-seasons
        """
        if verbose:
            print("Loading team stats...")
        team_stats = self.load_team_stats_dict()

        if verbose:
            print(f"Loaded stats for {len(team_stats)} teams")

        if seasons is None:
            # Find all cached league stats
            seasons = []
            for year in range(2020, 2026):
                season = f"{year}-{str(year+1)[-2:]}"
                key = f"league_stats_{season}"
                if self.cache.has(key):
                    seasons.append(season)

        all_data = []
        for i, season in enumerate(seasons):
            if verbose:
                print(f"[{i+1}/{len(seasons)}] Processing {season}...")
            df = self.process_season(season, team_stats)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        if verbose:
            print(f"Processed {len(combined)} player-seasons")

        return combined

    def build_player_features(self, combined_stats: pd.DataFrame) -> pd.DataFrame:
        """Build aggregated player features from season data.

        Creates per-player feature vectors with per-game averages.

        Args:
            combined_stats: Combined player-season data

        Returns:
            DataFrame with one row per player
        """
        # First compute per-game stats for each season
        df = combined_stats.copy()

        # Stats to convert to per-game
        counting_stats = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
                         "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                         "OREB", "DREB"]

        for col in counting_stats:
            if col in df.columns and "GP" in df.columns:
                df[col] = df[col] / df["GP"].replace(0, 1)

        # Columns to aggregate (now per-game)
        numeric_cols = [
            "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "pts_share", "ast_share", "reb_share", "stl_share", "blk_share",
            "fg3a_share", "min_share",
        ]

        # Filter to columns that exist
        agg_cols = [c for c in numeric_cols if c in df.columns]

        # Group by player and compute means across seasons
        player_features = df.groupby(["PLAYER_ID", "PLAYER_NAME"])[agg_cols].mean()
        player_features = player_features.reset_index()

        # Add season count and total games
        season_counts = combined_stats.groupby("PLAYER_ID").agg(
            season_count=("SEASON", "nunique"),
            total_gp=("GP", "sum")
        ).reset_index()
        player_features = player_features.merge(season_counts, on="PLAYER_ID")

        # Compute derived stats
        if "FGA" in player_features.columns and "FGM" in player_features.columns:
            player_features["fg_pct"] = player_features["FGM"] / player_features["FGA"].replace(0, 1)

        if "FG3A" in player_features.columns and "FG3M" in player_features.columns:
            player_features["fg3_pct"] = player_features["FG3M"] / player_features["FG3A"].replace(0, 1)

        if "FTA" in player_features.columns and "FTM" in player_features.columns:
            player_features["ft_pct"] = player_features["FTM"] / player_features["FTA"].replace(0, 1)

        # True shooting percentage approximation
        if all(c in player_features.columns for c in ["PTS", "FGA", "FTA"]):
            tsa = player_features["FGA"] + 0.44 * player_features["FTA"]
            player_features["ts_pct"] = player_features["PTS"] / (2 * tsa.replace(0, 1))

        return player_features

    def run(
        self,
        seasons: list[str] | None = None,
        output_path: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run the full feature pipeline.

        Args:
            seasons: Seasons to process (None for all cached)
            output_path: Path to save features (optional)
            verbose: Print progress

        Returns:
            Player feature DataFrame
        """
        if verbose:
            print("=== Feature Pipeline ===\n")

        # Process seasons
        combined = self.process_all_seasons(seasons, verbose)
        if combined.empty:
            print("No data to process!")
            return pd.DataFrame()

        # Build player features
        if verbose:
            print("\nBuilding player features...")
        features = self.build_player_features(combined)

        if verbose:
            print(f"Generated features for {len(features)} players")

        # Save if path provided
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            features.to_parquet(path, index=False)
            if verbose:
                print(f"Saved to {path}")
        else:
            # Save to default location
            default_path = self.cache.features_dir / "player_features.parquet"
            features.to_parquet(default_path, index=False)
            if verbose:
                print(f"Saved to {default_path}")

        # Also save the combined season data
        combined_path = self.cache.processed_dir / "player_seasons.parquet"
        combined.to_parquet(combined_path, index=False)
        if verbose:
            print(f"Saved season data to {combined_path}")

        if verbose:
            print("\n=== Pipeline complete ===")

        return features


def main():
    parser = argparse.ArgumentParser(description="Feature generation pipeline")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons to process")
    parser.add_argument("--output", type=str, help="Output path for features")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    pipeline = FeaturePipeline()
    pipeline.run(
        seasons=args.seasons,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
