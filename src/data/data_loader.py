"""Unified data loading interface.

Provides a single interface to load player and team data,
abstracting away the caching layer and API clients.
"""

import argparse

import pandas as pd

from .cache_manager import CacheManager
from .nba_api_client import NBAApiClient


class DataLoader:
    """Unified interface for loading player and team data.

    Handles:
    - Cache checking before API calls
    - Automatic caching of fetched data
    - Rate limiting via underlying clients
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache = CacheManager(cache_dir)
        self.client = NBAApiClient()

    def get_player_career(self, player_id: int) -> pd.DataFrame:
        """Load player's career stats (from cache or API)."""
        cached = self.cache.get_player_stats(player_id)
        if cached is not None:
            return cached

        data = self.client.get_player_career_stats(player_id)
        self.cache.store_player_stats(player_id, data)
        return data

    def get_player_info(self, player_id: int) -> pd.DataFrame:
        """Load player metadata (from cache or API)."""
        cached = self.cache.get_player_info(player_id)
        if cached is not None:
            return cached

        data = self.client.get_player_info(player_id)
        self.cache.store_player_info(player_id, data)
        return data

    def get_league_stats(self, season: str) -> pd.DataFrame:
        """Load league-wide player stats for a season."""
        key = f"league_stats_{season}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        data = self.client.get_league_player_stats(season)
        self.cache.store(key, data, entry_type="league_stats")
        return data

    def get_tracking_stats(self, season: str) -> pd.DataFrame:
        """Load tracking stats for a season (drives, catch-shoot, pull-up, passing, possessions)."""
        key = f"tracking_stats_{season}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        data = self.client.get_all_tracking_stats(season)
        if not data.empty:
            self.cache.store(key, data, entry_type="tracking_stats")
        return data

    def get_scoring_stats(self, season: str) -> pd.DataFrame:
        """Load scoring stats for a season (assisted/unassisted breakdown)."""
        key = f"scoring_stats_{season}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        data = self.client.get_league_scoring_stats(season)
        if not data.empty:
            self.cache.store(key, data, entry_type="scoring_stats")
        return data

    def get_league_stats_with_tracking(self, season: str) -> pd.DataFrame:
        """Load league stats merged with tracking and scoring stats for a season."""
        league = self.get_league_stats(season)

        if league is None or league.empty:
            return pd.DataFrame()

        merged = league.copy()

        # Merge tracking stats
        tracking = self.get_tracking_stats(season)
        if tracking is not None and not tracking.empty:
            merged = merged.merge(
                tracking.drop(columns=['PLAYER_NAME'], errors='ignore'),
                on='PLAYER_ID',
                how='left'
            )

        # Merge scoring stats (assisted/unassisted)
        scoring = self.get_scoring_stats(season)
        if scoring is not None and not scoring.empty:
            # Only keep the assisted/unassisted columns
            scoring_cols = ['PLAYER_ID', 'PCT_AST_2PM', 'PCT_UAST_2PM',
                           'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM']
            scoring_cols = [c for c in scoring_cols if c in scoring.columns]
            scoring_subset = scoring[scoring_cols]
            merged = merged.merge(scoring_subset, on='PLAYER_ID', how='left')

        return merged

    def get_team_stats(self, team_id: int) -> pd.DataFrame:
        """Load all season stats for a team."""
        key = f"team_{team_id}_all_seasons"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        data = self.client.get_team_season_stats(team_id)
        self.cache.store(key, data, entry_type="team_stats")
        return data

    def get_all_players(self) -> list[dict]:
        """Get list of all players with IDs."""
        return self.client.get_all_players()

    def get_active_players(self) -> list[dict]:
        """Get list of active players."""
        return self.client.get_active_players()

    def get_all_cached_player_stats(self) -> pd.DataFrame | None:
        """Load all cached player stats."""
        return self.cache.get_all_player_stats()

    def get_all_cached_player_info(self) -> pd.DataFrame | None:
        """Load all cached player info."""
        return self.cache.get_all_player_info()

    def pull_seasons(self, seasons: list[str], verbose: bool = True):
        """Pull league-wide stats for specified seasons.

        Args:
            seasons: List of season strings like ['2023-24', '2022-23']
            verbose: Print progress
        """
        for i, season in enumerate(seasons):
            if verbose:
                print(f"[{i+1}/{len(seasons)}] Fetching {season} league stats...")
            self.get_league_stats(season)
        if verbose:
            print("Done fetching league stats.")

    def pull_players(self, player_ids: list[int], verbose: bool = True):
        """Pull career stats and info for specified players.

        Args:
            player_ids: List of player IDs
            verbose: Print progress
        """
        total = len(player_ids)
        for i, pid in enumerate(player_ids):
            if verbose and (i % 50 == 0 or i == total - 1):
                print(f"[{i+1}/{total}] Fetching player {pid}...")

            # Skip if already cached
            if not self.cache.has_player_stats(pid):
                try:
                    self.get_player_career(pid)
                except Exception as e:
                    if verbose:
                        print(f"  Error fetching stats for {pid}: {e}")

            if not self.cache.has_player_info(pid):
                try:
                    self.get_player_info(pid)
                except Exception as e:
                    if verbose:
                        print(f"  Error fetching info for {pid}: {e}")

        if verbose:
            print("Done fetching player data.")

    def pull_teams(self, verbose: bool = True):
        """Pull stats for all teams."""
        teams = self.client.get_all_teams()
        for i, team in enumerate(teams):
            if verbose:
                print(f"[{i+1}/{len(teams)}] Fetching {team['full_name']}...")
            self.get_team_stats(team["id"])
        if verbose:
            print("Done fetching team stats.")

    def full_pull(
        self,
        start_year: int = 2020,
        end_year: int = 2024,
        verbose: bool = True,
    ):
        """Pull all data for the specified year range.

        This fetches:
        1. League-wide player stats for each season
        2. Team stats for all teams
        3. Career stats for players active in the date range

        Args:
            start_year: Start year (e.g., 2020 for 2020-21 season)
            end_year: End year (e.g., 2024 for 2024-25 season)
            verbose: Print progress
        """
        # Build season strings
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

        if verbose:
            print(f"=== Full data pull: {seasons[0]} to {seasons[-1]} ===\n")

        # 1. Fetch league stats for each season
        if verbose:
            print("Step 1: Fetching league-wide stats...")
        self.pull_seasons(seasons, verbose=verbose)

        # 2. Fetch team stats
        if verbose:
            print("\nStep 2: Fetching team stats...")
        self.pull_teams(verbose=verbose)

        # 3. Get player IDs from league stats and fetch their careers
        if verbose:
            print("\nStep 3: Collecting player IDs from league stats...")

        player_ids = set()
        for season in seasons:
            df = self.get_league_stats(season)
            if df is not None and "PLAYER_ID" in df.columns:
                player_ids.update(df["PLAYER_ID"].unique())

        player_ids = sorted(player_ids)
        if verbose:
            print(f"Found {len(player_ids)} unique players.\n")
            print("Step 4: Fetching player career stats and info...")

        self.pull_players(player_ids, verbose=verbose)

        if verbose:
            print("\n=== Full pull complete ===")


def main():
    parser = argparse.ArgumentParser(description="NBA data loader")
    parser.add_argument("--full-pull", action="store_true", help="Pull all historical data")
    parser.add_argument("--start-year", type=int, default=2020, help="Start year (default: 2020)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument("--seasons", nargs="+", help="Specific seasons to pull (e.g., 2023-24)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    loader = DataLoader()

    if args.full_pull:
        loader.full_pull(args.start_year, args.end_year, verbose=not args.quiet)
    elif args.seasons:
        loader.pull_seasons(args.seasons, verbose=not args.quiet)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
