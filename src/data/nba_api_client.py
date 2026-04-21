"""NBA API client for fetching player and team data from stats.nba.com."""

import time
from functools import wraps

import pandas as pd
from nba_api.stats.endpoints import (
    CommonPlayerInfo,
    LeagueDashPlayerStats,
    LeagueDashPtStats,
    PlayerAwards,
    PlayerCareerStats,
    PlayerDashboardByShootingSplits,
    TeamYearByYearStats,
)
from nba_api.stats.static import players, teams


_client_instance = None


def get_client() -> "NBAApiClient":
    """Get or create singleton NBAApiClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = NBAApiClient()
    return _client_instance


def rate_limit(min_interval=0.6):
    """Decorator to enforce minimum time between calls."""
    last_call = [0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result

        return wrapper

    return decorator


class NBAApiClient:
    """Client for fetching data from stats.nba.com via nba_api.

    Key endpoints used:
    - playercareerstats: Season-by-season stats
    - commonplayerinfo: Height, weight, draft info, position
    - teamyearbyyearstats: Team totals for composition calc
    - leaguedashplayerstats: League-wide player stats by season
    - draftcombinestats: Wingspan, vertical, etc. (2000+)
    """

    def __init__(self):
        self.all_players = players.get_players()
        self.all_teams = teams.get_teams()

    @rate_limit(0.6)
    def get_player_career_stats(self, player_id: int) -> pd.DataFrame:
        """Fetch career stats for a player.

        Returns DataFrame with season-by-season regular season stats.
        """
        career = PlayerCareerStats(player_id=player_id)
        # Get regular season totals
        df = career.get_data_frames()[0]  # SeasonTotalsRegularSeason
        df["PLAYER_ID"] = player_id
        return df

    @rate_limit(0.6)
    def get_player_info(self, player_id: int) -> pd.DataFrame:
        """Fetch player metadata (height, weight, position, draft info)."""
        info = CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[0]  # CommonPlayerInfo
        return df

    @rate_limit(0.6)
    def get_team_season_stats(self, team_id: int) -> pd.DataFrame:
        """Fetch all season stats for a team."""
        stats = TeamYearByYearStats(team_id=team_id)
        df = stats.get_data_frames()[0]
        return df

    @rate_limit(0.6)
    def get_league_player_stats(self, season: str) -> pd.DataFrame:
        """Fetch league-wide player stats for a season.

        Args:
            season: Season string like '2023-24'
        """
        stats = LeagueDashPlayerStats(season=season)
        df = stats.get_data_frames()[0]
        return df

    @rate_limit(0.6)
    def get_league_scoring_stats(self, season: str) -> pd.DataFrame:
        """Fetch league-wide scoring stats including assisted/unassisted.

        Returns stats like PCT_AST_2PM, PCT_UAST_2PM, PCT_AST_3PM, PCT_UAST_3PM, etc.
        """
        stats = LeagueDashPlayerStats(season=season, measure_type_detailed_defense='Scoring')
        df = stats.get_data_frames()[0]
        return df

    @rate_limit(0.6)
    def get_player_awards(self, player_id: int) -> pd.DataFrame:
        """Fetch all awards/accolades for a player.

        Returns DataFrame with columns: DESCRIPTION, SEASON, ALL_NBA_TEAM_NUMBER, etc.
        Key award types: 'NBA All-Star', 'All-NBA', 'NBA Champion', 'NBA Most Valuable Player'
        """
        awards = PlayerAwards(player_id=player_id)
        df = awards.get_data_frames()[0]
        return df

    @rate_limit(0.6)
    def get_tracking_stats(self, season: str, measure_type: str) -> pd.DataFrame:
        """Fetch league-wide tracking stats for a season.

        Args:
            season: Season string like '2023-24'
            measure_type: One of 'Drives', 'CatchShoot', 'PullUpShot', 'Passing', 'Possessions'

        Returns:
            DataFrame with tracking stats for all players
        """
        stats = LeagueDashPtStats(
            season=season,
            player_or_team='Player',
            pt_measure_type=measure_type
        )
        df = stats.get_data_frames()[0]
        return df

    @rate_limit(0.6)
    def get_player_shooting_splits(self, player_id: int, season: str) -> pd.DataFrame:
        """Fetch shooting splits including assisted/unassisted for a player-season.

        Returns the overall stats row with PCT_AST_2PM, PCT_UAST_2PM, etc.
        """
        splits = PlayerDashboardByShootingSplits(player_id=player_id, season=season)
        df = splits.get_data_frames()[0]  # Overall splits
        return df

    def get_all_tracking_stats(self, season: str) -> pd.DataFrame:
        """Fetch all tracking stat types and merge into one DataFrame.

        Fetches: Drives, CatchShoot, PullUpShot, Passing, Possessions
        Merges on PLAYER_ID.
        """
        measure_types = ['Drives', 'CatchShoot', 'PullUpShot', 'Passing', 'Possessions']

        merged = None
        for measure in measure_types:
            try:
                df = self.get_tracking_stats(season, measure)

                # Keep only relevant columns (drop duplicates like GP, W, L, MIN)
                keep_cols = ['PLAYER_ID', 'PLAYER_NAME']
                stat_cols = [c for c in df.columns if c not in
                            ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                             'GP', 'W', 'L', 'MIN']]
                keep_cols.extend(stat_cols)
                df = df[keep_cols]

                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df.drop(columns=['PLAYER_NAME']),
                                         on='PLAYER_ID', how='outer')
            except Exception as e:
                print(f"Warning: Could not fetch {measure} stats: {e}")
                continue

        return merged if merged is not None else pd.DataFrame()

    def get_all_players(self) -> list[dict]:
        """Get static list of all players with IDs."""
        return self.all_players

    def get_all_teams(self) -> list[dict]:
        """Get static list of all teams with IDs."""
        return self.all_teams

    def find_player_by_name(self, name: str) -> dict | None:
        """Find a player by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for player in self.all_players:
            if name_lower in player["full_name"].lower():
                return player
        return None

    def find_players_by_name(self, name: str) -> list[dict]:
        """Find all players matching name (case-insensitive partial match)."""
        name_lower = name.lower()
        return [p for p in self.all_players if name_lower in p["full_name"].lower()]

    def get_active_players(self) -> list[dict]:
        """Get list of currently active players."""
        return [p for p in self.all_players if p["is_active"]]
