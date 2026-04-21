"""Compute player's share of team production.

Composition stats normalize across eras by expressing player
contribution as percentage of team totals:
- pts_share: player_pts / team_pts
- ast_share: player_ast / team_ast
- reb_share: player_reb / team_reb
- 3pa_share: player_3pa / team_3pa

"25% of team assists" is comparable across decades.
"""

import pandas as pd


class CompositionStatsCalculator:
    """Calculate player's percentage of team stats."""

    # Mapping from player stat columns to team stat columns
    STAT_MAPPINGS = {
        "pts_share": ("PTS", "PTS"),
        "ast_share": ("AST", "AST"),
        "reb_share": ("REB", "REB"),
        "oreb_share": ("OREB", "OREB"),
        "dreb_share": ("DREB", "DREB"),
        "stl_share": ("STL", "STL"),
        "blk_share": ("BLK", "BLK"),
        "tov_share": ("TOV", "TOV"),
        "fga_share": ("FGA", "FGA"),
        "fgm_share": ("FGM", "FGM"),
        "fg3a_share": ("FG3A", "FG3A"),
        "fg3m_share": ("FG3M", "FG3M"),
        "fta_share": ("FTA", "FTA"),
        "ftm_share": ("FTM", "FTM"),
        "min_share": ("MIN", "MIN"),
    }

    def __init__(self):
        pass

    def calculate_from_league_and_team_stats(
        self,
        league_stats: pd.DataFrame,
        team_stats_dict: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """Calculate composition stats using league player stats and team stats.

        Args:
            league_stats: League-wide player stats for a season (from LeagueDashPlayerStats)
            team_stats_dict: Dict mapping team_id to team stats DataFrame

        Returns:
            DataFrame with composition stat columns added
        """
        result = league_stats.copy()

        # Initialize composition columns
        for col in self.STAT_MAPPINGS.keys():
            result[col] = 0.0

        # Get the season from the league stats if available, otherwise we need it passed in
        # LeagueDashPlayerStats doesn't have a season column, so we process one season at a time

        for idx, row in result.iterrows():
            team_id = row.get("TEAM_ID")
            if team_id is None or team_id not in team_stats_dict:
                continue

            team_df = team_stats_dict[team_id]
            if team_df.empty:
                continue

            # Calculate shares for each stat
            for share_col, (player_col, team_col) in self.STAT_MAPPINGS.items():
                player_val = row.get(player_col, 0) or 0
                # Team stats are totals - we need to find the matching season
                # For now, use the most recent season in team stats
                team_val = team_df[team_col].iloc[-1] if team_col in team_df.columns else 0

                if team_val and team_val > 0:
                    result.at[idx, share_col] = player_val / team_val
                else:
                    result.at[idx, share_col] = 0.0

        return result

    def calculate_for_season(
        self,
        league_stats: pd.DataFrame,
        team_stats_dict: dict[int, pd.DataFrame],
        season: str,
    ) -> pd.DataFrame:
        """Calculate composition stats for a specific season.

        Args:
            league_stats: League-wide player stats for the season
            team_stats_dict: Dict mapping team_id to team historical stats
            season: Season string like '2023-24'

        Returns:
            DataFrame with composition stats added
        """
        result = league_stats.copy()

        # Initialize composition columns
        for col in self.STAT_MAPPINGS.keys():
            result[col] = 0.0

        # Convert season to format used in team stats (e.g., "2023-24")
        for idx, row in result.iterrows():
            team_id = row.get("TEAM_ID")
            if team_id is None or team_id not in team_stats_dict:
                continue

            team_df = team_stats_dict[team_id]
            if team_df.empty:
                continue

            # Find matching season in team stats
            # Team stats use "YEAR" column like "2023-24"
            season_match = team_df[team_df["YEAR"] == season] if "YEAR" in team_df.columns else None

            if season_match is None or season_match.empty:
                continue

            team_season = season_match.iloc[0]

            # Calculate shares
            for share_col, (player_col, team_col) in self.STAT_MAPPINGS.items():
                player_val = row.get(player_col, 0) or 0
                team_val = team_season.get(team_col, 0) or 0

                if team_val and team_val > 0:
                    result.at[idx, share_col] = player_val / team_val
                else:
                    result.at[idx, share_col] = 0.0

        return result

    def calculate_from_career_stats(
        self,
        player_career: pd.DataFrame,
        team_stats_dict: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """Calculate composition stats for a player's career.

        Args:
            player_career: Player's season-by-season career stats
            team_stats_dict: Dict mapping team_id to team historical stats

        Returns:
            DataFrame with composition stats added for each season
        """
        result = player_career.copy()

        # Initialize composition columns
        for col in self.STAT_MAPPINGS.keys():
            result[col] = 0.0

        for idx, row in result.iterrows():
            team_id = row.get("TEAM_ID")
            season = row.get("SEASON_ID")  # Format like "2023-24"

            if team_id is None or team_id not in team_stats_dict:
                continue

            team_df = team_stats_dict[team_id]
            if team_df.empty:
                continue

            # Find matching season
            season_match = team_df[team_df["YEAR"] == season] if "YEAR" in team_df.columns else None

            if season_match is None or season_match.empty:
                continue

            team_season = season_match.iloc[0]

            # Calculate shares
            for share_col, (player_col, team_col) in self.STAT_MAPPINGS.items():
                player_val = row.get(player_col, 0) or 0
                team_val = team_season.get(team_col, 0) or 0

                if team_val and team_val > 0:
                    result.at[idx, share_col] = player_val / team_val
                else:
                    result.at[idx, share_col] = 0.0

        return result

    @staticmethod
    def get_composition_columns() -> list[str]:
        """Return list of composition stat column names."""
        return list(CompositionStatsCalculator.STAT_MAPPINGS.keys())
