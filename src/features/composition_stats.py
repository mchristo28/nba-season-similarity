"""Season-aligned shares based on actual team production."""

import pandas as pd

from .transforms import COUNTING_STATS, team_shares


class CompositionStatsCalculator:
    STAT_MAPPINGS = {f"{stat.lower()}_share": (stat, stat) for stat in COUNTING_STATS}

    @staticmethod
    def _teams(team_stats_dict):
        frames = []
        for team_id, frame in team_stats_dict.items():
            if "YEAR" not in frame:
                continue
            frame = frame.rename(columns={"YEAR": "SEASON"}).copy()
            frame["TEAM_ID"] = team_id
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def calculate_from_league_and_team_stats(self, league_stats, team_stats_dict):
        if "SEASON" not in league_stats:
            raise ValueError("SEASON is required to align team production")
        return team_shares(league_stats, self._teams(team_stats_dict), player_mode="Totals")

    def calculate_for_season(self, league_stats, team_stats_dict, season):
        return self.calculate_from_league_and_team_stats(
            league_stats.assign(SEASON=season), team_stats_dict
        )

    def calculate_from_career_stats(self, player_career, team_stats_dict):
        return self.calculate_from_league_and_team_stats(
            player_career.rename(columns={"SEASON_ID": "SEASON"}), team_stats_dict
        )

    @staticmethod
    def get_composition_columns():
        return list(CompositionStatsCalculator.STAT_MAPPINGS)
