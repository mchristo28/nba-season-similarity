"""Era and pace normalization for cross-era comparisons.

Optional adjustments to normalize stats across different NBA eras:
- Pace adjustment (possessions per game varied significantly)
- Rule change normalization
- League average relative stats
"""

import pandas as pd


class EraAdjuster:
    """Normalize stats across different NBA eras."""

    def __init__(self):
        pass

    def pace_adjust(self, stats: pd.DataFrame, target_pace: float = 100.0) -> pd.DataFrame:
        """Adjust stats to a standard pace (per 100 possessions).

        Args:
            stats: Player stats with pace column
            target_pace: Target pace to normalize to

        Returns:
            Pace-adjusted stats
        """
        raise NotImplementedError

    def league_relative(self, stats: pd.DataFrame, league_averages: pd.DataFrame) -> pd.DataFrame:
        """Express stats relative to league average for that season.

        Args:
            stats: Player stats
            league_averages: League averages by season

        Returns:
            Stats expressed as % of league average
        """
        raise NotImplementedError
