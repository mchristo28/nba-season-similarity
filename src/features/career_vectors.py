"""Year-by-year stat embeddings for trajectory matching.

Creates career-year features:
- Year 1 stats for all players (rookie seasons)
- Year 2 stats for all players (sophomore seasons)
- etc.

Enables matching "developing like Player X" patterns by comparing
year-by-year development rather than career averages.
"""

import pandas as pd

from .transforms import add_career_year, efficiency_stats, per_game


def fetch_player_rookie_years() -> dict[int, int]:
    """Fetch FROM_YEAR (rookie year) for all NBA players.

    Returns:
        Dict mapping player_id -> rookie_year (e.g., {2544: 2003} for LeBron)
    """
    import time

    from nba_api.stats.endpoints import commonallplayers

    print("Fetching player rookie years from NBA API...")
    time.sleep(0.6)  # Rate limit
    players = commonallplayers.CommonAllPlayers(is_only_current_season=0)
    df = players.get_data_frames()[0]

    rookie_years = {}
    for _, row in df.iterrows():
        player_id = row["PERSON_ID"]
        from_year = row["FROM_YEAR"]
        if pd.notna(from_year):
            rookie_years[player_id] = int(from_year)

    print(f"  Found rookie years for {len(rookie_years)} players")
    return rookie_years


class CareerVectorBuilder:
    """Build career trajectory vectors for similarity matching."""

    # Stats to include in career vectors (will be per-game)
    STAT_COLS = [
        "PTS",
        "AST",
        "REB",
        "STL",
        "BLK",
        "TOV",
        "MIN",
        "FGM",
        "FGA",
        "FG3M",
        "FG3A",
        "FTM",
        "FTA",
    ]

    # Composition stats
    COMPOSITION_COLS = [
        "pts_share",
        "ast_share",
        "reb_share",
        "min_share",
        "stl_share",
        "blk_share",
        "fg3a_share",
    ]

    # Derived stats
    DERIVED_COLS = ["fg_pct", "fg3_pct", "ft_pct", "ts_pct"]

    def __init__(self, rookie_years: dict[int, int] | None = None):
        """Initialize the builder.

        Args:
            rookie_years: Dict mapping player_id -> rookie_year. If None, will fetch from API.
        """
        self.rookie_years = rookie_years

    def add_career_year(
        self, player_seasons: pd.DataFrame, rookie_years: dict[int, int] | None = None
    ) -> pd.DataFrame:
        """Add career year number to each player-season.

        Uses actual rookie year to compute true career year:
        CAREER_YEAR = season_start_year - rookie_year + 1

        Args:
            player_seasons: DataFrame with PLAYER_ID, SEASON columns
            rookie_years: Dict mapping player_id -> rookie_year (overrides self.rookie_years)

        Returns:
            DataFrame with CAREER_YEAR column added (1 = rookie, 2 = sophomore, etc.)
        """
        df = player_seasons.copy()
        years = rookie_years or self.rookie_years

        if years is None:
            raise ValueError("rookie_years required - pass to __init__ or add_career_year")

        return add_career_year(df, years)

    def compute_per_game_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        return efficiency_stats(per_game(df, self.STAT_COLS))

    def build_career_year_features(
        self,
        player_seasons: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for each player's career years.

        Args:
            player_seasons: Raw season data with composition stats

        Returns:
            DataFrame with one row per player-career_year,
            containing per-game stats and composition metrics
        """
        # Add career year
        df = self.add_career_year(player_seasons)

        # Convert to per-game
        df = self.compute_per_game_stats(df)

        # Select feature columns
        feature_cols = (
            ["PLAYER_ID", "PLAYER_NAME", "SEASON", "CAREER_YEAR", "GP", "AGE"]
            + [c for c in self.STAT_COLS if c in df.columns]
            + [c for c in self.COMPOSITION_COLS if c in df.columns]
            + [c for c in self.DERIVED_COLS if c in df.columns]
        )

        return df[[c for c in feature_cols if c in df.columns]]

    def get_player_trajectory(
        self,
        career_features: pd.DataFrame,
        player_id: int,
    ) -> pd.DataFrame:
        """Get a player's full career trajectory."""
        return career_features[career_features["PLAYER_ID"] == player_id].sort_values("CAREER_YEAR")

    def get_players_by_career_length(
        self,
        career_features: pd.DataFrame,
        min_years: int,
        max_years: int | None = None,
    ) -> list[int]:
        """Get player IDs with career length in specified range."""
        career_lengths = career_features.groupby("PLAYER_ID")["CAREER_YEAR"].max()

        if max_years:
            mask = (career_lengths >= min_years) & (career_lengths <= max_years)
        else:
            mask = career_lengths >= min_years

        return career_lengths[mask].index.tolist()


def build_career_features(
    seasons_path: str = "data/processed/player_seasons.parquet",
    output_path: str = "data/features/career_year_features.parquet",
) -> pd.DataFrame:
    """Build and save career year features.

    Args:
        seasons_path: Path to player seasons data
        output_path: Path to save career features

    Returns:
        Career year features DataFrame
    """
    print("Loading player seasons...")
    seasons = pd.read_parquet(seasons_path)

    # Fetch actual rookie years from NBA API
    rookie_years = fetch_player_rookie_years()

    print("Building career year features...")
    builder = CareerVectorBuilder(rookie_years=rookie_years)
    features = builder.build_career_year_features(seasons)

    print(f"Built features for {features['PLAYER_ID'].nunique()} players")
    print(f"Career years range: 1 to {features['CAREER_YEAR'].max()}")

    # Show some examples
    sample_players = ["LeBron James", "Stephen Curry", "Jayson Tatum"]
    for name in sample_players:
        player_data = features[features["PLAYER_NAME"] == name]
        if not player_data.empty:
            years = player_data["CAREER_YEAR"].tolist()
            print(f"  {name}: Career years {min(years)}-{max(years)}")

    # Save
    features.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    return features


if __name__ == "__main__":
    build_career_features()
