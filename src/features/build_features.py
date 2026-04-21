"""Build feature set from comprehensive stats for similarity matching."""

import pandas as pd
from pathlib import Path


def convert_tracking_totals_to_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tracking stats from season totals to per-game values.

    Many tracking stats from NBA API are season totals, not per-game.
    This converts them to per-game for fair comparison across seasons.
    """
    # Stats that are season TOTALS and need to be converted to per-game
    # (excludes percentages and already-per-touch stats)
    total_stats = [
        # Drives
        "DRIVES", "DRIVE_FGA", "DRIVE_FGM", "DRIVE_PTS", "DRIVE_AST",
        "DRIVE_TOV", "DRIVE_FTA", "DRIVE_FTM", "DRIVE_PASSES", "DRIVE_PF",
        # Pull-up shots
        "PULL_UP_FGA", "PULL_UP_FGM", "PULL_UP_FG3A", "PULL_UP_FG3M", "PULL_UP_PTS",
        # Catch and shoot
        "CATCH_SHOOT_FGA", "CATCH_SHOOT_FGM", "CATCH_SHOOT_FG3A",
        "CATCH_SHOOT_FG3M", "CATCH_SHOOT_PTS",
        # Passing
        "PASSES_MADE", "PASSES_RECEIVED", "POTENTIAL_AST", "AST_PTS_CREATED",
        # Touches
        "TOUCHES", "FRONT_CT_TOUCHES", "ELBOW_TOUCHES", "PAINT_TOUCHES", "POST_TOUCHES",
        # Time of possession (in minutes, convert to per-game)
        "TIME_OF_POSS",
    ]

    # Only convert stats that exist in the dataframe
    stats_to_convert = [s for s in total_stats if s in df.columns]

    if "GP" not in df.columns:
        print("Warning: GP column not found, cannot convert to per-game")
        return df

    # Convert each stat to per-game
    for stat in stats_to_convert:
        # Avoid division by zero
        df[stat] = df[stat] / df["GP"].replace(0, 1)

    print(f"Converted {len(stats_to_convert)} tracking stats to per-game values")
    return df


def compute_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year delta features for trajectory matching.

    For each player, computes both:
    - Raw deltas (e.g., +3 PPG)
    - Percentage change (e.g., +25% scoring increase)

    This captures growth/development patterns.
    """
    # Stats to compute deltas for
    delta_stats = [
        "PTS", "AST", "REB", "STL", "BLK",  # counting stats
        "ts_pct", "efg_pct",  # efficiency
        "MIN", "FGA", "FG3A",  # volume/role
        "e_usg_pct", "e_off_rating", "e_def_rating",  # advanced
    ]

    # Only use stats that exist in the dataframe
    delta_stats = [s for s in delta_stats if s in df.columns]

    # Sort by player and career year
    df = df.sort_values(["PLAYER_ID", "CAREER_YEAR"])

    # Compute deltas within each player
    for stat in delta_stats:
        # Raw delta (e.g., +3 PPG)
        delta_col = f"{stat}_delta"
        df[delta_col] = df.groupby("PLAYER_ID")[stat].diff()

        # Percentage change (with floor to avoid division issues)
        # Use a minimum of 1.0 for counting stats, 0.01 for percentages
        pct_col = f"{stat}_pct_change"
        prev_val = df.groupby("PLAYER_ID")[stat].shift(1)

        if stat in ["ts_pct", "efg_pct", "e_usg_pct"]:
            # For percentage stats, use small floor
            floor = 0.01
        else:
            # For counting stats, use 1.0 floor
            floor = 1.0

        df[pct_col] = df[delta_col] / prev_val.clip(lower=floor)
        # Cap extreme values (e.g., going from 0 to 5 shouldn't be "infinite")
        df[pct_col] = df[pct_col].clip(lower=-2.0, upper=2.0)

    return df


def build_features(
    input_path: str = "data/processed/comprehensive_stats.parquet",
    output_path: str = "data/features/player_features.parquet",
) -> pd.DataFrame:
    """Build and save player features from comprehensive stats.

    Args:
        input_path: Path to comprehensive stats parquet
        output_path: Path to save features

    Returns:
        Features DataFrame
    """
    print("Loading comprehensive stats...")
    df = pd.read_parquet(input_path)

    print(f"Loaded {len(df)} player-seasons")

    # Convert tracking totals to per-game values
    print("Converting tracking stats to per-game...")
    df = convert_tracking_totals_to_per_game(df)

    # Add trajectory features (year-over-year deltas)
    print("Computing trajectory features...")
    df = compute_trajectory_features(df)

    # Columns we want to keep for features
    id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "SEASON", "CAREER_YEAR", "AGE", "GP"]

    # Basic box score stats
    box_score_cols = [
        "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "PLUS_MINUS",
    ]

    # Bio/physical (only height and weight - draft info and country don't affect production)
    bio_cols = ["height_inches", "weight"]

    # Shot location stats
    shot_cols = [c for c in df.columns if any(x in c.lower() for x in ["restricted", "paint", "mid_range", "corner", "above_the_break"])]

    # Hustle stats
    hustle_cols = [c for c in df.columns if any(x in c.lower() for x in ["contested", "deflection", "charge", "loose_ball", "screen_assist", "box_out"])]

    # Advanced/estimated metrics
    advanced_cols = [c for c in df.columns if c.startswith("e_") and not c.endswith("_rank")]

    # Team share stats
    share_cols = [c for c in df.columns if c.endswith("_share")]

    # Derived efficiency stats
    efficiency_cols = ["ts_pct", "efg_pct", "fg_pct", "fg3_pct", "ft_pct"]

    # Shot profile (% from each zone)
    profile_cols = [c for c in df.columns if c.startswith("pct_fga_") or c.startswith("fg_pct_")]

    # Tracking stats (drives, catch-shoot, pull-up, touches, passing, assisted/unassisted)
    tracking_cols = [c for c in df.columns if any(x in c for x in
        ['DRIVE', 'CATCH_SHOOT', 'PULL_UP', 'TOUCHES', 'TOUCH', 'PASSES', 'POTENTIAL_AST',
         'AST_PTS_CREATED', 'TIME_OF_POSS', 'DRIB', 'PCT_UAST', 'PCT_AST_'])]

    # Trajectory features (year-over-year deltas and percentage changes)
    trajectory_cols = [c for c in df.columns if c.endswith("_delta") or c.endswith("_pct_change")]

    # Combine all columns
    all_feature_cols = (
        id_cols +
        [c for c in box_score_cols if c in df.columns] +
        [c for c in bio_cols if c in df.columns] +
        [c for c in shot_cols if c in df.columns] +
        [c for c in hustle_cols if c in df.columns] +
        [c for c in advanced_cols if c in df.columns] +
        [c for c in share_cols if c in df.columns] +
        [c for c in efficiency_cols if c in df.columns] +
        [c for c in profile_cols if c in df.columns] +
        [c for c in tracking_cols if c in df.columns] +
        [c for c in trajectory_cols if c in df.columns]
    )

    # Remove duplicates while preserving order
    seen = set()
    unique_cols = []
    for c in all_feature_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            unique_cols.append(c)

    features = df[unique_cols].copy()

    print(f"Selected {len(unique_cols)} feature columns")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

    # Print summary
    print(f"\n=== Feature Summary ===")
    print(f"Total rows: {len(features)}")
    print(f"Unique players: {features['PLAYER_ID'].nunique()}")
    print(f"Career year range: {features['CAREER_YEAR'].min()} to {features['CAREER_YEAR'].max()}")
    print(f"\nFeature categories:")
    print(f"  ID columns: {len(id_cols)}")
    print(f"  Box score: {len([c for c in box_score_cols if c in df.columns])}")
    print(f"  Bio/physical: {len([c for c in bio_cols if c in df.columns])}")
    print(f"  Shot location: {len([c for c in shot_cols if c in df.columns])}")
    print(f"  Hustle: {len([c for c in hustle_cols if c in df.columns])}")
    print(f"  Advanced: {len([c for c in advanced_cols if c in df.columns])}")
    print(f"  Team shares: {len([c for c in share_cols if c in df.columns])}")
    print(f"  Tracking: {len([c for c in tracking_cols if c in df.columns])}")
    print(f"  Trajectory (deltas): {len([c for c in trajectory_cols if c in df.columns])}")

    return features


if __name__ == "__main__":
    build_features()
