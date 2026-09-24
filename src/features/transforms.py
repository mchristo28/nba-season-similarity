"""Shared stat transformations; units must be explicit at ingestion."""

import numpy as np
import pandas as pd

COUNTING_STATS = [
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
]


def per_game(df: pd.DataFrame, columns=COUNTING_STATS) -> pd.DataFrame:
    result = df.copy()
    if "GP" not in result:
        raise ValueError("GP is required for per-game conversion")
    games = result.GP.where(result.GP > 0)
    for column in columns:
        if column in result:
            result[column] = result[column] / games
    return result


def efficiency_stats(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for made, attempted, output in [
        ("FGM", "FGA", "fg_pct"),
        ("FG3M", "FG3A", "fg3_pct"),
        ("FTM", "FTA", "ft_pct"),
    ]:
        if made in result and attempted in result:
            result[output] = result[made] / result[attempted].where(result[attempted] > 0)
    if {"PTS", "FGA", "FTA"} <= set(result):
        attempts = result.FGA + 0.44 * result.FTA
        result["ts_pct"] = result.PTS / (2 * attempts.where(attempts > 0))
    if {"FGM", "FG3M", "FGA"} <= set(result):
        result["efg_pct"] = (result.FGM + 0.5 * result.FG3M) / result.FGA.where(result.FGA > 0)
    return result


def add_career_year(df: pd.DataFrame, rookie_years: dict) -> pd.DataFrame:
    result = df.copy()
    season_year = result.SEASON.str[:4].astype(int)
    rookie = result.PLAYER_ID.map(rookie_years)
    if rookie.isna().any():
        raise ValueError("Missing rookie years; cannot assign stable career-year identifiers")
    result["CAREER_YEAR"] = (season_year - rookie.astype(int) + 1).clip(lower=1)
    return result.sort_values(["PLAYER_ID", "CAREER_YEAR"])


def team_shares(
    df: pd.DataFrame, teams: pd.DataFrame, *, player_mode="PerGame", team_mode="Totals"
) -> pd.DataFrame:
    """Player per-game production / actual team per-game production.

    Multi-team aggregates cannot be attributed to their final team. Leave shares
    unavailable unless the caller supplies individual team stints instead.
    Legacy totals inputs use a season-total share (same units on both sides).
    """
    result = df.copy()
    stats = [c for c in COUNTING_STATS if c in result]
    for stat in stats:
        result[f"{stat.lower()}_share"] = np.nan
    if teams.empty:
        return result
    if teams.duplicated(["TEAM_ID", "SEASON"]).any():
        raise ValueError("Duplicate team-season totals")
    indexed = teams.set_index(["TEAM_ID", "SEASON"])
    keys = pd.MultiIndex.from_frame(result[["TEAM_ID", "SEASON"]])
    aligned = indexed.reindex(keys).reset_index(drop=True)
    # This mask is mutated below; pandas Copy-on-Write views can be read-only.
    valid_team = result.TEAM_ID.gt(0).to_numpy(copy=True)
    if "TEAM_COUNT" in result:
        valid_team &= result.TEAM_COUNT.eq(1).to_numpy()
    elif player_mode == "PerGame":
        # Without trade metadata, do not assume final-team attribution is valid.
        valid_team[:] = False
    for stat in stats:
        if stat not in aligned:
            continue
        denominator = aligned[stat].to_numpy(dtype=float)
        if player_mode == "PerGame" and team_mode == "Totals":
            games = aligned.GP.to_numpy(dtype=float)
            denominator = np.divide(
                denominator, games, out=np.full(len(games), np.nan), where=games > 0
            )
        elif player_mode != team_mode:
            raise ValueError("Unsupported player/team unit combination")
        valid = valid_team & np.isfinite(denominator) & (denominator > 0)
        result[f"{stat.lower()}_share"] = np.divide(
            result[stat].to_numpy(dtype=float),
            denominator,
            out=np.full(len(result), np.nan),
            where=valid,
        )
    return result
