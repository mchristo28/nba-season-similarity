"""Validate and atomically publish the season dataset used by the app."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "PLAYER_ID",
    "PLAYER_NAME",
    "SEASON",
    "CAREER_YEAR",
    "AGE",
    "GP",
    "MIN",
    "PTS",
    "AST",
    "REB",
    "TEAM_ABBREVIATION",
}


def validate_season_features(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Season dataset is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Season dataset is empty")
    keys = ["PLAYER_ID", "CAREER_YEAR"]
    if df[keys + ["SEASON", "PLAYER_NAME"]].isna().any().any():
        raise ValueError("Season identifiers cannot be missing")
    if df.duplicated(keys).any() or df.duplicated(["PLAYER_ID", "SEASON"]).any():
        raise ValueError("Expected one row per player-season and career year")
    if not df.SEASON.str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("Invalid season identifier")
    numeric = df.select_dtypes(include="number")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite feature values are not allowed")
    if (df[["GP", "MIN", "CAREER_YEAR"]] < 0).any().any():
        raise ValueError("Games, minutes and career year must be nonnegative")


def publish_features(df: pd.DataFrame, path: str | Path) -> None:
    validate_season_features(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
