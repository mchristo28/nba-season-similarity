"""Versioned, cached access to the published season snapshot."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.similarity.weighted_matcher import WeightedMatcher

project_root = Path(__file__).resolve().parents[2]


@st.cache_data
def load_cached_awards(version=None) -> pd.DataFrame | None:
    awards_path = project_root / "data/features/season_awards.parquet"
    if awards_path.exists():
        return pd.read_parquet(awards_path)
    return None


@st.cache_resource(max_entries=12)
def load_matcher(version=None, profile=None, peer_groups=()) -> WeightedMatcher | None:
    features_path = project_root / "data/features/player_features.parquet"
    if not features_path.exists():
        return None
    df = load_career_features(version)
    matcher = WeightedMatcher(profile=profile, peer_groups=peer_groups)
    matcher.fit(df)
    return matcher


@st.cache_data
def load_career_features(version=None) -> pd.DataFrame | None:
    features_path = project_root / "data/features/player_features.parquet"
    if not features_path.exists():
        return None
    frame = pd.read_parquet(features_path)
    positions_path = project_root / "data/features/player_positions.parquet"
    if positions_path.exists():
        frame = frame.merge(
            pd.read_parquet(positions_path),
            on=["PLAYER_ID", "SEASON"],
            how="left",
            validate="one_to_one",
        )
    return frame


def data_version(filename="player_features.parquet"):
    path = project_root / "data/features" / filename
    version = path.stat().st_mtime_ns if path.exists() else None
    if filename == "player_features.parquet":
        return version, data_version("player_positions.parquet")
    return version


@st.cache_data(max_entries=32)
def search_seasons(player_id, season_key, version=None, profile=None, peer_groups=(), **kwargs):
    return load_matcher(version, profile, peer_groups).find_similar_season(
        player_id, season_key, **kwargs
    )
