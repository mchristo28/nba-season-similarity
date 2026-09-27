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


@st.cache_resource(max_entries=4)
def load_matcher(version=None, profile=None) -> WeightedMatcher | None:
    features_path = project_root / "data/features/player_features.parquet"
    if not features_path.exists():
        return None
    df = load_career_features(version)
    matcher = WeightedMatcher(profile=profile)
    matcher.fit(df)
    return matcher


@st.cache_data
def load_career_features(version=None) -> pd.DataFrame | None:
    features_path = project_root / "data/features/player_features.parquet"
    if not features_path.exists():
        return None
    return pd.read_parquet(features_path)


def data_version(filename="player_features.parquet"):
    path = project_root / "data/features" / filename
    return path.stat().st_mtime_ns if path.exists() else None


@st.cache_data(max_entries=32)
def search_seasons(player_id, season_key, version=None, profile=None, **kwargs):
    return load_matcher(version, profile).find_similar_season(player_id, season_key, **kwargs)
