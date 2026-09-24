import pandas as pd
import pytest

from src.data.comprehensive_stats import ComprehensiveStatsPipeline
from src.data.data_loader import DataLoader
from src.data.nba_api_client import NBAApiClient
from src.features.feature_pipeline import FeaturePipeline


def test_comprehensive_ingestion_includes_tracking_and_scoring(tmp_path, monkeypatch):
    pipeline = ComprehensiveStatsPipeline(str(tmp_path))
    basic = pd.DataFrame({"PLAYER_ID": [1], "GP": [50], "PTS": [20], "SEASON": ["2020-21"]})
    monkeypatch.setattr(pipeline, "fetch_basic_stats", lambda season: basic.copy())
    for method in [
        "fetch_shot_locations",
        "fetch_hustle_stats",
        "fetch_advanced_metrics",
        "fetch_bio_stats",
    ]:
        monkeypatch.setattr(pipeline, method, lambda season: pd.DataFrame())
    monkeypatch.setattr(
        DataLoader,
        "get_tracking_stats",
        lambda self, s: pd.DataFrame({"PLAYER_ID": [1], "DRIVES": [100]}),
    )
    monkeypatch.setattr(
        DataLoader,
        "get_scoring_stats",
        lambda self, s: pd.DataFrame({"PLAYER_ID": [1], "PTS": [1000], "PCT_UAST_FGM": [0.6]}),
    )
    result = pipeline.fetch_season_data("2020-21")
    assert result.PTS.iloc[0] == 20
    assert result.DRIVES.iloc[0] == 100
    assert result.PCT_UAST_FGM.iloc[0] == 0.6
    assert result.columns.is_unique


def test_tracking_merges_do_not_suffix_overlapping_columns(monkeypatch):
    client = NBAApiClient()
    monkeypatch.setattr(
        client,
        "get_tracking_stats",
        lambda season, measure: pd.DataFrame(
            {"PLAYER_ID": [1], "PLAYER_NAME": ["A"], "AST": [2], measure: [3]}
        ),
    )
    result = client.get_all_tracking_stats("2020-21")
    assert "AST" in result and not any(c.endswith(("_x", "_y")) for c in result)


def test_partial_seasons_are_not_published(tmp_path, monkeypatch):
    pipeline = ComprehensiveStatsPipeline(str(tmp_path))
    monkeypatch.setattr(pipeline, "fetch_player_info", lambda: pd.DataFrame())
    monkeypatch.setattr(pipeline, "fetch_season_data", lambda season: pd.DataFrame())
    with pytest.raises(ValueError, match="refusing partial"):
        pipeline.pull_all_seasons("2020-21", "2021-22")


def test_aggregate_pipeline_cannot_overwrite_app_snapshot(tmp_path, monkeypatch):
    pipeline = FeaturePipeline(str(tmp_path))
    monkeypatch.setattr(pipeline, "process_all_seasons", lambda *args: pd.DataFrame({"PTS": [1]}))
    monkeypatch.setattr(pipeline, "build_player_features", lambda df: df)
    with pytest.raises(ValueError, match="reserved"):
        pipeline.run(output_path=str(tmp_path / "player_features.parquet"), verbose=False)
    pipeline.run(verbose=False)
    assert (tmp_path / "features/career_aggregate_features.parquet").exists()
    assert not (tmp_path / "features/player_features.parquet").exists()


def test_merge_preserves_base_units_and_rejects_duplicate_players():
    from src.data.merge_stats import merge_player_measurements

    base = pd.DataFrame({"PLAYER_ID": [1], "PTS": [20]})
    extra = pd.DataFrame({"player_id": [1], "pts": [1000], "drives": [100]})
    result = merge_player_measurements(base, extra)
    assert result.PTS.iloc[0] == 20 and result.drives.iloc[0] == 100
    assert "pts" not in result and "player_id" not in result
    with pytest.raises(pd.errors.MergeError):
        merge_player_measurements(base, pd.concat([extra, extra]))
