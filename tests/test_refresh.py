import pandas as pd
import pytest

from scripts.cache_awards import build_season_awards_lookup
from scripts.refresh_data import Snapshot, normalize


def test_failed_download_does_not_replace_snapshot_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.refresh_data.time.sleep", lambda _: None)
    snapshot = Snapshot(tmp_path)

    class Good:
        def __init__(self, **kwargs):
            pass

        def get_data_frames(self):
            return [pd.DataFrame({"PLAYER_ID": [1]})]

    snapshot.fetch("basic", Good, season="2024-25")
    original = snapshot.manifest_path.read_text()

    class Empty(Good):
        def get_data_frames(self):
            return [pd.DataFrame()]

    with pytest.raises(ValueError, match="empty required"):
        snapshot.fetch("basic", Empty, season="2025-26")
    assert snapshot.manifest_path.read_text() == original
    assert len(pd.read_parquet(tmp_path / "basic.parquet")) == 1
    # A successful response is reused when resuming the same snapshot.
    assert len(snapshot.fetch("basic", Good, season="2024-25")) == 1


def test_shot_locations_keep_identifiers_and_normalize_zones():
    source = pd.DataFrame(
        [[1, 3.4]],
        columns=pd.MultiIndex.from_tuples([("", "PLAYER_ID"), ("In The Paint (Non-RA)", "FGA")]),
    )
    frame = normalize(source, "shots")
    assert frame.columns.tolist() == ["PLAYER_ID", "in_the_paint_non_ra_fga"]
    assert frame.iloc[0, 1] == 3.4


def test_awards_recognize_nba_mvp_and_string_all_nba_teams(tmp_path):
    source = tmp_path / "awards.parquet"
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "SEASON": "2025-26", "DESCRIPTION": "NBA Most Valuable Player"},
            {
                "PLAYER_ID": 1,
                "SEASON": "2025-26",
                "DESCRIPTION": "All-NBA",
                "ALL_NBA_TEAM_NUMBER": "1",
            },
            {"PLAYER_ID": 2, "SEASON": "2025-26", "DESCRIPTION": "NBA Finals Most Valuable Player"},
            {
                "PLAYER_ID": 3,
                "SEASON": "2025-26",
                "DESCRIPTION": "NBA All-Star Most Valuable Player",
            },
        ]
    ).to_parquet(source)
    result = build_season_awards_lookup(str(source), str(tmp_path / "lookup.parquet"))
    assert len(result) == 1
    assert result.iloc[0].AWARDS == "👑🥇"


def test_refresh_rejects_lost_players_and_missing_tracking():
    from scripts.refresh_data import validate_refresh
    from tests.test_matching import frame

    data = frame()
    for column in [
        "height_inches",
        "weight",
        "DRIVES",
        "TOUCHES",
        "CATCH_SHOOT_FGA",
        "PULL_UP_FGA",
        "PASSES_MADE",
        "deflections",
    ]:
        data[column] = 1.0
    validate_refresh(data, data, 2020, 2020)
    with pytest.raises(ValueError, match="lost 1"):
        validate_refresh(data.iloc[:-1], data, 2020, 2020)
    with pytest.raises(ValueError, match="Insufficient TOUCHES"):
        validate_refresh(data.drop(columns="TOUCHES"), data, 2020, 2020)
    with pytest.raises(ValueError, match="all requested seasons"):
        validate_refresh(data, data, 2019, 2020)


def test_shot_endpoint_cache_round_trip(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr("scripts.refresh_data.time.sleep", lambda _: None)

    class Shots:
        def __init__(self, **kwargs):
            pass

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    [[1, 2.0]],
                    columns=pd.MultiIndex.from_tuples(
                        [("", np.str_("PLAYER_ID")), (np.str_("Restricted Area"), np.str_("FGA"))]
                    ),
                )
            ]

    snapshot = Snapshot(tmp_path)
    first = snapshot.fetch("shots", Shots)
    resumed = Snapshot(tmp_path).fetch("shots", Shots)
    pd.testing.assert_frame_equal(first, resumed)
    assert first.columns.tolist() == ["PLAYER_ID", "restricted_area_fga"]


def test_profile_fallback_preserves_season_measurements_and_leaves_unknown_missing():
    from scripts.refresh_data import backfill_physical

    class Profiles:
        def fetch(self, key, endpoint, **parameters):
            return pd.DataFrame({"HEIGHT": ["6-5"], "WEIGHT": ["0"]})

    source = pd.DataFrame(
        {"PLAYER_ID": [1, 1], "height_inches": [None, 78.0], "weight": [None, 205.0]}
    )
    result = backfill_physical(source, Profiles())
    assert result.height_inches.tolist() == [77.0, 78.0]
    assert result.weight.iloc[1] == 205
    assert pd.isna(result.weight.iloc[0])
    assert result.height_inches_source.notna().tolist() == [True, False]
