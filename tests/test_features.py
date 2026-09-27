import pandas as pd
import pytest

from src.features.build_features import convert_tracking_totals_to_per_game
from src.features.schema import publish_features
from src.features.transforms import add_career_year, efficiency_stats, per_game, team_shares
from tests.test_matching import frame


def test_actual_team_rate_and_traded_player_unknown():
    players = pd.DataFrame(
        {
            "TEAM_ID": [1, 1, 1],
            "SEASON": ["2020-21"] * 3,
            "TEAM_COUNT": [1, 1, 2],
            "PTS": [20.0, 10.0, 30.0],
            "GP": [82, 10, 50],
        }
    )
    teams = pd.DataFrame({"TEAM_ID": [1], "SEASON": ["2020-21"], "GP": [82], "PTS": [8200]})
    result = team_shares(players, teams)
    assert result.pts_share[:2].tolist() == [0.2, 0.1]
    assert pd.isna(result.pts_share.iloc[2])
    assert team_shares(players, pd.DataFrame()).pts_share.isna().all()


def test_totals_share_aligns_season():
    players = pd.DataFrame({"TEAM_ID": [1, 1], "SEASON": ["2020-21", "2021-22"], "PTS": [100, 100]})
    teams = pd.DataFrame({"TEAM_ID": [1, 1], "SEASON": ["2021-22", "2020-21"], "PTS": [1000, 2000]})
    assert team_shares(players, teams, player_mode="Totals").pts_share.tolist() == [0.05, 0.1]


def test_units_and_missing_efficiency():
    df = pd.DataFrame(
        {"GP": [10, 0], "PTS": [200, 0], "FGA": [100, 0], "FGM": [50, 0], "DRIVES": [120, 0]}
    )
    result = convert_tracking_totals_to_per_game(df)
    assert result.DRIVES.iloc[0] == 12 and result.PTS.iloc[0] == 200
    assert df.DRIVES.iloc[0] == 120
    assert pd.isna(result.DRIVES.iloc[1])
    assert efficiency_stats(per_game(df)).fg_pct.iloc[0] == 0.5
    assert pd.isna(efficiency_stats(df).fg_pct.iloc[1])


def test_publish_validates_before_replacing(tmp_path):
    path = tmp_path / "seasons.parquet"
    publish_features(frame(), path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        publish_features(frame().drop(columns="CAREER_YEAR"), path)
    assert path.read_bytes() == before


def test_missing_rookie_year_does_not_collapse_seasons():
    with pytest.raises(ValueError, match="rookie"):
        add_career_year(frame(), {})


def test_shot_zone_no_attempts_are_unknown_not_misses(tmp_path):
    from src.data.comprehensive_stats import ComprehensiveStatsPipeline

    data = pd.DataFrame(
        {
            "FGA": [0.0, 1.0],
            "restricted_area_fga": [0.0, 1.0],
            "restricted_area_fg_pct": [0.0, 0.0],
            "left_corner_3_fga": [0.0, 0.0],
            "right_corner_3_fga": [0.0, 0.0],
            "left_corner_3_fgm": [0.0, 0.0],
            "right_corner_3_fgm": [0.0, 0.0],
        }
    )
    result = ComprehensiveStatsPipeline(str(tmp_path)).compute_derived_stats(data)
    assert pd.isna(result.fg_pct_restricted.iloc[0])
    assert result.fg_pct_restricted.iloc[1] == 0
    assert result.fg_pct_corner3.isna().all()
    assert result.pct_fga_corner3.iloc[1] == 0
