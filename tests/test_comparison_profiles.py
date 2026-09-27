import numpy as np
import pandas as pd
import pytest

from src.features.comparison import RATE_STATS, TOTAL_STATS, attach_measurements
from src.features.transforms import efficiency_stats
from src.similarity.profiles import get_profile
from src.similarity.weighted_matcher import WeightedMatcher
from tests.test_matching import frame


def profile_frame(profile):
    data = frame()
    data["GP"] = 50
    data["MIN"] = 25
    for spec in get_profile(profile)["groups"].values():
        for column in spec["features"]:
            data[column] = [1.0, 1.0, 2.0, 3.0]
    return data


def test_exact_efficiency_does_not_use_rounded_per_game_counts():
    source = pd.DataFrame(
        {
            "FGM": [0.0],
            "FGA": [0.1],
            "FG3M": [0.0],
            "FG3A": [0.1],
            "FGM_TOTAL": [2],
            "FG3M_TOTAL": [2],
            "FGA_TOTAL": [3],
            "FG3A_TOTAL": [3],
            "PTS_TOTAL": [6],
            "FTA_TOTAL": [0],
        }
    )
    result = efficiency_stats(source)
    assert result.fg3_pct.iloc[0] == pytest.approx(2 / 3)
    assert result.ts_pct.iloc[0] == 1
    assert result.efg_pct.iloc[0] == 1
    source["FG3A_TOTAL"] = 0
    assert pd.isna(efficiency_stats(source).fg3_pct.iloc[0])


def test_official_percentages_are_preserved_without_totals():
    data = pd.DataFrame({"FG3M": [0.0], "FG3A": [0.1], "FG3_PCT": [0.667]})
    assert efficiency_stats(data).fg3_pct.iloc[0] == 0.667


def test_endpoint_units_and_missing_players_fail_closed():
    base = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], "PTS": [1.1]})
    totals = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], **{c: [33] for c in TOTAL_STATS}})
    rates = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], **{c: [19.8] for c in RATE_STATS}})
    advanced = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], "POSS": [500], "PACE": [98]})
    result = attach_measurements(base, totals, rates, advanced)
    assert result.PTS_TOTAL.iloc[0] == 33
    assert result.PTS_PER100.iloc[0] == 19.8
    assert result.PTS.iloc[0] == 1.1
    with pytest.raises(ValueError, match="every base player"):
        attach_measurements(base, totals.iloc[:0], rates, advanced)
    advanced["GP"] = 29
    with pytest.raises(ValueError, match="games differ"):
        attach_measurements(base, totals, rates, advanced)


@pytest.mark.parametrize("profile", ["style_historical", "style_tracking", "production"])
def test_raw_minutes_and_per_game_output_do_not_change_profile_distance(profile):
    data = profile_frame(profile)
    data.loc[1, ["MIN", "PTS", "AST", "REB"]] = [40, 40, 15, 15]
    matcher = WeightedMatcher(profile=profile).fit(data)
    results = matcher.find_similar_season(1, 1)
    assert results[0][0] == 2
    assert results[0][3] == 0
    assert matcher.season_coverage(1, 1, 2, 1) == 1


def test_style_ignores_accuracy_but_production_compares_relative_efficiency():
    data = profile_frame("style_historical")
    data["ts_relative"] = [0.1, -0.1, 0.0, 0.05]
    for column in ["PTS_PER100", "AST_PER100", "TOV_PER100", "OREB_PER100", "DREB_PER100"]:
        data[column] = [10, 10, 20, 30]
    style = WeightedMatcher(profile="style_historical").fit(data)
    production = WeightedMatcher(profile="production").fit(data)
    assert style.find_similar_season(1, 1)[0][3] == 0
    result = next(r for r in production.find_similar_season(1, 1) if r[0] == 2)
    assert result[3] > 0


def test_candidates_cannot_gain_from_missing_query_measurements():
    data = profile_frame("production")
    data.loc[1, "AST_PER100"] = np.nan
    matcher = WeightedMatcher(profile="production").fit(data)
    assert 2 not in [r[0] for r in matcher.find_similar_season(1, 1, min_coverage=0)]
    # A metric unknown for the query is excluded for every candidate in the same ranking.
    data.loc[0, "AST_PER100"] = np.nan
    matcher.fit(data)
    assert matcher.find_similar_season(1, 1, min_coverage=0)[0][0] == 2


def test_reference_excludes_short_samples_and_filters_do_not_change_scores():
    data = profile_frame("production")
    data.loc[3, ["GP", "PTS_PER100"]] = [1, 10000]
    matcher = WeightedMatcher(profile="production").fit(data)
    assert matcher.scalers["scoring_volume"]["scaler"].mean_[0] == pytest.approx(4 / 3)
    before = matcher.find_similar_season(1, 1)
    after = matcher.find_similar_season(1, 1, min_games=20, min_minutes=20, n=1)
    assert before[0][3] == after[0][3]


def test_tracking_year_boundary_age_filter_and_saved_profile(tmp_path):
    data = profile_frame("style_tracking")
    data.loc[0, "SEASON"] = "2012-13"
    matcher = WeightedMatcher(profile="style_tracking").fit(data)
    with pytest.raises(ValueError, match="starts in 2013"):
        matcher.find_similar_season(1, 1)
    data.loc[1, "AGE"] = 30
    matcher.fit(data)
    assert 2 not in [r[0] for r in matcher.find_similar_season(3, 1, max_age_difference=2)]
    saved = tmp_path / "matcher.pkl"
    matcher.save(saved)
    restored = WeightedMatcher.load(saved)
    assert restored.profile == "style_tracking"
    assert restored.FEATURE_GROUPS == matcher.FEATURE_GROUPS


def test_league_efficiency_uses_weighted_totals_and_ratios_cancel_workload():
    from src.features.comparison import comparison_features

    base = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], "SEASON": ["2020-21"], "FGA": [3.3]})
    totals = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], **{c: [100] for c in TOTAL_STATS}})
    totals["PTS"] = 120
    totals["FTA"] = 0
    rates = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], **{c: [20] for c in RATE_STATS}})
    advanced = pd.DataFrame({"PLAYER_ID": [1], "GP": [30], "POSS": [500], "PACE": [98]})
    data = attach_measurements(base, totals, rates, advanced)
    for col in [
        "PCT_UAST_FGM",
        "PASSES_MADE",
        "TOUCHES",
        "POTENTIAL_AST",
        "TIME_OF_POSS",
        "DRIVE_PASSES",
        "DRIVES",
        "DRIVE_FGA",
        "PULL_UP_FGA",
        "CATCH_SHOOT_FGA",
    ]:
        data[col] = 1.0
    teams = pd.DataFrame(
        {
            "TEAM_ID": [1, 2],
            "SEASON": ["2020-21"] * 2,
            "PTS": [120, 40],
            "FGA": [100, 50],
            "FTA": [0, 0],
        }
    )
    result = comparison_features(data, teams)
    assert result.league_ts_pct.iloc[0] == pytest.approx(160 / 300)
    assert result.ts_relative.iloc[0] == pytest.approx(0.6 - 160 / 300)
    assert result.seconds_per_touch.iloc[0] == 60
    doubled = data.copy()
    for column in ["PASSES_MADE", "TOUCHES", "POTENTIAL_AST", "TIME_OF_POSS"]:
        doubled[column] *= 2
    other = comparison_features(doubled, teams)
    for column in ["passes_per_touch", "potential_assists_per_pass", "seconds_per_touch"]:
        assert other[column].iloc[0] == result[column].iloc[0]
    with pytest.raises(ValueError, match="team totals"):
        comparison_features(data, teams.iloc[:0])
