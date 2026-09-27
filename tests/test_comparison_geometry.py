import numpy as np
import pandas as pd
import pytest

from src.app.comparison_display import profile_categories
from src.app.presentation import render_key_differences, render_stat_breakdown
from src.similarity.profiles import get_profile
from src.similarity.scoring import difference_band
from src.similarity.weighted_matcher import WeightedMatcher
from tests.test_comparison_profiles import profile_frame


def test_joint_distance_penalizes_concentrated_mismatch_and_explains_it():
    weights = {"physical": 1, "scoring_volume": 1}
    matcher = WeightedMatcher(weights, profile="production").fit(profile_frame("production"))
    # Candidate 2 has one large mismatch and one perfect match. Candidate 3 is
    # consistently closer. The old mean-of-group-distances preferred candidate 2.
    matcher._matrices["physical"] = np.array([[0, 0], [2, 2], [1.1, 1.1], [3, 3]])
    matcher._matrices["scoring_volume"] = np.array([[0], [0], [1.1], [3]])
    matches = matcher.find_similar_season(1, 1)
    assert [r[0] for r in matches[:2]] == [3, 2]
    assert matches[1][3] == pytest.approx(np.sqrt(2))
    explanation = matcher.explain_season(1, 1, 2, 1)
    assert sum(explanation["group_contributions"].values()) == pytest.approx(2)
    assert sum(v["contribution"] for v in explanation["features"].values()) == pytest.approx(2)
    assert matcher.compute_distance(1, 2)[0] == pytest.approx(np.sqrt(2))
    reverse = matcher.explain_season(2, 1, 1, 1)
    assert reverse == explanation
    rescaled = matcher.explain_season(1, 1, 2, 1, weights={k: v * 10 for k, v in weights.items()})
    assert rescaled == explanation


def test_explanation_handles_missing_and_disabled_without_creating_evidence():
    data = profile_frame("production")
    data.loc[0, "AST_PER100"] = np.nan
    matcher = WeightedMatcher({"playmaking": 1}, profile="production").fit(data)
    result = matcher.explain_season(1, 1, 3, 1)
    assert np.isnan(result["features"]["AST_PER100"]["distance"])
    assert result["features"]["AST_PER100"]["contribution"] == 0
    assert not result["features"]["height_inches"]["enabled"]
    assert result["features"]["height_inches"]["contribution"] == 0
    assert result["features"]["TOV_PER100"]["contribution"] == pytest.approx(
        result["distance"] ** 2
    )
    assert result["coverage"] == 0.5


@pytest.mark.parametrize(
    "gap,band",
    [
        (0, "close"),
        (0.499, "close"),
        (0.5, "noticeable"),
        (0.999, "noticeable"),
        (1, "large"),
        (3, "large"),
        (np.nan, "unavailable"),
        (np.inf, "unavailable"),
    ],
)
def test_color_boundaries(gap, band):
    assert difference_band(gap) == band


def test_color_uses_population_spread_not_percentage_of_measurement():
    data = profile_frame("production")
    data["height_inches"] = [84, 81, 78, 75]
    data["weight"] = [240, 210, 200, 180]
    matcher = WeightedMatcher(profile="production").fit(data)
    evidence = matcher.explain_season(1, 1, 2, 1)["features"]
    assert difference_band(evidence["height_inches"]["distance"]) == "noticeable"
    assert difference_band(evidence["weight"]["distance"]) == "large"
    categories = profile_categories({"physical": get_profile("production")["groups"]["physical"]})
    html = render_stat_breakdown(data.iloc[0], data.iloc[1], "A", "B", categories, evidence)
    assert 'class="match-mid"' in html and "Noticeable difference" in html
    assert 'class="match-weak"' in html and "Large difference" in html
    # Translating the measurements changes percentage gaps but not standardized gaps.
    shifted = data.copy()
    shifted["height_inches"] += 100
    other = WeightedMatcher(profile="production").fit(shifted).explain_season(1, 1, 2, 1)
    assert other["features"]["height_inches"]["distance"] == pytest.approx(
        evidence["height_inches"]["distance"]
    )
    context = render_stat_breakdown(data.iloc[0], data.iloc[1], "A", "B", categories)
    assert (
        "match-weak" not in context and "match-mid" not in context and "match-strong" not in context
    )


def test_context_zero_missing_and_disabled_are_neutral():
    categories = [{"name": "Defense", "stats": [("Blocks", "BLK_PER100", False)]}]
    rows = [pd.Series({"BLK_PER100": value}) for value in [0.0, 0.1, np.nan]]
    evidence = {"BLK_PER100": {"enabled": True, "distance": 0.1}}
    assert "Close" in render_stat_breakdown(*rows[:2], "A", "B", categories, evidence)
    evidence["BLK_PER100"]["distance"] = np.nan
    assert "Unavailable" in render_stat_breakdown(rows[0], rows[2], "A", "B", categories, evidence)
    evidence["BLK_PER100"]["enabled"] = False
    assert "Not scored" in render_stat_breakdown(*rows[:2], "A", "B", categories, evidence)


def test_highlights_explain_contribution_and_do_not_exaggerate_close_gaps():
    data = profile_frame("production")
    matcher = WeightedMatcher({"physical": 1}, profile="production").fit(data)
    identical = matcher.explain_season(1, 1, 2, 1)
    assert "No noticeable gaps" in render_key_differences(data.iloc[0], data.iloc[1], identical)
    distant = matcher.explain_season(1, 1, 4, 1)
    rendered = render_key_differences(data.iloc[0], data.iloc[3], distant)
    assert "Height" in rendered and "Weight" in rendered
    assert "PTS" not in rendered
