import numpy as np
import pandas as pd
import pytest

from src.features.positions import build_positions, position_groups, position_mask
from src.similarity.weighted_matcher import WeightedMatcher
from tests.test_comparison_profiles import profile_frame


def test_season_join_hybrids_and_team_label_variation():
    features = pd.DataFrame({"PLAYER_ID": [1, 1, 2], "SEASON": ["2020-21", "2021-22", "2020-21"]})
    rosters = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1],
            "SEASON": ["2020-21", "2020-21", "2021-22"],
            "POSITION": ["F", "F-C", "G-F"],
            "TeamID": [10, 20, 20],
        }
    )
    result = build_positions(features, rosters)
    assert len(result) == 3
    assert result.POSITION.tolist()[:2] == ["F-C", "G-F"]
    assert result.POSITION_LABELS.iloc[0] == "F/F-C"
    assert result.POSITION_LABEL_VARIATION.iloc[0]
    assert pd.isna(result.POSITION.iloc[2])
    assert position_mask(result, ("C",)).tolist() == [True, False, False]
    assert position_groups(None) == ()
    assert position_groups("Unknown") == ()


def test_peer_reference_filters_candidates_and_explains_same_geometry(tmp_path):
    data = profile_frame("style_tracking")
    data["POSITION"] = ["G", "G-F", "C", None]
    data["pct_fga_restricted"] = [0.1, 0.2, 0.8, 0.1]
    model = WeightedMatcher(profile="style_tracking", peer_groups=("G",)).fit(data)
    assert model.reference_count == 2
    assert model.find_similar_season(1, 1)[0][0] == 2
    assert len(model.find_similar_season(1, 1)) == 1
    explanation = model.explain_season(1, 1, 2, 1)
    assert explanation["features"]["pct_fga_restricted"]["distance"] == pytest.approx(2)
    assert explanation["distance"] == model.find_similar_season(1, 1)[0][3]
    assert sum(explanation["group_contributions"].values()) == pytest.approx(
        explanation["distance"] ** 2
    )
    assert model.explain_season(2, 1, 1, 1)["distance"] == explanation["distance"]
    # A result filter never refits the distribution.
    model.find_similar_season(1, 1, min_games=60)
    assert model.explain_season(1, 1, 2, 1)["distance"] == explanation["distance"]
    with pytest.raises(ValueError, match="no listed membership"):
        model.find_similar_season(4, 1)
    assert np.isinf(model.compute_distance(1, 3)[0])
    with pytest.raises(ValueError, match="Both seasons"):
        model.explain_season(1, 1, 3, 1)
    model.save(tmp_path / "peer.pkl")
    restored = WeightedMatcher.load(tmp_path / "peer.pkl")
    assert restored.peer_groups == ("G",)
    assert restored.find_similar_season(1, 1) == model.find_similar_season(1, 1)


def test_all_players_unchanged_by_position_data():
    data = profile_frame("style_tracking")
    original = WeightedMatcher(profile="style_tracking").fit(data)
    data["POSITION"] = ["G", "F", "C", None]
    enriched = WeightedMatcher(profile="style_tracking").fit(data)
    assert original.find_similar_season(1, 1) == enriched.find_similar_season(1, 1)
    for group in original.scalers:
        np.testing.assert_array_equal(original._matrices[group], enriched._matrices[group])


def test_peer_reference_cannot_be_empty_or_unrecognized():
    data = profile_frame("style_tracking")
    with pytest.raises(ValueError, match="Not enough listed"):
        WeightedMatcher(profile="style_tracking", peer_groups=("G",)).fit(data)
    with pytest.raises(ValueError, match="Unknown position"):
        WeightedMatcher(peer_groups=("PG",))
