import numpy as np
import pandas as pd
import pytest

from src.similarity.scoring import similarity_score
from src.similarity.weighted_matcher import WeightedMatcher


def frame():
    return pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "PLAYER_NAME": ["A", "B", "C", "D"],
            "SEASON": ["2020-21"] * 4,
            "CAREER_YEAR": [1] * 4,
            "AGE": [20] * 4,
            "GP": [50, 50, 5, 50],
            "MIN": [30, 30, 10, 30],
            "PTS": [20, 20, 22, 30],
            "AST": [4, 4, 5, 6],
            "REB": [5, 5, 6, 7],
            "TEAM_ABBREVIATION": ["A"] * 4,
            "e_usg_pct": [0.2, 0.2, 0.25, 0.3],
            "DRIVES": [10, np.nan, 11, 20],
        }
    )


def test_identical_match_and_zero_distance_weights():
    matcher = WeightedMatcher({"usage": 1}).fit(frame())
    results = matcher.find_similar_season(1, 1)
    assert results[0][0] == 2 and results[0][3] == 0
    assert matcher.compute_distance(1, 2)[0] == 0
    assert similarity_score(results[0][3]) == 100


def test_missing_is_not_zero_and_coverage_filters():
    matcher = WeightedMatcher({"drives": 1}).fit(frame())
    assert matcher.find_similar_season(1, 1, min_coverage=0.1)[0][0] == 3
    assert not matcher.find_similar_season(2, 1, min_coverage=0)
    assert matcher.season_coverage(1, 1, 2, 1) == 0


def test_group_rms_and_zero_weight():
    matcher = WeightedMatcher({"usage": 1}).fit(frame())
    # The weighted result is exactly the enabled single group.
    result = matcher.find_similar_season(1, 1)[1]
    assert result[3] == pytest.approx(result[4]["usage"])
    both = WeightedMatcher({"usage": 1, "scoring_volume": 1}).fit(frame())
    both._matrices = {
        "usage": np.array([[0], [0], [1], [2]]),
        "scoring_volume": np.array([[0, 0], [1, 1], [1, 1], [2, 2]]),
    }
    distances, _, _ = both._distances(0, np.array([1, 2]), both.weights)
    assert distances.tolist() == [0.5, 1.0]


@pytest.mark.parametrize(
    "weights", [{}, {"usage": 0}, {"usage": -1}, {"usage": float("nan")}, {"bad": 1}]
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        WeightedMatcher(weights)


def test_filters_and_persistence(tmp_path):
    matcher = WeightedMatcher({"usage": 1}).fit(frame())
    expected = matcher.find_similar_season(1, 1, min_games=20, min_minutes=20)
    assert [r[0] for r in expected] == [2, 4]
    path = tmp_path / "model.pkl"
    matcher.save(path)
    assert (
        WeightedMatcher.load(path).find_similar_season(1, 1, min_games=20, min_minutes=20)
        == expected
    )
    assert not matcher.find_similar_season(1, 1, season_start=2021)


def test_scale_is_fixed_bounded_monotone():
    assert [similarity_score(d) for d in [0, 0.5, 1, 2, float("inf")]] == pytest.approx(
        [100, 84.0896415, 50, 6.25, 0]
    )
    assert all(
        a >= b
        for a, b in zip(
            [similarity_score(x) for x in np.arange(0, 3, 0.1)],
            [similarity_score(x) for x in np.arange(0.1, 3.1, 0.1)],
        )
    )


def test_duplicate_schema_rejected():
    with pytest.raises(ValueError, match="one row"):
        WeightedMatcher().fit(pd.concat([frame(), frame().iloc[:1]]))


def test_pair_score_is_invariant_to_result_count_and_candidate_filters():
    matcher = WeightedMatcher({"usage": 1}).fit(frame())
    full = matcher.find_similar_season(1, 1, n=20)
    filtered = matcher.find_similar_season(1, 1, n=1, min_games=20)
    assert filtered[0] == full[0]


def test_experimental_strategies_handle_missing_measurements():
    from src.similarity.trajectory_matcher import HybridTrajectoryMatcher
    from src.similarity.trajectory_matching import AlignedTrajectoryMatcher

    first = frame()
    second = first.assign(SEASON="2021-22", CAREER_YEAR=2, PTS=first.PTS + 2)
    data = pd.concat([first, second], ignore_index=True)
    for cls in (AlignedTrajectoryMatcher, HybridTrajectoryMatcher):
        matcher = cls()
        matcher.fit(data)
        distance, _ = matcher.compute_trajectory_distance(1, 2)
        assert distance == 0


def _synthetic(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "PLAYER_ID": np.repeat(np.arange(n // 3), 3),
            "PLAYER_NAME": [f"P{i}" for i in np.repeat(np.arange(n // 3), 3)],
            "SEASON": np.tile(["2020-21", "2021-22", "2022-23"], n // 3),
            "CAREER_YEAR": np.tile([1, 2, 3], n // 3),
            "AGE": 20,
            "GP": 60,
            "MIN": 30.0,
            "PTS": 10.0,
            "AST": 3.0,
            "REB": 4.0,
            "TEAM_ABBREVIATION": "T",
            "e_usg_pct": rng.normal(0.2, 0.03, n),
        }
    )


def test_best_per_player_keeps_one_season_each_in_rank_order():
    matcher = WeightedMatcher({"usage": 1}).fit(_synthetic())
    everything = matcher.find_similar_season(0, 1, n=100)
    best = matcher.find_similar_season(0, 1, n=100, best_per_player=True)
    pids = [r[0] for r in best]
    assert len(pids) == len(set(pids)) and 0 in pids  # the subject's other seasons compete too
    first_seen = {}
    for r in everything:
        first_seen.setdefault(r[0], r)
    assert best == sorted(first_seen.values(), key=lambda r: r[3])


def test_pair_percentile_is_monotonic_and_bounded():
    matcher = WeightedMatcher({"usage": 1}).fit(_synthetic())
    values = [matcher.pair_percentile(d) for d in (0.0, 0.5, 1.0, 1e6)]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1.0 and values[-1] == 0.0


def test_closeness_bands_are_pool_percentiles():
    from src.similarity.scoring import closeness_band

    assert closeness_band(0.995)[0] == "very_close"
    assert closeness_band(0.96)[0] == "close"
    assert closeness_band(0.85)[0] == "moderate"
    assert closeness_band(0.3)[0] == "loose"
