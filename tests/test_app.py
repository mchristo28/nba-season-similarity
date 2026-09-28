from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.app.presentation import render_radar_svg, render_similarity_bars, score_label


def test_unavailable_bars_and_labels():
    html = render_similarity_bars({"usage": 0}, {"usage": 1, "drives": 1})
    assert "N/A" in html and "OFF" in html and ">100<" in html
    assert score_label(99) != "IDENTICAL MEASURED PROFILE"


def test_missing_radar_is_not_zero():
    assert "Not enough" in render_radar_svg(
        pd.Series(dtype=float), pd.Series(dtype=float), "A", "B"
    )


def test_full_app_controls():
    app_path = Path(__file__).resolve().parents[1] / "src/app/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    app.slider(key="w_usage").set_value(3).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.slider(key="w_usage").value == 1
    app.checkbox(key="exclude_same").check().run()
    app.radio(key="n_results").set_value(5).run()
    assert not app.exception
    app.selectbox(key="player_select").select("LeBron James").run()
    assert not app.exception
    assert (
        app.selectbox(key="season_select").value
        == len(app.selectbox(key="season_select").options) - 1
    )
    for slider in app.slider:
        if slider.key and slider.key.startswith("w_"):
            slider.set_value(0)
    app.run()
    assert not app.exception
    assert any("Enable at least one" in w.value for w in app.warning)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.selectbox(key="compare_select").options) == 5


def test_award_tokens():
    from src.app.streamlit_app import get_awards_pills

    awards = pd.DataFrame({"PLAYER_ID": [1], "SEASON": ["2020-21"], "AWARDS": ["🛡️6️⃣👑"]})
    assert set(get_awards_pills(1, "2020-21", awards)) == {"DPOY", "6MOY", "MVP"}


def test_comparison_modes_and_historical_boundary():
    app_path = Path(__file__).resolve().parents[1] / "src/app/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert app.radio(key="matching_detail").value == "Tracking (2013+)"
    assert app.checkbox(key="exclude_same").value
    app.radio(key="matching_mode").set_value("Production").run(timeout=30)
    assert not app.exception
    assert any("TS above league" in item.value for item in app.markdown)
    app.radio(key="matching_mode").set_value("Playing style").run(timeout=30)
    app.radio(key="matching_detail").set_value("Tracking (2013+)").run(timeout=30)
    assert not app.exception
    assert app.slider(key="w_drives").value == 1
    app.selectbox(key="player_select").select("LeBron James").run()
    app.selectbox(key="season_select").select(0).run()
    assert not app.exception
    assert any("Tracking comparisons start" in item.value for item in app.info)
    next(b for b in app.button if b.label == "Compare with historical data").click().run()
    assert not app.exception
    assert app.selectbox(key="compare_select").options
    app.selectbox(key="age_filter").select("Within 2 years of age").run()
    assert not app.exception


def test_relative_efficiency_is_displayed_in_percentage_points():
    from src.app.presentation import fmt_stat

    assert fmt_stat(0.04, "ts_relative", True) == "+4.0 pp"
    assert fmt_stat(-0.025, "ts_relative", True) == "-2.5 pp"


def test_whole_profile_explanation_and_tracking_shortcut():
    app_path = Path(__file__).resolve().parents[1] / "src/app/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    app.radio(key="matching_detail").set_value("Historical (2003+)").run()
    app.radio(key="matching_mode").set_value("Production").run()
    app.radio(key="matching_mode").set_value("Playing style").run()
    assert app.radio(key="matching_detail").value == "Historical (2003+)"
    app.selectbox(key="player_select").select("Lauri Markkanen").run()
    app.selectbox(key="season_select").select(7).run()
    assert not app.exception
    gg = next(
        i
        for i, option in enumerate(app.selectbox(key="compare_select").options)
        if "GG Jackson (2023-24)" in option
    )
    app.selectbox(key="compare_select").select(gg).run()
    text = "\n".join(item.value for item in app.markdown)
    assert "WHAT DRIVES THE DIFFERENCE" in text and "KEY DIFFERENCES" in text
    assert "Noticeable difference" in text and "Large difference" in text
    assert "VERY CLOSE" not in text and "SIMILARITY BY CATEGORY" not in text
    app.slider(key="w_physical").set_value(0).run()
    assert not app.exception
    assert any("Not scored" in item.value for item in app.markdown)
    button = next(b for b in app.button if b.label == "Compare with tracking detail")
    button.click().run()
    assert not app.exception
    assert app.radio(key="matching_detail").value == "Tracking (2013+)"


def test_position_peers_and_hybrid_switching():
    app_path = Path(__file__).resolve().parents[1] / "src/app/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert app.radio(key="reference_mode").value == "All players"
    app.selectbox(key="player_select").select("Keyonte George").run()
    app.radio(key="reference_mode").set_value("Position peers").run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox(key="peer_group").value == "Guard"
    assert any("Reference: Guard" in item.value for item in app.caption)
    app.selectbox(key="player_select").select("Lauri Markkanen").run()
    assert not app.exception and not app.error
    assert len(app.selectbox(key="peer_group").options) == 3
    app.selectbox(key="peer_group").select("Center / Big").run()
    assert not app.exception and not app.error
    assert any("Reference: Center / Big" in item.value for item in app.caption)
    # A previous hybrid's subgroup must not leak into the next subject's pool.
    app.selectbox(key="player_select").select("Austin Reaves").run()
    assert app.selectbox(key="peer_group").value == "Guard"
    assert not app.exception and not app.error
    app.radio(key="reference_mode").set_value("All players").run()
    assert not app.exception and not app.error


def test_unknown_season_position_stays_available_in_all_player_mode():
    app_path = Path(__file__).resolve().parents[1] / "src/app/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    app.selectbox(key="player_select").select("Lonzo Ball").run()
    app.radio(key="reference_mode").set_value("Position peers").run()
    assert not app.exception and not app.error
    assert any("no verified roster position" in item.value for item in app.info)
    app.radio(key="reference_mode").set_value("All players").run()
    assert not app.exception and not app.error
    assert app.selectbox(key="compare_select").options
