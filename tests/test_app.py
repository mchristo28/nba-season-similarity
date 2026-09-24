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
    app = AppTest.from_file("src/app/streamlit_app.py").run(timeout=30)
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
