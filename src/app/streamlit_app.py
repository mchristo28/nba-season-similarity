"""The Season Almanac — NBA Player-Season Similarity Tool.

Editorial broadsheet redesign. Preserves WeightedMatcher wiring.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from src.app.data_access import (
    data_version,
    load_cached_awards,
    load_career_features,
    load_matcher,
    search_seasons,
)
from src.app.presentation import (
    _clean,
    player_abbr,
    render_anchor_portrait,
    render_awards,
    render_colophon,
    render_masthead,
    render_radar_svg,
    render_results_table_html,
    render_section_head,
    render_similarity_bars,
    render_stat_breakdown,
    render_statline,
    score_color_hex,
    score_label,
)
from src.app.styles import CSS, FONT_LINKS
from src.features.registry import DEFAULT_WEIGHTS, DIMENSIONS
from src.similarity.scoring import similarity_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="The Season Almanac",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Load cached awards
# ---------------------------------------------------------------------------


AWARD_MAP = {
    "⭐": "ALL-STAR",
    "🥇": "ALL-NBA 1ST",
    "🥈": "ALL-NBA 2ND",
    "🥉": "ALL-NBA 3RD",
    "🏅": "ALL-NBA",
    "🏆": "CHAMPION",
    "👑": "MVP",
    "🌟": "ROY",
    "🛡️": "DPOY",
    "📈": "MIP",
    "6️⃣": "6MOY",
}


def get_awards_pills(player_id: int, season: str, cached_awards: pd.DataFrame | None) -> list[str]:
    if cached_awards is None:
        return []
    match = cached_awards[
        (cached_awards["PLAYER_ID"] == player_id) & (cached_awards["SEASON"] == season)
    ]
    if match.empty:
        return []
    emoji_str = match.iloc[0]["AWARDS"]
    pills = []
    for token, label in AWARD_MAP.items():
        if token in emoji_str:
            pills.append(label)
    return pills


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Inject fonts and CSS
    st.markdown(FONT_LINKS, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)

    # Load and validate before rendering controls.
    try:
        matcher = load_matcher(data_version())
        career_df = load_career_features(data_version())
        cached_awards = load_cached_awards(data_version("season_awards.parquet"))
    except (ValueError, OSError) as error:
        st.error(f"Cannot load the season dataset: {error}")
        return

    if matcher is None or career_df is None:
        st.error("No data found. Run the feature pipeline first.")
        return

    metadata_path = project_root / "data/features/player_features.json"
    try:
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    source_date = metadata.get("source_updated_at") or "unknown (stored snapshot)"
    latest = career_df.SEASON.max()
    latest_games = int(career_df.loc[career_df.SEASON == latest, "GP"].max())
    st.caption(
        f"Stored data snapshot · latest season {latest} · maximum {latest_games} games per player. "
        f"Source updated: {source_date}. This is not a live feed; incomplete seasons are included."
    )
    total_seasons = len(career_df)
    total_players = career_df["PLAYER_ID"].nunique()
    player_names = sorted(career_df["PLAYER_NAME"].unique())

    # ---- Session state defaults ----
    if "compare_idx" not in st.session_state:
        st.session_state.compare_idx = 0

    # Default weights
    default_weights = DEFAULT_WEIGHTS
    for key, val in default_weights.items():
        if f"w_{key}" not in st.session_state:
            st.session_state[f"w_{key}"] = val

    # ---- Masthead ----
    issue_no = str(213).zfill(3)
    render_masthead(
        total_seasons,
        total_players,
        issue_no,
        sum(len(v["columns"]) for v in matcher.scalers.values()),
    )

    # ---- Section I: The Subject ----
    render_section_head(
        "I",
        "THE SUBJECT",
        "Select a player-season to examine.",
        f"Explore {career_df.SEASON.min()} through {career_df.SEASON.max()}: "
        f"{total_seasons:,} seasons compared across "
        f"{sum(len(v['columns']) for v in matcher.scalers.values())} available matching features.",
    )

    col_anchor, col_editor = st.columns([1.2, 0.9], gap="large")

    with col_anchor:
        col_portrait, col_meta = st.columns([1, 2])

        with col_portrait:
            selected_player = st.selectbox(
                "PLAYER",
                options=player_names,
                index=player_names.index("Shai Gilgeous-Alexander")
                if "Shai Gilgeous-Alexander" in player_names
                else 0,
                key="player_select",
            )

        # Get player data
        player_data = career_df[career_df["PLAYER_NAME"] == selected_player].sort_values(
            "CAREER_YEAR"
        )
        player_id = player_data["PLAYER_ID"].iloc[0]
        seasons = player_data[
            ["CAREER_YEAR", "SEASON", "AGE", "PTS", "AST", "REB", "TEAM_ABBREVIATION"]
        ].to_dict("records")

        season_labels = [f"{s['SEASON']} · Yr {int(s['CAREER_YEAR'])}" for s in seasons]

        with col_meta:
            selected_season_idx = st.selectbox(
                "SEASON",
                options=range(len(season_labels)),
                format_func=lambda i: season_labels[i],
                index=len(season_labels) - 1,
                key="season_select",
            )

        anchor_season = seasons[selected_season_idx]
        anchor_year = int(anchor_season["CAREER_YEAR"])
        anchor_abbr = player_abbr(selected_player)
        anchor_team = anchor_season.get("TEAM_ABBREVIATION", "")

        # Get full anchor row
        anchor_row = player_data[player_data["CAREER_YEAR"] == anchor_year].iloc[0]

        with col_portrait:
            render_anchor_portrait(anchor_abbr, anchor_year, anchor_team)

        with col_meta:
            age = int(anchor_season["AGE"]) if pd.notna(anchor_season.get("AGE")) else "—"
            st.markdown(
                f"""<div style="font-family:var(--body); font-size:16px; color:var(--ink-60);
                            margin-top:4px;">Age {age}</div>""",
                unsafe_allow_html=True,
            )

        # Statline
        ts_pct = anchor_row.get("ts_pct", 0)
        usg_pct = anchor_row.get("e_usg_pct", 0)
        render_statline(
            anchor_season["PTS"], anchor_season["AST"], anchor_season["REB"], ts_pct, usg_pct
        )

        # Awards
        award_pills = get_awards_pills(player_id, anchor_season["SEASON"], cached_awards)
        render_awards(award_pills)

    with col_editor:
        # Editor's note
        st.markdown(
            _clean("""
        <div class="editor-note">
            <div class="editor-note-flag">EDITOR'S NOTE</div>
            <p>Similarity is computed across <b>eleven dimensions</b> — scoring, efficiency, shot profile,
            creation, drives, playmaking, ball handling, rebounding, defense, usage, and physical build.</p>
            <p>Stats are standardized and each dimension uses an average difference, so larger groups do not
            automatically outweigh smaller ones. Unavailable measurements are excluded. Adjust the weights below to tell the engine what matters.</p>
        </div>
        """),
            unsafe_allow_html=True,
        )

        # Weights panel
        with st.expander("§ A   MATCHING WEIGHTS", expanded=False):
            st.markdown(
                '<p class="weight-intro">Each dimension\'s distance is multiplied by its weight '
                "before combining into the overall score. Crank a slider up to prioritize that aspect of play.</p>",
                unsafe_allow_html=True,
            )

            custom_weights = {}
            for dim in DIMENSIONS:
                st.markdown(
                    f'<div class="weight-desc-line">{dim["desc"]}</div>', unsafe_allow_html=True
                )
                custom_weights[dim["key"]] = st.slider(
                    dim["label"],
                    0.0,
                    3.0,
                    step=0.25,
                    key=f"w_{dim['key']}",
                )

            def reset_weights():
                for key, val in default_weights.items():
                    st.session_state[f"w_{key}"] = val

            st.button("↺ RESET TO DEFAULTS", on_click=reset_weights)

    # Render controls before searching so state and results are consistent.
    with st.expander("Search filters"):
        n_results = st.radio("SHOW", [5, 10, 15, 20], index=1, horizontal=True, key="n_results")
        exclude_same = st.checkbox("Exclude other seasons by the same player", key="exclude_same")
        min_games = st.number_input("Minimum games", min_value=0, value=20, step=5)
        min_minutes = st.number_input(
            "Minimum minutes per game", min_value=0.0, value=10.0, step=1.0
        )
        min_coverage = st.slider("Minimum shared data coverage", 0, 100, 50, step=5) / 100
        first_year = int(career_df.SEASON.str[:4].min())
        last_year = int(career_df.SEASON.str[:4].max())
        year_range = st.slider(
            "Candidate season start years", first_year, last_year, (first_year, last_year)
        )
    st.caption(
        "Score guide: 100 = identical measured features; 84 ≈ half a standard deviation "
        "apart; 50 = one standard deviation apart; 6 ≈ two. Scores describe statistical "
        "closeness, not player quality, percentiles, or probabilities. Coverage is shown separately."
    )
    if not any(custom_weights.values()):
        st.warning("Enable at least one matching dimension.")
        return

    try:
        similar = search_seasons(
            player_id,
            season_key=anchor_year,
            n=n_results,
            compare_by="year",
            weights=custom_weights,
            min_games=min_games,
            min_minutes=min_minutes,
            min_coverage=min_coverage,
            exclude_same=exclude_same,
            season_start=year_range[0],
            season_end=year_range[1],
            version=data_version(),
        )

        results_data = []
        for pid, name, their_year, dist, group_dists in similar:
            if pid == player_id and their_year == anchor_year:
                continue
            if exclude_same and pid == player_id:
                continue
            if len(results_data) >= n_results:
                break

            score = similarity_score(dist)
            info = matcher.get_season_info(pid, their_year, "year")
            if not info:
                continue

            season_row = career_df[
                (career_df["PLAYER_ID"] == pid) & (career_df["CAREER_YEAR"] == their_year)
            ]
            if season_row.empty:
                continue
            season_row = season_row.iloc[0]

            results_data.append(
                {
                    "player_id": pid,
                    "name": name,
                    "abbr": player_abbr(name),
                    "team": season_row.get("TEAM_ABBREVIATION", ""),
                    "pos": "",
                    "season": info["season"],
                    "year": their_year,
                    "age": int(info.get("age", 0)) if info.get("age") else "—",
                    "pts": info["pts"],
                    "ast": info["ast"],
                    "reb": info["reb"],
                    "ts": float(season_row.get("ts_pct", 0))
                    if pd.notna(season_row.get("ts_pct"))
                    else float("nan"),
                    "usg": float(season_row.get("e_usg_pct", 0))
                    if pd.notna(season_row.get("e_usg_pct"))
                    else float("nan"),
                    "score": score,
                    "coverage": matcher.season_coverage(
                        player_id, anchor_year, pid, their_year, weights=custom_weights
                    ),
                    "group_distances": group_dists,
                    "career_year": their_year,
                }
            )
    except Exception as e:
        st.error(f"Error finding similar seasons: {e}")
        results_data = []

    if not results_data:
        st.info("No matches meet these filters. Try lowering the coverage or playing-time minimum.")
        render_colophon()
        return

    # ---- Section II: Nearest Neighbors ----
    render_section_head(
        "II",
        "NEAREST NEIGHBORS",
        "The closest historical seasons.",
        f"Ranked by similarity score (0–100). {anchor_abbr} {anchor_season['SEASON']} compared against "
        "eligible seasons under the selected filters.",
    )

    # Compare selection
    compare_idx = st.session_state.get("compare_idx", 0)
    if compare_idx >= len(results_data):
        compare_idx = 0

    compare_options = [
        f"{r['name']} ({r['season']}) — Score: {r['score']:.0f}" for r in results_data
    ]
    if compare_options:
        compare_idx = st.selectbox(
            "SELECT COMPARISON",
            options=range(len(compare_options)),
            format_func=lambda i: compare_options[i],
            index=compare_idx,
            key="compare_select",
        )

    # Render results table
    st.markdown(
        render_results_table_html(results_data, compare_idx),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="results-foot">Showing {len(results_data)} matches under the selected filters · '
        f"Use the dropdown above to select a comparison</div>",
        unsafe_allow_html=True,
    )

    if not results_data:
        render_colophon()
        return

    # ---- Section III: Anatomy of the match ----
    compare_data = results_data[compare_idx]
    compare_score = compare_data["score"]
    sc_hex = score_color_hex(compare_score)
    sc_label = score_label(compare_score)

    render_section_head(
        "III",
        f"RANK {str(compare_idx + 1).zfill(2)} · HEAD-TO-HEAD",
        "Anatomy of the match.",
        f"A full breakdown of how {compare_data['name']}'s {compare_data['season']} campaign compares "
        f"with {selected_player}'s {anchor_season['SEASON']}.",
    )

    # Comparison heads
    st.markdown(
        _clean(f"""
    <div class="comparison-panel">
        <div class="comparison-heads">
            <div class="ch-a">
                <div class="ch-kicker" style="color:var(--ink);">THE QUERY</div>
                <div class="ch-name">{selected_player}</div>
                <div class="ch-meta">{anchor_season["SEASON"]} · {anchor_team} · Age {age}</div>
            </div>
            <div class="ch-score">
                <div class="ch-score-num" style="color:{sc_hex};">{compare_score:.0f}</div>
                <div class="ch-score-lbl">SIMILARITY SCORE</div>
                <div class="ch-score-tag" style="color:{sc_hex};">— {sc_label} —</div>
            </div>
            <div class="ch-b">
                <div class="ch-kicker" style="color:var(--accent);">THE MATCH</div>
                <div class="ch-name">{compare_data["name"]}</div>
                <div class="ch-meta">{compare_data["season"]} · {compare_data["team"]} · Age {compare_data["age"]}</div>
            </div>
        </div>
    """),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Shared data coverage: {compare_data['coverage']:.0%} of weighted requested features. "
        "Unavailable stats are omitted, never treated as zero. Traded-season team shares are unavailable."
    )

    # Charts: Similarity bars + Radar
    col_bars, col_radar = st.columns([1.2, 1], gap="large")

    with col_bars:
        st.markdown(
            """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. B — SIMILARITY BY CATEGORY</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            render_similarity_bars(compare_data["group_distances"], custom_weights),
            unsafe_allow_html=True,
        )

    with col_radar:
        st.markdown(
            """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. C — PLAYER PROFILE OVERLAY</div>""",
            unsafe_allow_html=True,
        )

        compare_row = career_df[
            (career_df["PLAYER_ID"] == compare_data["player_id"])
            & (career_df["CAREER_YEAR"] == compare_data["career_year"])
        ]
        if not compare_row.empty:
            compare_row = compare_row.iloc[0]
            st.markdown(
                render_radar_svg(
                    anchor_row,
                    compare_row,
                    f"{anchor_abbr} {anchor_season['SEASON']}",
                    f"{compare_data['abbr']} {compare_data['season']}",
                ),
                unsafe_allow_html=True,
            )

    # Stat breakdown
    st.markdown(
        """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                color:var(--ink-60); margin:24px 0 14px; padding-bottom:6px; padding-top:22px;
                border-top:1px solid var(--ink-20);
                border-bottom:1px dotted var(--ink-20);">FIG. D — STAT-LINE BREAKDOWN</div>""",
        unsafe_allow_html=True,
    )

    if not compare_row.empty if isinstance(compare_row, pd.DataFrame) else True:
        label_a = f"{anchor_abbr} {anchor_season['SEASON'][2:]}"
        label_b = f"{compare_data['abbr']} {compare_data['season'][2:]}"
        st.markdown(
            render_stat_breakdown(anchor_row, compare_row, label_a, label_b),
            unsafe_allow_html=True,
        )

    # Close comparison panel
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Colophon ----
    render_colophon()


if __name__ == "__main__":
    main()
