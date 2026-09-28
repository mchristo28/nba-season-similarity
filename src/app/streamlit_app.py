"""The Season Almanac — NBA Player-Season Similarity Tool.

Editorial broadsheet redesign. Preserves WeightedMatcher wiring.
"""

import hashlib
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from src.app.comparison_display import profile_categories
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
    render_key_differences,
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
from src.features.positions import POSITION_GROUPS, position_groups
from src.similarity.profiles import get_profile
from src.similarity.scoring import MODEL_VERSION, similarity_score

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

    mode = st.session_state.get("matching_mode", "Playing style")
    detail = st.session_state.setdefault(
        "matching_detail", st.session_state.get("style_detail_preference", "Tracking (2013+)")
    )
    # Streamlit removes widget state while Production hides the style control.
    st.session_state["style_detail_preference"] = detail
    profile = (
        "production"
        if mode == "Production"
        else ("style_tracking" if detail == "Tracking (2013+)" else "style_historical")
    )
    profile_spec = get_profile(profile)
    dimensions = [{"key": key, **spec} for key, spec in profile_spec["groups"].items()]
    # Load and validate before rendering controls.
    try:
        matcher = load_matcher(data_version(), profile)
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
    source_date = (metadata.get("source_updated_at") or "unknown (stored snapshot)").split("T")[0]
    latest = career_df.SEASON.max()
    latest_games = int(career_df.loc[career_df.SEASON == latest, "GP"].max())
    st.caption(
        f"Stored data snapshot · latest season {latest} · maximum {latest_games} games per player. "
        f"Source fetched: {source_date}. This is a regular-season snapshot, not a live feed."
    )
    total_seasons = len(career_df)
    total_players = career_df["PLAYER_ID"].nunique()
    player_names = sorted(career_df["PLAYER_NAME"].unique())

    # ---- Session state defaults ----
    # Default weights
    default_weights = {key: spec["default_weight"] for key, spec in profile_spec["groups"].items()}
    if st.session_state.get("active_profile") != profile:
        for key, val in default_weights.items():
            st.session_state[f"w_{key}"] = val
        st.session_state["active_profile"] = profile
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
        len(dimensions),
    )
    st.radio("Compare", ["Playing style", "Production"], horizontal=True, key="matching_mode")
    if mode == "Playing style":
        st.radio(
            "Style data",
            ["Historical (2003+)", "Tracking (2013+)"],
            horizontal=True,
            key="matching_detail",
        )
    st.caption(profile_spec["description"])

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

    def reset_player_season():
        selected = st.session_state["player_select"]
        # Season values are list positions, not shared career/year identifiers.
        # A position carried from the previous player can select an unrelated era.
        st.session_state["season_select"] = int(career_df.PLAYER_NAME.eq(selected).sum()) - 1

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
                on_change=reset_player_season,
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

        listed_position = anchor_row.get("POSITION_LABELS")
        st.caption(
            f"Listed season position: {listed_position}"
            if pd.notna(listed_position)
            else "Listed season position: unavailable"
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
        st.markdown(
            _clean(f"""<div class="editor-note">
            <div class="editor-note-flag">EDITOR'S NOTE</div>
            <p><b>{profile_spec["label"]}</b></p>
            <p>{profile_spec["description"]}</p>
            <p>The score compares the whole measured profile. Larger mismatches carry more influence;
            category breakdowns explain where that difference comes from.</p>
            <p>Every candidate in this ranking must have the same measurements available as the selected season.
            Scores use a fixed reference of rotation-player seasons. Adjust the weights to emphasize what matters.</p>
            </div>"""),
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
            for dim in dimensions:
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

    reference_mode = st.radio(
        "Comparison pool",
        ["Position peers", "All players"],
        horizontal=True,
        key="reference_mode",
    )
    peer_groups = ()
    reference_label = "All players"
    memberships = position_groups(anchor_row.get("POSITION"))
    if reference_mode == "Position peers" and not memberships:
        st.info(
            "This season has no verified roster position. Using All players for this comparison."
        )
    elif reference_mode == "Position peers":
        context = (int(player_id), anchor_year)
        if st.session_state.get("peer_subject") != context or "peer_groups" not in st.session_state:
            st.session_state["peer_groups"] = [POSITION_GROUPS[g] for g in memberships]
            st.session_state["peer_subject"] = context
        selected_groups = st.multiselect(
            "Positions to include",
            list(POSITION_GROUPS.values()),
            key="peer_groups",
            help="Starts with this season's listed positions. Add groups to broaden the comparison. Hybrids belong to each listed group.",
        )
        peer_groups = tuple(g for g, label in POSITION_GROUPS.items() if label in selected_groups)
        if not set(peer_groups).intersection(memberships):
            st.info("Include at least one of this player's listed position groups to compare.")
            return
        reference_label = " + ".join(POSITION_GROUPS[g] for g in peer_groups)
        try:
            matcher = load_matcher(data_version(), profile, peer_groups)
        except ValueError as error:
            st.info(str(error))
            return
        eligible = career_df[career_df.SEASON.str[:4].astype(int) >= profile_spec["first_year"]]
        known = eligible.get("POSITION", pd.Series(index=eligible.index, dtype=str)).notna()
        st.caption(
            "Add positions above to broaden the pool. Season roster labels describe listed "
            "positions, not time spent playing each role. Unknown positions are excluded; "
            "choose All players to include them. "
            f"Positions available for {known.sum():,} of {len(eligible):,} seasons in this mode."
        )
    st.caption(
        f"Reference: {reference_label} · {matcher.reference_count:,} rotation-player seasons "
        "(20+ games, 15+ minutes per game). Scores and gap colors use this same reference; "
        "search filters do not change it. Compare scores within the same mode and pool."
    )

    # Render controls before searching so state and results are consistent.
    with st.expander("Search filters"):
        n_results = st.radio("SHOW", [5, 10, 15, 20], index=1, horizontal=True, key="n_results")
        exclude_same = st.checkbox(
            "Exclude other seasons by the same player", value=True, key="exclude_same"
        )
        min_games = st.number_input("Minimum games", min_value=0, value=20, step=5)
        min_minutes = st.number_input(
            "Minimum minutes per game", min_value=0.0, value=10.0, step=1.0
        )
        min_coverage = st.slider("Minimum shared data coverage", 0, 100, 80, step=5) / 100
        age_filter = st.selectbox(
            "Career-stage filter",
            ["Any age", "Within 2 years of age", "Within 5 years of age"],
            key="age_filter",
        )
        max_age_difference = {
            "Any age": None,
            "Within 2 years of age": 2,
            "Within 5 years of age": 5,
        }[age_filter]
        first_year = max(profile_spec["first_year"], int(career_df.SEASON.str[:4].min()))
        last_year = int(career_df.SEASON.str[:4].max())
        year_range = st.slider(
            "Candidate season start years", first_year, last_year, (first_year, last_year)
        )
    st.caption(
        "Score guide: 100 = identical measured features; 84 ≈ a combined gap of half a standard deviation; "
        "50 ≈ one; 6 ≈ two. Larger individual gaps have more influence. Scores describe statistical "
        "closeness, not player quality, percentiles, or probabilities. Compare scores within the selected mode. "
        f"Model {MODEL_VERSION}; coverage is shown separately."
    )
    if not any(custom_weights.values()):
        st.warning("Enable at least one matching dimension.")
        return

    if int(anchor_season["SEASON"][:4]) < profile_spec["first_year"]:
        st.info(
            "Tracking comparisons start in 2013–14. Choose Historical style data for this season."
        )

        def use_historical():
            st.session_state["matching_detail"] = "Historical (2003+)"

        st.button("Compare with historical data", on_click=use_historical)
        return
    if profile == "style_historical" and int(anchor_season["SEASON"][:4]) >= 2013:
        st.info(
            "This season supports richer tracking comparisons, including pull-up shots, "
            "catch-and-shoot attempts, drives and handling. Historical mode uses a broader, simpler profile."
        )

        def use_tracking():
            st.session_state["matching_detail"] = "Tracking (2013+)"

        st.button("Compare with tracking detail", on_click=use_tracking)
    if anchor_row.GP < 20 or anchor_row.MIN < 10:
        st.warning("This subject has a small playing-time sample; its profile may be unstable.")
    if anchor_row.get("FGA_TOTAL", 0) < 100 and mode == "Production":
        st.caption(
            "Shooting efficiency is based on fewer than 100 field-goal attempts. It is an observed result, not an estimate of shooting talent."
        )
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
            profile=profile,
            peer_groups=peer_groups,
            max_age_difference=max_age_difference,
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
                    "pos": season_row.get("POSITION", ""),
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

    # Remount when results change: the hosted selectbox can otherwise retain its
    # previous input label even after its selected value and options have changed.
    compare_options = [
        f"{r['name']} ({r['season']}) — Score: {r['score']:.0f}" for r in results_data
    ]
    selected_comparison = st.selectbox(
        "SELECT COMPARISON",
        options=compare_options,
        key="compare_select_"
        + hashlib.sha256("\n".join(compare_options).encode()).hexdigest()[:16],
    )
    compare_idx = compare_options.index(selected_comparison)

    # Render results table
    st.caption(
        "Table stats are per game. The selected mode’s normalized measurements appear in the breakdown below."
    )
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
    compare_row = career_df[
        (career_df["PLAYER_ID"] == compare_data["player_id"])
        & (career_df["CAREER_YEAR"] == compare_data["career_year"])
    ].iloc[0]
    explanation = matcher.explain_season(
        player_id,
        anchor_year,
        compare_data["player_id"],
        compare_data["career_year"],
        weights=custom_weights,
    )
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

    other_position = compare_row.get("POSITION_LABELS")
    other_position = other_position if pd.notna(other_position) else "unavailable"
    query_position = listed_position if pd.notna(listed_position) else "unavailable"
    st.caption(
        f"Listed positions: {selected_player} — {query_position}; "
        f"{compare_data['name']} — {other_position}. "
        f"Comparison pool: {reference_label}."
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
                <div class="ch-score-lbl">{profile_spec["label"]}</div>
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
        "All ranked candidates share the selected season’s available comparison measurements. "
        "Unavailable measurements are not treated as zero."
    )

    st.caption(
        "One score for the whole measured profile; it does not mean equivalent players. "
        "Close matches can still have meaningful differences."
    )
    col_bars, col_differences = st.columns([1.2, 1], gap="large")

    with col_bars:
        st.markdown(
            """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. B — WHAT DRIVES THE DIFFERENCE</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            render_similarity_bars(
                compare_data["group_distances"],
                custom_weights,
                dimensions,
                explanation["group_contributions"],
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            "Longer bars contribute more to the total measured difference. "
            "Colors describe gap size, not player quality."
        )

    with col_differences:
        st.markdown(
            """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. C — KEY DIFFERENCES</div>""",
            unsafe_allow_html=True,
        )

        st.caption(
            f"{anchor_abbr} → {compare_data['abbr']} · largest contributors among noticeable or large gaps"
        )
        st.markdown(
            render_key_differences(anchor_row, compare_row, explanation), unsafe_allow_html=True
        )

    # Stat breakdown
    st.markdown(
        """<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                color:var(--ink-60); margin:24px 0 14px; padding-bottom:6px; padding-top:22px;
                border-top:1px solid var(--ink-20);
                border-bottom:1px dotted var(--ink-20);">FIG. D — STAT-LINE BREAKDOWN</div>""",
        unsafe_allow_html=True,
    )

    if not compare_row.empty:
        label_a = f"{anchor_abbr} {anchor_season['SEASON'][2:]}"
        label_b = f"{compare_data['abbr']} {compare_data['season'][2:]}"
        st.markdown("**Measurements for this comparison mode**")
        st.caption(
            "Green: close · Yellow: noticeable difference · Red: large difference. "
            "Colors use the same reference scales as the score. Disabled measurements are not scored."
        )
        with st.expander("How gaps and colors are measured"):
            st.write(
                f"Each gap uses the {reference_label} rotation-player reference in this mode. "
                "Close means less than half a standard deviation; noticeable means half to less than one; "
                "large means one or more. These are display guidelines, not significance tests. "
                "Search filters do not change the scales. The overall score combines squared gaps, "
                "with the selected category weights; the largest differences count more."
            )

        st.markdown(
            render_stat_breakdown(
                anchor_row,
                compare_row,
                label_a,
                label_b,
                profile_categories(profile_spec["groups"]),
                evidence=explanation["features"],
            ),
            unsafe_allow_html=True,
        )
        with st.expander("Season context · raw stats and descriptive profile"):
            st.caption(
                "These tables and the profile overlay provide season context. "
                "They do not explain the score; differences are shown without similarity colors."
            )
            st.markdown(
                render_radar_svg(anchor_row, compare_row, label_a, label_b),
                unsafe_allow_html=True,
            )
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
