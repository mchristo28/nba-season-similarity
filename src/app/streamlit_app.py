"""NBA Single Season Comparison Tool.

Find the closest historical match for any player-season.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.similarity.weighted_matcher import WeightedMatcher
from src.data.nba_api_client import NBAApiClient


@st.cache_resource
def load_matcher() -> WeightedMatcher | None:
    """Load the weighted matcher (cached)."""
    features_path = Path("data/features/player_features.parquet")
    if not features_path.exists():
        return None

    df = pd.read_parquet(features_path)
    matcher = WeightedMatcher()
    matcher.fit(df)
    return matcher


@st.cache_data
def load_career_features() -> pd.DataFrame | None:
    """Load career features DataFrame (cached)."""
    features_path = Path("data/features/player_features.parquet")
    if not features_path.exists():
        return None
    return pd.read_parquet(features_path)


@st.cache_resource
def get_api_client() -> NBAApiClient:
    """Get NBA API client (cached)."""
    return NBAApiClient()


@st.cache_data
def load_cached_awards() -> pd.DataFrame | None:
    """Load pre-cached awards from parquet."""
    awards_path = Path("data/features/season_awards.parquet")
    if awards_path.exists():
        return pd.read_parquet(awards_path)
    return None


def get_player_season_awards(player_id: int, season: str, cached_awards: pd.DataFrame | None) -> str:
    """Get awards string for a player-season from cached data."""
    if cached_awards is None:
        return ""
    match = cached_awards[(cached_awards["PLAYER_ID"] == player_id) & (cached_awards["SEASON"] == season)]
    if not match.empty:
        return match.iloc[0]["AWARDS"]
    return ""


# Award emoji explanations for tooltips
AWARD_TOOLTIPS = {
    "⭐": "All-Star",
    "🥇": "All-NBA 1st Team",
    "🥈": "All-NBA 2nd Team",
    "🥉": "All-NBA 3rd Team",
    "🏅": "All-NBA",
    "🏆": "NBA Champion",
    "👑": "MVP",
    "🌟": "Rookie of the Year",
    "🛡️": "Defensive Player of the Year",
    "📈": "Most Improved Player",
    "6️⃣": "Sixth Man of the Year",
}


def awards_with_tooltips(awards_str: str) -> str:
    """Convert awards string to HTML with tooltips."""
    if not awards_str:
        return ""
    html_parts = []
    for emoji in awards_str:
        tooltip = AWARD_TOOLTIPS.get(emoji, "Award")
        html_parts.append(f'<span title="{tooltip}">{emoji}</span>')
    return "".join(html_parts)


def format_pct(val: float) -> str:
    if pd.isna(val):
        return "-"
    return f"{val:.1%}"


def format_stat(val: float, decimals: int = 1) -> str:
    if pd.isna(val):
        return "-"
    return f"{val:.{decimals}f}"


def get_similarity_color(val1: float, val2: float, is_pct: bool = False) -> str:
    """Get background color based on how similar two values are."""
    if val1 is None or val2 is None:
        return ""

    diff = abs(val1 - val2)
    avg = (abs(val1) + abs(val2)) / 2

    GREEN = "background-color: rgba(0, 128, 0, 0.15)"
    YELLOW = "background-color: rgba(255, 165, 0, 0.15)"
    RED = "background-color: rgba(220, 20, 60, 0.15)"

    if is_pct:
        if diff < 0.02:
            return GREEN
        elif diff < 0.05:
            return YELLOW
        else:
            return RED
    else:
        if avg < 2:
            if diff < 0.3:
                return GREEN
            elif diff < 0.7:
                return YELLOW
            else:
                return RED
        elif 80 <= avg <= 140:
            if diff < 3:
                return GREEN
            elif diff < 7:
                return YELLOW
            else:
                return RED
        else:
            rel_diff = diff / avg if avg > 0 else 0
            if rel_diff < 0.10:
                return GREEN
            elif rel_diff < 0.25:
                return YELLOW
            else:
                return RED


def create_similarity_bars(group_distances: dict):
    """Create a horizontal bar chart showing similarity scores by category.

    Uses the actual group distances from the matcher to show how similar
    the players are in each dimension (higher = more similar).
    """
    # Map matcher group names to display names (in order of importance)
    group_display = [
        ("scoring_volume", "Scoring Volume"),
        ("scoring_efficiency", "Efficiency"),
        ("shot_profile", "Shot Profile"),
        ("shot_creation", "Shot Creation"),
        ("drives", "Drives"),
        ("playmaking", "Playmaking"),
        ("ball_handling", "Ball Handling"),
        ("rebounding", "Rebounding"),
        ("defense", "Defense"),
        ("usage", "Usage"),
        ("physical", "Physical"),
    ]

    labels = []
    similarity_scores = []

    for group_key, display_name in group_display:
        dist = group_distances.get(group_key, 0)
        # Convert distance to similarity score (0-100)
        # Lower distance = higher similarity
        similarity = 100 / (1 + dist) if dist > 0 else 100
        labels.append(display_name)
        similarity_scores.append(similarity)

    # Reverse for horizontal bar chart (top to bottom)
    labels = labels[::-1]
    similarity_scores = similarity_scores[::-1]

    # Color based on similarity (green = high, yellow = medium, red = low)
    colors = []
    for score in similarity_scores:
        if score >= 70:
            colors.append('#2ecc71')  # Green
        elif score >= 50:
            colors.append('#f1c40f')  # Yellow
        else:
            colors.append('#e74c3c')  # Red

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=similarity_scores,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.0f}" for s in similarity_scores],
        textposition='inside',
        textfont=dict(color='white', size=12),
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            title="Similarity Score",
            tickvals=[0, 25, 50, 75, 100],
        ),
        yaxis=dict(title=""),
        margin=dict(l=120, r=20, t=30, b=40),
        height=350,
        showlegend=False,
    )

    return fig


def create_profile_radar(p1_row, p2_row, p1_name: str, p2_name: str, p1_season: str, p2_season: str):
    """Create a radar chart comparing two players' actual stat profiles."""

    def safe_get(row, col, default=0):
        val = row.get(col)
        return float(val) if pd.notna(val) else default

    def compute_dimension(row, dimension):
        """Compute a normalized 0-100 score for display."""
        if dimension == "Scoring":
            return min(100, (safe_get(row, "PTS") / 30) * 100)
        elif dimension == "Efficiency":
            ts = safe_get(row, "ts_pct") * 100
            return min(100, max(0, (ts - 45) / 25 * 100))
        elif dimension == "3PT Vol":
            return min(100, (safe_get(row, "FG3A") / 10) * 100)
        elif dimension == "Rim/Paint":
            # Restricted + paint area shots combined
            rim_paint = safe_get(row, "pct_fga_restricted") + safe_get(row, "pct_fga_paint")
            return min(100, rim_paint * 100)
        elif dimension == "Self-Create":
            return min(100, safe_get(row, "PCT_UAST_FGM") * 100)
        elif dimension == "Drives":
            return min(100, (safe_get(row, "DRIVES") / 15) * 100)
        elif dimension == "Playmaking":
            # Assist points created - value of playmaking
            return min(100, (safe_get(row, "AST_PTS_CREATED") / 25) * 100)
        elif dimension == "Rebounding":
            return min(100, (safe_get(row, "REB") / 15) * 100)
        elif dimension == "Defense":
            # Composite of hustle stats: contests + deflections + charges + loose balls
            hustle = (
                safe_get(row, "contested_shots") / 10 +  # ~10/game max
                safe_get(row, "deflections") / 4 +        # ~4/game max
                safe_get(row, "charges_drawn") / 0.5 +    # ~0.5/game max
                safe_get(row, "def_loose_balls_recovered") / 1  # ~1/game max
            ) / 4 * 100
            return min(100, hustle)
        elif dimension == "Usage":
            return min(100, safe_get(row, "e_usg_pct") * 100 * 3)
        return 50

    dimensions = [
        "Scoring", "Efficiency", "3PT Vol", "Rim/Paint", "Self-Create",
        "Drives", "Playmaking", "Rebounding", "Defense", "Usage"
    ]

    p1_values = [compute_dimension(p1_row, dim) for dim in dimensions]
    p2_values = [compute_dimension(p2_row, dim) for dim in dimensions]

    # Close the radar
    labels = list(dimensions) + [dimensions[0]]
    p1_values = p1_values + [p1_values[0]]
    p2_values = p2_values + [p2_values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=p1_values,
        theta=labels,
        fill='toself',
        name=f"{p1_name} ({p1_season})",
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)',
    ))

    fig.add_trace(go.Scatterpolar(
        r=p2_values,
        theta=labels,
        fill='toself',
        name=f"{p2_name} ({p2_season})",
        line=dict(color='#ff7f0e'),
        fillcolor='rgba(255, 127, 14, 0.3)',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=60, t=30, b=60),
        height=350,
    )

    return fig


def display_season_comparison(df: pd.DataFrame, p1_id: int, p2_id: int, p1_year: int, p2_year: int, group_distances: dict = None):
    """Display detailed comparison between two specific seasons."""
    p1_row = df[(df["PLAYER_ID"] == p1_id) & (df["CAREER_YEAR"] == p1_year)]
    p2_row = df[(df["PLAYER_ID"] == p2_id) & (df["CAREER_YEAR"] == p2_year)]

    if p1_row.empty or p2_row.empty:
        st.warning("Could not find season data")
        return

    p1_row = p1_row.iloc[0]
    p2_row = p2_row.iloc[0]

    p1_name = p1_row["PLAYER_NAME"]
    p2_name = p2_row["PLAYER_NAME"]
    p1_season = p1_row.get("SEASON", "")
    p2_season = p2_row.get("SEASON", "")

    # Create unique column names for the comparison table
    col1_name = p1_name[:15]
    col2_name = p2_name[:15]
    if col1_name == col2_name:
        col1_name = f"{p1_name[:12]} (1)"
        col2_name = f"{p2_name[:12]} (2)"

    # Header
    st.subheader("Player Comparison")

    # Two charts side by side: Similarity breakdown + Player profiles
    if group_distances:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Similarity by Category**")
            bars_fig = create_similarity_bars(group_distances)
            st.plotly_chart(bars_fig, use_container_width=True)
        with col2:
            st.markdown("**Player Profiles**")
            radar_fig = create_profile_radar(p1_row, p2_row, p1_name, p2_name, p1_season, p2_season)
            st.plotly_chart(radar_fig, use_container_width=True)

    # Stat comparison tables
    st.markdown("---")
    st.caption("Detailed Stats | Green = very similar | Yellow = moderate | Red = different")

    def get_val(row, col):
        val = row.get(col)
        return val if pd.notna(val) else None

    # Stat categories - aligned with the 11 matcher feature groups
    categories = {
        "Scoring Volume": [
            ("PPG", "PTS", False),
            ("FGA", "FGA", False),
            ("FTA", "FTA", False),
            ("3PA", "FG3A", False),
            ("MIN", "MIN", False),
            ("Pts Share", "pts_share", True),
            ("FGA Share", "fga_share", True),
        ],
        "Efficiency": [
            ("TS%", "ts_pct", True),
            ("eFG%", "efg_pct", True),
            ("FG%", "fg_pct", True),
            ("3P%", "fg3_pct", True),
            ("FT%", "ft_pct", True),
        ],
        "Shot Profile": [
            ("Rim %", "pct_fga_restricted", True),
            ("Paint %", "pct_fga_paint", True),
            ("Mid %", "pct_fga_midrange", True),
            ("Corner 3%", "pct_fga_corner3", True),
            ("Above 3%", "pct_fga_above_break3", True),
        ],
        "Shot Creation": [
            ("Unast 2P%", "PCT_UAST_2PM", True),
            ("Unast 3P%", "PCT_UAST_3PM", True),
            ("Pull-up FGA", "PULL_UP_FGA", False),
            ("Pull-up FG%", "PULL_UP_FG_PCT", True),
            ("C&S FGA", "CATCH_SHOOT_FGA", False),
            ("C&S FG%", "CATCH_SHOOT_FG_PCT", True),
        ],
        "Drives": [
            ("Drives", "DRIVES", False),
            ("Drive FGA", "DRIVE_FGA", False),
            ("Drive FG%", "DRIVE_FG_PCT", True),
            ("Drive PTS", "DRIVE_PTS", False),
            ("Drive AST", "DRIVE_AST", False),
            ("Drive TOV", "DRIVE_TOV", False),
        ],
        "Playmaking": [
            ("APG", "AST", False),
            ("TOV", "TOV", False),
            ("AST Share", "ast_share", True),
            ("Passes", "PASSES_MADE", False),
            ("Pot AST", "POTENTIAL_AST", False),
            ("AST Pts", "AST_PTS_CREATED", False),
        ],
        "Ball Handling": [
            ("Touches", "TOUCHES", False),
            ("Time Poss", "TIME_OF_POSS", False),
            ("Drib/Touch", "AVG_DRIB_PER_TOUCH", False),
            ("FrontCt", "FRONT_CT_TOUCHES", False),
            ("Elbow", "ELBOW_TOUCHES", False),
            ("Paint", "PAINT_TOUCHES", False),
        ],
        "Rebounding": [
            ("RPG", "REB", False),
            ("OREB", "OREB", False),
            ("DREB", "DREB", False),
            ("REB Share", "reb_share", True),
            ("OREB%", "e_oreb_pct", True),
            ("DREB%", "e_dreb_pct", True),
        ],
        "Defense": [
            ("STL", "STL", False),
            ("BLK", "BLK", False),
            ("Contests", "contested_shots", False),
            ("Deflections", "deflections", False),
            ("Charges", "charges_drawn", False),
            ("Loose Balls", "def_loose_balls_recovered", False),
            ("Box Out %", "pct_box_outs_def", True),
        ],
        "Usage": [
            ("USG%", "e_usg_pct", True),
        ],
        "Physical": [
            ("Height", "height_inches", False),
            ("Weight", "weight", False),
        ],
    }

    # Display in 3 columns
    cols = st.columns(3)
    col_idx = 0

    for cat_name, stats in categories.items():
        with cols[col_idx % 3]:
            st.markdown(f"**{cat_name}**")

            rows = []
            raw_values = []

            for stat_name, col_name, is_pct in stats:
                p1_val = get_val(p1_row, col_name)
                p2_val = get_val(p2_row, col_name)

                # Special formatting for height
                if col_name == "height_inches":
                    def fmt_height(h):
                        if h is None:
                            return "-"
                        return f"{int(h//12)}'{int(h%12)}\""
                    p1_fmt = fmt_height(p1_val)
                    p2_fmt = fmt_height(p2_val)
                elif col_name == "weight":
                    p1_fmt = f"{int(p1_val)} lbs" if p1_val is not None else "-"
                    p2_fmt = f"{int(p2_val)} lbs" if p2_val is not None else "-"
                elif is_pct:
                    p1_fmt = format_pct(p1_val) if p1_val is not None else "-"
                    p2_fmt = format_pct(p2_val) if p2_val is not None else "-"
                else:
                    p1_fmt = format_stat(p1_val) if p1_val is not None else "-"
                    p2_fmt = format_stat(p2_val) if p2_val is not None else "-"

                rows.append({
                    "Stat": stat_name,
                    col1_name: p1_fmt,
                    col2_name: p2_fmt,
                })
                raw_values.append((p1_val, p2_val, is_pct))

            comp_df = pd.DataFrame(rows)

            def style_row(row_idx):
                p1_val, p2_val, is_pct = raw_values[row_idx]
                color = get_similarity_color(p1_val, p2_val, is_pct)
                return ["", color, color]

            styled_df = comp_df.style.apply(lambda x: style_row(x.name), axis=1)
            st.dataframe(styled_df, hide_index=True, use_container_width=True)

        col_idx += 1


def main():
    st.set_page_config(
        page_title="NBA Season Comparison",
        page_icon="🏀",
        layout="wide",
    )

    st.title("NBA Single Season Comparison")
    st.markdown("*Find the closest historical match for any player-season*")

    # Load data
    matcher = load_matcher()
    career_df = load_career_features()
    cached_awards = load_cached_awards()

    if matcher is None or career_df is None:
        st.error("No data found. Run the feature pipeline first.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        n_results = st.slider("Results to show", 5, 20, 10)
        exclude_same_player = st.checkbox("Exclude same player", value=False)

        st.markdown("---")

        # Custom weights section
        default_weights = {
            "scoring_volume": 1.0,
            "scoring_efficiency": 1.0,
            "shot_profile": 1.0,
            "shot_creation": 1.0,
            "drives": 1.0,
            "playmaking": 1.0,
            "rebounding": 1.0,
            "defense": 1.0,
            "physical": 0.5,
            "usage": 1.0,
            "ball_handling": 1.0,
        }

        # Initialize session state with defaults if not present
        for key, val in default_weights.items():
            if f"w_{key}" not in st.session_state:
                st.session_state[f"w_{key}"] = val

        # Check if reset was requested (before widgets are created)
        if st.session_state.get("reset_weights", False):
            for key, val in default_weights.items():
                st.session_state[f"w_{key}"] = val
            st.session_state["reset_weights"] = False

        with st.expander("⚙️ Customize Matching Weights", expanded=False):
            st.caption("Boost categories to prioritize in matching")

            custom_weights = {}

            st.markdown("**Scoring**")
            custom_weights["scoring_volume"] = st.slider(
                "Volume (PTS, FGA, FTA, MIN)", 0.0, 3.0, step=0.25,
                key="w_scoring_volume"
            )
            custom_weights["scoring_efficiency"] = st.slider(
                "Efficiency (TS%, eFG%, FG%)", 0.0, 3.0, step=0.25,
                key="w_scoring_efficiency"
            )

            st.markdown("**Shot Profile & Creation**")
            custom_weights["shot_profile"] = st.slider(
                "Location (Rim, Paint, Mid, 3PT)", 0.0, 3.0, step=0.25,
                key="w_shot_profile"
            )
            custom_weights["shot_creation"] = st.slider(
                "Self-Creation (Unassisted %, Pull-up, C&S)", 0.0, 3.0, step=0.25,
                key="w_shot_creation"
            )
            custom_weights["drives"] = st.slider(
                "Drives (volume, efficiency)", 0.0, 3.0, step=0.25,
                key="w_drives"
            )

            st.markdown("**Playmaking**")
            custom_weights["playmaking"] = st.slider(
                "Passing (AST, TOV, potential AST)", 0.0, 3.0, step=0.25,
                key="w_playmaking"
            )
            custom_weights["ball_handling"] = st.slider(
                "Ball Handling (touches, time of poss)", 0.0, 3.0, step=0.25,
                key="w_ball_handling"
            )

            st.markdown("**Rebounding**")
            custom_weights["rebounding"] = st.slider(
                "Rebounds (REB, OREB, DREB)", 0.0, 3.0, step=0.25,
                key="w_rebounding"
            )

            st.markdown("**Defense**")
            custom_weights["defense"] = st.slider(
                "Defense (STL, BLK, contests, deflections, charges)", 0.0, 3.0, step=0.25,
                key="w_defense"
            )

            st.markdown("**Usage**")
            custom_weights["usage"] = st.slider(
                "Usage Rate (USG%)", 0.0, 3.0, step=0.25,
                key="w_usage"
            )

            st.markdown("**Physical**")
            custom_weights["physical"] = st.slider(
                "Size (height, weight)", 0.0, 3.0, step=0.25,
                key="w_physical"
            )

            if st.button("Reset to Defaults"):
                # Set flag to reset on next rerun (before widgets are created)
                st.session_state["reset_weights"] = True
                st.rerun()

        st.markdown("---")
        st.markdown("""
        **How it works:**

        Select a player and season to find the most similar historical seasons.

        Adjust weights above to prioritize what matters most to you.
        """)

    # Player search
    player_names = sorted(career_df["PLAYER_NAME"].unique())

    selected_player = st.selectbox(
        "Search for a player",
        options=[""] + player_names,
        format_func=lambda x: "Type to search..." if x == "" else x,
        key="player_select",
    )

    # Use session state value if selectbox returns empty (Streamlit rerun quirk)
    if not selected_player and st.session_state.get("player_select"):
        selected_player = st.session_state["player_select"]

    if selected_player:
        player_data = career_df[career_df["PLAYER_NAME"] == selected_player].sort_values("CAREER_YEAR")
        player_id = player_data["PLAYER_ID"].iloc[0]

        # Season selector
        seasons = player_data[["CAREER_YEAR", "SEASON", "AGE", "PTS", "AST", "REB"]].to_dict("records")
        season_options = [
            f"Year {s['CAREER_YEAR']} (Age {int(s['AGE'])}): {s['SEASON']} - {s['PTS']:.1f}/{s['AST']:.1f}/{s['REB']:.1f}"
            for s in seasons
        ]

        selected_season_idx = st.selectbox(
            "Select season",
            options=range(len(season_options)),
            format_func=lambda i: season_options[i],
            index=len(season_options) - 1,  # Default to most recent
        )

        selected_year = seasons[selected_season_idx]["CAREER_YEAR"]
        selected_season_str = seasons[selected_season_idx]["SEASON"]

        st.markdown("---")

        # Find similar seasons
        try:
            # Request extra results if filtering same player
            fetch_n = n_results + 20 if exclude_same_player else n_results

            similar = matcher.find_similar_season(
                player_id,
                season_key=selected_year,
                n=fetch_n,
                compare_by="year",
                weights=custom_weights,
            )

            similar_data = []
            for pid, name, their_year, dist, group_dists in similar:
                # Skip same player if toggle is on
                if exclude_same_player and pid == player_id:
                    continue

                # Stop once we have enough results
                if len(similar_data) >= n_results:
                    break

                similarity_score = 100 / (1 + dist)
                info = matcher.get_season_info(pid, their_year, "year")
                if info:
                    # Get player info for this specific season
                    season_row = career_df[(career_df["PLAYER_ID"] == pid) & (career_df["CAREER_YEAR"] == their_year)]
                    if season_row.empty:
                        continue
                    season_row = season_row.iloc[0]

                    height = season_row.get("height_inches", 0)
                    height_str = f"{int(height//12)}'{int(height%12)}\"" if height else "-"
                    age = season_row.get("AGE", 0)
                    age_str = f"{int(age)}" if age else "-"

                    # Get awards for this season (from cache)
                    season_str = info["season"]
                    awards_emojis = get_player_season_awards(pid, season_str, cached_awards)

                    similar_data.append({
                        "Player": name,
                        "Season": info["season"],
                        "Yr": their_year,
                        "Age": age_str,
                        "Awards": awards_emojis,
                        "Height": height_str,
                        "PTS": f"{info['pts']:.1f}",
                        "AST": f"{info['ast']:.1f}",
                        "REB": f"{info['reb']:.1f}",
                        "Score": f"{similarity_score:.0f}",
                        "player_id": pid,
                        "year": their_year,
                        "group_distances": group_dists,
                    })

            similar_df = pd.DataFrame(similar_data)

            if not similar_df.empty:
                st.subheader("Most Similar Seasons")

                # Selection dropdown
                compare_idx = st.selectbox(
                    "Select a player to compare",
                    options=range(len(similar_df)),
                    format_func=lambda i: f"{similar_df.iloc[i]['Player']} ({similar_df.iloc[i]['Season']}) - Score: {similar_df.iloc[i]['Score']}",
                )

                # Similar seasons table
                st.dataframe(
                    similar_df[["Player", "Season", "Yr", "Age", "Awards", "Height", "PTS", "AST", "REB", "Score"]],
                    hide_index=True,
                    use_container_width=True,
                    height=min(400, 35 + 35 * len(similar_df)),
                )

                compare_row = similar_df.iloc[compare_idx]

                # Detailed comparison below
                st.markdown("---")
                display_season_comparison(
                    career_df,
                    player_id,
                    compare_row["player_id"],
                    selected_year,
                    compare_row["year"],
                    group_distances=compare_row.get("group_distances"),
                )
            else:
                st.info("No similar seasons found")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        # Home
        st.markdown("---")
        st.markdown("""
        ### How to use

        1. **Search** for a player
        2. **Select** which season to analyze
        3. **See** the most similar historical seasons
        4. **Compare** stats side-by-side

        Uses 24 stats across scoring, efficiency, shot profile, playmaking,
        rebounding, defense, and impact metrics.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Players", f"{career_df['PLAYER_ID'].nunique():,}")
        with col2:
            st.metric("Seasons", f"{len(career_df):,}")
        with col3:
            st.metric("Data", f"{career_df['SEASON'].min()} - {career_df['SEASON'].max()}")


if __name__ == "__main__":
    main()
