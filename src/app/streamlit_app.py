"""The Season Almanac — NBA Player-Season Similarity Tool.

Editorial broadsheet redesign. Preserves WeightedMatcher wiring.
"""

import math
import re
import sys
from datetime import date
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import streamlit as st

from src.similarity.weighted_matcher import WeightedMatcher

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

DIMENSIONS = [
    {"key": "scoring_volume",     "label": "Scoring Volume",  "short": "SCR VOL",  "group": "Scoring",     "desc": "PTS, FGA, FTA, MIN"},
    {"key": "scoring_efficiency", "label": "Efficiency",      "short": "EFF",      "group": "Scoring",     "desc": "TS%, eFG%, FG%"},
    {"key": "shot_profile",       "label": "Shot Profile",    "short": "SHT PRO",  "group": "Shooting",    "desc": "Rim, Paint, Mid, 3PT"},
    {"key": "shot_creation",      "label": "Shot Creation",   "short": "CREATE",   "group": "Shooting",    "desc": "Unassisted %, Pull-up"},
    {"key": "drives",             "label": "Drives",          "short": "DRV",      "group": "Shooting",    "desc": "Drive volume + efficiency"},
    {"key": "playmaking",         "label": "Playmaking",      "short": "PLY",      "group": "Playmaking",  "desc": "AST, TOV, potential AST"},
    {"key": "ball_handling",      "label": "Ball Handling",   "short": "HAND",     "group": "Playmaking",  "desc": "Touches, time of poss"},
    {"key": "rebounding",         "label": "Rebounding",      "short": "REB",      "group": "Rebounding",  "desc": "REB, OREB, DREB"},
    {"key": "defense",            "label": "Defense",         "short": "DEF",      "group": "Defense",     "desc": "STL, BLK, contests, deflections"},
    {"key": "usage",              "label": "Usage",           "short": "USG",      "group": "Usage",       "desc": "USG%"},
    {"key": "physical",           "label": "Physical",        "short": "PHY",      "group": "Physical",    "desc": "Height, weight"},
]

TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#000000", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#0E2240",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#002D62",
    "LAC": "#C8102E", "LAL": "#552583", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#006BB6",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
    # Historical
    "SEA": "#00653A", "NJN": "#002A60", "NOH": "#0C2340", "NOK": "#0C2340",
    "VAN": "#00B2A9", "CHA": "#1D1160", "CHH": "#00788C",
}

STAT_CATEGORIES = [
    {
        "name": "Scoring Volume",
        "stats": [
            ("PTS", "PTS", False), ("FGA", "FGA", False), ("FTA", "FTA", False),
            ("3PA", "FG3A", False), ("MIN", "MIN", False),
            ("Pts Share", "pts_share", True), ("FGA Share", "fga_share", True),
        ],
    },
    {
        "name": "Efficiency",
        "stats": [
            ("TS%", "ts_pct", True), ("eFG%", "efg_pct", True),
            ("FG%", "fg_pct", True), ("3P%", "fg3_pct", True), ("FT%", "ft_pct", True),
        ],
    },
    {
        "name": "Shot Profile",
        "stats": [
            ("Rim %", "pct_fga_restricted", True), ("Paint %", "pct_fga_paint", True),
            ("Mid %", "pct_fga_midrange", True), ("Corner 3%", "pct_fga_corner3", True),
            ("Above 3%", "pct_fga_above_break3", True),
        ],
    },
    {
        "name": "Shot Creation",
        "stats": [
            ("Unast 2P%", "PCT_UAST_2PM", True), ("Unast 3P%", "PCT_UAST_3PM", True),
            ("Pull-up FGA", "PULL_UP_FGA", False), ("Pull-up FG%", "PULL_UP_FG_PCT", True),
            ("C&S FGA", "CATCH_SHOOT_FGA", False), ("C&S FG%", "CATCH_SHOOT_FG_PCT", True),
        ],
    },
    {
        "name": "Playmaking",
        "stats": [
            ("APG", "AST", False), ("TOV", "TOV", False), ("AST Share", "ast_share", True),
            ("Passes", "PASSES_MADE", False), ("Pot AST", "POTENTIAL_AST", False),
            ("AST Pts", "AST_PTS_CREATED", False),
        ],
    },
    {
        "name": "Defense",
        "stats": [
            ("STL", "STL", False), ("BLK", "BLK", False),
            ("Contests", "contested_shots", False), ("Deflections", "deflections", False),
            ("Charges", "charges_drawn", False), ("Loose Balls", "def_loose_balls_recovered", False),
        ],
    },
    {
        "name": "Physical",
        "stats": [
            ("Height", "height_inches", False), ("Weight", "weight", False),
        ],
    },
]


def player_abbr(name: str) -> str:
    """Derive 2-3 letter monogram from player name."""
    parts = name.split()
    if len(parts) == 1:
        return parts[0][:3].upper()
    # Handle hyphenated last names like Gilgeous-Alexander
    if "-" in parts[-1]:
        subparts = parts[-1].split("-")
        return (parts[0][0] + subparts[0][0] + subparts[1][0]).upper()
    if len(parts) == 2:
        return (parts[0][0] + parts[1][0]).upper()
    return (parts[0][0] + parts[1][0] + parts[2][0]).upper()


def score_color_hex(s: float) -> str:
    if s >= 65:
        return "#2a9d5c"
    if s >= 45:
        return "#c9a227"
    return "#c44536"


def _clean(html: str) -> str:
    """Strip leading whitespace from HTML lines to prevent Markdown code-block interpretation."""
    return re.sub(r"\n[ \t]+", "\n", html)


def score_label(s: float) -> str:
    if s >= 85:
        return "IDENTICAL"
    if s >= 65:
        return "VERY CLOSE"
    if s >= 50:
        return "SIMILAR"
    if s >= 35:
        return "LOOSE"
    return "DISTANT"


def fmt_stat(val, col_name: str, is_pct: bool) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if col_name == "height_inches":
        return f"{int(val // 12)}'{int(val % 12)}\""
    if col_name == "weight":
        return f"{int(val)}"
    if is_pct:
        return f"{val * 100:.1f}%"
    return f"{val:.1f}"


def compute_radar_value(row, dimension_key: str) -> float:
    """Compute a normalized 0-100 value for the radar chart."""
    def safe(col, default=0):
        v = row.get(col)
        return float(v) if pd.notna(v) else default

    if dimension_key == "scoring_volume":
        return min(100, (safe("PTS") / 35) * 100)
    elif dimension_key == "scoring_efficiency":
        ts = safe("ts_pct") * 100
        return min(100, max(0, (ts - 45) / 25 * 100))
    elif dimension_key == "shot_profile":
        return min(100, (safe("pct_fga_restricted") + safe("pct_fga_paint")) * 100)
    elif dimension_key == "shot_creation":
        return min(100, safe("PCT_UAST_FGM") * 100)
    elif dimension_key == "drives":
        return min(100, (safe("DRIVES") / 15) * 100)
    elif dimension_key == "playmaking":
        return min(100, (safe("AST_PTS_CREATED") / 25) * 100)
    elif dimension_key == "ball_handling":
        return min(100, (safe("TOUCHES") / 80) * 100)
    elif dimension_key == "rebounding":
        return min(100, (safe("REB") / 15) * 100)
    elif dimension_key == "defense":
        hustle = (
            safe("contested_shots") / 10 +
            safe("deflections") / 4 +
            safe("charges_drawn") / 0.5 +
            safe("def_loose_balls_recovered") / 1
        ) / 4 * 100
        return min(100, hustle)
    elif dimension_key == "usage":
        return min(100, safe("e_usg_pct") * 100 * 3)
    elif dimension_key == "physical":
        h = safe("height_inches", 78)
        return min(100, max(0, (h - 66) / 18 * 100))
    return 50


# ---------------------------------------------------------------------------
# Load cached awards
# ---------------------------------------------------------------------------

@st.cache_data
def load_cached_awards() -> pd.DataFrame | None:
    awards_path = Path("data/features/season_awards.parquet")
    if awards_path.exists():
        return pd.read_parquet(awards_path)
    return None


AWARD_MAP = {
    "⭐": "ALL-STAR", "🥇": "ALL-NBA 1ST", "🥈": "ALL-NBA 2ND", "🥉": "ALL-NBA 3RD",
    "🏅": "ALL-NBA", "🏆": "CHAMPION", "👑": "MVP", "🌟": "ROY",
    "🛡️": "DPOY", "📈": "MIP", "6️⃣": "6MOY",
}


def get_awards_pills(player_id: int, season: str, cached_awards: pd.DataFrame | None) -> list[str]:
    if cached_awards is None:
        return []
    match = cached_awards[(cached_awards["PLAYER_ID"] == player_id) & (cached_awards["SEASON"] == season)]
    if match.empty:
        return []
    emoji_str = match.iloc[0]["AWARDS"]
    pills = []
    for ch in emoji_str:
        if ch in AWARD_MAP:
            pills.append(AWARD_MAP[ch])
    return pills


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_matcher() -> WeightedMatcher | None:
    features_path = Path("data/features/player_features.parquet")
    if not features_path.exists():
        return None
    df = pd.read_parquet(features_path)
    matcher = WeightedMatcher()
    matcher.fit(df)
    return matcher


@st.cache_data
def load_career_features() -> pd.DataFrame | None:
    features_path = Path("data/features/player_features.parquet")
    if not features_path.exists():
        return None
    return pd.read_parquet(features_path)


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;900&family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
"""

CSS = """<style>
:root {
    --paper: #17140f;
    --paper-2: #1f1c17;
    --paper-3: #28241d;
    --ink: #f0ead6;
    --ink-80: rgba(240,234,214,.82);
    --ink-60: rgba(240,234,214,.60);
    --ink-40: rgba(240,234,214,.40);
    --ink-20: rgba(240,234,214,.22);
    --ink-10: rgba(240,234,214,.12);
    --ink-05: rgba(240,234,214,.06);
    --accent: #c65b2e;
    --accent-dim: rgba(198,91,46,0.18);
    --good: #2a9d5c;
    --warn: #c9a227;
    --bad: #c44536;
    --display: "Inter Tight", system-ui, sans-serif;
    --body: "Newsreader", "Source Serif Pro", Georgia, serif;
    --mono: "JetBrains Mono", "IBM Plex Mono", ui-monospace, monospace;
}

/* Override Streamlit's default backgrounds and text */
.stApp, [data-testid="stAppViewContainer"], .main .block-container,
[data-testid="stMainBlockContainer"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    background-image:
        radial-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        radial-gradient(rgba(255,255,255,0.015) 1px, transparent 1px);
    background-size: 3px 3px, 7px 7px;
    background-position: 0 0, 1px 2px;
}
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
.main .block-container { max-width: 1320px !important; padding-top: 28px !important; }

/* Override Streamlit widget styling */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
    color: var(--ink-60) !important;
    font-family: var(--mono) !important;
    font-size: 9.5px !important;
    letter-spacing: 0.26em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--paper-2) !important;
    border: 1px solid var(--ink-20) !important;
    color: var(--ink) !important;
    font-family: var(--body) !important;
}
[data-testid="stSelectbox"] > div > div > div {
    color: var(--ink) !important;
}
[data-testid="stExpander"] {
    background: var(--paper-2) !important;
    border: 1px solid var(--ink) !important;
}
[data-testid="stExpander"] summary {
    background: var(--paper-3) !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
}
[data-testid="stExpander"] summary span {
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    letter-spacing: 0.2em !important;
}

/* Slider styling */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--ink-20) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border-radius: 0 !important;
    border: 1px solid var(--ink) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    color: var(--ink) !important;
    font-family: var(--mono) !important;
}

/* Checkbox styling */
[data-testid="stCheckbox"] span[data-testid="stCheckboxLabel"] {
    color: var(--ink-80) !important;
    font-family: var(--body) !important;
    font-style: italic !important;
}

/* Button styling */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--ink) !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 10.5px !important;
    letter-spacing: 0.18em !important;
    border-radius: 0 !important;
    padding: 8px 14px !important;
}
.stButton > button:hover {
    background: var(--ink) !important;
    color: var(--paper) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ---------- Custom editorial classes ---------- */
.masthead {
    padding: 18px 0 26px;
    border-bottom: 3px double var(--ink);
    text-align: center;
}
.masthead-top {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.12em;
    color: var(--ink-60);
    padding-bottom: 18px;
    border-bottom: 1px solid var(--ink-20);
    margin-bottom: 22px;
}
.masthead-title {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
}
.masthead-ornament {
    font-family: var(--mono);
    color: var(--ink-40);
    font-size: 13px;
    letter-spacing: 0.6em;
}
.mh-line1 {
    display: block;
    font-family: var(--display);
    font-size: clamp(40px, 6vw, 84px);
    font-style: italic;
    font-weight: 500;
    letter-spacing: -0.04em;
}
.mh-line2 {
    display: block;
    font-family: var(--display);
    font-size: clamp(64px, 11vw, 148px);
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 0.85;
    margin-top: -6px;
}
.masthead-tagline {
    margin-top: 14px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.32em;
    color: var(--ink-80);
}
.masthead-stats {
    margin-top: 18px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--ink-60);
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.masthead-stats b {
    color: var(--ink);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.sep { color: var(--ink-40); }

/* Section headings */
.section-head { margin: 56px 0 22px; }
.section-head-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}
.section-num {
    font-family: var(--display);
    font-style: italic;
    font-size: 20px;
    color: var(--accent);
    font-weight: 700;
}
.section-rule {
    flex: 0 0 60px;
    height: 1px;
    background: var(--ink);
}
.section-kicker {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.24em;
    color: var(--ink-60);
}
.section-title {
    margin: 0 0 6px;
    font-family: var(--display);
    font-weight: 700;
    font-size: clamp(28px, 3.2vw, 42px);
    line-height: 1.05;
    letter-spacing: -0.04em;
    max-width: 900px;
}
.section-sub {
    max-width: 780px;
    margin: 8px 0 0;
    color: var(--ink-80);
    font-family: var(--body);
    font-size: 16px;
    font-style: italic;
    line-height: 1.5;
}

/* Anchor card */
.anchor-card {
    position: relative;
    background: var(--paper-2);
    border: 1.5px solid var(--ink);
    padding: 22px 24px 24px;
}
.anchor-label {
    position: absolute;
    top: -10px;
    left: 18px;
    background: var(--paper);
    padding: 0 10px;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.26em;
    color: var(--ink);
}
.portrait-frame {
    aspect-ratio: 3/4;
    background:
        repeating-linear-gradient(135deg, transparent 0 7px, rgba(240,234,214,0.05) 7px 8px),
        var(--paper-3);
    border: 1px solid var(--ink);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 14px;
    position: relative;
    max-width: 180px;
}
.portrait-frame::before {
    content: "";
    position: absolute;
    inset: 5px;
    border: 1px solid var(--ink-40);
    pointer-events: none;
}
.portrait-abbr {
    font-family: var(--display);
    font-weight: 900;
    font-size: 64px;
    line-height: 0.85;
    letter-spacing: -0.04em;
    color: var(--ink);
    position: relative;
    z-index: 1;
}
.portrait-num {
    align-self: flex-end;
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: 0.2em;
    color: var(--ink-60);
    position: relative;
    z-index: 1;
}
.portrait-caption {
    margin-top: 8px;
    display: flex;
    justify-content: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    color: var(--ink-60);
    text-transform: uppercase;
}
.statline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    margin-top: 6px;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--ink);
}
.stat-cell {
    padding: 10px 8px;
    text-align: center;
    border-right: 1px solid var(--ink-20);
}
.stat-cell:last-child { border-right: none; }
.stat-num {
    font-family: var(--display);
    font-weight: 700;
    font-size: 26px;
    line-height: 1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
}
.stat-lbl {
    margin-top: 3px;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.2em;
    color: var(--ink-60);
}
.awards-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.award-pill {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    padding: 4px 10px;
    border: 1px solid var(--ink);
    background: var(--paper);
    text-transform: uppercase;
    color: var(--ink);
}

/* Editor's note */
.editor-note {
    border-top: 3px double var(--ink);
    padding-top: 18px;
}
.editor-note-flag {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.26em;
    color: var(--accent);
    margin-bottom: 10px;
    font-weight: 700;
}
.editor-note p {
    margin: 0 0 12px;
    font-family: var(--body);
    font-size: 15px;
    line-height: 1.55;
    color: var(--ink-80);
}
.editor-note p:first-of-type::first-letter {
    font-family: var(--display);
    font-weight: 900;
    font-size: 52px;
    float: left;
    line-height: 0.85;
    padding: 4px 8px 0 0;
    color: var(--ink);
}

/* Comparison panel */
.comparison-panel {
    border: 1.5px solid var(--ink);
    background: var(--paper-2);
    padding: 28px 28px 24px;
}
.comparison-heads {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 24px;
    align-items: center;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--ink);
    margin-bottom: 24px;
}
.ch-a { text-align: left; }
.ch-b { text-align: right; }
.ch-kicker {
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.26em;
    margin-bottom: 4px;
}
.ch-a .ch-kicker { color: var(--ink); }
.ch-b .ch-kicker { color: var(--accent); }
.ch-name {
    font-family: var(--display);
    font-weight: 700;
    font-size: clamp(22px, 2.4vw, 30px);
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--ink);
}
.ch-meta {
    margin-top: 4px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.14em;
    color: var(--ink-60);
}
.ch-score { text-align: center; padding: 0 12px; }
.ch-score-num {
    font-family: var(--display);
    font-weight: 900;
    font-size: clamp(60px, 6vw, 84px);
    line-height: 0.85;
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
}
.ch-score-lbl {
    margin-top: 4px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.26em;
    color: var(--ink-60);
}
.ch-score-tag {
    margin-top: 4px;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.26em;
    font-weight: 700;
}

/* Similarity bars */
.simbars { display: flex; flex-direction: column; gap: 7px; }
.simbar-row { display: grid; grid-template-columns: 150px 1fr; gap: 14px; align-items: center; }
.simbar-label { text-align: right; }
.simbar-name { display: block; font-family: var(--body); font-size: 13.5px; font-weight: 600; line-height: 1.1; color: var(--ink); }
.simbar-group {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.16em;
    color: var(--ink-60);
    text-transform: uppercase;
}
.simbar-track {
    height: 22px;
    background: var(--paper-3);
    border: 1px solid var(--ink-20);
    position: relative;
    overflow: hidden;
}
.simbar-fill { height: 100%; transition: width 0.35s ease; }
.simbar-val {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

/* Stat breakdown */
.stat-breakdown {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 24px;
    margin-top: 14px;
}
.stat-cat-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.stat-cat-name {
    font-family: var(--display);
    font-style: italic;
    font-size: 16px;
    font-weight: 600;
    color: var(--ink);
}
.stat-cat-rule {
    flex: 1;
    height: 1px;
    background: var(--ink);
}
.detail-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--body);
    font-variant-numeric: tabular-nums;
}
.detail-table th {
    text-align: left;
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.2em;
    color: var(--ink-60);
    font-weight: 500;
    padding: 6px 8px;
    border-bottom: 1px solid var(--ink);
}
.detail-table th.r { text-align: right; }
.detail-table td {
    padding: 7px 8px;
    font-size: 13.5px;
    border-bottom: 1px solid var(--ink-10);
    color: var(--ink);
}
.detail-table .c-stat {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--ink-80);
}
.detail-table .c-val, .detail-table .c-diff { text-align: right; font-weight: 500; }
.detail-table .c-diff {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--ink-60);
    width: 56px;
}
tr.match-strong { background: rgba(42,157,92,0.10); }
tr.match-strong .c-diff { color: var(--good); }
tr.match-mid { background: rgba(201,162,39,0.08); }
tr.match-mid .c-diff { color: var(--warn); }
tr.match-weak { background: rgba(196,69,54,0.08); }
tr.match-weak .c-diff { color: var(--bad); }

/* Colophon */
.colophon {
    margin-top: 60px;
    padding-top: 14px;
    border-top: 3px double var(--ink);
}
.colophon-rule {
    width: 40%;
    height: 1px;
    background: var(--ink);
    margin: 0 auto 14px;
}
.colophon-body {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 32px;
    padding: 4px 0 18px;
}
.colo-head {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.26em;
    color: var(--accent);
    margin-bottom: 6px;
}
.colophon-body p {
    margin: 0;
    font-family: var(--body);
    font-size: 13px;
    line-height: 1.55;
    color: var(--ink-80);
    font-style: italic;
}
.colophon-body code {
    font-family: var(--mono);
    font-size: 11px;
    font-style: normal;
    background: var(--ink-05);
    padding: 1px 5px;
}
.colophon-foot {
    text-align: center;
    padding: 20px 0 4px;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.4em;
    color: var(--ink-40);
}

/* Results table footer */
.results-foot {
    padding: 12px 4px;
    font-family: var(--mono);
    font-size: 10.5px;
    letter-spacing: 0.14em;
    color: var(--ink-60);
}

/* Weight panel intro */
.weight-intro {
    font-family: var(--body);
    font-size: 13px;
    font-style: italic;
    color: var(--ink-60);
    line-height: 1.5;
    margin-bottom: 14px;
}
.weight-desc-line {
    font-family: var(--mono);
    font-size: 9.5px;
    color: var(--ink-60);
    letter-spacing: 0.08em;
    margin-top: 2px;
}

@media (max-width: 980px) {
    .comparison-heads { grid-template-columns: 1fr; gap: 22px; text-align: center; }
    .ch-a, .ch-b { text-align: center; }
    .colophon-body { grid-template-columns: 1fr; }
}
</style>
"""

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_masthead(total_seasons: int, total_players: int, issue_no: str):
    today = date.today()
    date_str = today.strftime("%A, %B %d, %Y").upper()
    st.markdown(_clean(f"""
    <div class="masthead">
        <div class="masthead-top">
            <span>VOL. III · ISSUE {issue_no}</span>
            <span>{date_str}</span>
            <span>EST. 2003</span>
        </div>
        <div class="masthead-title">
            <div class="masthead-ornament">★ ★ ★</div>
            <div>
                <span class="mh-line1">The Season</span>
                <span class="mh-line2">Almanac</span>
            </div>
            <div class="masthead-ornament">★ ★ ★</div>
        </div>
        <div class="masthead-tagline">
            A COMPARATIVE INDEX OF EVERY PLAYER-SEASON · 2003–2026
        </div>
        <div class="masthead-stats">
            <span><b>{total_seasons:,}</b> player-seasons</span>
            <span class="sep">·</span>
            <span><b>{total_players:,}</b> players</span>
            <span class="sep">·</span>
            <span><b>190</b> features</span>
            <span class="sep">·</span>
            <span><b>11</b> dimensions</span>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_section_head(number: str, kicker: str, title: str, sub: str = ""):
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(_clean(f"""
    <div class="section-head">
        <div class="section-head-row">
            <span class="section-num">§ {number}</span>
            <span class="section-rule"></span>
            <span class="section-kicker">{kicker}</span>
        </div>
        <h2 class="section-title">{title}</h2>
        {sub_html}
    </div>
    """), unsafe_allow_html=True)


def render_anchor_portrait(abbr: str, year: int, team: str):
    st.markdown(_clean(f"""
    <div class="portrait-frame">
        <div class="portrait-abbr">{abbr}</div>
        <div class="portrait-num">#{year}</div>
    </div>
    <div class="portrait-caption">
        <span>FIGURE A</span><span>·</span><span>{team}</span>
    </div>
    """), unsafe_allow_html=True)


def render_statline(pts, ast, reb, ts_pct, usg_pct):
    ts_val = f"{ts_pct * 100:.1f}" if pd.notna(ts_pct) else "—"
    usg_val = f"{usg_pct * 100:.1f}" if pd.notna(usg_pct) else "—"
    st.markdown(_clean(f"""
    <div class="statline">
        <div class="stat-cell"><div class="stat-num">{pts:.1f}</div><div class="stat-lbl">PTS</div></div>
        <div class="stat-cell"><div class="stat-num">{ast:.1f}</div><div class="stat-lbl">AST</div></div>
        <div class="stat-cell"><div class="stat-num">{reb:.1f}</div><div class="stat-lbl">REB</div></div>
        <div class="stat-cell"><div class="stat-num">{ts_val}</div><div class="stat-lbl">TS%</div></div>
        <div class="stat-cell"><div class="stat-num">{usg_val}</div><div class="stat-lbl">USG%</div></div>
    </div>
    """), unsafe_allow_html=True)


def render_awards(pills: list[str]):
    if not pills:
        return
    pills_html = "".join(f'<span class="award-pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="awards-row">{pills_html}</div>', unsafe_allow_html=True)


def render_results_table_html(results_data: list[dict], selected_idx: int) -> str:
    """Build the results table as raw HTML."""
    rows_html = []
    for i, r in enumerate(results_data):
        selected_cls = "selected" if i == selected_idx else ""
        rank_color = "color: var(--accent);" if i == selected_idx else "color: var(--ink-60);"
        team_color = TEAM_COLORS.get(r["team"], "#f0ead6")
        score = r["score"]
        sc = score_color_hex(score)
        bar_w = max(0, min(100, score))
        txt_color = "#0a0a0a" if score >= 45 else "#fff"
        btn_cls = "active" if i == selected_idx else ""
        btn_label = "▸ VIEWING" if i == selected_idx else "COMPARE"
        btn_bg = f"background: var(--accent); border-color: var(--accent); color: var(--ink);" if i == selected_idx else ""

        ts_display = f"{r['ts'] * 100:.1f}" if r.get("ts") and not math.isnan(r["ts"]) else "—"
        usg_display = f"{r['usg'] * 100:.1f}" if r.get("usg") and not math.isnan(r["usg"]) else "—"

        bg_style = "background: var(--accent-dim);" if i == selected_idx else ""
        border_style = "border-bottom-color: var(--accent);" if i == selected_idx else ""

        rows_html.append(f"""
        <tr style="cursor:pointer; transition: background 0.08s; {bg_style}"
            onmouseover="this.style.background='{bg_style or 'var(--ink-05)'}';"
            onmouseout="this.style.background='{bg_style.split(';')[0] + ';' if bg_style else ''}';"
            onclick="
                const params = new URLSearchParams(window.parent.location.search);
                params.set('compare_idx', '{i}');
                window.parent.history.replaceState(null, '', '?' + params.toString());
                window.parent.document.querySelectorAll('button[data-testid]').forEach(b => {{
                    if (b.innerText && b.innerText.includes('REFRESH')) b.click();
                }});
            ">
            <td style="width:44px; {border_style}">
                <span style="font-family: var(--display); font-weight: 700; font-size: 22px;
                             font-variant-numeric: tabular-nums; letter-spacing: -0.03em; {rank_color}">
                    {str(i + 1).zfill(2)}
                </span>
            </td>
            <td style="min-width:230px; {border_style}">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:34px; height:34px; border:1.5px solid {team_color}; color:{team_color};
                                display:grid; place-items:center; font-family:var(--mono); font-size:11px;
                                font-weight:700; letter-spacing:0.04em; flex-shrink:0; background:var(--paper);">
                        {r['abbr']}
                    </div>
                    <div>
                        <div style="font-family:var(--body); font-weight:600; font-size:15px; line-height:1.1; color:var(--ink);">
                            {r['name']}
                        </div>
                        <div style="font-family:var(--mono); font-size:10px; letter-spacing:0.16em; color:var(--ink-60); margin-top:2px;">
                            {r['team']} · {r.get('pos', '')}
                        </div>
                    </div>
                </div>
            </td>
            <td style="width:110px; {border_style}">
                <div style="display:flex; flex-direction:column;">
                    <span style="font-family:var(--body); font-weight:600; font-size:15px; letter-spacing:0.02em; color:var(--ink);">{r['season']}</span>
                    <span style="font-family:var(--mono); font-size:10px; color:var(--ink-60); letter-spacing:0.1em;">Y{r['year']}</span>
                </div>
            </td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r['age']}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; font-weight:600; font-size:15.5px; color:var(--ink); {border_style}">{r['pts']:.1f}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r['ast']:.1f}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r['reb']:.1f}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{ts_display}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{usg_display}</td>
            <td style="width:180px; {border_style}">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="flex:1; height:18px; background:var(--paper-3); border:1px solid var(--ink-20); position:relative; overflow:hidden;">
                        <div style="height:100%; width:{bar_w}%; background:{sc}; transition:width 0.3s ease;"></div>
                    </div>
                    <span style="font-family:var(--mono); font-size:13px; font-weight:700; width:28px; text-align:right;
                                 font-variant-numeric:tabular-nums; color:{sc};">{score:.0f}</span>
                </div>
            </td>
            <td style="width:104px; text-align:right; {border_style}">
                <span style="font-family:var(--mono); font-size:10px; letter-spacing:0.18em; padding:6px 10px;
                             border:1px solid var(--ink); cursor:pointer; white-space:nowrap; {btn_bg}">
                    {btn_label}
                </span>
            </td>
        </tr>
        """)

    return _clean(f"""
    <div style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-family:var(--body); font-variant-numeric:tabular-nums;">
        <thead>
            <tr>
                <th style="text-align:left; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">RK</th>
                <th style="text-align:left; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">PLAYER</th>
                <th style="text-align:left; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">SEASON</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">AGE</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">PTS</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">AST</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">REB</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">TS%</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">USG%</th>
                <th style="text-align:left; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);">SIMILARITY</th>
                <th style="text-align:right; font-family:var(--mono); font-size:10px; letter-spacing:0.2em; color:var(--ink-60);
                           font-weight:500; padding:14px 12px 10px; border-bottom:1.5px solid var(--ink); background:var(--paper);"></th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
    </div>
    """)


def render_similarity_bars(group_distances: dict) -> str:
    bars_html = []
    for dim in DIMENSIONS:
        dist = group_distances.get(dim["key"], 0)
        score = 100 / (1 + dist) if dist > 0 else 100
        sc = score_color_hex(score)
        txt_color = "#0a0a0a" if score >= 45 else "#fff"
        bars_html.append(f"""
        <div class="simbar-row">
            <div class="simbar-label">
                <span class="simbar-name">{dim['label']}</span>
                <span class="simbar-group">{dim['group']}</span>
            </div>
            <div class="simbar-track">
                <div class="simbar-fill" style="width:{score:.0f}%; background:{sc};"></div>
                <div class="simbar-val" style="color:{txt_color};">{score:.0f}</div>
            </div>
        </div>
        """)
    return _clean(f'<div class="simbars">{"".join(bars_html)}</div>')


def render_radar_svg(anchor_row, compare_row, anchor_label: str, compare_label: str) -> str:
    size = 360
    cx, cy = size / 2, size / 2
    r = 130
    N = len(DIMENSIONS)

    def point(val, i):
        angle = (2 * math.pi * i / N) - math.pi / 2
        rr = (val / 100) * r
        return cx + math.cos(angle) * rr, cy + math.sin(angle) * rr

    def axis_point(i, mult=1):
        angle = (2 * math.pi * i / N) - math.pi / 2
        return cx + math.cos(angle) * r * mult, cy + math.sin(angle) * r * mult

    # Compute values
    vals_a = [compute_radar_value(anchor_row, d["key"]) for d in DIMENSIONS]
    vals_b = [compute_radar_value(compare_row, d["key"]) for d in DIMENSIONS]

    # Grid rings
    rings = []
    for lvl in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{axis_point(i, lvl)[0]:.1f},{axis_point(i, lvl)[1]:.1f}" for i in range(N))
        dash = "" if lvl == 1.0 else 'stroke-dasharray="2 3"'
        rings.append(f'<polygon points="{pts}" fill="none" stroke="var(--ink-20)" stroke-width="0.75" {dash}/>')

    # Axes
    axes = []
    for i in range(N):
        x2, y2 = axis_point(i, 1)
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--ink-20)" stroke-width="0.5"/>')

    # Shapes
    path_a = " ".join(f"{'M' if i == 0 else 'L'}{point(v, i)[0]:.1f},{point(v, i)[1]:.1f}" for i, v in enumerate(vals_a)) + " Z"
    path_b = " ".join(f"{'M' if i == 0 else 'L'}{point(v, i)[0]:.1f},{point(v, i)[1]:.1f}" for i, v in enumerate(vals_b)) + " Z"

    # Dots
    dots_a = "".join(f'<circle cx="{point(v, i)[0]:.1f}" cy="{point(v, i)[1]:.1f}" r="2.5" fill="var(--ink)"/>' for i, v in enumerate(vals_a))
    dots_b = "".join(f'<circle cx="{point(v, i)[0]:.1f}" cy="{point(v, i)[1]:.1f}" r="2.5" fill="var(--accent)"/>' for i, v in enumerate(vals_b))

    # Labels
    labels = []
    for i, d in enumerate(DIMENSIONS):
        x, y = axis_point(i, 1.18)
        angle = (2 * math.pi * i / N) - math.pi / 2
        ta = "middle" if abs(math.cos(angle)) < 0.1 else ("start" if math.cos(angle) > 0 else "end")
        labels.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{ta}" dominant-baseline="middle" '
                      f'style="font-family:var(--mono); font-size:8.5px; letter-spacing:0.18em; fill:var(--ink-80); font-weight:600;">'
                      f'{d["short"]}</text>')

    svg = f"""
    <div style="display:flex; flex-direction:column; align-items:center;">
        <svg viewBox="0 0 {size} {size}" style="max-width:100%; height:auto;">
            {"".join(rings)}
            {"".join(axes)}
            <path d="{path_b}" fill="var(--accent)" fill-opacity="0.18" stroke="var(--accent)" stroke-width="1.75"/>
            <path d="{path_a}" fill="var(--ink)" fill-opacity="0.15" stroke="var(--ink)" stroke-width="1.75"/>
            {dots_a}
            {dots_b}
            {"".join(labels)}
        </svg>
        <div style="display:flex; flex-direction:column; gap:6px; margin-top:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:20px; height:4px; display:inline-block; background:var(--ink);"></span>
                <span style="font-family:var(--mono); font-size:11px; letter-spacing:0.08em; color:var(--ink);">{anchor_label}</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="width:20px; height:4px; display:inline-block; background:var(--accent);"></span>
                <span style="font-family:var(--mono); font-size:11px; letter-spacing:0.08em; color:var(--ink);">{compare_label}</span>
            </div>
        </div>
    </div>
    """
    return _clean(svg)


def render_stat_breakdown(anchor_row, compare_row, label_a: str, label_b: str) -> str:
    cats_html = []
    for cat in STAT_CATEGORIES:
        rows_html = []
        for stat_name, col_name, is_pct in cat["stats"]:
            v1 = anchor_row.get(col_name)
            v2 = compare_row.get(col_name)
            v1_f = fmt_stat(v1, col_name, is_pct)
            v2_f = fmt_stat(v2, col_name, is_pct)

            # Delta
            diff_str = "—"
            row_cls = ""
            if v1 is not None and v2 is not None and not (isinstance(v1, float) and math.isnan(v1)) and not (isinstance(v2, float) and math.isnan(v2)):
                d = float(v2) - float(v1)
                if is_pct:
                    diff_str = f"{'+' if d >= 0 else ''}{d * 100:.1f}"
                elif col_name in ("height_inches", "weight"):
                    diff_str = f"{'+' if d >= 0 else ''}{d:.0f}"
                else:
                    diff_str = f"{'+' if d >= 0 else ''}{d:.1f}"

                avg = (abs(float(v1)) + abs(float(v2))) / 2
                if avg > 0:
                    diff_pct = abs(float(v1) - float(v2)) / avg
                    if diff_pct < 0.05:
                        row_cls = "match-strong"
                    elif diff_pct < 0.15:
                        row_cls = "match-mid"
                    else:
                        row_cls = "match-weak"

            rows_html.append(f"""
            <tr class="{row_cls}">
                <td class="c-stat">{stat_name}</td>
                <td class="c-val">{v1_f}</td>
                <td class="c-val">{v2_f}</td>
                <td class="c-diff">{diff_str}</td>
            </tr>
            """)

        cats_html.append(f"""
        <div>
            <div class="stat-cat-head">
                <span class="stat-cat-name">{cat['name']}</span>
                <span class="stat-cat-rule"></span>
            </div>
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>STAT</th>
                        <th class="r">{label_a}</th>
                        <th class="r">{label_b}</th>
                        <th class="r">Δ</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows_html)}</tbody>
            </table>
        </div>
        """)

    return _clean(f'<div class="stat-breakdown">{"".join(cats_html)}</div>')


def render_colophon():
    st.markdown(_clean("""
    <footer class="colophon">
        <div class="colophon-rule"></div>
        <div class="colophon-body">
            <div>
                <div class="colo-head">COLOPHON</div>
                <p>
                    Data sourced from stats.nba.com via <code>nba_api</code> with secondary scraping
                    from Basketball Reference for pre-2010 gaps.
                    Features standardized per-group via <code>StandardScaler</code>.
                    Nearest neighbors via weighted Euclidean distance.
                </p>
            </div>
            <div>
                <div class="colo-head">COVERAGE</div>
                <p>
                    2003-04 through 2025-26. Regular season only.
                    Playoff performance is excluded by design.
                    Tracking-data availability varies by season; missing values filled with zeros.
                </p>
            </div>
            <div>
                <div class="colo-head">TYPESET IN</div>
                <p>
                    <b>Inter Tight</b> for display, <b>Newsreader</b> for body copy, and
                    <b>JetBrains Mono</b> for figures.
                </p>
            </div>
        </div>
        <div class="colophon-foot">— 30 —</div>
    </footer>
    """), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Inject fonts and CSS
    st.markdown(FONT_LINKS, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)

    # Load data
    matcher = load_matcher()
    career_df = load_career_features()
    cached_awards = load_cached_awards()

    if matcher is None or career_df is None:
        st.error("No data found. Run the feature pipeline first.")
        return

    total_seasons = len(career_df)
    total_players = career_df["PLAYER_ID"].nunique()
    player_names = sorted(career_df["PLAYER_NAME"].unique())

    # ---- Session state defaults ----
    if "compare_idx" not in st.session_state:
        st.session_state.compare_idx = 0

    # Default weights
    default_weights = {d["key"]: (0.5 if d["key"] == "physical" else 1.0) for d in DIMENSIONS}
    for key, val in default_weights.items():
        if f"w_{key}" not in st.session_state:
            st.session_state[f"w_{key}"] = val

    # ---- Masthead ----
    issue_no = str(213).zfill(3)
    render_masthead(total_seasons, total_players, issue_no)

    # ---- Section I: The Subject ----
    render_section_head(
        "I", "THE SUBJECT",
        "Select a player-season to examine.",
        "Pick any season from 2003-04 through 2025-26. The almanac will locate its nearest "
        "neighbors among more than eleven thousand candidate seasons — ranked by a weighted "
        "distance across 190 features."
    )

    col_anchor, col_editor = st.columns([1.2, 0.9], gap="large")

    with col_anchor:
        # Anchor card
        st.markdown('<div class="anchor-card"><div class="anchor-label">THE QUERY</div>', unsafe_allow_html=True)

        col_portrait, col_meta = st.columns([1, 2])

        with col_portrait:
            selected_player = st.selectbox(
                "PLAYER",
                options=player_names,
                index=player_names.index("Shai Gilgeous-Alexander") if "Shai Gilgeous-Alexander" in player_names else 0,
                key="player_select",
            )

        # Get player data
        player_data = career_df[career_df["PLAYER_NAME"] == selected_player].sort_values("CAREER_YEAR")
        player_id = player_data["PLAYER_ID"].iloc[0]
        seasons = player_data[["CAREER_YEAR", "SEASON", "AGE", "PTS", "AST", "REB", "TEAM_ABBREVIATION"]].to_dict("records")

        season_labels = [
            f"{s['SEASON']} · Yr {int(s['CAREER_YEAR'])}"
            for s in seasons
        ]

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
            st.markdown(f"""<div style="font-family:var(--body); font-size:16px; color:var(--ink-60);
                            margin-top:4px;">Age {age}</div>""", unsafe_allow_html=True)

        # Statline
        ts_pct = anchor_row.get("ts_pct", 0)
        usg_pct = anchor_row.get("e_usg_pct", 0)
        render_statline(anchor_season["PTS"], anchor_season["AST"], anchor_season["REB"], ts_pct, usg_pct)

        # Awards
        award_pills = get_awards_pills(player_id, anchor_season["SEASON"], cached_awards)
        render_awards(award_pills)

        st.markdown('</div>', unsafe_allow_html=True)  # close anchor-card

    with col_editor:
        # Editor's note
        st.markdown(_clean("""
        <div class="editor-note">
            <div class="editor-note-flag">EDITOR'S NOTE</div>
            <p>Similarity is computed across <b>eleven dimensions</b> — scoring, efficiency, shot profile,
            creation, drives, playmaking, ball handling, rebounding, defense, usage, and physical build.</p>
            <p>Each is standardized independently, so a one-unit difference in <i>defense</i> means the same
            as a one-unit difference in <i>playmaking</i>. Adjust the weights below to tell the engine what matters.</p>
        </div>
        """), unsafe_allow_html=True)

        # Weights panel
        with st.expander("§ A   MATCHING WEIGHTS", expanded=False):
            st.markdown('<p class="weight-intro">Each dimension\'s distance is multiplied by its weight '
                        'before combining into the overall score. Crank a slider up to prioritize that aspect of play.</p>',
                        unsafe_allow_html=True)

            custom_weights = {}
            for dim in DIMENSIONS:
                st.markdown(f'<div class="weight-desc-line">{dim["desc"]}</div>', unsafe_allow_html=True)
                custom_weights[dim["key"]] = st.slider(
                    dim["label"],
                    0.0, 3.0, step=0.25,
                    key=f"w_{dim['key']}",
                )

            if st.button("↺ RESET TO EVEN"):
                for key, val in default_weights.items():
                    st.session_state[f"w_{key}"] = val
                st.rerun()

    # ---- Find similar seasons ----
    n_results = st.session_state.get("n_results", 10)
    exclude_same = st.session_state.get("exclude_same", False)

    try:
        fetch_n = n_results + 30
        similar = matcher.find_similar_season(
            player_id,
            season_key=anchor_year,
            n=fetch_n,
            compare_by="year",
            weights=custom_weights,
        )

        results_data = []
        for pid, name, their_year, dist, group_dists in similar:
            if pid == player_id and their_year == anchor_year:
                continue
            if exclude_same and pid == player_id:
                continue
            if len(results_data) >= n_results:
                break

            similarity_score = 100 / (1 + dist)
            info = matcher.get_season_info(pid, their_year, "year")
            if not info:
                continue

            season_row = career_df[(career_df["PLAYER_ID"] == pid) & (career_df["CAREER_YEAR"] == their_year)]
            if season_row.empty:
                continue
            season_row = season_row.iloc[0]

            results_data.append({
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
                "ts": float(season_row.get("ts_pct", 0)) if pd.notna(season_row.get("ts_pct")) else float("nan"),
                "usg": float(season_row.get("e_usg_pct", 0)) if pd.notna(season_row.get("e_usg_pct")) else float("nan"),
                "score": similarity_score,
                "group_distances": group_dists,
                "career_year": their_year,
            })
    except Exception as e:
        st.error(f"Error finding similar seasons: {e}")
        results_data = []

    if not results_data:
        render_colophon()
        return

    # ---- Section II: Nearest Neighbors ----
    render_section_head(
        "II", "NEAREST NEIGHBORS",
        "The closest historical seasons.",
        f"Ranked by similarity score (0–100). {anchor_abbr} {anchor_season['SEASON']} compared against "
        f"{total_seasons - 1:,} candidate seasons."
    )

    # Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        n_results = st.radio(
            "SHOW",
            options=[5, 10, 15, 20],
            index=1,
            horizontal=True,
            key="n_results",
        )
    with ctrl_col2:
        exclude_same = st.checkbox(
            "Exclude other seasons by the same player",
            key="exclude_same",
        )

    # Re-filter if exclude_same changed
    if exclude_same:
        results_data = [r for r in results_data if r["player_id"] != player_id][:n_results]
    else:
        results_data = results_data[:n_results]

    # Compare selection
    compare_idx = st.session_state.get("compare_idx", 0)
    if compare_idx >= len(results_data):
        compare_idx = 0

    compare_options = [
        f"{r['name']} ({r['season']}) — Score: {r['score']:.0f}"
        for r in results_data
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
        f'<div class="results-foot">Showing {len(results_data)} of {total_seasons - 1:,} candidates · '
        f'Use the dropdown above to select a comparison</div>',
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
        f"with {selected_player}'s {anchor_season['SEASON']}."
    )

    # Comparison heads
    st.markdown(_clean(f"""
    <div class="comparison-panel">
        <div class="comparison-heads">
            <div class="ch-a">
                <div class="ch-kicker" style="color:var(--ink);">THE QUERY</div>
                <div class="ch-name">{selected_player}</div>
                <div class="ch-meta">{anchor_season['SEASON']} · {anchor_team} · Age {age}</div>
            </div>
            <div class="ch-score">
                <div class="ch-score-num" style="color:{sc_hex};">{compare_score:.0f}</div>
                <div class="ch-score-lbl">SIMILARITY SCORE</div>
                <div class="ch-score-tag" style="color:{sc_hex};">— {sc_label} —</div>
            </div>
            <div class="ch-b">
                <div class="ch-kicker" style="color:var(--accent);">THE MATCH</div>
                <div class="ch-name">{compare_data['name']}</div>
                <div class="ch-meta">{compare_data['season']} · {compare_data['team']} · Age {compare_data['age']}</div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    # Charts: Similarity bars + Radar
    col_bars, col_radar = st.columns([1.2, 1], gap="large")

    with col_bars:
        st.markdown("""<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. B — SIMILARITY BY CATEGORY</div>""",
                    unsafe_allow_html=True)
        st.markdown(render_similarity_bars(compare_data["group_distances"]), unsafe_allow_html=True)

    with col_radar:
        st.markdown("""<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                    color:var(--ink-60); margin-bottom:14px; padding-bottom:6px;
                    border-bottom:1px dotted var(--ink-20);">FIG. C — PLAYER PROFILE OVERLAY</div>""",
                    unsafe_allow_html=True)

        compare_row = career_df[
            (career_df["PLAYER_ID"] == compare_data["player_id"]) &
            (career_df["CAREER_YEAR"] == compare_data["career_year"])
        ]
        if not compare_row.empty:
            compare_row = compare_row.iloc[0]
            st.markdown(
                render_radar_svg(
                    anchor_row, compare_row,
                    f"{anchor_abbr} {anchor_season['SEASON']}",
                    f"{compare_data['abbr']} {compare_data['season']}",
                ),
                unsafe_allow_html=True,
            )

    # Stat breakdown
    st.markdown("""<div style="font-family:var(--mono); font-size:10.5px; letter-spacing:0.22em;
                color:var(--ink-60); margin:24px 0 14px; padding-bottom:6px; padding-top:22px;
                border-top:1px solid var(--ink-20);
                border-bottom:1px dotted var(--ink-20);">FIG. D — STAT-LINE BREAKDOWN</div>""",
                unsafe_allow_html=True)

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
