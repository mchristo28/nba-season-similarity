"""Pure formatting helpers and Streamlit presentation components."""

import math
import re
from datetime import date
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from src.app.comparison_display import BANDS, LABELS, TWO_DECIMAL_FEATURES
from src.features.registry import DIMENSIONS, STAT_CATEGORIES
from src.similarity.scoring import difference_band, similarity_score

TEAM_COLORS = {
    "ATL": "#E03A3E",
    "BOS": "#007A33",
    "BKN": "#000000",
    "CHA": "#1D1160",
    "CHI": "#CE1141",
    "CLE": "#860038",
    "DAL": "#00538C",
    "DEN": "#0E2240",
    "DET": "#C8102E",
    "GSW": "#1D428A",
    "HOU": "#CE1141",
    "IND": "#002D62",
    "LAC": "#C8102E",
    "LAL": "#552583",
    "MEM": "#5D76A9",
    "MIA": "#98002E",
    "MIL": "#00471B",
    "MIN": "#0C2340",
    "NOP": "#0C2340",
    "NYK": "#006BB6",
    "OKC": "#007AC1",
    "ORL": "#0077C0",
    "PHI": "#006BB6",
    "PHX": "#E56020",
    "POR": "#E03A3E",
    "SAC": "#5A2D81",
    "SAS": "#C4CED4",
    "TOR": "#CE1141",
    "UTA": "#002B5C",
    "WAS": "#002B5C",
    # Historical
    "SEA": "#00653A",
    "NJN": "#002A60",
    "NOH": "#0C2340",
    "NOK": "#0C2340",
    "VAN": "#00B2A9",
    "CHH": "#00788C",
}


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
    if s > similarity_score(0.5):
        return "#2a9d5c"
    if s > similarity_score(1.0):
        return "#c9a227"
    return "#c44536"


def _clean(html: str) -> str:
    """Strip leading whitespace from HTML lines to prevent Markdown code-block interpretation."""
    return re.sub(r"\n[ \t]{4,}", "\n", html)


def score_label(s: float) -> str:
    if s >= 100 - 1e-10:
        return "IDENTICAL MEASURED PROFILE"
    if s > similarity_score(0.5):
        return "SMALL MEASURED GAP"
    if s > similarity_score(1.0):
        return "MODERATE MEASURED GAP"
    return "LARGE MEASURED GAP"


def fmt_stat(val, col_name: str, is_pct: bool) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    if col_name == "height_inches":
        return f"{int(val // 12)}'{int(val % 12)}\""
    if col_name == "weight":
        return f"{int(val)}"
    if col_name == "ts_relative":
        return f"{float(val) * 100:+.1f} pp"
    if col_name in TWO_DECIMAL_FEATURES:
        return f"{float(val):.2f}"
    if is_pct:
        return f"{val * 100:.1f}%"
    return f"{val:.1f}"


def compute_radar_value(row, dimension_key: str) -> float:
    """Compute a normalized 0-100 value for the radar chart."""
    required = {
        "scoring_volume": ["PTS"],
        "scoring_efficiency": ["ts_pct"],
        "shot_profile": ["pct_fga_restricted", "pct_fga_paint"],
        "shot_creation": ["PCT_UAST_FGM"],
        "drives": ["DRIVES"],
        "playmaking": ["AST_PTS_CREATED"],
        "ball_handling": ["TOUCHES"],
        "rebounding": ["REB"],
        "defense": ["contested_shots", "deflections", "charges_drawn", "def_loose_balls_recovered"],
        "usage": ["e_usg_pct"],
        "physical": ["height_inches"],
    }
    if any(pd.isna(row.get(c)) for c in required.get(dimension_key, [])):
        return float("nan")

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
            (
                safe("contested_shots") / 10
                + safe("deflections") / 4
                + safe("charges_drawn") / 0.5
                + safe("def_loose_balls_recovered") / 1
            )
            / 4
            * 100
        )
        return min(100, hustle)
    elif dimension_key == "usage":
        return min(100, safe("e_usg_pct") * 100 * 3)
    elif dimension_key == "physical":
        h = safe("height_inches", 78)
        return min(100, max(0, (h - 66) / 18 * 100))
    return 50


def render_masthead(
    total_seasons: int,
    total_players: int,
    issue_no: str,
    feature_count: int,
    dimension_count: int = 11,
):
    today = date.today()
    date_str = today.strftime("%A, %B %d, %Y").upper()
    st.markdown(
        _clean(f"""
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
            <span><b>{feature_count}</b> matching features</span>
            <span class="sep">·</span>
            <span><b>{dimension_count}</b> dimensions</span>
        </div>
    </div>
    """),
        unsafe_allow_html=True,
    )


def render_section_head(number: str, kicker: str, title: str, sub: str = ""):
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    st.markdown(
        _clean(f"""
    <div class="section-head">
        <div class="section-head-row">
            <span class="section-num">§ {number}</span>
            <span class="section-rule"></span>
            <span class="section-kicker">{kicker}</span>
        </div>
        <h2 class="section-title">{title}</h2>
        {sub_html}
    </div>
    """),
        unsafe_allow_html=True,
    )


def render_anchor_portrait(abbr: str, year: int, team: str):
    st.markdown(
        _clean(f"""
    <div class="portrait-frame">
        <div class="portrait-abbr">{abbr}</div>
        <div class="portrait-num">#{year}</div>
    </div>
    <div class="portrait-caption">
        <span>FIGURE A</span><span>·</span><span>{team}</span>
    </div>
    """),
        unsafe_allow_html=True,
    )


def render_statline(pts, ast, reb, ts_pct, usg_pct):
    ts_val = f"{ts_pct * 100:.1f}" if pd.notna(ts_pct) else "—"
    usg_val = f"{usg_pct * 100:.1f}" if pd.notna(usg_pct) else "—"
    st.markdown(
        _clean(f"""
    <div class="statline">
        <div class="stat-cell"><div class="stat-num">{pts:.1f}</div><div class="stat-lbl">PTS</div></div>
        <div class="stat-cell"><div class="stat-num">{ast:.1f}</div><div class="stat-lbl">AST</div></div>
        <div class="stat-cell"><div class="stat-num">{reb:.1f}</div><div class="stat-lbl">REB</div></div>
        <div class="stat-cell"><div class="stat-num">{ts_val}</div><div class="stat-lbl">TS%</div></div>
        <div class="stat-cell"><div class="stat-num">{usg_val}</div><div class="stat-lbl">USG%</div></div>
    </div>
    """),
        unsafe_allow_html=True,
    )


def render_awards(pills: list[str]):
    if not pills:
        return
    pills_html = "".join(f'<span class="award-pill">{p}</span>' for p in pills)
    st.markdown(f'<div class="awards-row">{pills_html}</div>', unsafe_allow_html=True)


def render_results_table_html(results_data: list[dict], selected_idx: int) -> str:
    """Build the results table as raw HTML."""
    rows_html = []
    for i, r in enumerate(results_data):
        rank_color = "color: var(--accent);" if i == selected_idx else "color: var(--ink-60);"
        team_color = TEAM_COLORS.get(r["team"], "#f0ead6")
        score = r["score"]
        sc = score_color_hex(score)
        bar_w = max(0, min(100, score))
        btn_label = "▸ VIEWING" if i == selected_idx else "—"
        btn_bg = (
            "background: var(--accent); border-color: var(--accent); color: var(--ink);"
            if i == selected_idx
            else ""
        )

        ts_display = f"{r['ts'] * 100:.1f}" if r.get("ts") and not math.isnan(r["ts"]) else "—"
        usg_display = f"{r['usg'] * 100:.1f}" if r.get("usg") and not math.isnan(r["usg"]) else "—"

        bg_style = "background: var(--accent-dim);" if i == selected_idx else ""
        border_style = "border-bottom-color: var(--accent);" if i == selected_idx else ""

        rows_html.append(f"""
        <tr style="{bg_style}">
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
                        {r["abbr"]}
                    </div>
                    <div>
                        <div style="font-family:var(--body); font-weight:600; font-size:15px; line-height:1.1; color:var(--ink);">
                            {r["name"]}
                        </div>
                        <div style="font-family:var(--mono); font-size:10px; letter-spacing:0.16em; color:var(--ink-60); margin-top:2px;">
                            {r["team"]} · {r.get("coverage", 0):.0%} data coverage
                        </div>
                    </div>
                </div>
            </td>
            <td style="width:110px; {border_style}">
                <div style="display:flex; flex-direction:column;">
                    <span style="font-family:var(--body); font-weight:600; font-size:15px; letter-spacing:0.02em; color:var(--ink);">{r["season"]}</span>
                    <span style="font-family:var(--mono); font-size:10px; color:var(--ink-60); letter-spacing:0.1em;">Y{r["year"]}</span>
                </div>
            </td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r["age"]}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; font-weight:600; font-size:15.5px; color:var(--ink); {border_style}">{r["pts"]:.1f}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r["ast"]:.1f}</td>
            <td style="width:64px; text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); {border_style}">{r["reb"]:.1f}</td>
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
                             border:1px solid var(--ink); white-space:nowrap; {btn_bg}">
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


def render_similarity_bars(
    group_distances: dict, weights=None, dimensions=None, contributions=None
) -> str:
    bars_html = []
    for dim in dimensions if dimensions is not None else DIMENSIONS:
        dist = group_distances.get(dim["key"])
        disabled = weights is not None and weights.get(dim["key"], 0) == 0
        if disabled or dist is None or not math.isfinite(dist):
            state = "OFF" if disabled else "N/A"
            bars_html.append(
                f'<div class="simbar-row"><span>{dim["label"]}</span> <span>{state}</span></div>'
            )
            continue
        score = similarity_score(dist)
        label = dim["group"]
        if contributions is not None:
            label, _, sc = BANDS[difference_band(dist)]
            total = sum(contributions.values())
            score = 100 * contributions.get(dim["key"], 0) / total if total else 0
            value = f"{score:.0f}% of gap" if score >= 1 or score == 0 else "<1% of gap"
        else:
            sc = score_color_hex(score)
            value = f"{score:.0f}"
        txt_color = "var(--ink)"
        bars_html.append(f"""
        <div class="simbar-row">
            <div class="simbar-label">
                <span class="simbar-name">{dim["label"]}</span>
                <span class="simbar-group">{label}</span>
            </div>
            <div class="simbar-track">
                <div class="simbar-fill" style="width:{score:.0f}%; background:{sc};"></div>
                <div class="simbar-val" style="color:{txt_color};">{value}</div>
            </div>
        </div>
        """)
    return _clean(f'<div class="simbars">{"".join(bars_html)}</div>')


def render_key_differences(anchor, other, explanation):
    """Rank visible disagreements by their actual contribution to overall distance."""
    features = explanation["features"]
    selected = sorted(
        [
            c
            for c, v in features.items()
            if v["enabled"]
            and np.isfinite(v["distance"])
            and v["distance"] >= 0.5
            and v["contribution"] > 0
        ],
        key=lambda c: features[c]["contribution"],
        reverse=True,
    )[:3]
    if not selected:
        return "<p>No noticeable gaps in the shared, enabled measurements.</p>"
    items = []
    for column in selected:
        name, pct = LABELS[column]
        label, _, color = BANDS[difference_band(features[column]["distance"])]
        left, right = (
            fmt_stat(anchor.get(column), column, pct),
            fmt_stat(other.get(column), column, pct),
        )
        items.append(
            f"<p><b>{escape(name)}</b><br>{escape(left)} → {escape(right)}<br>"
            f'<span style="color:{color}">{label}</span></p>'
        )
    return "".join(items)


def render_radar_svg(anchor_row, compare_row, anchor_label: str, compare_label: str) -> str:
    size = 440
    cx, cy = size / 2, size / 2
    r = 140
    dimensions = [
        d
        for d in DIMENSIONS
        if np.isfinite(compute_radar_value(anchor_row, d["key"]))
        and np.isfinite(compute_radar_value(compare_row, d["key"]))
    ]
    if len(dimensions) < 3:
        return "Not enough shared measurements for a profile overlay."
    N = len(dimensions)

    def point(val, i):
        angle = (2 * math.pi * i / N) - math.pi / 2
        rr = (val / 100) * r
        return cx + math.cos(angle) * rr, cy + math.sin(angle) * rr

    def axis_point(i, mult=1):
        angle = (2 * math.pi * i / N) - math.pi / 2
        return cx + math.cos(angle) * r * mult, cy + math.sin(angle) * r * mult

    # Compute values
    vals_a = [compute_radar_value(anchor_row, d["key"]) for d in dimensions]
    vals_b = [compute_radar_value(compare_row, d["key"]) for d in dimensions]

    # Grid rings
    rings = []
    for lvl in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{axis_point(i, lvl)[0]:.1f},{axis_point(i, lvl)[1]:.1f}" for i in range(N))
        dash = "" if lvl == 1.0 else 'stroke-dasharray="2 3"'
        rings.append(
            f'<polygon points="{pts}" fill="none" stroke="var(--ink-20)" stroke-width="0.75" {dash}/>'
        )

    # Axes
    axes = []
    for i in range(N):
        x2, y2 = axis_point(i, 1)
        axes.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--ink-20)" stroke-width="0.5"/>'
        )

    # Shapes
    path_a = (
        " ".join(
            f"{'M' if i == 0 else 'L'}{point(v, i)[0]:.1f},{point(v, i)[1]:.1f}"
            for i, v in enumerate(vals_a)
        )
        + " Z"
    )
    path_b = (
        " ".join(
            f"{'M' if i == 0 else 'L'}{point(v, i)[0]:.1f},{point(v, i)[1]:.1f}"
            for i, v in enumerate(vals_b)
        )
        + " Z"
    )

    # Dots
    dots_a = "".join(
        f'<circle cx="{point(v, i)[0]:.1f}" cy="{point(v, i)[1]:.1f}" r="2.5" fill="var(--ink)"/>'
        for i, v in enumerate(vals_a)
    )
    dots_b = "".join(
        f'<circle cx="{point(v, i)[0]:.1f}" cy="{point(v, i)[1]:.1f}" r="2.5" fill="var(--accent)"/>'
        for i, v in enumerate(vals_b)
    )

    # Labels
    labels = []
    for i, d in enumerate(dimensions):
        x, y = axis_point(i, 1.18)
        angle = (2 * math.pi * i / N) - math.pi / 2
        ta = "middle" if abs(math.cos(angle)) < 0.1 else ("start" if math.cos(angle) > 0 else "end")
        labels.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{ta}" dominant-baseline="middle" '
            f'style="font-family:var(--mono); font-size:8.5px; letter-spacing:0.18em; fill:var(--ink-80); font-weight:600;">'
            f"{d['short']}</text>"
        )

    svg = f"""
    <div style="display:flex; flex-direction:column; align-items:center;">
        <p>Descriptive profile; only shared measurements shown. Axis scales differ from similarity scores.</p>
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


def render_stat_breakdown(
    anchor_row, compare_row, label_a: str, label_b: str, categories=None, evidence=None
) -> str:
    cats_html = []
    for cat in categories if categories is not None else STAT_CATEGORIES:
        rows_html = []
        for stat_name, col_name, is_pct in cat["stats"]:
            v1 = anchor_row.get(col_name)
            v2 = compare_row.get(col_name)
            v1_f = fmt_stat(v1, col_name, is_pct)
            v2_f = fmt_stat(v2, col_name, is_pct)

            # Delta
            diff_str = "—"
            row_cls = ""
            gap_cell = ""
            if evidence is not None:
                item = evidence.get(col_name)
                if item is None or not item["enabled"]:
                    band_label = "Not scored"
                else:
                    band_label, row_cls, _ = BANDS[difference_band(item["distance"])]
                gap_cell = f'<td class="c-gap">{band_label}</td>'
            if (
                v1 is not None
                and v2 is not None
                and not (isinstance(v1, float) and math.isnan(v1))
                and not (isinstance(v2, float) and math.isnan(v2))
            ):
                d = float(v2) - float(v1)
                if is_pct:
                    diff_str = f"{'+' if d >= 0 else ''}{d * 100:.1f}"
                elif col_name in TWO_DECIMAL_FEATURES:
                    diff_str = f"{d:+.2f}"
                elif col_name in ("height_inches", "weight"):
                    diff_str = f"{'+' if d >= 0 else ''}{d:.0f}"
                else:
                    diff_str = f"{'+' if d >= 0 else ''}{d:.1f}"

            rows_html.append(f"""
            <tr class="{row_cls}">
                <td class="c-stat">{stat_name}</td>
                <td class="c-val">{v1_f}</td>
                <td class="c-val">{v2_f}</td>
                <td class="c-diff">{diff_str}</td>
                {gap_cell}
            </tr>
            """)

        cats_html.append(f"""
        <div>
            <div class="stat-cat-head">
                <span class="stat-cat-name">{cat["name"]}</span>
                <span class="stat-cat-rule"></span>
            </div>
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>STAT</th>
                        <th class="r">{label_a}</th>
                        <th class="r">{label_b}</th>
                        <th class="r">Δ</th>
                        {"<th>GAP</th>" if evidence is not None else ""}
                    </tr>
                </thead>
                <tbody>{"".join(rows_html)}</tbody>
            </table>
        </div>
        """)

    extra_class = " comparison-inputs" if evidence is not None else ""
    return _clean(f'<div class="stat-breakdown{extra_class}">{"".join(cats_html)}</div>')


def render_colophon():
    st.markdown(
        _clean("""
    <footer class="colophon">
        <div class="colophon-rule"></div>
        <div class="colophon-body">
            <div>
                <div class="colo-head">COLOPHON</div>
                <p>
                    Data sourced from stats.nba.com via <code>nba_api</code>.
                    Features standardized per-group via <code>StandardScaler</code>.
                    Whole-profile comparisons via a joint weighted RMS of standardized differences.
                </p>
            </div>
            <div>
                <div class="colo-head">COVERAGE</div>
                <p>
                    2003-04 through 2025-26. Regular season only.
                    Playoff performance is excluded by design.
                    Tracking coverage varies by season; unavailable measurements are excluded.
                    Team shares use actual team totals. Multi-team shares are unavailable.
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
    """),
        unsafe_allow_html=True,
    )
