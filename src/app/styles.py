"""Season Almanac styles."""

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
[data-testid="stExpander"] summary span:not([data-testid="stIconMaterial"]) {
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    color: var(--ink-60) !important;
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
.comparison-inputs {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
}
.comparison-inputs .detail-table .c-gap { font-size: 11px; min-width: 65px; }
.comparison-inputs .detail-table .c-val { white-space: nowrap; }
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
tr.match-strong .c-diff, tr.match-strong .c-gap { color: var(--good); }
tr.match-mid { background: rgba(201,162,39,0.08); }
tr.match-mid .c-diff, tr.match-mid .c-gap { color: var(--warn); }
tr.match-weak { background: rgba(196,69,54,0.08); }
tr.match-weak .c-diff, tr.match-weak .c-gap { color: var(--bad); }

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
