# nba-season-similarity

Who had a season most like Shai Gilgeous-Alexander's 2024-25? According to 190 features across scoring, shot creation, playmaking, defense, and more, the closest matches are Jaylen Brown's 2025-26, De'Aaron Fox's 2020-21, and Ja Morant's 2022-23. This tool finds those answers.

Pick any player-season from 2003-04 through 2025-26, adjust which dimensions matter most, and see the most similar seasons in NBA history -- ranked, visualized, and broken down stat by stat.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-season-similarity.streamlit.app)

## Screenshots

**Search for a player and see the most similar seasons ranked by similarity score:**

![Results table showing similar seasons for Shai Gilgeous-Alexander](screenshots/screenshot_results.png)

**Drill into any match with similarity breakdowns and radar profile overlays:**

![Similarity bars and radar chart comparison](screenshots/screenshot_comparison.png)

## Features

- **190 features per player-season** spanning box score stats, tracking data, hustle stats, shooting zones, and team composition metrics
- **11 matchable dimensions**: scoring volume, efficiency, shot profile, shot creation, drives, playmaking, ball handling, rebounding, defense, usage, physical
- **Adjustable weights**: slider controls to emphasize what matters -- crank up "drives" to find players who attack the basket similarly, or boost "shot profile" to match shooting location distributions
- **Single-season granularity**: compare individual seasons, not career averages
- **Composition stats**: player production expressed as share of team totals (e.g., "25% of team assists"), making stats comparable across eras and team contexts
- **Side-by-side comparison**: color-coded stat tables (green/yellow/red for similarity), radar chart overlays, and per-dimension similarity scores
- **11,400+ player-seasons** from 2,350+ players, 2003-04 through 2025-26

## How it works

1. **Data collection**: Player stats pulled from stats.nba.com via `nba_api` -- career stats, shooting zones, hustle stats, tracking data, draft combine measurements
2. **Feature engineering**: Raw stats transformed into composition metrics (share of team production), shooting zone distributions, shot creation profiles, and defensive hustle composites
3. **Per-group standardization**: Each of the 11 feature groups is standardized independently (StandardScaler), so a 1-unit difference in "defense" means the same thing as a 1-unit difference in "playmaking"
4. **Weighted distance**: Euclidean distance computed per-group, then combined with user-adjustable weights into an overall similarity score

## Technical stack

- **Python 3.10+**
- **pandas / numpy** -- data processing and feature engineering
- **scikit-learn** -- StandardScaler for normalization, NearestNeighbors for similarity search
- **Streamlit** -- interactive web UI
- **Plotly** -- radar charts and similarity bar charts
- **nba_api** -- primary data source (stats.nba.com)
- **BeautifulSoup** -- secondary scraping for gaps (Basketball Reference)
- **Parquet / SQLite** -- caching layer so API calls aren't repeated

## Run locally

```bash
git clone https://github.com/mchristo28/nba-season-similarity.git
cd nba-season-similarity

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run src/app/streamlit_app.py
```

The repo includes pre-computed feature data (3.8 MB), so the app works immediately without needing to run the data pipeline.

To regenerate features from scratch (pulls fresh data from stats.nba.com):

```bash
python -m src.features.build_features
```

## Project structure

```
src/
  data/          # API clients, caching, data loading
  features/      # Composition stats, career vectors, feature pipeline
  similarity/    # Distance metrics, neighbor engine, weighted matcher
  app/           # Streamlit UI
data/
  features/      # Pre-computed feature vectors (included in repo)
```

## Known limitations

- **2003-04 onward only**: Tracking data (drives, touches, time of possession, shot zones, hustle stats) is only available from the mid-2000s. Pre-2003 players are excluded rather than compared on incomplete feature sets.
- **Tracking data availability varies**: Some tracking stats (e.g., drives, contested shots) started in different seasons. Missing values are filled with 0, which can slightly disadvantage early-2000s seasons in those dimensions.
- **No playoff data**: Comparisons are regular season only.
- **Single-season focus**: The tool compares individual seasons, not full careers. Career trajectory matching is partially built (`trajectory_matcher.py`) but not yet exposed in the UI.
