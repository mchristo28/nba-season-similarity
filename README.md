# The Season Almanac

Compare NBA player-seasons with weighted statistical dimensions, a transparent
0–100 similarity scale, and side-by-side profiles. The bundled snapshot includes
11,424 seasons from 2,353 players, spanning 2003–04 through a **partial 2025–26**.
It is stored data, not a live feed.

[Open the hosted app](https://nba-season-similarity.streamlit.app)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/streamlit_app.py
```

Dependencies are declared in `pyproject.toml`; `requirements.txt` installs the
project. The feature snapshot is included, so startup does not require API calls.

## Matching and scores

Eleven dimensions cover scoring, efficiency, shot location, creation, drives,
passing, touches, rebounding, defense, usage, and physical measurements. The UI
reports the actual available matching-feature count; the parquet's total column
count also includes metadata and features not used by this matcher.

Within each dimension, matching uses RMS differences of standardized features.
Group weights control their influence. Missing observations are excluded and
coverage is displayed separately. Games, minutes, season range, same-player
exclusion, and minimum shared coverage are configurable.

Scores use `100 × 2^(-distance²)`: 100 is identical on measured features, about 84
is a half-standard-deviation difference, and 50 is one standard deviation. Scores
are not percentages of identical play, percentiles, or grades. The closest match
for a unique player can still have a moderate score. See [scoring details](docs/scoring.md).

## Rebuild data

Rebuild from existing local processed data and cached actual team totals:

```bash
python -m src.features.build_features
```

This requires `data/processed/comprehensive_stats.parquet` and uses team-total
caches in `data/raw`. If team totals are unavailable, shares remain missing.
Provide `--team-stats path/to/team_totals.parquet` to supply a table with `TEAM_ID`,
`SEASON`, `GP`, and season-total stat columns.

Fetch a new source snapshot and publish validated features:

```bash
python -m src.features.build_features --fetch --start-year 2003 --end-year 2025
```

This makes many rate-limited NBA API requests and may take substantial time.
Optional endpoint failures leave missing measurements; missing entire seasons or
invalid output schemas prevent publishing the app dataset. `--input` and `--output`
select alternate paths. Feature publication is atomic. The adjacent JSON records
build time and, for a fresh fetch, source update time. Rebuilding old inputs does
not claim to refresh them.

The older `src.features.feature_pipeline` produces **career aggregates** in
`career_aggregate_features.parquet`; it cannot overwrite the app's season file.

## Development

```bash
pip install '.[dev,trajectory]'
ruff check src tests
python -m pytest -q
python scripts/audit_scores.py
```

Tests cover matching invariants, missingness, score reference points, units, team
shares, schema protection, ingestion merges, and Streamlit control interactions.
CI runs on Python 3.10 and 3.12.

## Structure

- `src/features/registry.py`: shared feature groups, weights, and display metadata.
- `src/features/transforms.py`: units, efficiency, career-year, and team-share transformations.
- `src/similarity/weighted_matcher.py`: vectorized season and career matching.
- `src/similarity/scoring.py`: shared aggregation and 0–100 score mapping.
- `src/app/streamlit_app.py`: page state and orchestration.
- `src/app/presentation.py`, `styles.py`: display components and styling.
- `src/data/`: NBA clients, cached data access, and comprehensive ingestion.

Experimental career strategies have explicit names: `AlignedTrajectoryMatcher`
in `trajectory_matching.py` and `HybridTrajectoryMatcher` in `trajectory_matcher.py`.
Their historical `TrajectoryMatcher` imports remain aliases. DTW is an optional
`trajectory` extra; the hybrid matcher falls back to aligned matching without it.
These strategies and the outcome projector are not exposed in the season UI.
Basketball Reference scraping and era adjustment modules remain unimplemented
extension points; the app does not use them or claim their data.

## Data limitations

- The included 2025–26 snapshot is incomplete; the app displays its maximum player games.
- Drives/touches are absent before 2020–21 in this snapshot; other coverage varies.
- Multi-team season shares are unavailable without team-stint data.
- Regular-season comparisons only. Scores do not adjust for every era or rule change.
- Statistical similarity is sensitive to the selected dimensions and available data.
