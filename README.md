# The Season Almanac

Compare NBA player-seasons with weighted statistical dimensions, a transparent
0–100 similarity scale, and side-by-side profiles. The bundled snapshot includes
11,501 seasons from 2,385 players, spanning 2003–04 through the **complete 2025–26
regular season**, refreshed September 24, 2026.
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

Choose **Playing style** or **Production**. Historical style uses 15 tendency and
role measurements across eight dimensions from 2003–04 onward. Tracking style
adds creation, drive decisions and handling per touch (23 features, 2013–14 onward).
Production uses 11 features: per-100-possession output, league-relative shooting
efficiency, usage, and size. Exact season totals prevent rounded-per-game shooting
errors. The breakdown shows the actual measurements used by each mode.

Features are standardized against rotation-player seasons (20+ games, 15+ MPG).
Group RMS distances prevent large dimensions from automatically dominating.
Every ranked candidate must have the same query-observed comparison inputs;
missing measurements never count as zero. Other seasons by the same player are
excluded by default. Games, minutes, candidate seasons, age difference, coverage,
and group weights are configurable. Scores should be compared within one mode.

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
caches in `data/raw`. Exact player totals, NBA per-100 rates, possessions, and
actual team totals are required; missing normalization inputs prevent publishing.
Provide `--team-stats path/to/team_totals.parquet` to supply a table with `TEAM_ID`,
`SEASON`, `GP`, and season-total stat columns.

Fetch a new source snapshot and publish validated features:

```bash
python scripts/refresh_data.py --snapshot-dir data/raw/refresh_YYYYMMDD --start-year 2003 --end-year 2025
python scripts/refresh_data.py --snapshot-dir data/raw/awards_YYYYMMDD --end-year 2025 --awards-only
```

Use a new directory for each refresh; reuse that directory only to resume an
interrupted run. Successful responses and their fetch timestamps are cached there.
The refresh retries failed calls and refuses publication when required endpoints,
seasons, existing players, or tracking coverage are missing. It updates the local
processed inputs and actual team totals too, so subsequent rebuilds use fresh data.

The awards command refreshes every player who appeared in the selected season,
retains retired players' historical awards, and rebuilds the displayed badges.
All requests must succeed before publishing. Commit the resulting files under
`data/features` to deploy them. The adjacent JSON files record source fetch dates,
coverage, and refresh scope. Rebuilding existing inputs does not claim to refresh them.

The older `src.features.feature_pipeline` produces **career aggregates** in
`career_aggregate_features.parquet`; it cannot overwrite the app's season file.

## Development

```bash
pip install '.[dev,trajectory]'
ruff check src tests
python -m pytest -q
python scripts/audit_scores.py --profile all
```

Tests cover matching invariants, missingness, score reference points, units, team
shares, schema protection, ingestion merges, and Streamlit control interactions.
CI runs on Python 3.10 and 3.12.

## Structure

- `src/similarity/profiles.py`: explicit style and production feature sets.
- `src/features/comparison.py`: exact totals, per-possession units, and league-relative efficiency.
- `src/features/registry.py`: shared display metadata and legacy feature groups.
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
Basketball Reference scraping and the older `EraAdjuster` class remain unused
extension points. The app’s implemented pace and league-relative adjustments live
in `src/features/comparison.py`.

## Data limitations

- The included 2025–26 regular season is complete (all 30 teams played 82 games).
- Tracking starts in 2013–14. Hustle starts with partial 2015–16 coverage; some source measurements remain unavailable.
- Missing season heights/weights use NBA player-profile measurements where available; fallback player-seasons are listed in the metadata.
- Multi-team season shares are unavailable without team-stint data.
- Regular-season comparisons only. Scores do not adjust for every era or rule change.
- Detailed play types, defensive assignments and estimated shooting skill are not yet included.
- Statistical similarity is sensitive to the selected dimensions and available data.
