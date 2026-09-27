# How comparison scores work (model 3.0)

A score describes closeness under the selected comparison mode. It is not a grade,
probability, impact estimate, or forecast of equal player quality.

| Mode | Features | Dimensions | Scope |
| --- | --- | --- | --- |
| Playing style — historical | 15 | 8 | 2003–04 onward; shot mix, unassisted makes, foul drawing, assist involvement, turnovers, rebound mix, defensive activity, usage, size |
| Playing style — tracking | 23 | 10 | 2013–14 onward; adds pull-up/catch-and-shoot mix, choices on drives, passing and handling per touch |
| Production | 11 | 7 | 2003–04 onward; scoring, assists, turnovers, offensive/defensive rebounds, steals and blocks per 100 possessions, league-relative efficiency, usage, size |

Style does not use shooting percentages or points per game. Production does not
use minutes or per-game volume as matching features. Size has a smaller default
weight in production; every enabled dimension can be reweighted. These are
statistical comparisons: defensive activity is not a complete defensive-talent
assessment, and movement, screening, and detailed play types remain future work.

## Measurements and units

FG%, 3P%, FT%, eFG%, and TS% are calculated from exact season totals. TS uses the
usual approximation `PTS / (2 × (FGA + 0.44 × FTA))`. No-attempt percentages are
missing. Raw per-game counts and shot-zone attempt distributions are derived from
exact totals, so rounding for display does not affect the calculations.

Production uses the NBA's per-100-on-court-possession measurements. Relative TS
is player TS minus that season's league TS, where league TS is calculated from
summed team totals. A displayed `+4.0 pp` means four percentage points above that
season's league. This accounts for pace and league shooting context, not every
rule change, opponent, or lineup effect. Style compares actual tendencies rather
than era-relative percentiles; users can constrain candidate seasons.

Assist involvement is `AST / (AST + FGA + 0.44 × FTA + TOV)`; turnover tendency is
`TOV / (FGA + 0.44 × FTA + TOV)`. These describe observed involvement, not causal
playmaking talent. Offensive rebound mix is `OREB / REB`. Tracking ratios use
counts in matching units: passes per touch, potential assists per pass, possession
seconds per touch, and pass/shot attempts per drive. These are named ratios, not
claims that every drive is assigned to one exhaustive, exclusive outcome.

Exact totals remain in the dataset for auditing and sample-size interpretation.
There is no unvalidated shrinkage or estimated skill in this release. Small
subject playing-time samples trigger a warning; production also flags fewer than
100 field-goal attempts. Very small event samples can still be noisy even when
the season has many games.

## Distance and score

1. Fit feature means and standard deviations on a fixed reference of seasons
   with at least 20 games and 15 MPG, within the selected mode's date coverage.
   All query and candidate values use that reference. Changing search filters
   does not refit the scales.
2. Compute a root mean square of standardized differences within each dimension.
3. Average dimension distances by the selected weights. Exact zero distances
   retain their weight. Dimensions unknown for the query are excluded.
4. Map distance `d` to `100 × 2^(-d²)`.

| Average group RMS distance | Score | Interpretation |
| --- | --- | --- |
| 0 | 100 | Identical measured profile |
| 0.25 | 95.8 | Very close |
| 0.5 | 84.1 | Close |
| 1 | 50 | Moderate difference |
| 1.5 | 21.0 | Large difference |
| 2 | 6.3 | Very large difference |

This mapping is a transparent design calibration, not an empirically established
measure of basketball equivalence. Top results are not rescaled to 100. Compare
scores within one mode and weight setting; changing mode, data, or model version
changes the reference and/or measurements. The UI identifies model version 3.0.

## Comparable evidence

Within a search, each candidate must have every query-observed measurement in
the enabled dimensions. If the query lacks a measurement, it is omitted for all
candidates. A candidate cannot improve its ranking simply by lacking a statistic
that is available for the query. This can reduce the number of eligible matches.

Coverage is the weighted fraction of the mode's requested features measured for
both seasons. It stays constant across eligible candidates in a search. The
minimum-coverage filter therefore also checks whether the query has enough data
for that mode. Tracking mode requires a 2013–14-or-later query and candidates;
historical mode uses its fixed feature set across the full date range. No missing
measurement is interpreted as zero. Zero counts with a positive denominator are
valid observations; zero-denominator ratios are unknown.

Different-player matches are the default. Games, MPG, age difference, candidate
year range, and shared coverage can be filtered. The ordinary result table shows
per-game stats for context. The comparison breakdown separately lists the actual
normalized inputs; the radar remains a descriptive chart with its own scales.

The original 71-feature matcher is retained as a backward-compatible Python API
when no profile is supplied. The web app and audit command explicitly select the
new profiles. Existing saved legacy models still load with their original setup.

Run `python scripts/audit_scores.py --profile all` for a repeatable, offline audit
across guards, wings, passing bigs, rim-running centers, and defensive facilitators.
It reports results and coverage; plausible names are a review aid, not ground truth
or a target used to tune weights. Tests separately verify exact units, independence
from raw minutes, mode behavior, historical boundaries, and missingness invariants.
