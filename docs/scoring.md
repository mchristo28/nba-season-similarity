# How comparison scores work (model 3.2)

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
3. Combine **squared** dimension distances by the selected weights, then take
   the square root. This is one weighted Euclidean distance across the complete
   observed profile, with each dimension's weight divided among its observed
   features. Large mismatches are harder to offset with several small matches.
   Exact zero distances
   retain their weight. Dimensions unknown for the query are excluded.
4. Map distance `d` to `100 × 2^(-d²)`.

For standardized feature differences `z_i` in group `g`, with `n_g` shared
features and group weight `w_g`:

`d² = sum_g [(w_g / sum(w)) × (sum_i z_i² / n_g)]`

Groups organize weights and explanations; the final score is not an arithmetic
average of category scores. This diagonal metric does not learn nonlinear roles
or remove every correlation. See the [aggregate-method review](comparison-geometry-review.md)
for the evaluated correlation-aware alternative and why it was not deployed.

| Joint weighted RMS distance | Score | In standardized units |
| --- | --- | --- |
| 0 | 100 | Identical measured profile |
| 0.25 | 95.8 | Very small gap |
| 0.5 | 84.1 | Half a standard deviation across the profile |
| 1 | 50 | One standard deviation |
| 1.5 | 21.0 | One and a half |
| 2 | 6.3 | Two |

**Scores are not percentiles.** Two random rotation-player seasons are typically about
1.3 SD apart (squared standardized differences average 2 per feature), so a typical
random pair scores about 30. Measured on random pairs from each pool:

| Pool | Median random pair | 90th percentile | 99th percentile |
| --- | --- | --- | --- |
| Tracking, all players | 31 | 61 | 78 |
| Tracking, guards | 29 | 56 | 74 |
| Production, all players | 32 | 68 | 86 |
| Historical style, all players | 32 | 62 | 80 |

A score of 50 is therefore already closer than roughly 95% of random pairs, and the
top match for a well-defined player commonly scores 80–90. The app labels matches by
this pool-relative rank, not the raw score: **very close** beats at least 99% of random
pairs, **close** 95%, **moderate** 80%, otherwise **loose**. The rank is estimated with
a deterministic sample of 300 random reference seasons under the current weights
(`WeightedMatcher.pair_percentile`) and is shown as "Closer than X% of random pairs".

This mapping is a transparent design calibration, not an empirically established
measure of basketball equivalence. Top results are not rescaled to 100. Compare
scores within one mode and weight setting; changing mode, data, or model version
changes the reference and/or measurements. The UI identifies model version 3.2. The pool-relative labels changed in the UI only; distances and scores are unchanged.

## Colors and explanations

Feature colors use the same absolute standardized gaps as the distance:
less than 0.5 SD is **Close** (green), 0.5 to less than 1 is **Noticeable difference**
(yellow), and 1 or more is **Large difference** (red). These are explicitly chosen
display boundaries, not fitted basketball-quality labels or significance tests.
No normal-distribution assumption is needed to express a difference in SD units.
Missing inputs are unavailable; disabled inputs are not scored and remain neutral.

Category bars show each group's share of the **total squared difference**, not
separate similarity scores. Each feature contributes
`(w_g / sum(w)) × z_i² / n_g`; summing contributions recovers `d²` exactly.
Longer bars mean more influence on the gap; their colors describe category RMS
gap size. An exact match has zero contributions throughout. Key differences show
the three largest contributing features whose gaps are at least 0.5 SD.

The normalized input table includes text labels as well as colors. Raw season
stats and the descriptive radar are in a separate context expander; raw stats do
not receive similarity colors. A tracking shortcut offers the richer mode for
eligible queries without silently changing the selected mode.

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

Different-player matches are the default, and only each player's closest season is listed unless
the user turns that filter off (repeated seasons of one player otherwise crowd the top results). Games, MPG, age difference, candidate
year range, and shared coverage can be filtered. The ordinary result table shows
per-game stats for context. The comparison breakdown separately lists the actual
normalized inputs; the radar remains a descriptive chart with its own scales.
New visits default to tracking style. Selecting a pre-2013 query offers a button
to switch to historical data; an explicitly chosen mode is never silently replaced.

The original 71-feature matcher is retained as a backward-compatible Python API
when no profile is supplied. The web app and audit command explicitly select the
new profiles. Existing saved legacy models still load with their original setup.

Run `python scripts/audit_scores.py --profile all` for a repeatable, offline audit
across guards, wings, passing bigs, rim-running centers, and defensive facilitators.
It reports results and coverage; plausible names are a review aid, not ground truth
or a target used to tune weights. Tests separately verify exact units, independence
from raw minutes, mode behavior, historical boundaries, and missingness invariants.

## Position reference (3.2)

Position peers is the default; the pool starts with the query’s listed groups.
Positions to include allows adding other groups. All players preserves the 3.1
distances and rankings, and is the automatic fallback for an unknown query position.
Position peers fits the same
model to rotation-player seasons belonging to a selected roster position group,
and restricts candidates to that group. A G-F or F-C season belongs to both listed
groups; the user can select either group or their union. The chosen reference is
fixed for the entire search and head-to-head explanation, regardless of the
candidate filters. Scores are not interchangeable across different references.

Positions are joined on player ID and season, using all NBA team rosters requested
for that season. Unknown positions stay unavailable, not inferred from height or
current positions. Forward / Wing is deliberately broad: an F listing alone cannot
distinguish a small forward from a power forward or describe time spent in a role.
The coverage audit and source dates are in `data/features/player_positions.json`.
