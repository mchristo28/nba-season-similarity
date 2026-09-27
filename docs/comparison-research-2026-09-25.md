# Comparison model review — September 25, 2026

This review describes the model at commit `1589ed7`, before the model 3.0 implementation. See [current scoring](scoring.md) for shipped behavior.

The application has a useful statistical-similarity foundation, but its single score blends playing style, production, workload, and efficiency. The next improvement should clarify those different questions and correct measurement issues before increasing the number of inputs. Most proposed additions are technically obtainable with the existing NBA API stack: all 11 initial endpoint probes returned data. These were read-only experiments; production behavior was not changed.

**How other approaches define a comparison**

There is no single universal definition of a player comp. Basketball Reference describes its career similarity as matching career quality and shape, not playing style. That answers a different question from this application's single-season search. [Basketball Reference methodology](https://www.basketball-reference.com/about/similar.html).

Basketball Index separates style, offensive role, defensive role, skill, and impact. Its style tool describes 85 variables. Its offensive-role methodology uses deployment and skill indicators; defensive roles use matchup assignments and rim responsibilities. This supports using basketball jobs rather than rigid listed positions as the primary context for style comparisons. It does not mean we should reproduce its proprietary grades. [Style tool](https://www.bball-index.com/data-tools-package/), [offensive roles](https://www.bball-index.com/offensive-archetypes/), [defensive roles](https://www.bball-index.com/defensive-roles/).

Impact is a separate modeling objective. EPM's methodology combines statistical information with adjusted plus-minus and explicitly addresses teammate context and sample noise. Raw on/off or a player's team's defensive rating should not become a direct measure of individual defensive talent. [EPM methodology](https://dunksandthrees.com/about/epm).

**Findings in this repository, in priority order**

1. **Correct percentage calculations.** `src/features/transforms.py:36` divides rounded per-game makes by rounded attempts, then computes TS% and eFG% from similarly rounded inputs. The matcher uses those derived lowercase columns even though official FG%, 3P%, and FT% fields are already present. In the current dataset, Shaun Livingston 2011–12 and Marc Gasol 2015–16 have official 3P% of .667 but derived `fg3_pct` of 0.0. These are tiny shooting samples, which also demonstrates why uncertainty matters. Preserve official percentages or compute from exact season totals; retain exact attempt counts for subsequent reliability adjustments. Do not recover totals by multiplying already-rounded averages by games.

2. **Separate style from production.** Per-game points, attempts, touches, drives, and passes all depend on opportunity. A reserve and starter with similar tendencies can look different simply because their minutes differ. Offer “Playing style” using action frequencies and rates, and “Production” using per-possession output plus efficiency. Treat minutes and offensive burden as explicit context, filters, or a separately weighted role dimension. Do not use a raw cosine comparison of signed z-scores as a shortcut for style.

3. **Adjust cross-era comparisons.** The matcher pools 2003–2026 observations into one `StandardScaler`; `src/features/era_adjustments.py` is unimplemented. Fresh team totals give league TS% of 51.62% in 2003–04 and 58.14% in 2025–26. Thus 56% TS represents very different league-relative efficiency. Use per-100-possession production, league-relative efficiency, and separately labeled era-relative tendencies. Keep absolute style available too: being an unusually frequent three-point shooter in 2003 is not identical to taking the same actual shot mix as a modern player. Compute league efficiency from summed totals, not the unweighted average of player percentages.

4. **Reduce duplicate influence.** In 3,300 player-seasons from 2016–17 onward with at least 20 games and 15 MPG, pairwise Pearson correlations are: assists versus assist-created points .998; points versus point share .995; rebounds versus rebound share .995; points versus FGA .976; drives versus drive FGA .964. Group RMS prevents large groups from automatically outweighing smaller groups, but does not remove repeated signals within or across groups. Prune redundant measures and test alternative feature sets; do not assume correlation alone makes every pair interchangeable. Also reconsider fitting the reference scale on every appearance: 2,000 of 11,501 seasons have fewer than 20 games, although search filters usually exclude them as candidates.

5. **Make historical rankings use comparable evidence.** Missing values are correctly excluded, but a 65%-coverage historical pair and a 100%-coverage modern pair are scored using different evidence. A partly observed dimension retains its full group weight. Coverage disclosure helps, but does not solve ranking comparability. Provide a historical mode with a fixed common feature set and a richer modern mode with explicit required coverage. Avoid imputing uncollected tracking as zero or mechanically treating missingness as dissimilarity. Include an explanation of which evidence contributed to each match.

6. **Account for sample reliability.** Minimum games and MPG do not guarantee enough three-point attempts, defended shots, or pick-and-roll possessions. Retain event denominators, establish per-metric minimums, and evaluate shrinkage toward appropriate priors for estimated-skill views. Show observed values alongside estimates; do not quietly replace raw season performance. Consider multi-season profiles as a separate option. A bootstrap needs game-level data, not repeated resampling of an already aggregated season row.

7. **Add missing basketball roles.** The current inputs describe on-ball creation relatively well, but do not directly distinguish isolation, pick-and-roll handler, roll man, post scorer, cutter, handoff receiver, or off-screen shooter. Defensive counts do not adequately identify perimeter assignments versus rim protection. Rebound chances distinguish opportunity from conversion. Screening is especially inexpensive to improve: `screen_assists` is already stored for 5,602 seasons but is absent from the matching registry. Movement speed is only an activity proxy, not a direct measure of off-ball gravity or athletic ability.

**Data feasibility: actual endpoint probes**

Results below were fetched September 25, 2026. Unless stated otherwise they are for the 2025–26 regular season. A successful sample proves present access to that request, not complete historical coverage or a guaranteed service contract.

| Addition | Tested source | Result | Assessment |
| --- | --- | --- | --- |
| Pace-adjusted production | `LeagueDashPlayerStats`, `Per100Possessions` | 582 players in 2025–26; 442 in 2003–04 | High feasibility; verify intervening seasons during ingestion |
| Pace and possession context | Same endpoint, `Advanced` | 442 players in 2003–04, including PACE/POSS and official efficiency | High; choose consistent units and denominators |
| Offensive play types | `SynergyPlayTypes`, offensive `PRBallHandler` | 347 team-player rows / 319 players in 2025–26; 272 / 255 in 2015–16; also 240 rows in 2013–14 and 274 in 2014–15 | High for sampled seasons; audit other actions and years |
| Rim-defense activity/outcomes | `LeagueDashPtDefend`, `Less Than 6Ft` | 576 players | High access; medium modeling difficulty because assignment and samples matter |
| Rebound opportunities/conversion | `LeagueDashPtStats`, `Rebounding` | 578 players, with chances and contested/uncontested counts | High; separate opportunity and conversion |
| Offensive/defensive movement | Same endpoint, `SpeedDistance` | 582 players | High access; useful context, weak direct talent indicator |
| Shot openness | `LeagueDashPlayerPtShot`, wide-open filter | 566 players, with attempt frequencies and percentages | High; obtain all relevant bins and retain denominators |
| Defensive assignments | `LeagueSeasonMatchups` | 334 opponent matchup rows for Nikola Jokić | Feasible; full-league historical coverage and role inference need a pilot |
| Wingspan and standing reach | `DraftCombinePlayerAnthro`, 2025 combine | 79 prospects with wingspan measurements | Technically easy, population coverage limited; draft measurements are not a complete current-player dataset |
| Screening | Existing published hustle data | 5,602 nonmissing screen-assist rows | Available now; consider rates, samples, and era coverage |

The play-type response has multiple team stints for some players. Combine counts and recompute rates with the appropriate possession denominators; never average percentages blindly or join these rows directly onto a unique player-season table. The smallest returned pick-and-roll sample was 10 possessions; an absent row cannot safely be assumed to mean zero usage. A blank play-type request returned **zero rows**, so ingestion must request explicit action types and validate each response. The client documentation is useful but its listed schema is older than the actual player response. [Client endpoint documentation](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/synergyplaytypes.md).

Play-by-play offers a path to lineup context, possession starts, and on/off comparisons, but needs significantly more ingestion and validation than season endpoints. PBP Stats provides tooling for possession reconstruction and richer events. This route was researched, not live-tested in this review. [PBP Stats documentation](https://pbpstats.readthedocs.io/en/stable/index.html).

Detailed gravity, scheme recognition, and comparable film scouting grades are not established by these probes. Proprietary Basketball Index or EPM data would require investigating programmatic access and permitted app usage; do not make those dependencies for the first iteration. Public summary endpoints already support most of the proposed first phase. The cited subscription tool descriptions do not establish redistribution rights.

**Recommended implementation sequence**

- **First release:** fix exact percentages; retain event totals; add pace/era normalization; separate style and production presets; reduce correlated inputs; offer historical versus modern feature sets; make different-player comps the default while retaining self-comparison as an option. Add age/career-stage filters for users seeking developmental peers.
- **Second release:** ingest explicit offensive play types, activate screening, add rebound chances, shot-openness bins, and rim-defense activity. Build transparent role descriptions from action frequencies before attempting learned role clusters.
- **Later research:** defensive assignment roles, multi-season stability, lineup context, playoffs as a separate sample, and independently validated career forecasting. A season similarity score is not a forecast of development or a claim that two players have equal value.

Validate before changing production rankings: exact totals versus official ratios; per-possession unit checks; no duplicate player-seasons after traded-player aggregation; role-diverse basketball review cases; sensitivity to weights and missingness; and held-out seasons for any learned parameters. For forecast experiments, split by time and fit transformations on training data only. Judge whether each mode answers its stated question rather than forcing famous player pairs or higher scores. Keep the interpretable 0–100 mapping unless validation supplies a reason to change it, and version the model/reference population so changes are explainable.

Raw probe responses and scripts are retained locally under `data/raw/comparison_research_20260925/`; a compact results record accompanies this report. No production model or app changes were made during this analysis.
