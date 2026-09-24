# How comparison scores work

A similarity score describes closeness of the measured statistical profiles. It
is not a grade, probability, percentile, or forecast of equal player quality.

1. Standardize each feature using the bundled player-season population's mean
   and standard deviation. Missing values remain missing and do not enter the fit.
2. For each dimension, take the root mean square (RMS) of the differences in
   shared measurements. Unlike a raw Euclidean norm, this does not automatically
   give a dimension more influence just because it has more features.
3. Average the dimension distances using the requested weights. Exact matches
   remain in the denominator. Dimensions without any shared data are omitted.
4. Convert the weighted distance `d` to `100 × 2^(-d²)`.

| Average group RMS distance | Score | Interpretation |
| --- | --- | --- |
| 0 | 100 | Identical on the measured features |
| 0.25 standard deviations | 95.8 | Very close |
| 0.5 | 84.1 | Close |
| 1 | 50 | Moderate difference |
| 1.5 | 21.0 | Large difference |
| 2 | 6.3 | Very large difference |

This is a transparent design calibration, not an empirically established measure
of basketball equivalence. It is fixed for all searches; top results are not
rescaled to 100. The best available match for an unusual player may be moderate.
Changing the selected player, result count, or candidate filters does not change
an individual pair's score. Updating the reference dataset or feature weights can.

The old formula, `100 / (1 + distance)`, used unnormalized group distances and had
no documented reference points. Its scores near 50 did not mean "50% alike."
The new numbers change because both the distance calculation and the display
mapping have been corrected, not just because the display range was stretched.

## Coverage and limits

Coverage is the weighted fraction of all requested features measured for **both**
seasons. The default search requires 50%; users can raise this to demand more
complete comparisons. A 100 score with 60% coverage means identical on the
available measurements, not proof of identical full profiles. Different coverage
can change the dimensions represented in a ranking; compare coverage alongside
scores, or raise the minimum for stricter searches. Category bars use the same
score mapping. They display N/A for unavailable groups and OFF for disabled ones.

In the included snapshot, drives and touches are absent before 2020–21. Missing
stats are excluded rather than interpreted as zero. Team shares use player
per-game production divided by actual team per-game production; aggregate seasons
with multiple teams have missing shares because final-team attribution would be
incorrect. Missing team minutes are also left unavailable. These are explicit
limitations, not estimated data.

A full category breakdown displays raw measurements separately from scores. The
radar is descriptive, with its own axis scales, and omits axes lacking shared data.

Run `python scripts/audit_scores.py` to inspect sample scores, coverage, and search
times against the bundled snapshot. It does not contact the NBA API.
