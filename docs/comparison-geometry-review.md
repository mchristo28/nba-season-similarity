# Distribution and aggregate-method review — model 3.1

This release addresses the Lauri Markkanen 2024–25 / GG Jackson 2023–24 example
without tuning weights to that pair. Run `python scripts/audit_geometry.py` to
reproduce the [full numeric audit](comparison-geometry-audit.json), which records
the dataset hash, distributions, correlations, method comparisons, and sensitivity.
The season data was not refreshed or changed for this release.

## Distribution audit and colors

The reference contains 7,103 rotation-player seasons for historical style and
production, and 4,237 for tracking (20+ games, 15+ MPG, within mode coverage).
Every profile input is audited for missingness, zero mass, skew, standard deviation,
quantiles, and IQR-based spread. The largest absolute skew is 1.87. Across profiles,
the ratio `(IQR / 1.349) / SD` ranges from 0.715 to 1.284. These diagnostics do not
prove normality or perfect reliability, but provide no strong basis for replacing
all existing scales with nonlinear transforms in this release. Counts already use
per-possession rates; tendencies use ratios. Event-sample reliability remains a
separate limitation.

Use the same standardized gaps for the model and the primary table:

| Gap | Color | Label |
| --- | --- | --- |
| Less than 0.5 SD | Green | Close |
| 0.5 to less than 1 SD | Yellow | Noticeable difference |
| 1 SD or more | Red | Large difference |

These are deliberate display boundaries anchored to half/full reference standard
deviations, not learned cutoffs, probabilities, or hypothesis tests. No assumption
of Gaussian data is needed for the units. The audit records random-pair band
fractions to make their behavior across distributions inspectable; it does not
force equal proportions of green/yellow/red across different statistics.

Historical height SD is 3.438 inches. Therefore 84 versus 81 inches is a 0.873-SD
gap (yellow), independent of the arbitrary zero point of the measurement. Lauri's
240 versus GG's 210 pounds is 1.156 SD (red). A 0 versus 0.1 block-rate difference
can be small despite its enormous relative percentage. Zero-denominator measurements
remain unknown. Raw contextual tables have no similarity colors.

## Aggregate comparison

Three methods were tested on the same candidates and nine query seasons spanning
scorers, creators, passing bigs, rim protection and defensive facilitators:

1. **Previous mean of group RMS distances.** An exact match on one dimension can
   offset a large mismatch in another relatively cheaply.
2. **Joint weighted RMS (selected).** Equivalent to a diagonal weighted Euclidean
   distance across the full observed feature vector. Category weights are divided
   among their features. Squared gaps add before the final square root; category
   scores are never averaged. Weights and features are unchanged.
3. **Regularized Mahalanobis variant.** Ledoit–Wolf covariance on complete reference
   rows after standardization. A diagonal matrix of square-root feature weights
   surrounds the inverse covariance. Normalize by `trace(Q × covariance)` so that
   independent reference pairs have comparable expected squared distance (~2).
   This is one reasonable weighted variant, not an exhaustive search of covariance
   methods. [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).

The audit uses complete candidates so every tested method sees identical evidence;
the deployed query-observed missingness policy remains unchanged. Covariance fitting
uses complete reference rows, while per-feature standardization uses available
reference observations. No unmeasured tracking is imputed.

Stability was checked by fitting only through 2022–23, keeping query/candidate
seasons fixed, and comparing the top five with those from the full reference.
This is a reference-sensitivity check, **not forecasting validation**. Nine cases
are diagnostic examples, not representative labeled ground truth.

| Mode | Previous: retained top 5 | Joint RMS | Covariance-aware |
| --- | ---: | ---: | ---: |
| Historical style | 4.89 | 4.78 | 4.67 |
| Tracking style | 4.78 | 4.67 | 2.44 |
| Production | 4.67 | 5.00 | 4.44 |

Separately increasing each category's weight by 20% yielded mean top-five retention
of 4.64, 4.79 and 4.83 for joint RMS across the three modes; the covariance variant
yielded 4.68, 4.70 and 4.21. These checks support the simpler method's stability;
they do not establish that its top matches are objectively correct.

Strong correlations remain: height/weight ~0.81, tracking time/dribbles per touch
~0.86, and production points/usage ~0.94. Shot-share constraints also create nearly
dependent combinations even without extreme individual pair correlations. The
shrunk covariance condition numbers are approximately 2,805, 3,259 and 487.
Correlation-aware scoring changed some results substantially and was especially
sensitive to reference changes for tracking. It also complicates missing evidence,
custom weights and simple positive per-feature explanations. Keep it experimental
until broader validation justifies those tradeoffs. Do not claim the selected
diagonal method has learned joint basketball roles or eliminated redundancy.

## Concrete result and UI changes

For Lauri 2024–25 versus GG 2023–24, with default weights:

| Mode | Previous score / rank | Joint RMS score / rank |
| --- | --- | --- |
| Historical style | 92.22 / 1 | 87.32 / 2 |
| Tracking style | 82.08 / 7 | 72.16 / 15 |
| Production | 94.18 / 7 | 92.02 / 8 |

The historical top result becomes Ersan Ilyasova 2016–17 (87.47), only narrowly
ahead of GG. This is not evidence that GG must be a bad comp or that the new winner
is uniquely correct. The improvement is a more coherent treatment of large gaps
and an explanation that exposes those gaps rather than hiding them behind “92.”

The UI now shows one mode-qualified score and contributions to its squared
difference. Category bars no longer imply independent “100 matches.” The key
differences show the largest contributing features with at least a noticeable gap.
The main table has standardized colors and textual labels; disabled and unknown
inputs stay neutral. A separate expander contains raw season context and the old
descriptive radar. Tracking is the default for new visits, with an explicit switch
to historical data for pre-2013 queries. Existing selections are preserved. A
shortcut offers tracking again for eligible queries in historical mode.

Model 3.1 retains `100 × 2^(-d²)`. The meaning of `d` changes, so rankings/scores
can change. Reference filters, category weights, candidate filters and feature
definitions are unchanged. Ordinary `WeightedMatcher()` without a profile keeps
its legacy aggregation. New profile models refit under the current model version.

Validation covers concentrated mismatches, exact contribution sums, symmetry for
shared evidence, weight-rescaling invariance, missing and disabled features, color
boundaries, translation-invariant measurement gaps, neutral context, career-distance
consistency for a single period, and app interactions. Existing coverage, units,
historical-boundary, filter-invariance and persistence tests continue to apply.


## Addendum: outlier clipping (evaluated, not deployed)

Several features are right-skewed (free-throw rate, blocks, corner-3 share) with maximum
standardized values of 6–8 SD. Since squared distance amplifies outliers, clipping
standardized values at ±3, ±4 and ±5 SD was compared with the deployed model on 300 random
queries per profile (top-10 overlap with the unclipped ranking):

| Clip | Tracking mean overlap | Production mean overlap |
| --- | --- | --- |
| ±3 SD | 0.977 | 0.979 |
| ±4 SD | 0.996 | 0.998 |
| ±5 SD | 0.999 | 1.000 |

At ±4 SD only about 0.3% of top-10 lists fall below 0.7 overlap, so rankings are not driven by
outliers. Clipping would break the exact-unit and additive-explanation guarantees for little
change in results, so it was not adopted.
