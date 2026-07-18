# Model 01 Bayesian-filter results

## Frozen scope

These results were rebuilt on 18 July 2026 after the Model 01 closure audit.
The deterministic history begins in August 2005, the filter runs through
16 July 2026, and formal forecast evaluation covers 97 classified targets from
January 2018 through May 2026. A forecast is eligible only if its checkpoint
date is strictly earlier than `label_available_at`. Same-date
`pre_confirmation` checkpoints remain in the state audit but are not scored.

The full methodology is in [`bayesian_filter.md`](bayesian_filter.md); exact
rows are published in
[`evaluation_summary.json`](../../../results/published/m01_bayesian_filter/evaluation_summary.json)
and
[`sensitivity_metrics.csv`](../../../results/published/m01_bayesian_filter/sensitivity_metrics.csv).

## Fixed-checkpoint performance

| Checkpoint | Specification | NLL | Brier | MAP accuracy | Balanced accuracy | Macro F1 | Entropy |
|---|---|---:|---:|---:|---:|---:|---:|
| Month start | Student-$t$ evidence | 1.0464 | 0.5725 | 62.89% | 33.92% | 0.3231 | 1.1861 |
| Month start | Transition only | 1.0741 | 0.5807 | 61.86% | 33.55% | 0.3043 | 1.2171 |
| Month end | Student-$t$ evidence | **0.9170** | **0.5156** | 64.95% | 34.60% | 0.3488 | 0.9392 |
| Month end | Transition only | 0.9762 | 0.5291 | **65.98%** | **34.97%** | **0.3529** | 1.0477 |

The event model improves both proper probability scores at month start and
month end. At month end, NLL improves by 0.0592 and Brier skill is 2.55% versus
the paired transition-only replay. It does not improve the hard-label hit or
balanced-accuracy metrics. The correct interpretation is a modest improvement
in probability assignment, not clearly superior classification.

The first four numbered monthly ICSA checkpoints contain 97 targets each. Their
baseline NLL values are 1.0042, 0.9817, 0.9930, and 0.8832; the corresponding
Brier scores are 0.5590, 0.5539, 0.5651, and 0.4946. A fifth release exists in
32 months and has NLL 0.8899 and Brier 0.5280. These slices are descriptive:
release number is correlated with calendar structure and should not be read as
a controlled causal dose-response experiment.

## Metric interpretation

- **NLL** is mean negative log probability assigned to the realized regime;
  confident mistakes are penalized heavily.
- **Brier** is the unscaled sum of four squared probability errors, averaged
  across forecasts; its range is 0 to 2.
- **MAP accuracy** is the fraction of largest-probability class calls that are
  correct.
- **Balanced accuracy** is mean recall across realized regimes and therefore
  gives rare quadrants equal weight.
- **Macro F1** averages per-regime harmonic precision/recall.
- **Entropy** measures posterior sharpness in nats; lower entropy is not
  automatically better accuracy.
- **Axis Brier scores** aggregate the quadrants into growth-up and inflation-up
  binary events.
- **Expected calibration error** is a bin-dependent summary of forecast-versus-
  realized frequency and is diagnostic rather than a proper score.

## Sensitivity summary

The baseline fixes Student-$t$ degrees of freedom at 7, mean pseudo-count at 5,
Ledoit-Wolf covariance, and scale multiplier at 1.0. Fifteen alternatives were
specified before the final replay and change one choice at a time.

At month end:

- degrees-of-freedom/Gaussian variants span NLL 0.9058--0.9185 and Brier
  0.5080--0.5157; the Gaussian alternative is best in this realized sample;
- mean pseudo-count variants span NLL 0.9139--0.9311 and Brier 0.5146--0.5182;
- covariance variants are nearly invariant, with NLL 0.9171--0.9174 and Brier
  0.5154--0.5159;
- scale 0.75 gives NLL 0.9120/Brier 0.5133, while scale 1.25 gives
  0.9228/0.5172.

These differences do not justify replacing the baseline after observing the
evaluation period. In particular, selecting the Gaussian alternative because
it is best on these same 97 targets would be post-selection overfitting.

## Latest posterior

At the 16 July 2026 cutoff, April and May are confirmed; June and July remain
filtered distributions.

| Regime | June 2026 | July 2026 | August transition forecast |
|---|---:|---:|---:|
| Growth up / inflation up | 72.24% | 59.66% | 52.10% |
| Growth down / inflation up | 15.43% | 20.09% | 19.34% |
| Growth up / inflation down | 12.33% | 19.98% | 23.07% |
| Growth down / inflation down | 0.00% | 0.26% | 5.49% |

The August column is the July marginal propagated through the transition matrix;
it has not received August-reference evidence. The exact joint path and status
fields are in
[`latest_posterior.json`](../../../results/published/m01_bayesian_filter/latest_posterior.json).

## Limitations visible in the results

The four realized classes are imbalanced, the sample contains only 97 targets,
and adjacent monthly errors are dependent. The block likelihoods also assume
conditional independence across releases; repeated ICSA observations can make
the filter sharper without supplying fully independent information. No
confidence interval is attached to the forecast-score difference in frozen
Model 01. These results are therefore a benchmark for later architectures, not
evidence of a production-grade forecasting advantage.
