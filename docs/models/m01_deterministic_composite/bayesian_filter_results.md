# Model 01 Bayesian-filter results

## Frozen run

These results were generated from the frozen specification in
[`m01_event_driven_bayesian_filter.yaml`](../../../configs/models/m01_event_driven_bayesian_filter.yaml).
The information cutoff is July 16, 2026. Forecast evaluation begins with the
January 2018 target month and ends with May 2026, the latest classified target
available at the cutoff. Every fixed-checkpoint comparison contains the same
97 target months.

The results are descriptive research output. The sensitivity variants were
predeclared and were not used to replace or retune the baseline after seeing
the evaluation sample.

## Fixed-checkpoint evaluation

| Checkpoint | Model | NLL | Brier | MAP accuracy | Balanced accuracy | Classwise ECE | Mean entropy | Brier skill |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Month start | Student-t evidence | 1.0548 | 0.5785 | 60.82% | 33.17% | 0.1419 | 1.1887 | 1.41% |
| Month start | Transition only | 1.0827 | 0.5868 | 60.82% | 33.17% | 0.1506 | 1.2197 | 0.00% |
| Month end | Student-t evidence | 0.9224 | 0.5193 | 64.95% | 34.60% | 0.0990 | 0.9431 | 2.57% |
| Month end | Transition only | 0.9819 | 0.5330 | 65.98% | 34.97% | 0.1106 | 1.0518 | 0.00% |
| Pre-confirmation | Student-t evidence | 0.9351 | 0.5324 | 63.92% | 29.84% | 0.0935 | 0.8768 | -2.64% |
| Pre-confirmation | Transition only | 0.9558 | 0.5187 | 68.04% | 36.59% | 0.1042 | 1.0394 | 0.00% |

NLL is mean negative log probability assigned to the realized regime. Brier is
the unscaled four-class squared probability error. Brier skill is measured
against the paired transition-only forecast; positive is better. Entropy is a
sharpness diagnostic, not an accuracy score.

The evidence model improves both proper probability scores at month end: NLL
falls by 0.0595 and Brier score falls by 2.57% relative to transition only.
Its MAP accuracy is slightly lower, illustrating why hard-classification
accuracy alone is an incomplete evaluation of a probability forecast.

Immediately before deterministic confirmation, the evidence model still has
better NLL but worse Brier score, MAP accuracy, and balanced accuracy. The
posterior is sharper at that checkpoint, but some of the additional confidence
is misplaced. This is evidence of residual overconfidence or dependence among
release blocks rather than an unqualified improvement.

## ICSA release checkpoints

| ICSA release number in target month | Months | Evidence NLL | Transition NLL | Evidence Brier | Transition Brier | Brier skill |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 97 | 1.0125 | 1.0739 | 0.5650 | 0.5812 | 2.79% |
| 2 | 97 | 0.9906 | 1.0733 | 0.5605 | 0.5819 | 3.68% |
| 3 | 97 | 1.0009 | 1.0962 | 0.5711 | 0.5956 | 4.12% |
| 4 | 97 | 0.8896 | 0.9784 | 0.4993 | 0.5267 | 5.20% |
| 5 | 32 | 0.8975 | 0.9408 | 0.5332 | 0.5358 | 0.47% |

These checkpoints contain all evidence processed earlier on the same date, not
an isolated causal effect estimate for ICSA. They nevertheless show that the
event-driven posterior generally improves proper probability scores as weekly
information accumulates. The fifth-release slice is smaller because many
calendar months contain only four eligible ICSA publications.

## Sensitivity checks

The tables below report the month-end slice. Complete results for month start,
month end, pre-confirmation, and ICSA release number are stored in
[`sensitivity_metrics.csv`](../../../results/published/m01_bayesian_filter/sensitivity_metrics.csv).

### Student-t degrees of freedom

| Distribution | NLL | Brier |
|---|---:|---:|
| Student-t, 3 df | 0.9208 | 0.5180 |
| Student-t, 5 df | 0.9217 | 0.5190 |
| **Student-t, 7 df baseline** | **0.9224** | **0.5193** |
| Student-t, 10 df | 0.9231 | 0.5194 |
| Student-t, 30 df | 0.9239 | 0.5195 |
| Gaussian limit | 0.9112 | 0.5118 |

The conclusion is not fragile to moderate changes in tail thickness. The
Gaussian variant is best on this historical slice, but it remains a
sensitivity result rather than a basis for replacing the predeclared 7-df
baseline.

### Regime-mean shrinkage

| Kappa | NLL | Brier |
|---:|---:|---:|
| 0 | 0.9193 | 0.5183 |
| 2 | 0.9202 | 0.5186 |
| **5 baseline** | **0.9224** | **0.5193** |
| 10 | 0.9268 | 0.5202 |
| 20 | 0.9365 | 0.5219 |

Stronger pooling gradually weakens month-end performance, but the differences
are modest. Kappa remains valuable during early or rare-regime fits because it
gives unobserved regimes the pooled mean instead of an unidentified location.

### Covariance shrinkage

| Shared covariance treatment | NLL | Brier |
|---|---:|---:|
| Empirical | 0.9225 | 0.5191 |
| **Ledoit-Wolf baseline** | **0.9224** | **0.5193** |
| 25% spherical shrinkage | 0.9226 | 0.5194 |
| 50% spherical shrinkage | 0.9227 | 0.5195 |
| 75% spherical shrinkage | 0.9227 | 0.5195 |

The historical conclusion is essentially unchanged across these covariance
treatments. For the univariate ICSA block, Ledoit-Wolf and empirical covariance
are mathematically identical because the spherical target equals the scalar
variance. Covariance sensitivity is meaningful only for the multivariate
monthly blocks.

### Residual scale

| Standard-deviation multiplier | NLL | Brier |
|---:|---:|---:|
| 0.75 | 0.9174 | 0.5169 |
| **1.00 baseline** | **0.9224** | **0.5193** |
| 1.25 | 0.9283 | 0.5210 |

The narrower likelihood performs somewhat better at month end, but the
pre-confirmation deterioration in the main results cautions against simply
making the posterior sharper. A later model should address cross-block
dependence directly rather than tune the scale on this evaluation sample.

## Evidence utilization

The baseline replay contains 1,476 auditable event vectors:

| Outcome | Count |
|---|---:|
| Applied | 1,244 |
| Insufficient causal training history | 127 |
| Already-confirmed target; recorded no-op | 101 |
| Incomplete same-event vector | 4 |

Only 14 JOLTS vectors moved the path posterior; 94 otherwise usable JOLTS
vectors arrived after their target month had already been deterministically
confirmed. This is a substantive architectural finding: JOLTS is often too
late to nowcast the same reference month's target. It has not been silently
remapped to a future regime.

The weekly claims likelihood uses ICSA only. It applies 568 ICSA innovations
after a 29-event causal warm-up. CCSA remains in the evidence data artifact but
does not enter this likelihood or posterior.

## Latest posterior

As of July 16, 2026, April and May 2026 are confirmed. The unconfirmed
posterior marginals are:

| Target month | Growth up / inflation up | Growth down / inflation up | Growth up / inflation down | Growth down / inflation down |
|---|---:|---:|---:|---:|
| June 2026 | 71.96% | 15.57% | 12.46% | 0.00% |
| July 2026 | 59.26% | 20.23% | 20.25% | 0.26% |
| August 2026 transition forecast | 51.63% | 19.42% | 23.47% | 5.48% |

The complete machine-readable result, including full precision and
confirmation status, is
[`latest_posterior.json`](../../../results/published/m01_bayesian_filter/latest_posterior.json).

## Interpretation

The first implementation provides evidence that the leading-release layer adds
modest probabilistic value over the transition-only model, particularly by
month end and after several ICSA releases. It does not establish strong rare-
regime classification performance, and it becomes too sharp at the final
pre-confirmation checkpoint under some metrics. Those mixed results are more
useful than a single headline accuracy number: they identify calibration and
cross-block dependence as the next modeling problems to solve.
