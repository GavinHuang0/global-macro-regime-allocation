# Model 01 inference results

These are the frozen results for the promoted
`event_driven_bayesian_filter`. The filter runs through 16 July 2026, and the
formal evaluation contains 97 classified reference months from January 2018
through May 2026. A checkpoint is scored only when its information date is
strictly earlier than the target's `label_available_at`.

Methodology is defined in [`bayesian_filter.md`](bayesian_filter.md). Exact
metrics are in the
[`evaluation summary`](../../../results/published/m01_bayesian_filter/evaluation_summary.json).

## Promoted model versus transition only

| Checkpoint | Method | NLL | Brier | MAP accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Month start | Promoted event model | **1.0464** | **0.5725** | **62.89%** | **33.92%** | **0.3231** |
| Month start | `transition_only` | 1.0741 | 0.5807 | 61.86% | 33.55% | 0.3043 |
| Month end | Promoted event model | **0.9170** | **0.5156** | 64.95% | 34.60% | 0.3488 |
| Month end | `transition_only` | 0.9762 | 0.5291 | **65.98%** | **34.97%** | **0.3529** |

NLL is mean negative log probability assigned to the realized quadrant.
Brier is the unscaled sum of four squared probability errors, averaged across
forecasts. Lower is better for both.

At month end, the event model improves NLL by 0.0592 and has 2.55% Brier skill
relative to the matched transition-only replay. It does not improve hard-state
accuracy or balanced accuracy. The result supports a modest improvement in
probability assignment, not superior classification.

No bootstrap interval is published for the inference-score difference. The
sample is small, classes are imbalanced, and adjacent forecast errors are
dependent. Point differences should therefore remain descriptive.

## Latest posterior

At cutoff \(d=\) 16 July 2026, April and May are confirmed; June and July
remain filtered distributions. The table reports
\(p_{m,r\mid d}\).

| Quadrant | June 2026 | July 2026 | August transition forecast |
|---|---:|---:|---:|
| Growth up / inflation up | 72.24% | 59.66% | 52.10% |
| Growth down / inflation up | 15.43% | 20.09% | 19.34% |
| Growth up / inflation down | 12.33% | 19.98% | 23.07% |
| Growth down / inflation down | 0.00% | 0.26% | 5.49% |

The August column propagates the July marginal through the transition matrix;
it has not received August-reference evidence. Exact path probabilities,
status fields, and full precision are in
[`latest_posterior.json`](../../../results/published/m01_bayesian_filter/latest_posterior.json).

## Interpretation

The leading-release layer is directionally useful on the two proper
probability scores, but its gain over `transition_only` is modest and is not
accompanied by sampling uncertainty. Repeated claims and correlated release
blocks can add sharpness without equivalent independent information. Model 01
is therefore a frozen research benchmark rather than evidence of a
production-grade forecast advantage.

Prespecified sensitivity results remain available in
[`sensitivity_metrics.csv`](../../../results/published/m01_bayesian_filter/sensitivity_metrics.csv);
they are diagnostics and do not change the promoted baseline.
