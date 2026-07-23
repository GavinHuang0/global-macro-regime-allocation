# Model 01 data dictionary

This document maps Model 01's machine artifacts to their roles. Detailed
mathematics live in the stage specifications; configurations and manifests are
the authoritative serialization and provenance contracts.

## Conventions

\(m\) is a reference month, \(d\) an information cutoff,
\(r\in\mathcal R\) a quadrant, and \(p_{m,r\mid d}\) its posterior
probability. Portfolio artifacts use rebalance index \(t\), return vector
\(\mathbf x_t\), pretrade and target weights \(\mathbf w_t^{-}\) and
\(\mathbf w_t\), moments \(\boldsymbol\mu_t\) and
\(\boldsymbol\Sigma_t\), and realized cost \(K_t\). Literal field names and
paths appear in backticks.

## Deterministic regime data

Configuration:
[`m01_deterministic_composite.yaml`](../../../configs/models/m01_deterministic_composite.yaml)

Manifest:
[`m01_deterministic_composite.json`](../../../data/manifests/m01_deterministic_composite.json)

| Artifact | Location | Core contract |
|---|---|---|
| `first_release_components_long.csv` | `data/processed/m01_deterministic_composite/` | One component/reference-month row with `series_id`, `release_date`, same-vintage levels, transformation, value, lag, and source |
| `composite_features_and_regimes.csv` | `data/processed/m01_deterministic_composite/` | Component z-scores, raw and smoothed axis scores, `regime_id`, `label_available_at`, and `data_status` |
| `regime_history.csv` | `results/published/m01_deterministic_composite/` | Compact public score, regime, availability, and status history |
| `latest_confirmed.json` | `results/published/m01_deterministic_composite/` | Latest fully classified month and scores |

The principal regime fields are:

| Field | Meaning |
|---|---|
| `reference_month` | Economic month, normalized to month start |
| `growth_raw`, `inflation_raw` | Equal-weight four-component axis scores |
| `growth_smoothed`, `inflation_smoothed` | Trailing three-month Model 01 scores |
| `regime_id` | Stable quadrant identifier |
| `label_available_at` | First date on which every prerequisite was available |
| `data_status` | Classification or explicit unavailable reason |

## Transition data

Configuration:
[`m01_deterministic_composite_transition.yaml`](../../../configs/models/m01_deterministic_composite_transition.yaml)

| Artifact | Location | Core contract |
|---|---|---|
| `transition_pairs.csv` | `data/processed/m01_deterministic_composite/` | Every adjacent candidate pair, label dates, inclusion flag, and exclusion reason |
| `transition_model.json` | `results/published/m01_deterministic_composite/` | State order, cutoff, counts, Dirichlet parameters, posterior-predictive matrix, intervals, hashes, and diagnostics |
| `transition_matrix.csv` | `results/published/m01_deterministic_composite/` | Long-form 16-cell view of counts, probabilities, and intervals |
| `latest_transition_prior.json` | `results/published/m01_deterministic_composite/` | Transition-only prior for the month after the latest eligible confirmed state |

Consumers must use the serialized `state_order`; they must not infer matrix
position from alphabetical ordering. `mle_probability` is diagnostic.
`posterior_predictive_probability` is the production transition law.

## Leading-evidence data

Configuration:
[`m01_non_defining_release_evidence.yaml`](../../../configs/models/m01_non_defining_release_evidence.yaml)

Manifest:
[`m01_non_defining_release_evidence.json`](../../../data/manifests/m01_non_defining_release_evidence.json)

All files live under
`data/processed/m01_non_defining_release_evidence/`.

| Artifact | Core contract |
|---|---|
| `first_release_observations.csv` | Acquired first appearances, release/reference dates, values, provider, lag, and feature eligibility |
| `non_defining_release_events.csv` | Full long event table, including warm-up and unusable rows |
| `non_defining_release_features.csv` | Available-only feature subset consumed by inference |

Stable event fields include `event_id`, `event_group_id`, `release_block`,
`release_date`, `reference_date`, `reference_month`, `feature_name`,
`series_id`, `transformed_value`, `feature_value`, and `feature_status`.
Claims rows additionally retain expanding AR counts, coefficients, forecasts,
and innovations.

## Bayesian-filter data

Configuration:
[`m01_event_driven_bayesian_filter.yaml`](../../../configs/models/m01_event_driven_bayesian_filter.yaml)

Manifest:
[`m01_event_driven_bayesian_filter.json`](../../../data/manifests/m01_event_driven_bayesian_filter.json)

Detailed files under `data/processed/m01_bayesian_filter/` are:

| Artifact | Core contract |
|---|---|
| `checkpoint_index.csv` | Saved state identity, information date, phase, path months, associated event/confirmation, and normalization |
| `joint_path_checkpoints.csv.gz` | Exactly 256 path-probability rows per checkpoint |
| `marginal_checkpoints.csv` | Four quadrant marginals for each path coordinate and next-month transition forecast |
| `event_update_audit.csv` | Event status, fit identity, likelihoods, prior/posterior marginals, and update diagnostics |
| `likelihood_fit_audit.csv` | Causal counts, means, covariance, Student-\(t\) shape, and numerical checks |
| `forecast_predictions.csv` | Forecasts, eventual labels, and explicit scoring eligibility |
| `evaluation_metrics.csv`, `calibration_bins.csv` | Proper scores, classification metrics, entropy, and reliability inputs |
| `sensitivity_specifications.csv`, `sensitivity_metrics.csv` | Prespecified one-at-a-time variants and matched results |

Public counterparts under `results/published/m01_bayesian_filter/` are
[`latest_posterior.json`](../../../results/published/m01_bayesian_filter/latest_posterior.json),
[`evaluation_summary.json`](../../../results/published/m01_bayesian_filter/evaluation_summary.json),
and
[`sensitivity_metrics.csv`](../../../results/published/m01_bayesian_filter/sensitivity_metrics.csv).

## ETF and allocation data

The market-data manifest is
[`us_cross_asset_etf_universe_v1.json`](../../../data/manifests/us_cross_asset_etf_universe_v1.json).
The strategy uses `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`;
`AGG` is benchmark-only.

Allocation configuration:
[`m01_regime_allocation_backtest.yaml`](../../../configs/models/m01_regime_allocation_backtest.yaml)

Manifest:
[`m01_regime_allocation_backtest.json`](../../../data/manifests/m01_regime_allocation_backtest.json)

Detailed files under
`data/processed/m01_regime_allocation_backtest/` include:

| Artifact | Core contract |
|---|---|
| `signal_table.csv` | Selected month-start posterior for each portfolio reference month |
| `holding_period_returns.csv` | Common adjusted-open holding returns and dates |
| `regime_estimate_audit.csv` | Eligible samples, conditional means, shrinkage, and covariance inputs |
| `optimizer_audit.csv` | Moments, pretrade estimate, solver outcome, constraints, and feasibility |
| `monthly_weights.csv` | Long-form targets by method |
| `monthly_strategy_returns.csv` | Gross return, turnover, cost, net return, and NAV |
| `daily_nav.csv.gz` | Pretrade open, post-trade open, close, and terminal-open NAV points |
| `performance_metrics.csv` | Return, risk, drawdown, turnover, and cost metrics |
| `comparison_uncertainty.csv` | Paired circular block-bootstrap comparisons |
| `sensitivity_metrics.csv` | Prespecified policy variants |

Public files under `results/published/m01_regime_allocation_backtest/` include
[`latest_allocation.json`](../../../results/published/m01_regime_allocation_backtest/latest_allocation.json),
[`backtest_summary.json`](../../../results/published/m01_regime_allocation_backtest/backtest_summary.json),
[`performance_summary.csv`](../../../results/published/m01_regime_allocation_backtest/performance_summary.csv),
[`monthly_returns.csv`](../../../results/published/m01_regime_allocation_backtest/monthly_returns.csv),
[`monthly_weights.csv`](../../../results/published/m01_regime_allocation_backtest/monthly_weights.csv),
[`comparison_uncertainty.csv`](../../../results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv),
and
[`sensitivity_metrics.csv`](../../../results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv).

## Provenance and security

Manifests record configuration, input, implementation, and output hashes where
applicable. Raw and detailed processed data remain local research artifacts;
compact publication files are tracked separately. Provider credentials are
read from the environment and are never serialized into cache identities,
manifests, URLs, or logs.
