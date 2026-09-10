# Model 01: deterministic composites and monthly allocation

Model 01 (`m01_deterministic_composite`) is the repository's frozen monthly
baseline. It constructs point-in-time growth and inflation regimes, updates a
four-month regime path as releases arrive, and converts the current-month
posterior into a constrained ETF allocation. It is a research model, not
investment advice.

## Status and promoted baselines

| Layer | Promoted baseline | Frequency | Status |
|---|---|---|---|
| Inference | `event_driven_bayesian_filter`: fixed-$`\nu=7`$ Student-$`t`$ release likelihoods around a causal first-order transition prior | Event driven; monthly state | Frozen |
| Allocation | `posterior_optimized`: posterior mixture moments and a constrained long-only optimizer | Monthly | Frozen |

The inference baseline is compared only with its matched `transition_only`
replay in the main documentation. The allocation baseline is compared with
`pooled_mean_optimizer`, `static_60_spy_40_agg`, and `equal_weight`.

## Canonical notation

- $`m`$ is a macroeconomic reference month, $`d`$ is an information cutoff,
  $`r\in\mathcal R`$ is one of the four growth–inflation quadrants, and
  $`p_{m,r\mid d}=\Pr(R_m=r\mid\mathcal D_d)`$.
- $`t`$ indexes portfolio rebalances and holding periods,
  $`\mathbf x_t`$ is the strategy-asset return vector,
  $`\mathbf w_t^{-}`$ is the pretrade weight vector, and $`\mathbf w_t`$ is
  the target.
- $`\boldsymbol\mu_t`$ and $`\boldsymbol\Sigma_t`$ are the posterior return
  moments used at rebalance $`t`$, and $`K_t`$ is realized trading cost.

Reference months and information dates are different objects. An observation
about month $`m`$ enters the model only on its recorded publication date.
Literal model IDs, field names, tickers, and paths appear in backticks.

## Architecture

```text
point-in-time defining releases
        -> deterministic monthly regime
        -> causal first-order transition prior

point-in-time non-defining releases
        -> block Student-t likelihoods
        -> four-month event-driven posterior
        -> current-month marginal
        -> monthly posterior allocation
```

The deterministic target uses four growth and four inflation components. Each
first-release transformation is standardized on at least 60 strictly earlier
observations, the four z-scores on each axis are equally weighted, and each
axis is averaged over the current and preceding two months. The signs of the
smoothed scores define the four states.

The inference layer maintains probabilities over $`4^4=256`$ four-month paths.
At a month roll it applies a causal expanding transition matrix with symmetric
Dirichlet-$`0.5`$ smoothing. Non-defining releases update the applicable path
coordinate using block-specific Student-$`t`$ likelihoods. Completed
deterministic labels are then imposed as exact end-of-day confirmations.

The portfolio signal is the `post_month_roll` current-month marginal from the
first calendar day. The target executes at the first common adjusted open and
is held to the next month's first common adjusted open. The strategy estimates
regime-conditioned returns causally, integrates the full posterior, and solves
a long-only expected-return problem with a 10% ex-ante volatility cap,
position and group caps, and five-basis-point one-way costs.

## Inference result

The formal evaluation contains 97 classified reference months from January 2018
through May 2026. Lower NLL and Brier scores are better.

| Checkpoint | Method | NLL | Brier | MAP accuracy |
|---|---|---:|---:|---:|
| Month start | Promoted event model | 1.0464 | 0.5725 | 62.89% |
| Month start | `transition_only` | 1.0741 | 0.5807 | 61.86% |
| Month end | Promoted event model | **0.9170** | **0.5156** | 64.95% |
| Month end | `transition_only` | 0.9762 | 0.5291 | **65.98%** |

The release model modestly improves probability scores but does not improve
month-end hard-quadrant accuracy. See
[`bayesian_filter_results.md`](bayesian_filter_results.md).

## Allocation result

The net backtest contains 102 complete monthly holdings from January 2018
through June 2026.

| Method | Total return | CAGR | Annualized volatility | Zero-rate Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| `posterior_optimized` | **124.15%** | **9.96%** | 10.17% | **0.988** | -21.16% |
| `pooled_mean_optimizer` | 120.69% | 9.76% | 10.17% | 0.970 | -21.92% |
| `static_60_spy_40_agg` | 117.41% | 9.57% | 11.21% | 0.874 | -21.60% |
| `equal_weight` | 65.13% | 6.08% | 6.37% | 0.960 | **-14.77%** |

Posterior minus pooled mean is $`0.184\%`$ annualized, with a paired six-month
block-bootstrap 95% interval of $`[-0.364\%,0.795\%]`$. The interval includes
zero. Model 01 therefore does not establish a statistically reliable
allocation edge from the posterior. See
[`portfolio_backtest_results.md`](portfolio_backtest_results.md).

## Current specification

- [Regime definition](regime_definition.md)
- [Transition model](transition_model.md)
- [Leading-evidence data](leading_evidence_data.md)
- [Bayesian filter](bayesian_filter.md)
- [Bayesian-filter results](bayesian_filter_results.md)
- [Portfolio allocation](portfolio_allocation.md)
- [Portfolio backtest results](portfolio_backtest_results.md)
- [Data dictionary](data_dictionary.md)

The historical closure record is preserved under
[`docs/archive/m01/`](../../archive/m01/README.md) and is not part of the
current specification.

## Reproduction and artifacts

From the repository root:

```powershell
python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_evidence --provider auto
python -m regime_allocation.cli.build_m01_transition
python -m regime_allocation.cli.build_m01_inference
python -m regime_allocation.cli.build_m01_backtest
```

Machine-readable configurations are under `configs/models/`, provenance under
`data/manifests/`, detailed local tables under `data/processed/`, and compact
public outputs under `results/published/`. The main manifests are:

- [`m01_deterministic_composite.json`](../../../data/manifests/m01_deterministic_composite.json)
- [`m01_non_defining_release_evidence.json`](../../../data/manifests/m01_non_defining_release_evidence.json)
- [`m01_event_driven_bayesian_filter.json`](../../../data/manifests/m01_event_driven_bayesian_filter.json)
- [`m01_regime_allocation_backtest.json`](../../../data/manifests/m01_regime_allocation_backtest.json)

## Principal limitations

The target is a constructed release-based quadrant, not an observed economic
state. The transition and likelihood laws are deliberately simple, release
blocks are not fully conditionally independent, and the 97-target inference
sample spans few macro cycles. Portfolio results depend materially on noisy
expected returns, a compact US-centric ETF universe, selected constraints,
provider-adjusted market history, and stylized costs.
