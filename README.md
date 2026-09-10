# Global Macro Regime Detection and Allocation

A research pipeline for estimating US growth and inflation regimes from
historical data vintages and translating their probabilities into cross-asset
ETF allocations.

<!-- M02-LATEST:START -->

## Model 02 weekly research snapshot

**Research only · FRESH at generation.** Signal week: **2026-09-07**.

Data freshness: All required input families passed release\-age checks at the signal cutoff\.

| Snapshot timing | As of |
|---|---|
| Information cutoff | `2026-09-07T00:00:00-04:00` |
| Latest macro release used | `2026-09-04` |
| Final price close used | `2026-09-04` |
| Generated | `2026-09-10T02:19:46Z` |
| Valid until (exclusive) | `2026-09-14T00:00:00-04:00` |

The cutoff is Monday 00:00 America/New_York; releases on the signal day are excluded.
Price coverage records the final close used by estimation and pretrade holdings.

| Asset | Research target |
|---|---:|
| SPY | 35.0% |
| IEF | 0.0% |
| TIP | 0.0% |
| HYG | 15.0% |
| BIL | 0.0% |
| GLD | 25.0% |
| LQD | 25.0% |

| Macro quadrant | Posterior probability |
|---|---:|
| Growth up / inflation up | 23.7% |
| Growth down / inflation up | 33.4% |
| Growth up / inflation down | 24.4% |
| Growth down / inflation down | 18.5% |

| Model estimate | Value |
|---|---:|
| Forecast weekly return | +0.17% |
| Ex ante annualized volatility | 9.83% |

Optimization outcome: **optimal**. Return and risk are model estimates, not realized performance or guaranteed outcomes.

Model: `student_t_7_reduced_core` · Release: `m02_weekly_live_v1`.
This research output is not investment advice.

<!-- M02-LATEST:END -->

[Snapshot JSON](results/live/m02_weekly/latest_allocation.json) ·
[Performance CSV](results/live/m02_weekly/performance_summary.csv) ·
[Run provenance](results/live/m02_weekly/run_manifest.json) ·
[Weekly update guide](docs/operations/weekly_updates.md)

[Overview](#overview) · [Methodology](#methodology) · [Results](#results) ·
[Quick start](#quick-start) · [Documentation](#documentation)

## Overview

Macroeconomic releases arrive with delays, describe different reference periods,
and are revised after publication. A historical model built from today's data
can therefore use information that an investor would not have had at the time.

This project reconstructs the available information on each decision date,
estimates the current month's growth and inflation state, and turns that
estimate into a portfolio of US-listed ETFs. The research asks whether macro
regime probabilities improve allocation beyond a model that uses pooled
historical asset returns.

There are two model lines:

| Model | Macro state | Portfolio cadence | Role |
|---|---|---|---|
| [M01](docs/models/m01_deterministic_composite/README.md) | Four discrete regimes from smoothed growth and inflation composites | Monthly | Frozen benchmark |
| [M02](docs/models/m02_soft_composite/README.md) | Continuous growth and inflation scores with probabilistic quadrant reporting | Weekly | Current research model |

M02's macro state is **monthly**; its posterior is updated as releases arrive
and sampled for weekly allocation. The current inference configuration is
`student_t_7_reduced_core`, paired with the `posterior_optimized` allocator.

## Methodology

```mermaid
flowchart LR
    A[Release vintages] --> B[Growth and inflation state]
    B --> C[Quadrant probabilities]
    C --> D[Return and risk estimates]
    D --> E[Constrained ETF weights]
```

### Data and regime definitions

The macro inputs come from FRED/ALFRED historical vintages. For each monthly
change, the pipeline uses the first eligible release and the preceding level
visible in that same vintage. Standardization uses at least 60 earlier valid
observations, so later data cannot change an earlier feature's scale.

| Axis | M02 defining components |
|---|---|
| Growth | Payroll growth, industrial-production growth, real-consumption growth, and the negative change in unemployment |
| Inflation | Core CPI, core PCE prices, core finished-goods producer prices, and average hourly earnings growth |

Each axis is an equal-weight average of its four standardized components. M01
uses payroll level changes and smooths its composites over three months before
assigning a discrete quadrant. M02 uses payroll log growth, retains unsmoothed
scores, and estimates their distribution before all components have been released.

The signs of the scores define the four growth–inflation quadrants shown in
the snapshot. Here, **up** and **down** refer to the standardized score's position
relative to zero; they do not directly mean that GDP is expanding or that the
inflation rate is accelerating.

### Bayesian inference

M02 maintains a joint state for the current and three preceding months. An
expanding vector autoregression, VAR(1), describes how growth and inflation
evolve between months. Partial defining releases update the relevant monthly
state; once all eight components are available, the completed score is imposed
as an exact observation.

Three additional evidence blocks update the state between defining releases:

- **Labor:** the innovation in initial unemployment claims.
- **Consumption:** real retail activity jointly with its implicit-price component.
- **Business investment:** core capital-goods shipments jointly with the
  orders-to-shipments pipeline.

These observations use Student-t likelihoods with seven degrees of freedom to
reduce sensitivity to outliers. The filter approximates the resulting updates
with Gaussian moments. A separate mapping accounts for historical revisions
and disagreement among components when translating the state into quadrant
probabilities. Model fits use only information available before the release
being processed.

The [inference specification](docs/models/m02_soft_composite/inference.md)
defines the state equations, update order, probability mapping, and evaluation.

### Portfolio construction

The allocator estimates weekly returns conditional on each completed macro
quadrant. A training observation is eligible only after both its holding-period
return and macro label are known. Estimation requires at least 260 labeled weeks;
quadrant means are shrunk toward the pooled mean using 104 equivalent weeks of
prior weight. Ledoit–Wolf shrinkage estimates a shared within-quadrant
covariance; a separate between-quadrant term captures uncertainty about
conditional means.

The expected return is the average of the four conditional estimates, weighted
by the current posterior:

```math
\boldsymbol\mu_t
=\sum_{r\in\mathcal R}p_{t,r}\,\widetilde{\boldsymbol\mu}_{t,r}.
```

Here, $`p_{t,r}`$ is the probability of quadrant $`r`$ at decision $`t`$,
and $`\widetilde{\boldsymbol\mu}_{t,r}`$ is its shrunk weekly return estimate.
The optimizer maximizes expected weekly return after estimated trading costs,
subject to full investment, long-only holdings, a **10% estimated annualized
volatility ceiling**, and position and group limits.

| ETF | Exposure | Maximum weight |
|---|---|---:|
| SPY | US large-cap equities | 35% |
| IEF | Intermediate US Treasuries | 50% |
| TIP | Inflation-linked US Treasuries | 40% |
| HYG | High-yield corporate bonds | 25% |
| BIL | Treasury bills / cash proxy | 100% |
| GLD | Gold | 25% |
| LQD | Investment-grade corporate bonds | 40% |

Monday signals exclude all releases published that day. Targets execute at the
first common adjusted open of the week and are held until the next week's first
common open. The backtest tracks each portfolio's drifted holdings and charges
**5 basis points per dollar traded**. Incomplete holding periods are excluded
from performance.

See the [allocation specification](docs/models/m02_soft_composite/portfolio_allocation.md)
for the objective, covariance construction, group limits, execution, and costs.

## Results

The archived M02 allocation study covers **445 complete weekly holdings**, from
1 January 2018 through the week of 6 July 2026. Returns below are net of modeled
transaction costs; Sharpe ratios use a zero risk-free rate.

| M02 strategy | CAGR | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Posterior optimizer | 9.86% | 10.42% | 0.956 | -21.75% |
| Pooled-mean optimizer | 9.83% | 10.50% | 0.946 | -21.99% |
| 60% SPY / 40% AGG | 9.60% | 11.62% | 0.847 | -22.02% |
| Equal weight | 6.09% | 6.34% | 0.964 | -14.69% |

The pooled-mean optimizer provides the central comparison: it uses the same
allocation constraints without the current macro posterior. The posterior's
annualized arithmetic return advantage is **0.019 percentage points**, with a
95% paired block-bootstrap interval of **[-0.088, +0.130] percentage points**.
The study therefore does not establish a reliable allocation advantage from
macro conditioning.

M02's inference comparisons also remain inconclusive: uncertainty intervals
include zero relative to both a transition-only model and a model using only
partial defining releases. M01 modestly improves probability scores over its
transition baseline, but its allocation comparison is likewise inconclusive.

The [M01 results](docs/models/m01_deterministic_composite/README.md),
[M02 inference results](docs/models/m02_soft_composite/inference.md#causal-replay-result),
and [M02 backtest report](docs/models/m02_soft_composite/portfolio_backtest_results.md)
contain the full comparisons. The [rolling performance summary](results/live/m02_weekly/performance_summary.csv)
extends the allocation replay with refreshed data; the archived study remains
fixed for reference.

### Limitations

- The evidence graph and allocation policy were selected using the same
  historical period as the reported comparisons. There is no untouched
  confirmatory holdout, and only a small number of independent macro cycles.
- Linear dynamics, approximate Gaussian updates, and the mapping from a
  monthly macro state to weekly returns restrict what the model can capture.
- ETF prices use Yahoo Finance's mutable adjusted history. Simulated execution
  and fixed trading costs omit market impact, taxes, and capacity; the estimated
  volatility ceiling does not guarantee realized risk.

## Quick start

Requires **Python 3.11+**. From the repository root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the package:

```bash
python -m pip install -e ".[dev]"
```

Configure `FRED_API_KEY` in the process environment using the
[credential setup guide](docs/operations/weekly_updates.md), then run:

```bash
python -m regime_allocation.cli.update_m02_weekly
```

The command acquires macro vintages and adjusted ETF prices, replays the model
and portfolio history, and writes the allocation, performance summary, and
provenance to `results/live/m02_weekly/`. It also refreshes the snapshot above.
Use `--compute-only` to keep a run under `outputs/` for inspection without
updating the published files.

To run the unit tests:

```bash
python -m pytest -q tests/unit
```

The editable installation picks up changes under `src/regime_allocation/`.
Historical reproduction additionally requires the input snapshots recorded in
the manifests; raw and processed data are excluded from Git. The
[data guide](docs/data_access.md) explains acquisition and provenance, and the
[weekly update guide](docs/operations/weekly_updates.md) covers scheduling and
GitHub Actions.

## Documentation

| Guide | Contents |
|---|---|
| [Documentation index](docs/README.md) | All model specifications, data dictionaries, and operating guides |
| [Research architecture](docs/architecture/research_overview.md) | Shared notation, model definitions, and published baselines |
| [M01 model card](docs/models/m01_deterministic_composite/README.md) | Discrete regimes and the frozen monthly benchmark |
| [M02 model card](docs/models/m02_soft_composite/README.md) | Continuous-state inference and weekly allocation |
| [Research archive](docs/archive/README.md) | Sensitivities, feature attribution, and alternative allocation experiments |

Code lives in [`src/regime_allocation/`](src/regime_allocation/), model settings
in [`configs/models/`](configs/models/), and validation in [`tests/`](tests/).
Archived results are under [`results/published/`](results/published/), with
input and implementation hashes in [`data/manifests/`](data/manifests/).

## Attribution

An original course project attributed to Gavin Huang, Zekai Yao, Mianchen Zhang, and Serin Gleave used the idea of classifying macro regimes to optimize FICC allocations. M01 and M02 are Gavin Huang's independent
implementations of a redesigned architecture. The [original manuscript](legacy/README.md)
is retained in the repository.

This product uses the FRED API but is not endorsed or certified by the Federal
Reserve Bank of St. Louis. All outputs are for research, not investment advice.
