# Global Macro Regime Detection and Allocation

A Python research system that turns macroeconomic releases into probabilistic
growth/inflation states and constrained ETF allocations, with an explicit clock
for what the model could know and when it could trade.

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

## What this project demonstrates

- **Data engineering:** FRED/ALFRED vintage acquisition, first-release features,
  release calendars, and explicit information cutoffs.
- **Statistical modeling:** event-driven Bayesian updates, robust Student-t
  likelihoods, continuous-state VAR dynamics, and probabilistic evaluation.
- **Portfolio research:** causal return estimation, covariance shrinkage,
  constrained optimization, drifted holdings, and transaction-cost accounting.
- **Software engineering:** a modular Python package, command-line pipelines,
  declarative configurations, unit/integration tests, and SHA-256 provenance.
- **Research discipline:** pooled-mean ablations, benchmark portfolios, paired
  block-bootstrap uncertainty, and documented experiments that were not promoted.

## Two model lines, one research question

Does probabilistic macro information improve allocation beyond a pooled estimate
of asset returns? Both models use `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and
`LQD`; `AGG` is reserved for the 60/40 comparison.

| Model | Inference | Allocation | Status |
|---|---|---|---|
| [Model 01](docs/models/m01_deterministic_composite/README.md) | Hard macro quadrants + Bayesian filter | Monthly | Frozen benchmark |
| [Model 02](docs/models/m02_soft_composite/README.md) | Continuous scores + reduced-core robust filter | Weekly | Promoted; not frozen |

The promoted Model 02 inference is `student_t_7_reduced_core`; its allocation
method is `posterior_optimized`. A weekly refresh extends the dated research
output. Method changes require a named release under the
[versioning contract](docs/architecture/model_versioning.md).

```text
Vintage releases → macro posterior → causal return estimates
                → constrained weights → costs and portfolio path
```

## What the published evidence supports

These are **historical publication results**, separate from the weekly snapshot
above. Model 01 covers January 2018–June 2026 (102 months); Model 02 covers
1 January 2018–6 July 2026 (445 weeks).

| Published strategy | CAGR | Annualized volatility | Sharpe |
|---|---:|---:|---:|
| Model 01 posterior-optimized | 9.96% | 10.17% | 0.988 |
| Model 02 posterior-optimized | 9.86% | 10.42% | 0.956 |
| Model 02 pooled-mean comparison | 9.83% | 10.50% | 0.946 |

Model 02's posterior adds only **0.019 percentage points** of annualized
arithmetic return over pooled mean; its 95% paired block-bootstrap interval is
**[-0.088, +0.130] percentage points**. Neither model establishes reliable
incremental allocation alpha from the posterior. Neither has an untouched
confirmatory holdout. Adjusted ETF histories are mutable provider snapshots,
and modeled costs omit taxes, market impact, and capacity.

Read the [research overview](docs/architecture/research_overview.md) for the
equations, full benchmark tables, inference results, and limitations, or inspect
the [Model 02 published results](docs/models/m02_soft_composite/portfolio_backtest_results.md).

## Run and edit locally

Requires Python 3.11+ and a FRED API key for the weekly refresh. Set the Windows
user variable using the [setup guide](docs/operations/weekly_updates.md#run-locally-on-windows),
then run these commands from the cloned repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:FRED_API_KEY = [Environment]::GetEnvironmentVariable('FRED_API_KEY', 'User')
.\.venv\Scripts\python.exe -m regime_allocation.cli.update_m02_weekly
.\.venv\Scripts\python.exe -m pytest -q tests/unit
```

The editable install picks up changes under `src/regime_allocation/`. Raw and
processed data are excluded from Git; a clone contains the published artifacts,
not every input needed to reproduce them. The
[weekly update guide](docs/operations/weekly_updates.md) explains refreshes and
automation; [data access](docs/data_access.md) covers credentials and provenance.

## Explore the repository

- [Documentation index](docs/README.md): model cards, mathematics, and data dictionaries.
- [Research archive](docs/archive/README.md): sensitivities, attribution, and allocation experiments.
- [Maintenance guide](docs/architecture/repository_maintenance.md): repository layout and artifact policy.
- [Source package](src/regime_allocation/) · [Tests](tests/) · [Model configurations](configs/models/).
- [Original course project](legacy/README.md) · [Security policy](SECURITY.md).

## Attribution

The original course project was produced by Zekai Yao, Mianchen Zhang, Gavin
Huang, and Serin Gleave. Model 01 and Model 02 are Gavin Huang's subsequent
independent reimplementation and extensions. This product uses the FRED API but
is not endorsed or certified by the Federal Reserve Bank of St. Louis.

All probabilities, allocations, and results are research outputs, not investment advice.
