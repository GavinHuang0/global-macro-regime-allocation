# Global Macro Regime Detection and Allocation

An independent, point-in-time rebuild of a team course project on macro-regime
detection and regime-conditioned asset allocation.

## Current status

Model 01 now implements the deterministic regime definition, transition,
leading-evidence, likelihood, Bayesian-filter, portfolio-allocation, and
backtest layers:

- release-coherent first-release monthly macro features;
- four equal-weight growth and inflation components;
- strictly lagged expanding standardization;
- trailing three-month smoothing;
- deterministic four-quadrant regime labels;
- an expanding, first-order transition matrix with Jeffreys-prior smoothing;
- five non-defining release blocks with 12 point-in-time evidence features;
- expanding log-AR(1) innovations for initial and continued claims;
- a canonical publication-date/reference-period event table;
- regime-specific shrunken means and one Ledoit-Wolf covariance per block;
- 7-df Student-t release likelihoods, using only strictly earlier labeled data;
- a four-month joint-path posterior with causal monthly transitions;
- atomic daily release updates and end-of-day deterministic confirmations;
- transition-only comparisons, calibration diagnostics, and frozen sensitivity checks;
- posterior-weighted, shrunk monthly ETF return estimates with total-covariance risk;
- a cost-aware long-only optimizer with asset, group, and ex-ante volatility caps;
- causal first-adjusted-open execution and drift-aware transaction-cost accounting;
- a causal pooled-mean optimizer ablation plus equal-weight, legacy Sharpe-MAP,
  and static `SPY`/`AGG` 60/40 comparisons;
- frozen allocation sensitivities for mean shrinkage, volatility caps,
  position/group caps, and joint optimizer-policy/realized-cost assumptions;
- paired six-month circular block-bootstrap comparison intervals;
- a 26-year point-in-time component panel from June 2000 through May 2026;
- local processed research files plus publication-safe regime outputs.

The evidence build contains 6,955 normalized first-release observations, 3,968
event rows, and 3,275 causally available feature rows through July 16, 2026.
It preserves delayed release batches and archive backfills explicitly rather
than treating either as ordinary weekly or monthly observations.

The acquired source panel spans 312 reference months from June 2000 through
May 2026. After the 60-observation warm-up, the public history spans July 2005
through May 2026: 247 months are classified and four are explicitly unavailable.
The gap from October 2025 through January 2026 originates with unpublished
October core-CPI and unemployment observations during the federal shutdown and
then propagates through the frozen one-month-change and three-month-smoothing
rules. Model 01 does not impute those observations or redistribute their weights.

The portfolio backtest contains 102 complete monthly holding periods from
January 2018 through June 2026. Published Model 01 outputs now include confirmed
historical labels, a transition-only prior, a release-conditioned posterior,
causal forecast metrics, posterior-weighted allocations, benchmark comparisons,
and frozen sensitivity results.

## Model versions

| ID | Architecture | Status |
|---|---|---|
| `m01_deterministic_composite` | Deterministic targets plus event-driven Bayesian nowcast | Complete through allocation, backtest, and public reporting |
| `m02_continuous_state` | Continuous latent growth/inflation state with quadrant probabilities | Planned |
| `m03_switching_state_space` | Regime-switching continuous state-space model | Planned |

Each later model receives its own configuration and model package. Data
contracts, evaluation, backtesting, and reporting remain shared so comparisons
are apples-to-apples.

## Model 01 in one paragraph

Growth combines nonfarm payrolls, industrial production, real personal
consumption, and the sign-reversed unemployment-rate change. Inflation combines
core CPI, core PCE prices, core finished-goods PPI, and average hourly earnings.
Monthly first-release transformations are standardized against expanding
history ending in the prior month, equally weighted within each axis, and
averaged over the current and preceding two months. The signs of those two
smoothed scores identify one of four regimes.

The complete mathematics, vintage rule, source substitutions, and limitations
are in the Model 01
[`regime definition`](docs/models/m01_deterministic_composite/regime_definition.md),
[`transition model`](docs/models/m01_deterministic_composite/transition_model.md),
[`leading-evidence data`](docs/models/m01_deterministic_composite/leading_evidence_data.md),
and [`event-driven Bayesian filter`](docs/models/m01_deterministic_composite/bayesian_filter.md)
specifications. Actual historical findings are in
[`bayesian_filter_results.md`](docs/models/m01_deterministic_composite/bayesian_filter_results.md).

The allocation layer combines the causal current-month posterior with
24-pseudo-month-shrunk regime return means and a Ledoit-Wolf risk model. It then
solves a fully invested long-only problem net of estimated turnover cost,
subject to individual, group, and 10% ex-ante annualized volatility caps. See
the frozen
[`portfolio methodology`](docs/models/m01_deterministic_composite/portfolio_allocation.md)
and the actual
[`portfolio backtest results`](docs/models/m01_deterministic_composite/portfolio_backtest_results.md).

## Current published result

The newest fully confirmed reference month is May 2026:

- regime: `growth_up_inflation_up`;
- growth score: `0.017332`;
- inflation score: `0.443875`;
- label available: June 25, 2026.

See the machine-readable
[`latest_confirmed.json`](results/published/m01_deterministic_composite/latest_confirmed.json),
the complete public
[`regime_history.csv`](results/published/m01_deterministic_composite/regime_history.csv),
and the frozen
[`data manifest`](data/manifests/m01_deterministic_composite.json). This is a
confirmed historical target, not a daily posterior or investable recommendation.

The transition layer publishes the auditable
[`transition model`](results/published/m01_deterministic_composite/transition_model.json),
its long-form
[`transition matrix`](results/published/m01_deterministic_composite/transition_matrix.csv),
and a
[`transition-only next-month prior`](results/published/m01_deterministic_composite/latest_transition_prior.json).
The prior is conditional on the latest eligible confirmed regime and contains
no event-level evidence.

At the June 25, 2026 information cutoff, the transition-only prior for the June
2026 reference month is:

| Regime ID | Probability |
|---|---:|
| `growth_up_inflation_up` | 69.05% |
| `growth_down_inflation_up` | 15.71% |
| `growth_up_inflation_down` | 12.86% |
| `growth_down_inflation_down` | 2.38% |

These probabilities are the Jeffreys-smoothed row conditioned on May's
confirmed `growth_up_inflation_up` regime. They are a baseline prior, not a
June posterior or a trading recommendation.

The event-driven filter publishes
[`latest_posterior.json`](results/published/m01_bayesian_filter/latest_posterior.json).
At the July 16, 2026 cutoff, its July 2026 marginal is:

| Regime ID | Probability |
|---|---:|
| `growth_up_inflation_up` | 59.26% |
| `growth_down_inflation_up` | 20.23% |
| `growth_up_inflation_down` | 20.25% |
| `growth_down_inflation_down` | 0.26% |

On the 97-month evaluation sample beginning in January 2018, the month-end
posterior achieves NLL 0.9224 and Brier score 0.5193, versus 0.9819 and 0.5330
for the paired transition-only model. The improvement is modest and the model
does not outperform at every checkpoint or metric; see the results document
and published
[`sensitivity metrics`](results/published/m01_bayesian_filter/sensitivity_metrics.csv).

### Current allocation and backtest

The July 1, 2026 `post_month_roll` allocation signal produced this research
target:

| ETF | Target weight |
|---|---:|
| `SPY` | 35% |
| `HYG` | 15% |
| `GLD` | 25% |
| `LQD` | 25% |
| `IEF`, `TIP`, `BIL` | 0% |

The allocation input assigned 56.78% probability to growth up / inflation up,
19.11% to growth down / inflation up, 19.54% to growth up / inflation down, and
4.57% to growth down / inflation down. This is the month-start signal used for
execution, not the later July 16 event-updated posterior above. The model
estimated 0.694% one-month return and 9.65% annualized volatility. These are
research estimates, not promised outcomes or investment advice.

Historical net performance from January 2018 through June 2026 was:

| Method | Total return | CAGR | Ann. vol. | Zero-rate Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Posterior optimized | 124.27% | 9.97% | 10.14% | 0.991 | -21.16% |
| Pooled-mean optimizer | 120.69% | 9.76% | 10.17% | 0.970 | -21.92% |
| Static 60% `SPY` / 40% `AGG` | 117.41% | 9.57% | 11.21% | 0.874 | -21.60% |
| Equal weight | 65.13% | 6.08% | 6.37% | 0.960 | -14.77% |
| Legacy Sharpe-MAP | 50.11% | 4.89% | 4.72% | 1.037 | -11.07% |

The optimized method had the highest historical absolute return, while the
legacy method had the highest zero-rate Sharpe and shallowest drawdown. The
pooled-mean optimizer is the cleaner causal ablation: it removes the current
posterior while retaining the same return sample, risk estimator, constraints,
costs, and execution. The posterior method's annualized arithmetic mean
advantage over it was only 0.187%, with a paired block-bootstrap 95% interval
of [-0.413%, 0.814%] and \(\Pr(\Delta>0)=0.737\). Versus static 60/40, the
corresponding estimate was 0.255%, interval [-2.616%, 3.085%], and
\(\Pr(\Delta>0)=0.572\). Both intervals span zero, so these results do not
support a claim of statistically reliable superiority. The reported
probabilities are bootstrap fractions, not p-values.

The optimized weights were concentrated: `SPY` remained at its 35% cap
throughout the completed sample and `BIL` remained at zero. Scaling non-`BIL`
asset and group caps to 0.8 or 1.2 materially changed return, volatility, and
turnover: the 0.8 multiplier produced 97.70% total return, 8.35% CAGR, 8.91%
annualized volatility, and 31.10% annualized one-way turnover, while 1.2
produced 146.02%, 11.17%, 10.94%, and 53.56%, respectively. These are
diagnostics, not evidence that the looser cap is superior. The outcome depends
on the declared constraint policy. Cost
variants jointly change optimizer regularization and realized charges; they
are not pure execution-cost stresses. Reported turnover uses half-\(L^1\), so
initial formation is shown as 50% one-way turnover even though 100% of NAV is
purchased. Daily drawdown starts from post-formation-cost NAV, while that cost
remains included in return and terminal wealth.

Machine-readable public outputs include the
[`latest allocation`](results/published/m01_regime_allocation_backtest/latest_allocation.json),
[`performance summary`](results/published/m01_regime_allocation_backtest/performance_summary.csv),
[`monthly returns`](results/published/m01_regime_allocation_backtest/monthly_returns.csv),
[`monthly weights`](results/published/m01_regime_allocation_backtest/monthly_weights.csv),
and allocation
[`sensitivity metrics`](results/published/m01_regime_allocation_backtest/sensitivity_metrics.csv),
plus the paired
[`comparison uncertainty`](results/published/m01_regime_allocation_backtest/comparison_uncertainty.csv).

## Repository layout

```text
configs/
  models/                    frozen model definitions
data/
  raw/                       local provider downloads; ignored by Git
  processed/                 local research tables; ignored by Git
  manifests/                 tracked provenance and hashes
docs/
  architecture/              cross-model design decisions
  models/                    model-specific methodology
legacy/                      preserved team paper and surviving script
results/
  published/                 small, reviewable public outputs
src/regime_allocation/
  backtest/                  shared execution, accounting, and performance metrics
  data/                      point-in-time providers and vintage logic
  features/                  shared feature transformations
  models/                    separately versioned architectures
  portfolio/                 return estimation, baselines, optimization, and pipeline
  cli/                       reproducible commands
tests/                       unit, leakage, and integration tests
```

## Rebuild Model 01

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m regime_allocation.cli.build_m01_dataset --provider auto
python -m regime_allocation.cli.build_m01_evidence --provider auto
python -m regime_allocation.cli.build_m01_transition
python -m regime_allocation.cli.build_m01_inference
python -m regime_allocation.cli.build_m01_backtest
pytest
```

`auto` prefers the official FRED API whenever `FRED_API_KEY` is available and
otherwise selects the retained keyless ALFRED client. Both providers produce the
same validated, deterministic observation-by-vintage ZIP contract. Compatible
existing ALFRED matrices are checked before any network request, so adding the
secret does not discard or redownload data already gathered. An explicit FRED
refresh writes under a separate `data/raw/fred_api/` tree and cannot overwrite
the historical `data/raw/alfred/` artifacts.

Use `--provider fred --refresh` to require authenticated acquisition, or
`--provider alfred` to force the keyless path. Provider selection, cache
provenance, and the providers actually used are recorded separately in newly
generated manifests. See [`docs/data_access.md`](docs/data_access.md) for the
exact retrieval and credential contract.

## Legacy project and attribution

The original team manuscript and the only surviving Bayesian orchestration
script are in [`legacy/`](legacy/README.md). They are retained for provenance,
not as working Model 01 code. The team project was produced by Zekai Yao,
Mianchen Zhang, Gavin Huang, and Serin Gleave. This repository documents Gavin
Huang's later independent reimplementation, corrections, and extensions.

## Secrets and broker credentials

The FRED API key is optional locally because the ALFRED fallback remains
available. The authenticated GitHub workflow reads only the repository Actions
secret named `FRED_API_KEY`, scopes it to the retrieval step, and never passes
it as a command-line argument. Schwab tokens and local API keys must remain
outside this repository and outside Codex's readable environment; `.gitignore`
alone prevents commits but does not deny a local process read access. See
[`SECURITY.md`](SECURITY.md) before adding any credentialed provider.

## FRED attribution

This product uses the FRED® API but is not endorsed or certified by the
Federal Reserve Bank of St. Louis. Individual series may also be subject to
third-party usage restrictions; the project preserves public series links and
publishes derived, reviewable outputs rather than credentialed API responses.
