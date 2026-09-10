# Research architecture and published baselines

This overview describes the shared research principles, model specifications,
notation, and published results. The performance below belongs to the
**published 2026 research snapshots and their stated sample windows**. The
weekly research output is reported separately with its own dates and inputs.

For the dated weekly research output, see the
[project README](../../README.md). Use the [documentation index](../README.md)
for the full model specifications and operating guides.

Model 01 is the frozen monthly benchmark. Model 02 is the promoted weekly model,
with a recorded historical publication and a separately updated research run.
Later model releases must preserve existing publications and identify their changes.

## Promoted baselines

| Model | Promoted inference baseline | Promoted allocation baseline | Frequency | Status |
|---|---|---|---|---|
| Model 01 | Event-driven fixed-$`\nu=7`$ Bayesian filter over deterministic hard regimes | `posterior_optimized` | Monthly | Frozen benchmark |
| Model 02 | `student_t_7_reduced_core` | `posterior_optimized` | Weekly | Promoted; versioned model with a retained historical publication |

Only these baselines are part of the main model contract. Pooled-mean,
equal-weight, and 60/40 portfolios remain as essential allocation comparisons.
Other experimental variants are documented in the research archive and did
not replace a promoted baseline.

## Shared research contract

Both models use the same principles:

1. **Point-in-time macro data.** A transformation uses the first eligible
   release and the prior level visible in that same vintage.
2. **Causal fitting.** A signal at information cutoff $`d`$ uses only returns,
   labels, and releases available by its declared cutoff.
3. **Explicit execution.** Targets trade at the first common adjusted open of
   the holding period; the next common open ends the period.
4. **Own-path accounting.** Each strategy has its own drifted pretrade
   holdings, turnover, five-basis-point one-way cost, and daily NAV path.
5. **Content-addressed outputs.** Manifests record configurations, inputs,
   implementation files, and published artifacts by SHA-256.

The end-to-end flow is:

```text
point-in-time releases
    -> growth/inflation state inference
    -> quadrant probabilities
    -> causal return moments
    -> constrained target weights
    -> transaction-cost-aware backtest
```

## Canonical notation

| Symbol | Meaning |
|---|---|
| $`m`$ | Macro reference month |
| $`d`$ | Information cutoff |
| $`r\in\mathcal R`$ | One of four growth/inflation quadrants |
| $`p_{m,r\mid d}`$ | Probability of quadrant $`r`$ for month $`m`$, given information through $`d`$ |
| $`\boldsymbol z_m`$, $`\boldsymbol Z_m`$ | Model 02 completed score and corresponding random state |
| $`t`$ | Portfolio rebalance and holding-period index |
| $`\mathbf x_t`$ | Asset simple-return vector |
| $`\mathbf w_t^{-}`$, $`\mathbf w_t`$ | Pretrade and target weights |
| $`\boldsymbol\mu_t`$, $`\boldsymbol\Sigma_t`$ | Expected-return vector and covariance used by the allocator |
| $`K_t`$ | Realized trading cost |
| $`A`$ | Annualization factor: 12 for Model 01 and 52 for Model 02 |

The canonical quadrant order is:

1. growth up / inflation up;
2. growth down / inflation up;
3. growth up / inflation down; and
4. growth down / inflation down.

Model 01 has a hard monthly regime $`R_m`$. Model 02 instead defines the
completed continuous score

```math
\boldsymbol z_m=(G_m,I_m)^\top
```

and models its uncertain state as $`\boldsymbol Z_m`$, whose reporting
distribution is integrated over the same four quadrants.
Model 01 and Model 02 inference scores are not directly comparable: Model 01
scores eventual one-hot regimes, while Model 02 scores continuous states and
a soft Gaussian quadrant map.

## Model 01

### Inference

Model 01 forms equal-weight growth and inflation composites from eight
first-release macro components. Transformations are standardized with strictly
lagged expanding moments, averaged within each axis, and smoothed over three
months. The signs of the two smoothed composites define $`R_m`$.

An expanding first-order Markov transition model evolves a four-month joint
regime path. Five leading-release blocks update that path with
regime-conditioned multivariate Student-$`t`$ likelihoods using fixed
$`\nu=7`$. Completed composite regimes enter later as exact end-of-day
confirmations.

At the month-end checkpoint over 97 scored targets, the promoted filter has
NLL 0.917 and Brier score 0.516, versus 0.976 and 0.529 for transition only.
Hard-label accuracy is similar, so the evidence supports modestly better
probability assignment rather than clearly better classification.

### Allocation

The promoted monthly allocator:

- uses `SPY`, `IEF`, `TIP`, `HYG`, `BIL`, `GLD`, and `LQD`;
- keeps `AGG` only for the 60/40 comparison;
- estimates monthly open-to-open regime returns from at least 60 labeled
  months;
- shrinks regime means toward the pooled mean with 24 pseudo-months;
- uses a shared Ledoit–Wolf within-regime covariance;
- is long-only, fully invested, and capped at 10% annualized volatility; and
- maximizes expected one-month return net of estimated trading cost.

The formal sample contains 102 complete months from January 2018 through June
2026.

| Method | CAGR | Ann. volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Posterior-optimized | **9.96%** | 10.17% | **0.988** | -21.16% |
| Pooled-mean | 9.76% | 10.17% | 0.970 | -21.92% |
| Static 60/40 | 9.57% | 11.21% | 0.874 | -21.60% |
| Equal-weight | 6.08% | **6.37%** | 0.960 | **-14.77%** |

Posterior minus pooled mean is +0.184% annualized arithmetic return with a 95%
paired block-bootstrap interval of [-0.364%, +0.795%]. The interval includes
zero; Model 01 does not establish a reliable incremental allocation edge from
its posterior.

See the [Model 01 model card](../../docs/models/m01_deterministic_composite/README.md).

## Model 02

### Inference

Model 02 removes Model 01's three-month smoothing and hard latent-regime
target. It evolves the continuous random state $`\boldsymbol Z_m`$ through an expanding
causal VAR(1) and a rolling four-month joint Gaussian state.

The promoted `student_t_7_reduced_core` baseline combines:

- sequential partial releases of the score-defining components;
- exact end-of-day conditioning when a monthly score is complete;
- fixed-$`\nu=7`$ robust non-defining release updates; and
- three reduced-core evidence models: initial claims, joint real-retail and
  implicit-price evidence, and joint capital-goods activity and pipeline
  evidence.

At the primary `before_any_defining_release` checkpoint, 183 target months are
scored. Mean score-center NLPD is 26.854, quadrant cross-entropy 1.536,
quadrant Brier distance 0.150, and hard-quadrant accuracy 37.70%. Its essential
comparisons with transition-only and partial-only are small and their
12-month block-bootstrap intervals include zero. The baseline is promoted as
the current operating specification, not as a statistically superior model.

### Allocation

The promoted weekly allocator uses the same seven strategy ETFs and policy caps
as Model 01, but aligns forecast, estimation, and holding horizons to one week:

- Monday-anchored first-common-open to next-week first-common-open returns;
- at least 260 labeled weeks;
- 104 pseudo-weeks of regime-mean shrinkage;
- 52x covariance annualization; and
- the posterior available on Monday before same-day releases.

The formal sample contains 445 complete weeks from 1 January 2018 through
6 July 2026.

| Method | CAGR | Ann. volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Posterior-optimized | **9.86%** | 10.42% | 0.956 | -21.75% |
| Pooled-mean | 9.83% | 10.50% | 0.946 | -21.99% |
| Static 60/40 | 9.60% | 11.62% | 0.847 | -22.02% |
| Equal-weight | 6.09% | **6.34%** | **0.964** | **-14.69%** |

Posterior minus pooled mean is +0.019% annualized arithmetic return with a 95%
paired block-bootstrap interval of [-0.088%, +0.130%]. The posterior is the
promoted allocation baseline, but this sample does not establish a reliable
incremental edge over pooled mean.

Model 02's published manifests identify the source and data snapshots used for
its recorded results. Exact reproduction requires those inputs; fresh provider
downloads can produce different historical values. Subsequent model changes
are recorded as separate versioned releases.

See the [Model 02 model card](../../docs/models/m02_soft_composite/README.md).

## Repository layout

```text
configs/models/          declarative model and backtest contracts
data/manifests/          content hashes and provenance
data/processed/          derived local data; ignored by Git
docs/models/             promoted model specifications and results
docs/archive/            development history outside the main model contract
results/published/       public machine-readable outputs
results/live/            dated outputs from the weekly research update
scripts/                 market-data acquisition and provider comparison utilities
src/regime_allocation/   data, features, models, portfolio, backtest, and CLI code
tests/                   unit and integration contracts
```

Primary documents:

- [Model 01](../../docs/models/m01_deterministic_composite/README.md)
- [Model 02](../../docs/models/m02_soft_composite/README.md)
- [Model versioning](../../docs/architecture/model_versioning.md)
- [Data access and credentials](../../docs/data_access.md)
- [Security policy](../../SECURITY.md)
- [Research archive](../../docs/archive/README.md)
- [Legacy course-project artifacts](../../legacy/README.md)

Primary machine-readable outputs:

- [Model 01 latest posterior](../../results/published/m01_bayesian_filter/latest_posterior.json)
- [Model 01 latest allocation](../../results/published/m01_regime_allocation_backtest/latest_allocation.json)
- [Model 01 allocation performance](../../results/published/m01_regime_allocation_backtest/performance_summary.csv)
- [Model 02 current inference registry](../../results/published/m02_soft_composite/current/current_registry.csv)
- [Model 02 latest marginals](../../results/published/m02_soft_composite/current/baseline_latest_marginals.csv)
- [Model 02 latest allocation](../../results/published/m02_regime_allocation_backtest/latest_allocation.json)
- [Model 02 allocation performance](../../results/published/m02_regime_allocation_backtest/performance_summary.csv)

## Installation and reproduction

Python 3.11 or later is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Model 01's stages are listed below. The allocation stage also requires the
ETF history described in the [data access guide](../data_access.md). The
[Model 01 reproduction workflow](../../.github/workflows/reproduce-model-01.yml)
records the complete acquisition and build sequence.

```powershell
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m01_dataset --provider auto
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m01_evidence --provider auto
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m01_transition
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m01_inference
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m01_backtest
```

Model 02's historical publication and allocation stages consume prepared
upstream score, probability-map, transition, evidence, and ETF artifacts.
Their specifications and dependencies are described in the
[Model 02 model card](../models/m02_soft_composite/README.md).

```powershell
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m02_current_baseline
.\.venv\Scripts\python.exe -m regime_allocation.cli.build_m02_backtest
```

With the required data and artifacts available, run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`--provider auto` uses the FRED API when `FRED_API_KEY` is available and the
keyless ALFRED path otherwise. Credentials are read only from the environment
and are never written to manifests, cache identifiers, published URLs, or logs. See
[data access](../../docs/data_access.md) for the complete policy.

The [weekly operations guide](../operations/weekly_updates.md) describes the
separate live-update command. It creates dated research outputs without
replacing the archival publication.

## Limitations

- Neither model has an untouched confirmatory holdout.
- Macro cycles and realized regimes are few and imbalanced.
- Release likelihoods and return mappings simplify dependence and parameter
  uncertainty.
- ETF constraints and the chosen universe materially affect allocation.
- Adjusted ETF history is a mutable provider snapshot, not point-in-time market
  data.
- Modeled costs omit taxes, market impact, capacity, and operational failure.

Published probabilities, targets, and historical results are research
artifacts, not live trading recommendations.

## Attribution

The original course project was produced by Zekai Yao, Mianchen Zhang, Gavin
Huang, and Serin Gleave. Model 01 and Model 02 are Gavin Huang's subsequent
independent reimplementation and extensions. This product uses the FRED API but
is not endorsed or certified by the Federal Reserve Bank of St. Louis.
