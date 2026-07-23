# Model 02 weekly portfolio allocation and backtest specification

## Scope

This stage applies the promoted Model 02 baseline,
`student_t_7_reduced_core`, to the frozen Model 01 portfolio policy. Its
machine-readable contract is
[`m02_regime_allocation_backtest.yaml`](../../../configs/models/m02_regime_allocation_backtest.yaml),
and its stage ID is `weekly_posterior_regime_allocation_backtest`.

The intended experiment moves the forecast, rebalance, and holding clocks to
the same weekly frequency while preserving frequency-equivalent Model 01
policy choices. The seven strategy ETFs, benchmark-only `AGG`, Ledoit-Wolf
covariance, 10% ex-ante volatility cap, position and group caps,
five-basis-point one-way cost, comparators, and one-at-a-time sensitivities are
otherwise identical to Model 01. The 60-month minimum and 24-pseudo-month mean
shrinkage become 260 completed weeks and 104 pseudo-weeks, respectively.

## Weekly causal signal and execution

Each reference week is identified by its calendar Monday. The promoted filter
is replayed causally and sampled on that Monday immediately after a deterministic
month roll, when applicable, but before any same-day release, partial defining
update, exact-score confirmation, or probability-map vintage. The probability
map is restricted to information available strictly before the Monday signal.

The target marginal is the newest month in Model 02's four-month joint state.
The target trades at the first common adjusted open for all eight simulation
assets in the Monday-anchored week and exits at the next week's first common
adjusted open. A Monday holiday therefore delays execution to Tuesday without
adding Monday releases to the saved signal. This is the weekly analogue of
Model 01 retaining its first-calendar-day signal across a weekend or holiday.

The ETF history begins partway through the Monday-anchored week of 31 December
2007. That leading partial week is excluded from estimation rather than being
treated as a complete weekly holding period.

The first formal signal is 1 January 2018. The final week is published as a
live research target when its next weekly execution open is unavailable, but
it is excluded from every performance statistic.

## Return estimation and Model 02 labels

Each estimator observation is the adjusted-open return from the first common
session in one Monday-anchored week to the first common session in the next.
The observation receives the hard regime label for the calendar month
containing its Monday `reference_week`. A weekly observation enters a fit only
when both its ending execution open and that month's exact-score label were
available by the Sunday immediately before the current Monday signal. This
dual-availability rule excludes both unfinished holdings and labels learned
too late.

Every fit requires at least 260 eligible labeled weeks. Each regime mean is
shrunk toward the pooled weekly mean with 104 pseudo-weeks, the exact
frequency equivalents of Model 01's 60-month minimum and 24 pseudo-months.
The covariance is estimated from the same causally eligible weekly sample and
annualized by 52.

Model 02 does not publish hard regimes as its inference output. For this
portfolio-only estimation layer, each completed exact composite score receives
the quadrant implied by the signs of its growth and inflation scores; zero is
assigned to the corresponding “up” side, matching Model 02's evaluation
helper. The label becomes usable only on `score_available_at`. These labels
train historical return moments; the current allocation always integrates the
full promoted soft probability vector and never replaces it with a MAP state.

## Optimization, comparators, and costs

At every weekly signal, the posterior-weighted shrunk weekly means and shared
within-regime plus between-regime covariance enter the same long-only Model 01
optimizer. Its return term is the expected one-week portfolio return, and its
turnover penalty is the estimated cost of the single rebalance needed to reach
the candidate target. All individual and group caps, SLSQP tolerances, and the
fallback order—feasible pretrade holdings, constrained minimum variance, then
all `BIL`—are unchanged.

Pretrade target formation uses the previous weekly execution open and the last
adjusted close strictly before the Monday signal. Realized accounting drifts
the prior holdings to the actual new execution open. The cost rate is

$$
K_t=\sum_a 0.0005\left|w_{t,a}-w_{t,a}^{-,\mathrm{exec}}\right|,
$$

and the net weekly return is `(1 - K_t) * (1 + gross_return) - 1`. The initial
formation trade is charged. Equal weight, legacy Sharpe-MAP, static 60/40, and
the pooled-mean optimizer rebalance on the same weekly dates and pay the same
realized costs.

### Exploratory pooled anchor

The published comparison also includes the derived target
`pooled_anchor_posterior_25pct`:

$$
w_t^{\mathrm{anchor}}
=0.75w_t^{\mathrm{pooled}}+0.25w_t^{\mathrm{posterior}}.
$$

This construction leaves the original posterior and pooled-mean strategies
unchanged. It is simulated as its own consolidated portfolio, with its own
holdings drift between execution opens and its own trades to the next blended
target. The same five-basis-point one-way cost is charged on that realized
anchor turnover; its backtest is not a weighted average of the two source
strategies' reported net returns.

As a convex combination of two feasible long-only targets, the anchor inherits
the linear full-investment, asset-cap, and group-cap constraints. It is not a
separate optimizer solution, however, and no 10% ex-ante volatility constraint
has been independently audited for the blend under one common covariance
estimate. The 25% posterior sleeve was introduced after reviewing the
corrected same-history results. It is therefore exploratory, post-result, and
non-promoted; it does not replace the corrected posterior baseline.

## Evaluation and artifacts

Performance metrics use 52 periods per year. Paired uncertainty uses the same
circular-block design and 10,000 resamples as Model 01, with 26-week blocks as
the frequency-equivalent of six months. The selected baseline was promoted
after same-history development, so the backtest is descriptive and is not an
untouched holdout. The exploratory anchor receives paired comparisons against
both the posterior baseline and pooled-mean ablation, but those intervals do
not turn its post-result 25% sleeve into a prespecified test.

Detailed artifacts live in `data/processed/m02_regime_allocation_backtest/`.
Public results live in `results/published/m02_regime_allocation_backtest/`, and
[`m02_regime_allocation_backtest.json`](../../../data/manifests/m02_regime_allocation_backtest.json)
hashes the promoted-baseline lineage, policy template, implementation, inputs,
and outputs.

The principal limitations remain uncertain expected returns, mutable adjusted
ETF history, a selected narrow universe, idealized opening execution, fixed
costs without market impact or taxes, plug-in regime probabilities and return
moments, and a finite sample with few independent macro cycles. Several weekly
observations share each monthly hard label, so the larger observation count
does not create an equivalent number of independent macro regimes. Weekly
trading also creates more opportunities for target instability and cost
accumulation.
