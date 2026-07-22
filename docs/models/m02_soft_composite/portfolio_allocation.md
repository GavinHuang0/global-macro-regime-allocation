# Model 02 weekly portfolio allocation and backtest specification

## Scope

This stage applies the promoted Model 02 baseline,
`student_t_7_reduced_core`, to the frozen Model 01 portfolio policy. Its
machine-readable contract is
[`m02_regime_allocation_backtest.yaml`](../../../configs/models/m02_regime_allocation_backtest.yaml),
and its stage ID is `weekly_posterior_regime_allocation_backtest`.

The intended experiment changes the rebalance cadence, not the portfolio
policy. The seven strategy ETFs, benchmark-only `AGG`, monthly
regime-conditioned return estimator, 60-month minimum, 24-pseudo-month mean
shrinkage, Ledoit-Wolf covariance, 10% ex-ante volatility cap, position and
group caps, five-basis-point one-way cost, comparators, and one-at-a-time
sensitivities are identical to Model 01.

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

The first formal signal is 1 January 2018. The final week is published as a
live research target when its next weekly execution open is unavailable, but
it is excluded from every performance statistic.

## Return estimation and Model 02 labels

Changing the estimator to weekly returns would also change the meaning of the
60-month minimum, 24 pseudo-months, expected-return horizon, transaction-cost
regularization, and covariance scaling. The stage therefore retains Model
01's monthly first-open-to-next-first-open return model and 12x covariance
annualization. Weekly re-estimation uses only monthly returns and labels
available by the Sunday immediately before the signal.

Model 02 does not publish hard regimes as its inference output. For this
portfolio-only estimation layer, each completed exact composite score receives
the quadrant implied by the signs of its growth and inflation scores; zero is
assigned to the corresponding “up” side, matching Model 02's evaluation
helper. The label becomes usable only on `score_available_at`. These labels
train historical return moments; the current allocation always integrates the
full promoted soft probability vector and never replaces it with a MAP state.

## Optimization, comparators, and costs

At every weekly signal, the posterior-weighted shrunk monthly means and shared
within-regime plus between-regime covariance enter the same long-only Model 01
optimizer. All individual and group caps, SLSQP tolerances, and the fallback
order—feasible pretrade holdings, constrained minimum variance, then all
`BIL`—are unchanged.

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

## Evaluation and artifacts

Performance metrics use 52 periods per year. Paired uncertainty uses the same
circular-block design and 10,000 resamples as Model 01, with 26-week blocks as
the frequency-equivalent of six months. The selected baseline was promoted
after same-history development, so the backtest is descriptive and is not an
untouched holdout.

Detailed artifacts live in `data/processed/m02_regime_allocation_backtest/`.
Public results live in `results/published/m02_regime_allocation_backtest/`, and
[`m02_regime_allocation_backtest.json`](../../../data/manifests/m02_regime_allocation_backtest.json)
hashes the promoted-baseline lineage, policy template, implementation, inputs,
and outputs.

The principal limitations remain uncertain expected returns, mutable adjusted
ETF history, a selected narrow universe, idealized opening execution, fixed
costs without market impact or taxes, plug-in regime probabilities and return
moments, and a finite sample with few independent macro cycles. Weekly trading
also creates more opportunities for target instability and cost accumulation.

