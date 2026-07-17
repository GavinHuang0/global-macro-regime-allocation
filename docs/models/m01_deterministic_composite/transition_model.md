# Model 01 transition model

## Scope

Model 01 uses a first-order, time-homogeneous Markov chain to turn the current
distribution over deterministic macro regimes into a prior for the following
reference month. The transition model is deliberately simple: it has no market
covariates, duration dependence, calendar effects, or time-varying parameters.
Its role is to provide a reproducible baseline before event-level likelihoods
are introduced.

The word *fixed* describes the form of the transition law. At any information
cutoff, one $4\times4$ matrix applies to every eligible transition. Historical
counts are nevertheless re-estimated through an expanding window as additional
regime labels become available. The matrix is therefore not estimated with the
full sample and silently reused at earlier backtest dates.

This layer produces a transition prior, not a daily posterior. A posterior will
require the later event-update layer to condition that prior on releases known
at the relevant timestamp.

## 1. State space and frozen order

The state space is the four deterministic quadrants defined in
[`regime_definition.md`](regime_definition.md). Every matrix and serialized
array uses this permanent order:

1. `growth_up_inflation_up`;
2. `growth_down_inflation_up`;
3. `growth_up_inflation_down`;
4. `growth_down_inflation_down`.

The model JSON stores this order explicitly. Long-form matrix rows and prior
probability records also carry their stable regime IDs and labels. Consumers
must not infer an order from display labels, alphabetical sorting, dictionary
iteration, or numerical encodings.

Let $R_m\in\{1,2,3,4\}$ denote the regime for reference month $m$. Define

$$
A_{ij}=P(R_{m+1}=j\mid R_m=i).
$$

Each row of $A$ is a categorical distribution and therefore satisfies

$$
A_{ij}\geq0,
\qquad
\sum_{j=1}^{4}A_{ij}=1.
$$

The first-order assumption is

$$
P(R_{m+1}\mid R_{m-3:m})=P(R_{m+1}\mid R_m).
$$

Retaining a joint distribution over four months supports delayed observations
and fixed-lag revision of earlier states. It does not make Model 01 a
fourth-order Markov chain.

## 2. Causal transition eligibility

The historical input is the point-in-time regime history. For each classified
month $m$, let $T_m$ be its `label_available_at` date. A candidate pair
$(m,m+1)$ is eligible at information cutoff $\tau$ only when all of the
following are true:

1. the two reference months are exactly one calendar month apart;
2. both rows have nonmissing regime IDs;
3. both labels were available by the cutoff:

$$
T_m\leq\tau
\quad\text{and}\quad
T_{m+1}\leq\tau.
$$

Equivalently, define the pair availability date

$$
T_{m\rightarrow m+1}=\max(T_m,T_{m+1});
$$

the transition becomes usable when $T_{m\rightarrow m+1}\leq\tau$.

The implementation never bridges a missing label. For example, if October
through January are unavailable, September-to-February is not treated as a
one-month transition. The current frozen public history contains 251 calendar
months, 247 classified months, and one four-month unavailable block. It
therefore contains 245 eligible consecutive transitions at the latest full
cutoff: the 250 adjacent pairs minus the five pairs that touch the block.

This timing rule matters in a walk-forward evaluation. A regime belongs to a
reference month but is not known on that month's final calendar day. Its
outgoing or incoming transition may enter the expanding count matrix only after
the destination and source labels have both become available.

## 3. Transition counts

At cutoff $\tau$, define

$$
N_{ij}^{(\tau)}
=
\sum_m
\mathbf 1\!\left[
R_m=i,
R_{m+1}=j,
T_{m\rightarrow m+1}\leq\tau,
\text{pair }(m,m+1)\text{ is eligible}
\right].
$$

The outgoing row total is

$$
N_i^{(\tau)}=\sum_{j=1}^{4}N_{ij}^{(\tau)}.
$$

For diagnostics, the unsmoothed maximum-likelihood estimate is

$$
\widehat A_{ij}^{\mathrm{MLE},(\tau)}
=
\frac{N_{ij}^{(\tau)}}{N_i^{(\tau)}}
$$

when $N_i^{(\tau)}>0$. It is undefined for an empty row and can assign an
unseen transition a probability of exactly zero, so it is not the production
estimate.

## 4. Dirichlet smoothing

Each row receives an independent symmetric Dirichlet prior:

$$
A_{i,\cdot}
\sim
\operatorname{Dirichlet}(\alpha,\alpha,\alpha,\alpha),
\qquad
\alpha=0.5.
$$

The value $\alpha=0.5$ is the symmetric Jeffreys prior for a categorical row.
It contributes two total pseudo-counts to each row, prevents an unobserved
transition from becoming impossible, and remains weak relative to a row with a
substantial history.

Conditional on the eligible counts, the row posterior is

$$
A_{i,\cdot}\mid\mathcal D_\tau
\sim
\operatorname{Dirichlet}
\left(
N_{i1}^{(\tau)}+0.5,
N_{i2}^{(\tau)}+0.5,
N_{i3}^{(\tau)}+0.5,
N_{i4}^{(\tau)}+0.5
\right).
$$

Model 01 uses the posterior-predictive mean

$$
\overline A_{ij}^{(\tau)}
=
\frac{N_{ij}^{(\tau)}+0.5}
{N_i^{(\tau)}+4(0.5)}
=
\frac{N_{ij}^{(\tau)}+0.5}
{N_i^{(\tau)}+2}
$$

as its transition probability. A row with no eligible history is therefore the
uniform distribution $(0.25,0.25,0.25,0.25)$ rather than an error or a vector
of zeros.

The marginal posterior for a single entry is

$$
A_{ij}\mid\mathcal D_\tau
\sim
\operatorname{Beta}
\left(
N_{ij}^{(\tau)}+0.5,
N_i^{(\tau)}-N_{ij}^{(\tau)}+3(0.5)
\right).
$$

Published 95% equal-tailed credible intervals use the 2.5th and 97.5th
percentiles of this beta distribution. These are marginal intervals for
individual cells, not simultaneous intervals for an entire row.

## 5. Expanding estimation

For a live build at cutoff $\tau$, the model uses every eligible transition
available by $\tau$. For a historical forecast, it recomputes
$\overline A^{(\tau)}$ from only the transitions then available. It never fits
one matrix using the final sample and applies that matrix to earlier dates.

The following items are fixed across expanding fits:

- the four state IDs and their order;
- the first-order, time-homogeneous transition graph;
- the symmetric Dirichlet prior with $\alpha=0.5$;
- the consecutive-month and availability rules.

Only the eligible transition counts change. The public artifact records both
the information cutoff and the latest destination reference month admitted to
the fit so that a reference period cannot be confused with a knowledge date.

These choices are frozen separately from the deterministic data specification
in
`configs/models/m01_deterministic_composite_transition.yaml`. Separating the
stage configuration allows the same point-in-time regime history to be rebuilt
without silently refitting or changing the transition law.

## 6. Shifting the four-month path distribution

Suppose that immediately before the monthly shift the joint distribution is

$$
q_m(a,b,c,d)
=
P(R_{m-3}=a,R_{m-2}=b,R_{m-1}=c,R_m=d\mid\mathcal D_\tau).
$$

The prior over the shifted path is

$$
q_{m+1}^{-}(b,c,d,e)
=
\sum_{a=1}^{4}
q_m(a,b,c,d)\overline A_{de}^{(\tau)}.
$$

This operation marginalizes the oldest state, preserves the three overlapping
months, and appends $R_{m+1}$. It maps a normalized nonnegative distribution on
$4^4=256$ paths to another normalized nonnegative distribution on 256 paths.

The next-month marginal follows directly:

$$
P(R_{m+1}=e\mid\mathcal D_\tau)
=
\sum_{d=1}^{4}
P(R_m=d\mid\mathcal D_\tau)\overline A_{de}^{(\tau)}.
$$

The full marginal distribution of $R_m$ is propagated. Replacing it with its
most likely state before the multiplication would discard uncertainty and is
not part of Model 01.

## 7. Reproducibility outputs

The transition build writes a pair-level audit table to
`data/processed/m01_deterministic_composite/transition_pairs.csv`. It contains
every adjacent row pair considered by the estimator, including excluded pairs
and their reasons. Like the other processed research tables, this file is
reproducible but not committed as a public result.

The build publishes three small, credential-free artifacts under
`results/published/m01_deterministic_composite/`:

- `transition_model.json`, a self-contained machine-readable specification and
  fit summary;
- `transition_matrix.csv`, a long-form table with one row per directed state
  pair;
- `latest_transition_prior.json`, the transition-only forecast for the month
  immediately after the latest eligible confirmed regime.

The JSON records the artifact schema version, model and stage IDs, state order,
information cutoff, input history and configuration hashes, eligible and
excluded pair counts, raw count matrix, MLE diagnostic, posterior parameters,
posterior-predictive matrix, and marginal credible intervals. The CSV includes
the row totals and repeats the cell-level quantities in a convenient review
format. Exact fields are defined in
[`data_dictionary.md`](data_dictionary.md).

The owning model ID remains `m01_deterministic_composite`; the independently
identifiable processing stage is `fixed_first_order_transition`.

The latest prior conditions on one confirmed deterministic source regime and is
the corresponding row of the posterior-predictive transition matrix. It states
the source month, target month, and information cutoff explicitly. It is not a
joint-path prior and does not condition on any current-month release evidence.
It must not be interpreted as a current-month posterior. The event layer
will publish separately named posterior snapshots.

## 8. Evaluation plan

Walk-forward evaluation will score the one-month predictive distributions with
multiclass log loss and Brier score, plus calibration diagnostics. Classification
accuracy is secondary because it discards probability quality. The required
baselines are:

1. expanding unconditional regime frequencies;
2. pure last-regime persistence;
3. the unsmoothed transition MLE;
4. the Dirichlet-smoothed transition model specified here.

Sensitivity analysis will report $\alpha\in\{0,0.5,1\}$, but the final test
period must not be used to choose the most flattering value.

## 9. Limitations

- The three-month trailing regime score uses overlapping inputs, which
  mechanically raises diagonal transition probabilities and apparent regime
  durations. The matrix describes the frozen label process; it is not a pure
  estimate of structural economic persistence.
- A first-order Markov chain implies geometrically distributed durations. It
  cannot distinguish a regime that began last month from one that has persisted
  for a year.
- Time homogeneity assumes the same transition law across monetary-policy,
  volatility, and structural economic environments. Subperiod matrices must be
  reported before treating that assumption as empirically adequate.
- Rows are assigned independent Dirichlet priors. The model does not pool
  similar transitions across origin states or learn economic restrictions on
  which quadrant changes are more plausible.
- The posterior-predictive mean carries point estimates into the path filter.
  Parameter uncertainty is shown through cell intervals but is not integrated
  through every later posterior calculation in this baseline.
- Rare regimes and transitions can still have wide uncertainty despite
  smoothing. Raw counts and row totals must always accompany probabilities.
- Transition estimates inherit every vintage, transformation, missing-data,
  and threshold choice in the deterministic regime definition.

Duration-dependent, semi-Markov, covariate-dependent, or time-varying
transition laws are materially different architectures and belong in later
model versions rather than hidden options inside this baseline.
