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

## Notation

The notation follows [`bayesian_filter.md`](bayesian_filter.md#notation): $m$
is a target reference month, $\ell$ is another historical reference month,
$d$ is a knowledge date, and $\mathcal D_d$ is the information available by
that date. $R_m\in\mathcal R$ is the monthly regime, where $\mathcal R$ is the
four-state set. The four-month path is
$S_m=(R_{m-3},R_{m-2},R_{m-1},R_m)$, $s$ denotes one candidate path, and
$q_{m,d}(s)=\Pr(S_m=s\mid\mathcal D_d)$. Bold uppercase letters denote
matrices.

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

Let $R_m\in\{1,2,3,4\}$ denote the regime for reference month $m$. Define the
transition matrix $\mathbf A=(A_{ij})$ by

$$
A_{ij}=\Pr(R_{m+1}=j\mid R_m=i).
$$

Each row of $\mathbf A$ is a categorical distribution and therefore satisfies

$$
A_{ij}\geq0,
\qquad
\sum_{j=1}^{4}A_{ij}=1.
$$

The first-order assumption is

$$
\Pr(R_{m+1}\mid R_{m-3:m})=\Pr(R_{m+1}\mid R_m).
$$

Retaining a joint distribution over four months supports delayed observations
and fixed-lag revision of earlier states. It does not make Model 01 a
fourth-order Markov chain.

## 2. Causal transition eligibility

The historical input is the point-in-time regime history. For each classified
month $m$, let $T_m$ be its `label_available_at` date. A candidate pair
$(\ell,\ell+1)$ is eligible at knowledge date $d$ only when all of the
following are true:

1. the two reference months are exactly one calendar month apart;
2. both rows have nonmissing regime IDs;
3. both labels were available by the cutoff:

$$
T_\ell\leq d
\quad\text{and}\quad
T_{\ell+1}\leq d.
$$

Equivalently, define the pair availability date

$$
T_{\ell\rightarrow \ell+1}=\max(T_\ell,T_{\ell+1});
$$

the transition becomes usable when $T_{\ell\rightarrow \ell+1}\leq d$.

The implementation never bridges a missing label. For example, if October
through January are unavailable, September-to-February is not treated as a
one-month transition. The current frozen public history contains 250 calendar
months, 246 classified months, and one four-month unavailable block. It
therefore contains 244 eligible consecutive transitions at the latest full
cutoff: the 249 adjacent pairs minus the five pairs that touch the block.

This timing rule matters in a walk-forward evaluation. A regime belongs to a
reference month but is not known on that month's final calendar day. Its
outgoing or incoming transition may enter the expanding count matrix only after
the destination and source labels have both become available.

## 3. Transition counts

At knowledge date $d$, define

$$
N_{ij}^{(d)}
=
\sum_\ell
\mathbf 1\!\left[
R_\ell=i,
R_{\ell+1}=j,
T_{\ell\rightarrow\ell+1}\leq d,
\text{pair }(\ell,\ell+1)\text{ is eligible}
\right].
$$

The indicator $\mathbf 1[\cdot]$ equals one when every condition inside the
brackets is true and zero otherwise, so $N_{ij}^{(d)}$ is an ordinary count of
causally usable transitions from state $i$ to state $j$.

The outgoing row total is

$$
N_i^{(d)}=\sum_{j=1}^{4}N_{ij}^{(d)}.
$$

For diagnostics, the unsmoothed maximum-likelihood estimate is

$$
\widehat A_{ij}^{\mathrm{MLE},(d)}
=
\frac{N_{ij}^{(d)}}{N_i^{(d)}}
$$

when $N_i^{(d)}>0$. It is undefined for an empty row and can assign an
unseen transition a probability of exactly zero, so it is not the production
estimate.

## 4. Dirichlet smoothing

Each row receives an independent symmetric Dirichlet prior:

$$
\mathbf A_{i,\cdot}
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
\mathbf A_{i,\cdot}\mid\mathcal D_d
\sim
\operatorname{Dirichlet}
\left(
N_{i1}^{(d)}+0.5,
N_{i2}^{(d)}+0.5,
N_{i3}^{(d)}+0.5,
N_{i4}^{(d)}+0.5
\right).
$$

Model 01 uses the posterior-predictive mean: the next-transition probability
obtained by averaging each unknown transition probability over its Dirichlet
posterior. It is

$$
\overline A_{ij}^{(d)}
=
\frac{N_{ij}^{(d)}+0.5}
{N_i^{(d)}+4(0.5)}
=
\frac{N_{ij}^{(d)}+0.5}
{N_i^{(d)}+2}
$$

as its transition probability. A row with no eligible history is therefore the
uniform distribution $(0.25,0.25,0.25,0.25)$ rather than an error or a vector
of zeros.

The marginal posterior for a single entry is

$$
A_{ij}\mid\mathcal D_d
\sim
\operatorname{Beta}
\left(
N_{ij}^{(d)}+0.5,
N_i^{(d)}-N_{ij}^{(d)}+3(0.5)
\right).
$$

Published 95% equal-tailed credible intervals use the 2.5th and 97.5th
percentiles of this beta distribution. These are marginal intervals for
individual cells, not simultaneous intervals for an entire row.

## 5. Expanding estimation

For a live build at knowledge date $d$, the model uses every eligible
transition available by $d$. For a historical forecast, it recomputes
$\overline{\mathbf A}^{(d)}$ from only the transitions then available. It
never fits one matrix using the final sample and applies that matrix to earlier
dates.

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

Suppose that immediately before a roll from month $m$ to month $m+1$ on
knowledge date $d$, the joint distribution is

$$
q_{m,d^-}(s)=\Pr(S_m=s\mid\mathcal D_{d^-}).
$$

For candidate states $r_{m-3},\ldots,r_{m+1}\in\mathcal R$, the prior over the
shifted path is

$$
\begin{aligned}
q_{m+1,d}^{-}(r_{m-2},r_{m-1},r_m,r_{m+1})
={}&\sum_{r_{m-3}\in\mathcal R}
q_{m,d^-}(r_{m-3},r_{m-2},r_{m-1},r_m)\\
&\times\overline A_{r_m,r_{m+1}}^{(d^-)}.
\end{aligned}
$$

This operation marginalizes the oldest state, preserves the three overlapping
months, and appends $R_{m+1}$. It maps a normalized nonnegative distribution on
$4^4=256$ paths to another normalized nonnegative distribution on 256 paths.

The next-month marginal follows directly:

$$
\Pr(R_{m+1}=j\mid\mathcal D_{d^-})
=
\sum_{i\in\mathcal R}
\Pr(R_m=i\mid\mathcal D_{d^-})
\overline A_{ij}^{(d^-)}.
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
publishes separately named posterior snapshots under `m01_bayesian_filter`.

## 8. Evaluation

The Dirichlet-smoothed transition model is replayed as the no-leading-evidence
baseline for every event-filter checkpoint. It uses the same causal month
rolls, expanding cutoff, initialization, and deterministic confirmations as the
event model. The January 2018--May 2026 results are reported in
[`bayesian_filter_results.md`](bayesian_filter_results.md).

An expanding unconditional-frequency model, hard persistence rule, and
unsmoothed transition MLE remain useful future diagnostics but are not part of
the frozen headline comparison. Likewise, alternative Dirichlet alpha values
were not added after observing the evaluation period. The absence of those
additional baselines is a limitation, not evidence that the selected transition
law is optimal.

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
