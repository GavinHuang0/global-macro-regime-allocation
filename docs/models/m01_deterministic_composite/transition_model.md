# Model 01 transition model

The `fixed_first_order_transition` stage supplies the causal monthly prior used
by the Model 01 filter. Its contract is
[`configs/models/m01_deterministic_composite_transition.yaml`](../../../configs/models/m01_deterministic_composite_transition.yaml).

## State and timing

Let \(m\) be a reference month, \(d\) an information cutoff, and
\(R_m\in\mathcal R\) the deterministic regime. The permanent state order is:

1. `growth_up_inflation_up`
2. `growth_down_inflation_up`
3. `growth_up_inflation_down`
4. `growth_down_inflation_down`

The transition law is first-order and time homogeneous:

\[
A_{ij}=\Pr(R_{m+1}=j\mid R_m=i),
\qquad
\Pr(R_{m+1}\mid R_{m-3:m})=\Pr(R_{m+1}\mid R_m).
\]

The form is fixed, but its counts are recomputed on an expanding causal
window. A final-sample matrix is never reused at an earlier historical cutoff.

## Eligible transitions

Let \(T_m\) be month \(m\)'s `label_available_at`. A historical pair
\((\ell,\ell+1)\) is usable at cutoff \(d\) only when:

- the reference months are consecutive;
- both regimes exist; and
- \(T_\ell\le d\) and \(T_{\ell+1}\le d\).

Missing labels are never bridged. The causal count is

\[
N_{ij}^{(d)}
=\sum_\ell
\mathbf 1\!\left[
R_\ell=i,\ R_{\ell+1}=j,\
\max(T_\ell,T_{\ell+1})\le d
\right],
\qquad
N_i^{(d)}=\sum_j N_{ij}^{(d)}.
\]

At the latest published cutoff, the 250-row history yields 244 eligible
consecutive transitions. Five adjacent pairs touch the four-month unavailable
block and are excluded.

## Smoothed transition probabilities

Each origin row has an independent symmetric Dirichlet prior with
\(\alpha=0.5\):

\[
\mathbf A_{i,\cdot}
\sim\operatorname{Dirichlet}(0.5,0.5,0.5,0.5).
\]

The production posterior-predictive probability is

\[
\overline A_{ij}^{(d)}
=\frac{N_{ij}^{(d)}+0.5}{N_i^{(d)}+2}.
\]

An empty row is therefore uniform rather than assigning zero probability to
unseen transitions. Published 95% intervals are marginal beta-posterior
intervals for individual cells; they are not simultaneous row intervals.

## Four-month path roll

The inference stage retains
\(S_m=(R_{m-3},R_{m-2},R_{m-1},R_m)\) with path probability
\(q_{m,d}(s)\). At a roll to \(m+1\),

\[
q_{m+1,d}^{-}(r_{m-2:m+1})
=\sum_{r_{m-3}\in\mathcal R}
q_{m,d^-}(r_{m-3:m})
\overline A_{r_m,r_{m+1}}^{(d^-)}.
\]

This marginalizes the oldest state, preserves the three overlapping months,
and appends the new state. The corresponding quadrant marginal is

\[
p_{m+1,r\mid d^-}
=\sum_{i\in\mathcal R}
p_{m,i\mid d^-}\overline A_{ir}^{(d^-)}.
\]

The full probability distribution is propagated; Model 01 never substitutes
the most likely state before the roll. Retaining four months supports delayed
release updates but does not make the transition law fourth-order.

## Latest fit

The published fit uses information through 25 June 2026 and includes
transitions through the May 2026 destination month. The self-transition
posterior-predictive probabilities are 69.34% for
`growth_up_inflation_up`, 46.43% for `growth_down_inflation_up`, 66.44% for
`growth_up_inflation_down`, and 56.45% for
`growth_down_inflation_down`. These probabilities partly reflect the
three-month overlap in the regime definition.

## Artifacts

- Local pair audit:
  `data/processed/m01_deterministic_composite/transition_pairs.csv`
- [Transition model](../../../results/published/m01_deterministic_composite/transition_model.json)
- [Transition matrix](../../../results/published/m01_deterministic_composite/transition_matrix.csv)
- [Latest transition-only prior](../../../results/published/m01_deterministic_composite/latest_transition_prior.json)

The event filter replays this same transition stage with all release
likelihoods suppressed as its sole main inference comparator,
`transition_only`.

## Limitations

Three-month score overlap raises apparent persistence. A first-order,
time-homogeneous law cannot represent duration dependence, structural change,
or market-conditioned transitions. Cell uncertainty is published but not
integrated through every later path update.
