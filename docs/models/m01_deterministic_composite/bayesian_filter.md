# Model 01 event-driven Bayesian filter

The promoted inference stage is `event_driven_bayesian_filter`. It converts the
causal transition prior into a release-by-release posterior over four monthly
regimes. Its frozen contract is
[`configs/models/m01_event_driven_bayesian_filter.yaml`](../../../configs/models/m01_event_driven_bayesian_filter.yaml);
results are in [`bayesian_filter_results.md`](bayesian_filter_results.md).

## State and information clock

Let \(m\) be a reference month, \(d\) an information cutoff,
\(r\in\mathcal R\) a quadrant, and

\[
p_{m,r\mid d}=\Pr(R_m=r\mid\mathcal D_d).
\]

The filter retains the joint path
\(S_m=(R_{m-3},R_{m-2},R_{m-1},R_m)\). If \(s\) is one of its
\(4^4=256\) possible values, write
\(q_{m,d}(s)=\Pr(S_m=s\mid\mathcal D_d)\). The marginals
\(p_{m,r\mid d}\) are obtained by summing path probabilities.

At initialization, the oldest state is uniform and is propagated with the
causal transition matrix. Labels available strictly before the initial cutoff
are then imposed. At each month roll, the oldest coordinate is marginalized
and one new coordinate is appended using the expanding Dirichlet-smoothed
transition law in [`transition_model.md`](transition_model.md).

Each calendar date is processed in this fixed order:

1. month roll, if applicable;
2. one atomic group of usable non-defining releases; and
3. one atomic end-of-day group of deterministic confirmations.

This order is conservative for date-only source data. Same-date
pre-confirmation states are retained for reconstruction but are not scored as
forecasts.

## Promoted evidence likelihood

The point-in-time features are defined in
[`leading_evidence_data.md`](leading_evidence_data.md).

| Block | Baseline vector | Dimension |
|---|---|---:|
| `weekly_claims` | `initial_claims_innovation` | 1 |
| `jolts` | Openings, hires, quits, and layoffs-rate changes | 4 |
| `retail_sales` | Total and ex-motor-vehicle log changes | 2 |
| `housing` | Starts and permits log changes | 2 |
| `durable_goods` | Total orders and core capital-goods orders log changes | 2 |

`continued_claims_innovation` remains in the source data but is not used by
the promoted filter. Monthly blocks require complete vectors.

For a release from block \(b\) published on date \(d\), the causal training set
contains only complete historical vectors \(\mathbf y_e\) satisfying:

\[
\mathcal T_b(d)
=\left\{
e:b(e)=b,\ d_e<d,\ T_{m(e)}<d
\right\},
\]

where \(d_e\) is the event's publication date and \(T_{m(e)}\) is the target
label's availability date. Today's events and labels confirmed today cannot
train today's likelihood. At least 24 complete labeled vectors are required.

Let \(n_{b,r}\) be the number of training events in quadrant \(r\),
\(\overline{\mathbf y}_{b,r}\) their mean, and
\(\overline{\mathbf y}_b\) the block-wide mean. The regime location is
shrunk using five pseudo-observations:

\[
\widetilde{\boldsymbol\mu}_{b,r}
=\frac{
n_{b,r}\overline{\mathbf y}_{b,r}
+5\overline{\mathbf y}_b
}{n_{b,r}+5}.
\]

Residuals around these locations are pooled across regimes. One Ledoit–Wolf
covariance \(\widehat{\mathbf C}_b\) is estimated for the block and shared by
all four quadrants. The promoted likelihood is multivariate Student-\(t\) with
\(\nu=7\). Because its shape matrix is not its covariance,

\[
\widehat{\boldsymbol\Psi}_b
=\frac{\nu-2}{\nu}\widehat{\mathbf C}_b
=\frac57\widehat{\mathbf C}_b,
\]

\[
\mathbf y_e\mid R_{m(e)}=r
\sim t_7\!\left(
\widetilde{\boldsymbol\mu}_{b,r},
\widehat{\boldsymbol\Psi}_b
\right).
\]

Unavailable or numerically invalid fits are skipped with a recorded reason;
the implementation never substitutes a future or full-sample fit.

## Release update and confirmation

For a usable event \(e\), candidate path \(s\) selects the likelihood attached
to its regime for the event's reference month:

\[
L_e(s)
=f_{b(e)}\!\left(
\mathbf y_e\mid
\widetilde{\boldsymbol\mu}_{b(e),r_{m(e)}(s)},
\widehat{\boldsymbol\Psi}_{b(e)},7
\right).
\]

All usable events on date \(d\) are combined and normalized once:

\[
q_{m,d}^{+}(s)
=\frac{
q_{m,d}^{-}(s)\prod_{e:d_e=d}L_e(s)
}{
\sum_{s'}q_{m,d}^{-}(s')\prod_{e:d_e=d}L_e(s')
}.
\]

The calculation is performed in log space. The product is a conditional-
independence approximation, not a claim that release blocks are economically
independent.

When the deterministic label for month \(\ell\) becomes available, paths
inconsistent with it receive zero mass:

\[
C_\ell(s)=\mathbf 1[r_\ell(s)=R_\ell],
\qquad
q_{m,d}^{+}(s)
\propto q_{m,d}^{-}(s)
\prod_{\ell:T_\ell=d}C_\ell(s).
\]

All same-day confirmations are imposed atomically. Missing labels are never
imputed or clamped.

## Saved states and evaluation

The baseline retains `initial`, `pre_month_roll`, `post_month_roll`,
`pre_release_group`, `post_release_group`, `pre_confirmation_group`,
`post_confirmation_group`, `month_end`, and `latest` checkpoints. Each saved
state includes the full 256-path distribution and four path marginals.

Formal evaluation starts in January 2018. A forecast is eligible only when:

- the target is classified;
- the checkpoint occurs strictly before `label_available_at`; and
- the probability vector is finite, nonnegative, and normalized.

The sole headline comparator is `transition_only`, which replays the same
initialization, month rolls, expanding transition fits, confirmations, and
evaluation rows while suppressing all release likelihoods.

For \(H\) eligible forecasts with eventual state \(o_h\), the primary proper
scores are

\[
\operatorname{NLL}
=-\frac1H\sum_{h=1}^{H}\log p_{h,o_h},
\]

\[
\operatorname{Brier}
=\frac1H\sum_{h=1}^{H}\sum_{r\in\mathcal R}
\left(p_{h,r}-\mathbf 1[o_h=r]\right)^2.
\]

The Brier score is the unscaled four-class form with range \([0,2]\). Lower is
better for both metrics. MAP accuracy, balanced accuracy, macro F1, axis
Brier scores, calibration, and entropy are secondary diagnostics.

## Artifacts

Detailed reproducible tables live under
`data/processed/m01_bayesian_filter/`:

- `checkpoint_index.csv`
- `joint_path_checkpoints.csv.gz`
- `marginal_checkpoints.csv`
- `event_update_audit.csv`
- `likelihood_fit_audit.csv`
- `forecast_predictions.csv`
- `evaluation_metrics.csv`
- `calibration_bins.csv`
- `sensitivity_specifications.csv`
- `sensitivity_metrics.csv`

Public outputs are:

- [Latest posterior](../../../results/published/m01_bayesian_filter/latest_posterior.json)
- [Evaluation summary](../../../results/published/m01_bayesian_filter/evaluation_summary.json)
- [Sensitivity summary](../../../results/published/m01_bayesian_filter/sensitivity_metrics.csv)
- [Manifest](../../../data/manifests/m01_event_driven_bayesian_filter.json)

Sensitivity artifacts retain the prespecified one-at-a-time alternatives, but
none replaces the promoted fixed-\(\nu=7\), five-pseudo-observation,
Ledoit–Wolf baseline.

## Limitations

Separate block likelihoods and repeated weekly releases are not fully
conditionally independent and can make the posterior too sharp. One covariance
per block excludes regime-varying volatility, while fixed \(\nu=7\) and mean
shrinkage are modeling choices. Daily timestamps do not resolve intraday
ordering. Transition, likelihood, and label uncertainty are not fully
integrated, and 97 evaluated reference months provide limited evidence across rare
quadrants and macro cycles.
