# Model 02: soft composites and event-driven Gaussian filtering

Model 02 is a modest architectural revision of Model 01. It preserves
economically specified, equal-weight growth and inflation composites but does
not smooth them over three months and does not convert them to hard quadrant
labels. A second stage maps these scores to soft quadrant weights using
disagreement and revision uncertainty. A third stage uses an expanding VAR(1)
to evolve the continuous score center. The inference stage adds point-in-time
non-defining releases, structured linear-Gaussian observation models, and a
rolling four-month joint Gaussian filter. The complete causal replay is
published through 20 July 2026. Allocation and a Model 02 backtest have not yet
been implemented.

## 1. Information clock and first-release rule

Let $m$ denote a monthly reference period, $k$ a component series, and
$v_k(m)$ the first eligible historical vintage in which the observation for
month $m$ appeared. The eligibility gate requires a nonnegative publication
lag no greater than 92 days. At the beginning of a real-time archive, only the
latest reference period exposed by the first snapshot is retained; older rows
in that snapshot are archive backfills rather than contemporaneous releases.

Both the current level and the preceding month's level are read from the same
vintage $v_k(m)$. Writing these levels as $x_{k,m}^{(v_k(m))}$ and
$x_{k,m-1}^{(v_k(m))}$ prevents a later revision of month $m-1$ from
entering month $m$'s first-release transformation.

The authenticated FRED API is the primary acquisition provider for the
published Model 02 snapshot. The keyless ALFRED provider and Model 01 data
remain intact in separate cache and output namespaces.

## 2. Components and transformations

The definition deliberately retains Model 01's four components per axis so
the architectural comparison is interpretable. The only component-transform
change is payrolls: Model 02 uses percentage growth rather than an absolute
change in thousands of employees.

| Axis | Component | FRED series | First-release transformation |
|---|---|---|---|
| Growth | Nonfarm payroll employment | `PAYEMS` | $100\log(x_m/x_{m-1})$ |
| Growth | Industrial production | `INDPRO` | $100\log(x_m/x_{m-1})$ |
| Growth | Real personal consumption | `PCEC96` | $100\log(x_m/x_{m-1})$ |
| Growth | Unemployment rate | `UNRATE` | $-(x_m-x_{m-1})$ |
| Inflation | Core CPI | `CPILFESL` | $100\log(x_m/x_{m-1})$ |
| Inflation | Core PCE price index | `PCEPILFE` | $100\log(x_m/x_{m-1})$ |
| Inflation | Core finished-goods PPI | `PPILFE`, then `WPSFD4131` | $100\log(x_m/x_{m-1})$ |
| Inflation | Average hourly earnings | `AHETPI` | $100\log(x_m/x_{m-1})$ |

For every log-transformed component,

$$
u_{k,m}
=100\log\left(
\frac{x_{k,m}^{(v_k(m))}}
{x_{k,m-1}^{(v_k(m))}}
\right).
$$

For unemployment,

$$
u_{k,m}
=-\left(
x_{k,m}^{(v_k(m))}
-x_{k,m-1}^{(v_k(m))}
\right).
$$

The negative sign makes a declining unemployment rate contribute positively to
the growth composite.

## 3. Strictly lagged standardization

Let

$$
\mathcal H_{k,m-1}
=\{\ell<m:u_{k,\ell}\text{ is available}\}
$$

be the set of earlier available transformed observations. With
$n_{k,m-1}=|\mathcal H_{k,m-1}|$, the causal expanding estimates are

$$
\widehat\mu_{k,m-1}
=\frac{1}{n_{k,m-1}}
\sum_{\ell\in\mathcal H_{k,m-1}}u_{k,\ell},
$$

$$
\widehat\sigma_{k,m-1}^{\,2}
=\frac{1}{n_{k,m-1}-1}
\sum_{\ell\in\mathcal H_{k,m-1}}
\left(u_{k,\ell}-\widehat\mu_{k,m-1}\right)^2.
$$

The standardized component is

$$
z_{k,m}
=\frac{u_{k,m}-\widehat\mu_{k,m-1}}
{\widehat\sigma_{k,m-1}},
\qquad n_{k,m-1}\ge 60.
$$

Month $m$ therefore never contributes to its own mean or standard deviation.
Missing observations remain missing and do not reduce the required history.

## 4. Equal-weight unsmoothed scores

Let $\mathcal G$ and $\mathcal I$ denote the four growth and four inflation
components. The Model 02 scores are

$$
G_m=\frac14\sum_{k\in\mathcal G}z_{k,m},
\qquad
I_m=\frac14\sum_{k\in\mathcal I}z_{k,m}.
$$

All four terms must be available. The implementation uses no dynamic
reweighting and does not fill a missing component.

Unlike Model 01, there is no additional transformation

$$
\frac{G_m+G_{m-1}+G_{m-2}}{3}
$$

or its inflation analogue. The unsmoothed monthly scores are the final outputs
of the score-definition stage. That stage assigns no sign threshold, quadrant
label, or maximum-probability state. Section 8 describes the separate
uncertainty-mapping stage that converts these scores into four soft quadrant
probabilities.

## 5. Availability dates

Each axis has its own causal availability date:

$$
T_m^G
=\max\{d_{k,\ell}: k\in\mathcal G,\ \ell\le m,\
u_{k,\ell}\text{ is available}\},
$$

with $T_m^I$ defined analogously. Here $d_{k,\ell}$ is the publication date
of a prerequisite first-release observation. The joint score date is

$$
T_m=\max(T_m^G,T_m^I)
$$

when both scores exist. This distinguishes the economic reference month from
the date on which the scores became knowable.

## 6. Authenticated data result

The current snapshot was retrieved through the authenticated FRED API with a
knowledge cutoff of 20 July 2026.

| Item | Result |
|---|---:|
| Requested reference range | January 2000-May 2026 |
| Monthly rows | 317 |
| Complete growth/inflation score pairs | 249 |
| First complete score pair | July 2005 |
| Latest complete score pair | May 2026 |
| Latest joint score availability date | 25 June 2026 |
| Latest growth score | 0.027394 |
| Latest inflation score | 0.410476 |

Authenticated first-release `PCEPILFE` begins with the July 2000 reference
month, released on 28 August 2000. Consequently, January-June 2000 remain in
the panel but have no complete inflation composite. A January or February 2000
eight-component score cannot be produced without changing the feature set or
using a non-point-in-time backfill, neither of which this model does.

The 60-prior-observation standardization requirement makes July 2005 the first
complete score month. October and November 2025 remain unavailable because the
federal shutdown left required core-CPI and unemployment transformations
missing. These gaps are explicit in the public score history.

The expanding z-scores are intentionally unbounded in this stage. The April
2020 payroll observation has a z-score of $-103.0223$ and drives that month's
growth composite to $-51.8317$. This is not a calculation error: it is the
consequence of applying the frozen classical standardization rule to an
extraordinary observation. The baseline probability map preserves this
unbounded score specification. A later sensitivity analysis should compare a
robust scale, clipping rule, or heavy-tailed link because this observation can
mechanically saturate its historical quadrant probabilities.

## 7. Score artifacts

Run the score builder with the authenticated provider after placing
`FRED_API_KEY` in the process environment:

```powershell
build-m02-scores --provider fred
```

The key is never accepted as a command-line argument or written to an
artifact. The existing keyless fallback remains available as
`build-m02-scores --provider alfred`.

Configuration:

- `configs/models/m02_soft_composite.yaml`

Model code:

- `src/regime_allocation/models/m02_soft_composite/scores.py`
- `src/regime_allocation/cli/build_m02_scores.py`

Local processed audits, excluded from Git:

- `data/processed/m02_soft_composite/first_release_components_long.csv`
- `data/processed/m02_soft_composite/composite_scores.csv`

Tracked provenance and public outputs:

- `data/manifests/m02_soft_composite.json`
- `results/published/m02_soft_composite/score_history.csv`
- `results/published/m02_soft_composite/latest_scores.json`

The manifest contains provider identifiers, public source URLs, non-secret
query metadata, input and output hashes, extraction diagnostics, and coverage
counts. It contains no API credential or authenticated request URL.

## 8. Gaussian quadrant mapping

Let the observed score vector be

$$
\boldsymbol s_m=
\begin{bmatrix}
G_m\\
I_m
\end{bmatrix}.
$$

Model 02 introduces the reporting variable

$$
\boldsymbol U_m
=\boldsymbol s_m+\boldsymbol\epsilon_m^{\mathrm{map}},
\qquad
\boldsymbol\epsilon_m^{\mathrm{map}}
\sim\mathcal N(\boldsymbol 0,\boldsymbol\Omega_{\mathrm{map},m}),
$$

where

$$
\boldsymbol\Omega_{\mathrm{map},m}
=\boldsymbol\Omega_{\mathrm{disagreement},m}
+\boldsymbol\Omega_{\mathrm{revision},m}.
$$

The resulting regime weights are probabilities that
$\boldsymbol U_m$ lies in each growth/inflation quadrant. They are
not hard labels, transition forecasts, or event-updated Bayesian posteriors.

### 8.1 Component-disagreement uncertainty

For one axis $a\in\{G,I\}$, let $z_{a,j,m}$ be its four standardized
components and

$$
S_{a,m}=\frac14\sum_{j=1}^{4}z_{a,j,m}.
$$

Deleting component $j$ and renormalizing the other three gives

$$
S_{a,m}^{(-j)}
=\frac13\sum_{\ell\ne j}z_{a,\ell,m}.
$$

The monthly delete-one jackknife variance is

$$
v_{a,m}^{\mathrm{JK}}
=\frac34\sum_{j=1}^{4}
\left(S_{a,m}^{(-j)}-S_{a,m}\right)^2
=\frac{\operatorname{sampleVar}(z_{a,1:4,m})}{4}.
$$

At score cutoff $T_m$, the estimator pools only complete score months that
were knowable strictly before $T_m$:

$$
\widehat v_{a,m}^{\mathrm{dis}}
=\frac{1}{|\mathcal H_m|}
\sum_{\ell\in\mathcal H_m}v_{a,\ell}^{\mathrm{JK}}.
$$

The disagreement covariance is deliberately diagonal:

$$
\boldsymbol\Omega_{\mathrm{disagreement},m}
=
\begin{bmatrix}
\widehat v_{G,m}^{\mathrm{dis}}&0\\
0&\widehat v_{I,m}^{\mathrm{dis}}
\end{bmatrix}.
$$

The implementation requires at least 24 earlier complete months. It averages
the correctly scaled monthly jackknife variances; it does not treat the four
leave-one-out replicates as independent observations.

### 8.2 Fixed-horizon revision uncertainty

Let $r_{k,m}$ be component $k$'s first-release date. For horizon
$h\in\{3,12\}$ months, the exact as-of vintage is

$$
v_{k,m}^{(h)}=r_{k,m}+h\text{ calendar months}.
$$

The authenticated FRED API is queried at that exact date. Both month $m$ and
month $m-1$ are read from the same vintage before applying the original
component transformation. This produces $u_{k,m}^{(h)}$. Holding the original
first-release standardization scale fixed, the component revision error is

$$
e_{k,m}^{(h)}
=\frac{u_{k,m}^{(h)}-u_{k,m}^{(0)}}
{\widehat\sigma_{k,m-1}}.
$$

The corresponding score-scale error vector is

$$
\boldsymbol e_m^{(h)}
=
\begin{bmatrix}
\frac14\sum_{k\in\mathcal G}e_{k,m}^{(h)}\\[3pt]
\frac14\sum_{k\in\mathcal I}e_{k,m}^{(h)}
\end{bmatrix}.
$$

All eight components are required; missing revisions never cause dynamic
reweighting. At cutoff $T_m$, only complete errors whose horizon dates are
strictly earlier than $T_m$ enter the centered sample covariance:

$$
\boldsymbol\Omega_{\mathrm{revision},m}^{(h)}
=\frac{1}{N_{m,h}-1}
\sum_{\ell\in\mathcal R_{m,h}}
\left(\boldsymbol e_\ell^{(h)}-\overline{\boldsymbol e}_{m,h}\right)
\left(\boldsymbol e_\ell^{(h)}-\overline{\boldsymbol e}_{m,h}\right)^\top.
$$

At least 24 mature bivariate errors are required. The production map uses the
12-month covariance; the 3-month covariance is published as a sensitivity.
They are not added or stacked because the two revision errors are nested and
strongly dependent.

### 8.3 Closed-form quadrant weights

Write

$$
\boldsymbol\Omega_{\mathrm{map},m}
=
\begin{bmatrix}
\sigma_G^2&\rho\sigma_G\sigma_I\\
\rho\sigma_G\sigma_I&\sigma_I^2
\end{bmatrix},
$$

and define

$$
q_G=\Phi\!\left(-\frac{G_m}{\sigma_G}\right),
\qquad
q_I=\Phi\!\left(-\frac{I_m}{\sigma_I}\right),
$$

$$
q_{GI}=\Phi_2\!\left(
-\frac{G_m}{\sigma_G},
-\frac{I_m}{\sigma_I};\rho
\right).
$$

The four raw Gaussian masses are

$$
\begin{aligned}
w_{G\uparrow,I\uparrow}&=1-q_G-q_I+q_{GI},\\
w_{G\downarrow,I\uparrow}&=q_G-q_{GI},\\
w_{G\uparrow,I\downarrow}&=q_I-q_{GI},\\
w_{G\downarrow,I\downarrow}&=q_{GI}.
\end{aligned}
$$

Numerical probabilities are reported as

$$
p_{r,m}=\frac{w_{r,m}}{\sum_{r'}w_{r',m}}.
$$

The normalization only removes numerical integration error; theoretically the
four weights already sum to one. The implementation uses a bivariate-normal
CDF, not Monte Carlo simulation.

## 9. Probability-map results

With a 24-month causal warm-up, the 12-month baseline produces 214 monthly
probability vectors from June 2008 through May 2026. The 3-month sensitivity
produces 223 vectors from September 2007 through May 2026. Exact revision-error
coverage is:

| Horizon | Complete errors | First month | Latest mature month |
|---|---:|---:|---:|
| 3 months | 246 | July 2005 | February 2026 |
| 12 months | 239 | July 2005 | May 2025 |

For May 2026, the production covariance decomposition is

$$
\boldsymbol\Omega_{\mathrm{disagreement}}
=
\begin{bmatrix}
1.690407&0\\
0&0.437102
\end{bmatrix},
$$

$$
\boldsymbol\Omega_{\mathrm{revision}}
=
\begin{bmatrix}
0.052083&-0.001852\\
-0.001852&0.045987
\end{bmatrix},
$$

and therefore

$$
\boldsymbol\Omega_{\mathrm{map}}
=
\begin{bmatrix}
1.742490&-0.001852\\
-0.001852&0.483089
\end{bmatrix}.
$$

The May 2026 baseline weights are:

| Quadrant | Probability |
|---|---:|
| Growth up / inflation up | 36.70% |
| Growth down / inflation up | 35.56% |
| Growth up / inflation down | 14.13% |
| Growth down / inflation down | 13.61% |

The implied marginal probabilities are 50.83% for growth up and 72.26% for
inflation up. The 3-month sensitivity is close: 36.93%, 35.74%, 13.90%, and
13.43% in the same order.

Disagreement contributes about 97.0% of the latest growth mapping variance and
90.5% of the inflation mapping variance. This is partly a consequence of the
arithmetic historical pool retaining the COVID component dispersion. The
zero-mean perturbation also discards the measured revision-error means, which
are published as diagnostics.

Adding the two covariance components assumes disagreement and revision errors
are independent. Because both originate from the same releases, this is an
explicit approximation rather than an identified variance decomposition.

## 10. Probability-map artifacts

Run:

```powershell
build-m02-probabilities --provider fred
```

Configuration and model code:

- `configs/models/m02_probability_map.yaml`
- `src/regime_allocation/models/m02_soft_composite/revisions.py`
- `src/regime_allocation/models/m02_soft_composite/probability_map.py`
- `src/regime_allocation/cli/build_m02_probability_map.py`

Tracked outputs:

- `data/manifests/m02_probability_map.json`
- `results/published/m02_soft_composite/probability_map/quadrant_probabilities.csv`
- `results/published/m02_soft_composite/probability_map/revision_horizon_sensitivity.csv`
- `results/published/m02_soft_composite/probability_map/latest_quadrant_probabilities.json`
- `results/published/m02_soft_composite/probability_map/uncertainty_summary.json`

Component-level revisions, axis revision errors, monthly jackknife diagnostics,
and the full two-specification mapping history are retained under
`data/processed/m02_soft_composite/probability_map/` and excluded from Git.
The probability-map manifest records the Python, NumPy, pandas, SciPy,
scikit-learn, and PyYAML versions used for the published numerical snapshot.

## 11. VAR(1) next-month transition

### 11.1 State interpretation

Define the continuous state evolved by the transition model as the released
composite-score center

$$
\boldsymbol Z_m
=
\boldsymbol s_m
=
\begin{bmatrix}G_m\\I_m\end{bmatrix}.
$$

Let $\mathcal D_{T_m}$ denote all information available by the source score's
availability date $T_m$. Once the composite is released, this model observes
the score center exactly:

$$
p(\boldsymbol Z_m\mid\mathcal D_{T_m})
=
\delta_{\boldsymbol s_m}(\boldsymbol Z_m),
$$

where $\delta_{\boldsymbol s_m}$ denotes a point mass at
$\boldsymbol s_m$. The Gaussian perturbation from Section 8 belongs to a
separate reporting variable,

$$
\boldsymbol U_m
=
\boldsymbol Z_m+\boldsymbol\epsilon^{\mathrm{map}}_m,
\qquad
\boldsymbol\epsilon^{\mathrm{map}}_m
\sim
\mathcal N(\boldsymbol 0,\boldsymbol\Omega_{\mathrm{map},m}).
$$

Quadrant probabilities describe $\boldsymbol U_m$, not uncertainty about the
already released value of $\boldsymbol Z_m$. This separation lets the model
retain soft quadrant weights without pretending that disagreement and revision
risk are measurement error in the exact VAR state.

The coefficient and innovation estimates are obtained by treating the
deterministic centers $\boldsymbol s_m$ as the observed VAR history. The
Gaussian mapping covariance enters only when translating a latent score
forecast into reporting quadrants.

The transition law is

$$
\boldsymbol Z_{m+1}
=\boldsymbol c
+\boldsymbol A\boldsymbol Z_m
+\boldsymbol\eta_{m+1},
\qquad
\boldsymbol\eta_{m+1}
\sim\mathcal N(\boldsymbol 0,\boldsymbol Q).
$$

The innovation $\boldsymbol\eta_{m+1}$ is assumed conditionally independent of
$\boldsymbol Z_m$. Here $\boldsymbol A$ is a VAR dynamics coefficient matrix;
unlike Model 01's Markov matrix, its entries need not be nonnegative and its
rows do not sum to one.

The baseline assumes first-order, time-homogeneous linear dynamics. It includes
an intercept but no time trend, ridge penalty, stationarity projection, or
parameter-uncertainty adjustment.

### 11.2 Causal expanding estimation

For a historical transition from reference month $\ell-1$ to $\ell$, define

$$
\boldsymbol y_\ell
=
\begin{bmatrix}
G_\ell\\
I_\ell
\end{bmatrix},
\qquad
\boldsymbol d_\ell
=
\begin{bmatrix}
1\\
G_{\ell-1}\\
I_{\ell-1}
\end{bmatrix}.
$$

At source cutoff $T_m$, a pair is eligible only when:

1. $\ell-1$ and $\ell$ are exact consecutive calendar months;
2. both first-release score pairs are complete;
3. the response month satisfies $\ell\le m$;
4. both scores were available by $T_m$.

The pair ending at $m$ may enter the fit because $\boldsymbol s_m$ is known
when the next-month prior is generated. Month $m+1$ never enters. Missing months
are not removed and compressed into an artificial one-month transition.

With $N_m$ eligible pairs, stack the row vectors $\boldsymbol d_\ell^\top$ in
$\boldsymbol D_m$ and the response rows $\boldsymbol y_\ell^\top$ in
$\boldsymbol Y_m$. Multivariate OLS gives

$$
\widehat{\boldsymbol\Theta}_m
=
(\boldsymbol D_m^\top\boldsymbol D_m)^{-1}
\boldsymbol D_m^\top\boldsymbol Y_m.
$$

The first row of $\widehat{\boldsymbol\Theta}_m$ is
$\widehat{\boldsymbol c}_m^\top$ and the remaining two rows are
$\widehat{\boldsymbol A}_m^\top$. If

$$
\widehat{\boldsymbol E}_m
=\boldsymbol Y_m
-\boldsymbol D_m\widehat{\boldsymbol\Theta}_m,
$$

the innovation covariance is estimated as

$$
\widehat{\boldsymbol Q}_m
=
\frac{
\widehat{\boldsymbol E}_m^\top\widehat{\boldsymbol E}_m
}{N_m-3}.
$$

The denominator subtracts the three regressors: the intercept and two lagged
scores. At least 60 eligible transitions are required before a prior is
published. The design rank, condition number, residual degrees of freedom,
equation-level $R^2$, residual means, and spectral radius are retained for every
fit.

### 11.3 Gaussian propagation and quadrant prior

Conditional on the plug-in estimates and the exact source score,

$$
\boldsymbol\mu_{m+1}^{-}
=\widehat{\boldsymbol c}_m
+\widehat{\boldsymbol A}_m\boldsymbol s_m,
$$

$$
\boldsymbol P_{m+1}^{Z,-}
=\widehat{\boldsymbol Q}_m.
$$

Thus

$$
\boldsymbol Z_{m+1}\mid\mathcal D_{T_m}
\sim
\mathcal N\!\left(
\boldsymbol\mu_{m+1}^{-},
\boldsymbol P_{m+1}^{Z,-}
\right).
$$

The target month's realized mapping covariance cannot be known at $T_m$.
For this standalone one-step artifact, let $k(m)$ identify the latest baseline
mapping row whose reference month is no later than $m$ and whose availability
date is no later than $T_m$. The stage uses

$$
\boldsymbol\Omega_{m+1}^{\mathrm{proxy}}
=
\boldsymbol\Omega_{\mathrm{map},k(m)}
$$

as an explicit proxy for the target reporting perturbation. Assuming that
perturbation is independent of the next VAR innovation gives

$$
\boldsymbol U_{m+1}\mid\mathcal D_{T_m}
\sim
\mathcal N\!\left(
\boldsymbol\mu_{m+1}^{-},
\boldsymbol P_{m+1}^{U,-}
\right),
$$

$$
\boldsymbol P_{m+1}^{U,-}
=
\widehat{\boldsymbol Q}_m
+\boldsymbol\Omega_{m+1}^{\mathrm{proxy}}.
$$

The four next-month priors are obtained by applying the bivariate-normal
quadrant integrals from Section 8.3 to this reporting distribution.

The proxy is a reporting approximation, not a claim that the target month's
future revision and disagreement covariance is already observed. In
particular, the stage never uses
$\widehat{\boldsymbol A}_m\boldsymbol\Omega_{\mathrm{map},m}
\widehat{\boldsymbol A}_m^\top$: that expression would incorrectly treat the
source month's reporting perturbation as uncertainty in the exact state
$\boldsymbol Z_m$.

### 11.4 Current production result

The 60-pair warm-up produces 189 causal priors. The first uses July 2010 as its
source and August 2010 as its target. The latest uses May 2026 as its source and
June 2026 as its target.

The latest fit uses 247 eligible adjacent-month score pairs. Every included
pair spans consecutive months, but the complete history is not one uninterrupted
sequence because the October-November 2025 score gaps remove three transitions.
The fit may use deterministic-score pairs beginning in 2005; publication of a
prior additionally requires the source month's 12-month Gaussian map, whose
history starts later.

$$
\widehat{\boldsymbol c}
=
\begin{bmatrix}
-0.274160\\
0.089846
\end{bmatrix},
$$

$$
\widehat{\boldsymbol A}
=
\begin{bmatrix}
0.178493&0.596859\\
0.055320&0.579750
\end{bmatrix},
$$

$$
\widehat{\boldsymbol Q}
=
\begin{bmatrix}
11.500901&-0.984899\\
-0.984899&0.411123
\end{bmatrix}.
$$

The fitted dynamics matrix has spectral radius $0.649807$. The latest causal
mapping proxy is the May 2026 baseline covariance,

$$
\boldsymbol\Omega_{2026\text{-}06}^{\mathrm{proxy}}
=
\begin{bmatrix}
1.742490&-0.001852\\
-0.001852&0.483089
\end{bmatrix}.
$$

For June 2026, the latent and reporting priors share the mean

$$
\boldsymbol\mu_{2026\text{-}06}^{-}
=
\begin{bmatrix}
-0.024274\\
0.329335
\end{bmatrix},
$$

$$
\boldsymbol P_{2026\text{-}06}^{Z,-}
=
\begin{bmatrix}
11.500901&-0.984899\\
-0.984899&0.411123
\end{bmatrix},
$$

$$
\boldsymbol P_{2026\text{-}06}^{U,-}
=
\begin{bmatrix}
13.243391&-0.986750\\
-0.986750&0.894211
\end{bmatrix}.
$$

The resulting quadrant prior is:

| June 2026 quadrant | Prior probability |
|---|---:|
| Growth up / inflation up | 27.29% |
| Growth down / inflation up | 36.33% |
| Growth up / inflation down | 22.44% |
| Growth down / inflation down | 13.94% |

The source score became complete on 25 June 2026. The prior is therefore a
late-June update for the June reference month, not a forecast available on
1 June. Across the 189 historical priors, none was available by the first day
of its target month, 157 were available by target month-end, and 32 arrived
after target month-end. Later event-level evaluation must preserve these
timestamps. A genuinely beginning-of-month prior will require recursive
multi-step propagation from the latest state available at that earlier cutoff.
Each published row is therefore labeled `as_of_exact_source_score_release` and
retains its own `prior_available_at` timestamp.

### 11.5 COVID diagnostic and modeling limitations

The unbounded April 2020 growth score produces the only expanding fit with a
spectral radius above one: $3.0988$. Its one-step growth prior mean is
$-160.77$. After the following observations enter, the fitted dynamics become
stable again, but the shock raises the latest growth innovation variance to
$11.50$.

This is not look-ahead leakage or a serialization error. It is the direct
consequence of combining the frozen unbounded score definition with ordinary
least squares. The baseline deliberately does not winsorize the shock, impose a
Student-$t$ innovation, apply robust regression, or rescale eigenvalues. Those
alternatives should be implemented as named sensitivities.

Other limitations are:

- coefficients and $\boldsymbol Q$ are plug-in estimates; parameter uncertainty
  is not integrated into the prior;
- the target reporting covariance is approximated by the latest causally
  available mapping covariance; its realized future value is unknown at the
  forecast cutoff;
- a time-homogeneous linear VAR cannot represent asymmetric or
  duration-dependent dynamics; and
- the standalone transition artifact produces only a one-step monthly
  marginal. Section 14 describes the separate filter stage that uses the same
  VAR law to construct and roll a coherent joint Gaussian over four months.

The filter never constructs joint path probabilities by multiplying monthly
marginal quadrant weights. It preserves the VAR-induced cross-month covariance
of the score centers and approximates the 256 quadrant paths jointly, subject
to the mapping-error approximation documented in Section 16.

### 11.6 Transition artifacts

Run:

```powershell
build-m02-transition
```

Configuration and code:

- `configs/models/m02_var1_transition.yaml`
- `src/regime_allocation/models/m02_soft_composite/var_transition.py`
- `src/regime_allocation/cli/build_m02_transition.py`

Tracked outputs:

- `data/manifests/m02_var1_transition.json`
- `results/published/m02_soft_composite/transition/next_month_priors.csv`
- `results/published/m02_soft_composite/transition/latest_next_month_prior.json`
- `results/published/m02_soft_composite/transition/transition_summary.json`

The pair-level eligibility audit is retained under
`data/processed/m02_soft_composite/var1_transition/` and excluded from Git.
The builder verifies both upstream file hashes and cross-stage lineage, and it
requires the score centers and availability dates in the probability map to
match the score-stage artifact exactly before fitting or propagation.

## 12. Non-defining release evidence

### 12.1 Information-set rule

Let $e$ denote a release event, $d_e$ its publication date, and $q(e)$ its
economic reference month. Those dates are different objects. Every event is
assigned to the month it describes; the filter never changes $q(e)$ merely
because the release arrives during a later calendar month.

For retrospective monthly series, a first appearance is eligible only when it
occurs no earlier than the reference-month end and no more than 92 days later.
At an archive's first snapshot, only the latest exposed reference period is
retained; older observations are classified as bootstrap history. Current and
prior levels used in a change are read from the same vintage.

The one exception is `EXPINF1YR`, an estimate labeled for month $m$ and
published during month $m$. It is eligible only when its first appearance lies
between the first and final calendar days of $m$. Its `release_lag_days` is
still measured relative to month-end for schema compatibility, so a valid row
normally has a negative lag. That negative value records contemporaneous
timing; it does not move the observation backward in time.

Except for a documented binary control, transformed evidence is standardized
using an expanding mean and sample standard deviation estimated from feature
releases strictly earlier than $d_e$. The default warm-up is 60 observations;
short-history inflation expectations and the mortgage-rate control use 24.
Rows in warm-up remain in the audit table but are unavailable to the emission
model.

The current evidence artifact has 3,731 feature rows, of which 3,001 pass all
feature-construction warm-ups, from 20 July 2000 through 16 July 2026. Existing
Model 01 point-in-time rows are reused only after their manifest and event-file
hashes are verified. The newly acquired series are `MORTGAGE30US`, `ANXAVS`,
`EXPINF1YR`, and `WPSID61`; these use the authenticated FRED API. The API key is
read only from the process environment and is never written to a cache key,
manifest, log, or result.

### 12.2 Evidence blocks and transformations

| Observation model | FRED series | Response or control | Transformation |
|---|---|---|---|
| Weekly labor stress | `ICSA` | Initial-claims innovation | Causal innovation from an expanding AR(1) for log claims, then causal standardization |
| Monthly labor demand | `JTSJOR` | Job-openings-rate change | Same-vintage monthly difference, then causal standardization |
| Monthly labor demand | `JTSHIR` | Hires-rate change | Same-vintage monthly difference, then causal standardization |
| Monthly labor demand | `JTSQUR` | Quits-rate change | Same-vintage monthly difference, then causal standardization |
| Monthly labor demand | `JTSLDR` | Layoffs/discharges-rate change | Same-vintage monthly difference, then causal standardization |
| Consumer demand | `RSFSXMV` | Retail sales excluding motor vehicles | Same-vintage monthly log change, then causal standardization |
| Consumer demand | `RSAFS` minus `RSFSXMV` | Motor-vehicle sales | Same-vintage level subtraction followed by a monthly log change and causal standardization |
| Housing activity | `HOUST` | Housing starts | Same-vintage monthly log change, then causal standardization |
| Housing activity | `PERMIT` | Building permits | Same-vintage monthly log change, then causal standardization |
| Housing activity | `MORTGAGE30US` | Observed rate control | Difference between current- and prior-reference-month averages of weekly first releases available by the housing event, then causal standardization |
| Housing activity | `MORTGAGE30US` | Methodology control | Indicator for use of the methodology introduced on 17 November 2022; not standardized |
| Business investment | `NEWORDER` | Core capital-goods orders | Same-vintage monthly log change, then causal standardization |
| Business investment | `ANXAVS` | Core capital-goods shipments | Same-vintage monthly log change, then causal standardization |
| Inflation expectations | `EXPINF1YR` | One-year inflation-expectations change | Same-vintage monthly difference, then causal standardization |
| Inflation input costs | `WPSID61` | Intermediate-materials price change | Same-vintage monthly log change, then causal standardization |

The two consumer-demand coordinates are deliberately disjoint: one excludes
motor vehicles, while the other reconstructs motor-vehicle sales by subtracting
same-vintage levels before taking the log change. The housing rate control uses
only weekly values published by the housing event date. If either the current or
prior reference month lacks an eligible weekly observation, the control is
unavailable rather than imputed.

For weekly claims, let $c_\tau=\log x_\tau$, where $x_\tau$ is the initial
claims level for reference week $\tau$. Immediately before publication date
$d$, the model estimates

$$
c_\tau=\alpha_d+\phi_d c_{\tau-1}+u_\tau
$$

from weekly observations released strictly before $d$. The event feature is
the fitted innovation

$$
\widehat u_\tau
=c_\tau-\widehat\alpha_d-\widehat\phi_d c_{\tau-1},
$$

standardized using earlier innovations only. Observations sharing one release
date use a common frozen pre-date fit and cannot train one another.

### 12.3 Evidence artifacts

Run:

```powershell
build-m02-evidence
```

Configuration and code:

- `configs/models/m02_release_evidence.yaml`
- `src/regime_allocation/features/m02_release_evidence.py`
- `src/regime_allocation/cli/build_m02_evidence.py`

Tracked provenance:

- `data/manifests/m02_release_evidence.json`

The long event table, available-feature table, and newly acquired
first-release audit are retained under
`data/processed/m02_soft_composite/release_evidence/` and excluded from Git.
No posterior update is performed by this data stage.

## 13. Linear-Gaussian observation models

### 13.1 Observation equation

For observation model $b$, let $\boldsymbol y_e\in\mathbb R^{p_b}$ contain
the response features released by event $e$, and let
$\boldsymbol v_e\in\mathbb R^{c_b}$ contain any observed controls. Conditional
on the score center for the event's actual reference month,

$$
\boldsymbol y_e
=\boldsymbol a_b
+\boldsymbol H_b\boldsymbol Z_{q(e)}
+\boldsymbol C_b\boldsymbol v_e
+\boldsymbol\epsilon_e,
\qquad
\boldsymbol\epsilon_e
\sim\mathcal N(\boldsymbol 0,\boldsymbol R_b).
$$

Here $\boldsymbol a_b$ is an intercept vector, $\boldsymbol H_b$ contains the
growth and inflation loadings, $\boldsymbol C_b$ contains control coefficients,
and $\boldsymbol R_b$ is one residual covariance matrix shared by all historical
events fitted under model $b$. A multi-response release is fitted jointly at
the covariance stage, so contemporaneous dependence within that release is not
discarded. If only a subset of its responses is observed, the update uses the
corresponding rows of $\boldsymbol H_b$ and principal submatrix of
$\boldsymbol R_b$.

The two inflation-pressure responses have separate observation models because
they arrive asynchronously. They share an economic block label for reporting,
but the filter does not manufacture a simultaneous vector.

### 13.2 Causal supervised training set

The observation equation is supervised by score centers that become known
after their component releases. Define $T_{q(e)}$ as the complete score's
availability date and $C_e$ as the latest availability date among required
controls, with $C_e=d_e$ when the model has no controls. The row becomes
eligible for future emission fits at

$$
D_e^{\mathrm{train}}
=\max(d_e,T_{q(e)},C_e).
$$

At a scored release date $d$, the training set is

$$
\mathcal T_b(d)
=\left\{
e:b(e)=b,\ D_e^{\mathrm{train}}<d,
\ \boldsymbol y_e,\boldsymbol Z_{q(e)},\boldsymbol v_e
\text{ are complete}
\right\}.
$$

The strict inequality prevents today's release, a score completed today, or a
control first known today from training today's model. Using an eventual score
as a historical response is therefore not look-ahead leakage: that row enters
only after the score was actually knowable.

### 13.3 Structured ridge estimates

For response coordinate $r$, the conditional-mean coefficients solve

$$
\min_{a_{b,r},\boldsymbol h_{b,r},\boldsymbol c_{b,r}}
\sum_{e\in\mathcal T_b(d)}
\left(
y_{e,r}-a_{b,r}
-\boldsymbol h_{b,r}^{\top}\boldsymbol Z_{q(e)}
-\boldsymbol c_{b,r}^{\top}\boldsymbol v_e
\right)^2
+\lambda_b
\sum_{j\in\{G,I\}}w_{b,r,j}h_{b,r,j}^{2}.
$$

Intercepts and observed controls are unpenalized. A configured exact-zero mask
can remove a loading rather than merely shrink it. A control that is constant
in a causal training fold is temporarily dropped because it is collinear with
the intercept; it becomes active once both values have appeared. This is
particularly important for the post-November-2022 housing methodology
indicator.

The base penalty is selected from

$$
\lambda_b\in\{0.001,0.01,0.1,1,10,100\}
$$

by minimum mean rolling-origin predictive Gaussian negative log likelihood.
Every row with the same training-availability timestamp is held out atomically.
If losses are equal to numerical tolerance, the larger penalty is selected.

The structural penalty multipliers are:

| Observation model | Growth multiplier | Inflation multiplier | Exact restriction |
|---|---:|---:|---|
| Weekly labor stress | 1 | 0 | Inflation loading fixed to zero |
| Openings, hires, layoffs/discharges | 1 | 10 | None |
| Quits | 1 | 1 | None |
| Consumer demand | 1 | 1 | None |
| Housing activity | 1 | 1 | None |
| Business investment | 1 | 10 | None |
| Inflation expectations | 10 | 1 | None |
| Inflation input costs | 10 | 1 | None |

Thus the zero shown for weekly labor stress is not an unpenalized inflation
coefficient; the exact-zero mask removes that coefficient altogether.

The final residuals are

$$
\widehat{\boldsymbol\epsilon}_e
=\boldsymbol y_e
-\widehat{\boldsymbol a}_b
-\widehat{\boldsymbol H}_b\boldsymbol Z_{q(e)}
-\widehat{\boldsymbol C}_b\boldsymbol v_e.
$$

One Ledoit–Wolf covariance estimate is fitted to these residuals for each
observation model. Its eigenvalues are then floored at $10^{-8}$ to guarantee a
numerically positive-definite $\widehat{\boldsymbol R}_b$. This is covariance
shrinkage, distinct from the structured ridge shrinkage applied to the mean
loadings.

Weekly labor stress requires at least 104 observations and 24 distinct target
months before fitting. The other minimum training counts are 36 for labor
demand, housing, business investment, and input costs; 60 for consumer demand;
and 24 for inflation expectations. Each model also has a smaller frozen
rolling-validation warm-up and a minimum number of held-out observations, as
recorded in configuration.

## 14. Rolling four-month joint Gaussian

### 14.1 State and initialization

The filter retains four consecutive monthly score centers as one
eight-dimensional vector:

$$
\boldsymbol Z_{m-3:m}
=\begin{bmatrix}
\boldsymbol Z_{m-3}\\
\boldsymbol Z_{m-2}\\
\boldsymbol Z_{m-1}\\
\boldsymbol Z_m
\end{bmatrix},
$$

with posterior

$$
\boldsymbol Z_{m-3:m}\mid\mathcal D_d
\sim\mathcal N(\boldsymbol\mu_d,\boldsymbol P_d).
$$

The replay starts at the earliest date for which a 60-pair causal VAR fit and a
four-month consecutive path can be formed. Its oldest score is initialized
exactly, and the same causal VAR fit supplies the three initialization edges.
Any later path score already public before the start date is then conditioned
on exactly. This deterministic initialization is retained in the checkpoint
audit.

### 14.2 Month roll

At the first day of a new month, the VAR fit uses only score pairs whose
`pair_available_at` timestamp is strictly earlier than that day. The oldest
month is dropped, the three overlapping monthly coordinates are copied, and a
new month is appended by

$$
\boldsymbol Z_{m+1}
=\widehat{\boldsymbol c}_d
+\widehat{\boldsymbol A}_d\boldsymbol Z_m
+\boldsymbol\eta_{m+1},
\qquad
\boldsymbol\eta_{m+1}\sim
\mathcal N(\boldsymbol 0,\widehat{\boldsymbol Q}_d).
$$

If $\boldsymbol Z_m$ is not yet exact, its uncertainty and cross-covariances
are propagated. For any retained coordinate vector $\boldsymbol W$,

$$
\operatorname{Cov}(\boldsymbol Z_{m+1},\boldsymbol W\mid\mathcal D_d)
=\widehat{\boldsymbol A}_d
\operatorname{Cov}(\boldsymbol Z_m,\boldsymbol W\mid\mathcal D_d),
$$

and

$$
\operatorname{Var}(\boldsymbol Z_{m+1}\mid\mathcal D_d)
=\widehat{\boldsymbol A}_d
\operatorname{Var}(\boldsymbol Z_m\mid\mathcal D_d)
\widehat{\boldsymbol A}_d^\top
+\widehat{\boldsymbol Q}_d.
$$

Only $\widehat{\boldsymbol Q}_d$ is new process uncertainty, and it is added
only to the new month's $2\times2$ block. Mapping covariance is absent from the
roll.

### 14.3 Release update

Let $\boldsymbol E_q$ be the $2\times8$ selector that extracts
$\boldsymbol Z_{q(e)}$ from $\boldsymbol Z_{m-3:m}$. After subtracting the
fitted intercept and observed controls, define

$$
\boldsymbol y_e^*
=\boldsymbol y_e
-\widehat{\boldsymbol a}_b
-\widehat{\boldsymbol C}_b\boldsymbol v_e,
\qquad
\boldsymbol H_e^{\mathrm{path}}
=\widehat{\boldsymbol H}_b\boldsymbol E_q.
$$

The pre-update innovation and its covariance are

$$
\boldsymbol\nu_e
=\boldsymbol y_e^*
-\boldsymbol H_e^{\mathrm{path}}\boldsymbol\mu_d^{-},
$$

$$
\boldsymbol S_e
=\boldsymbol H_e^{\mathrm{path}}\boldsymbol P_d^{-}
\left(\boldsymbol H_e^{\mathrm{path}}\right)^\top
+\widehat{\boldsymbol R}_b.
$$

The gain and posterior mean are

$$
\boldsymbol K_e
=\boldsymbol P_d^{-}\left(\boldsymbol H_e^{\mathrm{path}}\right)^\top
\boldsymbol S_e^{-1},
\qquad
\boldsymbol\mu_d^{+}
=\boldsymbol\mu_d^{-}+\boldsymbol K_e\boldsymbol\nu_e.
$$

The covariance uses the numerically stable Joseph form:

$$
\boldsymbol P_d^{+}
=\left(\boldsymbol I-\boldsymbol K_e\boldsymbol H_e^{\mathrm{path}}\right)
\boldsymbol P_d^{-}
\left(\boldsymbol I-\boldsymbol K_e\boldsymbol H_e^{\mathrm{path}}\right)^\top
+\boldsymbol K_e\widehat{\boldsymbol R}_b\boldsymbol K_e^\top.
$$

Because the state is joint across months, evidence about one reference month
can revise all four monthly marginals through their cross-covariances.

Events outside the rolling four-month window are recorded and skipped. An
event whose target score is already exact still produces a predictive
innovation and likelihood diagnostic, but its state update is a no-op. This is
not an error: JOLTS, in particular, is often published after the complete
Model 02 score for the same reference month. The implementation does not move
such evidence to a newer month to make it appear useful.

## 15. Exact score observations and quadrant readout

### 15.1 End-of-day exact conditioning

When the complete score $\boldsymbol s_q$ becomes available, it is treated as
an exact observation of $\boldsymbol Z_q$. With selector
$\boldsymbol E_q$, exact Gaussian conditioning is

$$
\boldsymbol K_q^{\mathrm{exact}}
=\boldsymbol P^{-}\boldsymbol E_q^\top
\left(\boldsymbol E_q\boldsymbol P^{-}\boldsymbol E_q^\top\right)^{-1},
$$

$$
\boldsymbol\mu^{+}
=\boldsymbol\mu^{-}
+\boldsymbol K_q^{\mathrm{exact}}
\left(\boldsymbol s_q-\boldsymbol E_q\boldsymbol\mu^{-}\right),
$$

$$
\boldsymbol P^{+}
=\boldsymbol P^{-}
-\boldsymbol K_q^{\mathrm{exact}}
\boldsymbol E_q\boldsymbol P^{-}.
$$

The conditioned month's covariance rows and columns are set exactly to zero.
Repeating an identical exact score is idempotent; a conflicting repeated value
raises an error.

The within-day order is fixed: month roll, release-group update, then exact
score conditioning. All observation models released on the same day are fitted
at the common pre-release cutoff, then applied in stable model-ID order. A
mapping covariance that first becomes available on the exact-score date can be
used only in the post-exact readout, never in an earlier checkpoint.

### 15.2 Soft monthly probabilities

For any monthly marginal

$$
\boldsymbol Z_q\mid\mathcal D_d
\sim\mathcal N(\boldsymbol\mu_{q,d},\boldsymbol P_{q,d}),
$$

the reporting variable is

$$
\boldsymbol U_q\mid\mathcal D_d
\sim\mathcal N\!\left(
\boldsymbol\mu_{q,d},
\boldsymbol P_{q,d}
+\boldsymbol\Omega_{\mathrm{map},q}^{(d)}
\right),
$$

where $\boldsymbol\Omega_{\mathrm{map},q}^{(d)}$ is the latest causally
eligible mapping estimate at checkpoint $d$. Quadrant probabilities are the
four bivariate-normal masses from Section 8.3. Even after
$\boldsymbol Z_q=\boldsymbol s_q$ is exact, the reporting probabilities can
remain soft because $\boldsymbol\Omega_{\mathrm{map},q}^{(d)}$ describes
classification uncertainty rather than uncertainty about the released number.

### 15.3 Joint quadrant paths

The 256 four-month paths are approximated from the full eight-dimensional
Gaussian rather than by multiplying four monthly marginals. The baseline uses

$$
\boldsymbol P_d^{U}
=\boldsymbol P_d
+\operatorname{blockdiag}\!\left(
\boldsymbol\Omega_{m-3}^{(d)},
\boldsymbol\Omega_{m-2}^{(d)},
\boldsymbol\Omega_{m-1}^{(d)},
\boldsymbol\Omega_m^{(d)}
\right).
$$

A scrambled Sobol sequence with a fixed checkpoint-derived seed and 16,384
draws estimates all path masses reproducibly. The block-diagonal addition
assumes mapping perturbations are independent across months. The score-center
cross-time covariance in $\boldsymbol P_d$ is preserved, but cross-month
dependence in disagreement or revision errors is omitted. This is an explicit
approximation and should be tested in a later sensitivity.

## 16. Checkpoints and forecast evaluation

The replay stores pre- and post-month-roll, start-of-score-availability-day,
pre- and post-release-group, pre- and post-exact-score, month-end, and latest
checkpoints. Every checkpoint
contains the joint score-center moments, monthly marginals, exact-score mask,
causal mapping reference, and, at selected checkpoints, the 256 joint path
probabilities. Two filters are replayed in parallel:

- `evidence_filter` uses VAR rolls, release updates, and exact scores;
- `transition_only` uses the same VAR rolls and exact scores but suppresses
  release evidence.

The source artifacts identify availability dates but do not reliably identify
intraday release times. The primary forecast is therefore frozen at the start
of the completed score's availability day, after a deterministic month roll if
applicable but before every data release sharing that date. This is labeled
`strict_pre_day`. A second forecast, labeled
`post_release_pre_exact_sensitivity`, is scored after same-day release blocks
but before exact conditioning. It is a timing sensitivity rather than the
primary out-of-sample result. If several exact scores share a date, both
information sets are frozen for the entire atomic score batch before any one
score is conditioned.

For either information cutoff, let
$\widehat{\boldsymbol\mu}_m$ and $\widehat{\boldsymbol P}_m$ be the forecast
moments for the exact score $\boldsymbol s_m$. The score-center negative log
predictive density is

$$
\operatorname{NLPD}_m
=\frac12\left[
2\log(2\pi)
+\log\det\widehat{\boldsymbol P}_m
+(\boldsymbol s_m-\widehat{\boldsymbol\mu}_m)^\top
\widehat{\boldsymbol P}_m^{-1}
(\boldsymbol s_m-\widehat{\boldsymbol\mu}_m)
\right].
$$

Lower NLPD rewards forecasts that are both accurate and appropriately
uncertain. Axis-specific RMSE reports the square root of the mean squared error
for growth and inflation score centers separately.

Let $\widehat{\boldsymbol p}_m$ be the forecast quadrant distribution and
$\boldsymbol p_m^*$ the soft map obtained from the exact released score. The
probability metrics are

$$
\operatorname{CE}_m
=-\sum_{r=1}^{4}p_{m,r}^*\log\widehat p_{m,r},
$$

$$
\operatorname{Brier}_m
=\sum_{r=1}^{4}
\left(\widehat p_{m,r}-p_{m,r}^*\right)^2,
$$

$$
\operatorname{KL}_m
=\sum_{r=1}^{4}p_{m,r}^*
\log\left(\frac{p_{m,r}^*}{\widehat p_{m,r}}\right).
$$

All three are lower-is-better. Cross-entropy measures predictive log loss,
Brier distance measures squared probability error, and KL divergence measures
information lost by using the forecast in place of the exact-score map. These
targets are soft reporting distributions, not one-hot quadrant labels.

The published causal replay contains 188 completed-score months. At the primary
strict-pre-day cutoff, the evidence filter compares with the otherwise
identical transition-only filter as follows:

| Lower-is-better metric | Evidence filter | Transition only | Evidence minus transition |
|---|---:|---:|---:|
| Growth-score RMSE | 3.441 | 4.238 | $-0.797$ |
| Inflation-score RMSE | 0.852 | 0.733 | $+0.119$ |
| Mean Gaussian NLPD | 28.031 | 26.247 | $+1.785$ |
| Median Gaussian NLPD | 1.770 | 1.817 | $-0.047$ |
| Mean quadrant cross-entropy | 1.573 | 1.527 | $+0.045$ |
| Mean quadrant Brier distance | 0.160 | 0.151 | $+0.008$ |
| Mean quadrant KL divergence | 0.515 | 0.470 | $+0.045$ |

These results do not establish an overall probabilistic improvement. The
evidence filter has lower monthly NLPD in 134 months, higher NLPD in 45 months,
and the same NLPD in 9 months, so its typical density update is modestly useful.
However, its April 2020 evidence-minus-transition NLPD is approximately
$+783.1$. That single catastrophic miss makes its mean NLPD worse even though
the paired monthly median favors the evidence filter. The apparent full-sample
growth-RMSE improvement is also crisis-sensitive: after excluding March through
June 2020, growth RMSE is 0.893 for evidence versus 0.745 for transition only.

Excluding all of calendar 2020 changes the mean NLPD comparison in favor of the
evidence filter by approximately 0.201, but growth RMSE is worse by 0.165 and
quadrant cross-entropy remains worse by approximately 0.011. Excluding the
longer 2020Q1--2021Q1 interval produces small improvements in both score RMSEs
and NLPD, while the quadrant-score differences are nearly zero. These are
diagnostic exclusions, not replacement headline results. Together they show
frequent small density gains, severe crisis sensitivity, and essentially no
stable quadrant-probability edge.

The post-release/pre-exact sensitivity is close to the strict-pre-day result.
It is not used as the primary result because release artifacts are dated but do
not provide dependable intraday ordering.

## 17. Release-block dependence diagnostics

The event likelihood factorizes across separately submitted observation
models. This is an approximation, so the replay records each event's causal
pre-update predictive innovation, marginal predictive standard deviation, and
Cholesky-whitened innovation. Every observation sharing a publication date is
diagnosed against the same frozen pre-release state; diagnostic residuals are
therefore not artifacts of the stable order used to apply likelihood factors.

For cross-model diagnostics, weekly `ICSA` innovations are averaged within
their reference month. The weekly serial sequence retains every observation,
including catch-up observations sharing one publication date, and orders them
by their original weekly reference date. Monthly response series are not
averaged; duplicate response/month rows are rejected. Pearson and Spearman
correlations are tested
at lags $-1$, $0$, and $+1$, with the convention

$$
\text{right reference month}
=\text{left reference month}+\text{lag}.
$$

Benjamini–Hochberg q-values control the false-discovery rate across the full
reported cross-model family. Responses already fitted jointly in the same
observation model are omitted from redundant pairwise tests because
$\boldsymbol R_b$ represents their contemporaneous covariance. Distinct models
remain eligible even if they share an economic block, so inflation expectations
and input costs are tested against one another.

Ljung–Box tests diagnose residual serial dependence at weekly lags 1, 4, and 8
for `ICSA`, and monthly lags 1, 3, and 6 for monthly responses. Serial-test
p-values receive a separate Benjamini–Hochberg adjustment. Missing reference
months and same-publication batch sizes are recorded rather than compressed
into artificial consecutive lags.

These tests cannot establish independence. A large p-value may reflect a short
sample, low power, changing fitted emissions, or noisy measurements. Pairwise
correlations are also exploratory when either residual series retains serial
dependence. Material residual dependence should motivate a joint observation
model, a shared latent disturbance, or a conservative likelihood tempering
sensitivity; it should not be hidden by selectively dropping diagnostics.

In the full replay, 342 cross-model tests and 36 serial tests have sufficient
data. Benjamini--Hochberg adjustment at $q<0.05$ rejects 64 cross-model nulls
and 28 serial-independence nulls. The cross-model count comprises 59 Pearson
and only 5 Spearman rejections. Dependence is therefore not safely negligible,
but the discrepancy between linear and rank correlations shows that crisis
observations and outliers drive much of the result. As an explicit diagnostic,
excluding 2020 leaves only 2 significant Pearson tests and no significant
Spearman tests; the maximum absolute Pearson correlation falls from 0.930 to
0.446. The baseline must consequently remain described as a
conditionally-independent Gaussian approximation, not a calibrated joint
release model.

## 18. Filter artifacts, status, and limitations

The frozen filter specification is:

- `configs/models/m02_event_driven_bayesian_filter.yaml`

Core implementation:

- `src/regime_allocation/models/m02_soft_composite/gaussian_emissions.py`
- `src/regime_allocation/models/m02_soft_composite/joint_filter.py`
- `src/regime_allocation/models/m02_soft_composite/walkforward.py`
- `src/regime_allocation/models/m02_soft_composite/dependence_diagnostics.py`
- `src/regime_allocation/cli/build_m02_inference.py`

The complete replay contains 3,573 checkpoints, 1,406 replayed
scored-or-audited events, 633
emission fits, 191 expanding VAR fits, 1,711 residual coordinates, and 752
evaluation rows (376 paired comparisons). Detailed checkpoint, event, fit,
residual, evaluation, and dependence audits are stored under
`data/processed/m02_soft_composite/bayesian_filter/`. Public summaries are under
`results/published/m02_soft_composite/bayesian_filter/`, with lineage in
`data/manifests/m02_event_driven_bayesian_filter.json`. In particular:

- `latest_posterior.json` reports the latest four-month Gaussian state and
  quadrant readout;
- `monthly_current_posteriors.csv` provides the public posterior history;
- `evaluation_summary.csv` reports both timing cutoffs and both filter variants;
- `emission_summary.csv` reports the fitted block models; and
- `dependence_summary.json` summarizes cross-model and serial diagnostics.

As of 20 July 2026, the July evidence-filter quadrant probabilities are 0.340
for growth down/inflation up, 0.251 for growth up/inflation up, 0.236 for growth
up/inflation down, and 0.172 for growth down/inflation down. Their entropy is
1.358, close to the four-quadrant maximum $\log 4\approx1.386$; the published
state is deliberately diffuse rather than a hard regime call.

Important limitations of the baseline are:

- the VAR and observation equations are linear Gaussian and use plug-in
  parameters rather than integrating coefficient or covariance uncertainty;
- the April 2020 growth outlier materially inflates VAR process variance and
  can influence ordinary least-squares emission loadings;
- the emission covariance is constant within an observation model and does not
  capture time-varying volatility or heavy tails; it is estimated from the
  final fit's in-sample residuals, so plug-in predictive variance can remain
  optimistic despite Ledoit–Wolf shrinkage;
- separate observation models are treated as conditionally independent within
  an update day, even though the residual diagnostics may reveal dependence;
- multiple weekly claims observations in a catch-up release are separate
  likelihood factors, so residual serial dependence can make the posterior too
  concentrated;
- the baseline marginalizes a partial multivariate response with the matching
  covariance submatrix; it does not yet condition a later coordinate on an
  earlier asynchronous coordinate from the same correlated release model;
- inflation expectations and input-cost series have short real-time histories,
  limiting estimation power;
- many same-reference-month JOLTS releases arrive after the score is exact and
  therefore contribute diagnostics rather than a live update;
- the mapping readout assumes cross-month independence of disagreement and
  revision perturbations, and probability-distance evaluation can reflect both
  score-center forecast error and drift between forecast and target mapping
  covariance vintages; and
- Model 02 allocation and backtesting have not yet been implemented.

The crisis observation is not removed silently. Robust standardization,
winsorized scores, Student-$t$ innovations, robust VAR estimation, observation
variance regimes, or likelihood tempering should each be implemented as a
named sensitivity so their effect can be compared with this frozen Gaussian
baseline.
