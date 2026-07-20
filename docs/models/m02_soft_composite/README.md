# Model 02: soft composite score definition

Model 02 is a modest architectural revision of Model 01. It preserves
economically specified, equal-weight growth and inflation composites but does
not smooth them over three months and does not convert them to hard quadrant
labels. A second implemented stage maps these scores to soft quadrant weights
using disagreement and revision uncertainty. Score transition dynamics,
release likelihoods, Bayesian updates, allocation, and a backtest have not yet
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

Model 02 introduces a perturbed score

$$
\widetilde{\boldsymbol s}_m
=\boldsymbol s_m+\boldsymbol\varepsilon_m,
\qquad
\boldsymbol\varepsilon_m
\sim\mathcal N(\boldsymbol 0,\boldsymbol\Omega_{\mathrm{map},m}),
$$

where

$$
\boldsymbol\Omega_{\mathrm{map},m}
=\boldsymbol\Omega_{\mathrm{disagreement},m}
+\boldsymbol\Omega_{\mathrm{revision},m}.
$$

The resulting regime weights are probabilities that
$\widetilde{\boldsymbol s}_m$ lies in each growth/inflation quadrant. They are
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
