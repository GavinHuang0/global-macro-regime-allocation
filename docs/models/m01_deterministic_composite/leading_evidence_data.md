# Model 01 leading-evidence event data

## Scope and stage boundary

This stage acquires and transforms the five release blocks that provide
non-defining or leading evidence about Model 01's monthly regime. It produces a
causal, point-in-time event table consumed by the separately specified
[`event-driven Bayesian filter`](bayesian_filter.md). This data stage does
**not** estimate a release likelihood, multiply an event into the
four-month path distribution, publish a daily posterior, or change a
deterministic regime label.

The eight series that define the deterministic growth and inflation composites
are not downloaded a second time here. Their first-release values,
same-vintage transformations, release dates, and label-availability dates
already exist in the Model 01 component data described in
[`regime_definition.md`](regime_definition.md). A later inference stage can
adapt those records into target-confirming events. Keeping that work separate
also preserves the distinction between:

- **leading evidence**, which can improve the transition-only prior before the
  target-defining data arrive; and
- **target confirmation**, which observes some of the data used to construct
  the deterministic label itself.

The new source universe is deliberately modest. Every block is obtainable from
FRED/ALFRED, has an explicit real-time release history, and can be explained
without a proprietary consensus-forecast archive.

The frozen stage configuration is
[`configs/models/m01_non_defining_release_evidence.yaml`](../../../configs/models/m01_non_defining_release_evidence.yaml).
It requests observations and vintages from January 1, 1994 through July 16,
2026. The event outputs retain reference dates from June 1, 2000 onward. The
earlier acquisition rows supply causal AR and standardization warm-up history;
they are not presented as additional out-of-sample events. Actual available
feature coverage still begins only when a series has a genuinely
contemporaneous first release and has satisfied its causal standardization
requirements.

## Notation

The document uses $m$ for a monthly reference period, $\ell$ for another
historical month, and $d$ for the calendar date on which information becomes
available. Event $e$ has publication date $d_e$, reference month $m(e)$, and
release block $b(e)$. Index $k$ identifies a source series. When one rule
applies to either a month or a week, $\rho$ denotes that generic source
reference period. Data vintage $v$
is written as a parenthesized superscript, as in $x_{k,m}^{(v)}$, to distinguish
it from an exponent. The regime and path notation follows
[`bayesian_filter.md`](bayesian_filter.md#notation).

The keyless ALFRED transport has later series-specific archive floors where
older requests do not exist: August 15, 2013 for both claims series; June 13,
2001 for both retail-sales series; December 1, 1996 for housing starts; and
January 1, 2000 for permits and both durable-goods series. These are transport
constraints, not synthetic observations. They do not truncate authenticated
FRED output-type-4 retrieval, and every provider still passes through the same
92-day and archive-bootstrap eligibility rules.

The completed local ALFRED build contains 6,955 normalized observations, 3,968
canonical event rows, and 3,275 available feature rows. Its event history runs
from July 20, 2000 through July 16, 2026. Series-level coverage remains visible
in the manifest; eligible claims events begin in August 2013 and JOLTS events
begin in June 2010 rather than being backfilled to 2000.

## 1. Frozen release blocks and features

The five blocks contain 12 series and produce 12 long-form features.

| Release block | Release ID | Feature name | FRED series | Frequency | Feature construction |
|---|---:|---|---|---|---|
| `weekly_claims` | 180 | `initial_claims_innovation` | [`ICSA`](https://fred.stlouisfed.org/series/ICSA) | Weekly | Causal standardized innovation from an expanding log-AR(1) |
| `weekly_claims` | 180 | `continued_claims_innovation` | [`CCSA`](https://fred.stlouisfed.org/series/CCSA) | Weekly | Causal standardized innovation from an expanding log-AR(1) |
| `jolts` | 192 | `job_openings_rate_change` | [`JTSJOR`](https://fred.stlouisfed.org/series/JTSJOR) | Monthly | Same-vintage level difference, then a causal 60-observation expanding z-score |
| `jolts` | 192 | `hires_rate_change` | [`JTSHIR`](https://fred.stlouisfed.org/series/JTSHIR) | Monthly | Same-vintage level difference, then a causal 60-observation expanding z-score |
| `jolts` | 192 | `quits_rate_change` | [`JTSQUR`](https://fred.stlouisfed.org/series/JTSQUR) | Monthly | Same-vintage level difference, then a causal 60-observation expanding z-score |
| `jolts` | 192 | `layoffs_discharges_rate_change` | [`JTSLDR`](https://fred.stlouisfed.org/series/JTSLDR) | Monthly | Same-vintage level difference, then a causal 60-observation expanding z-score |
| `retail_sales` | 9 | `retail_sales_log_change` | [`RSAFS`](https://fred.stlouisfed.org/series/RSAFS) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |
| `retail_sales` | 9 | `retail_sales_ex_motor_vehicles_log_change` | [`RSFSXMV`](https://fred.stlouisfed.org/series/RSFSXMV) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |
| `housing` | 27 | `housing_starts_log_change` | [`HOUST`](https://fred.stlouisfed.org/series/HOUST) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |
| `housing` | 27 | `building_permits_log_change` | [`PERMIT`](https://fred.stlouisfed.org/series/PERMIT) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |
| `durable_goods` | 95 | `durable_goods_orders_log_change` | [`DGORDER`](https://fred.stlouisfed.org/series/DGORDER) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |
| `durable_goods` | 95 | `core_capital_goods_orders_log_change` | [`NEWORDER`](https://fred.stlouisfed.org/series/NEWORDER) | Monthly | Same-vintage log change, then a causal 60-observation expanding z-score |

The JOLTS rates use differences rather than percentage changes because their
published units are already rates. The retail, housing, and orders series are
positive-valued levels or flows, so their source transforms use log changes.
Each monthly source transform is then standardized against its own strictly
earlier expanding history. Claims levels are persistent and serially
correlated; their separate innovation and standardization rule is specified in
Section 3.

These variables are indicators, not replacements for the deterministic target.
Retail and orders are nominal and can reflect both real demand and prices.
JOLTS is published with a lag and is better interpreted as labor-demand
evidence than as a genuinely leading high-frequency signal. The later
evaluation must measure each block's incremental probabilistic value rather
than assuming that its economic label guarantees predictive value.

## 2. Point-in-time first-release rule

Let $x_{k,\rho}^{(v)}$ denote the value for series $k$ and source reference
period $\rho$ as visible in vintage $v$. Candidate publication dates come from
the authenticated FRED API's initial-release records when `FRED_API_KEY` is
available; the retained keyless ALFRED path supplies the same normalized
real-time contract. Provider choice must not change the feature mathematics.

For each series and period, first find the observation's earliest appearance in
the acquired archive:

$$
\widetilde v_k(\rho)=\min\left\{v\in V_k:
x_{k,\rho}^{(v)}\text{ exists}
\right\}.
$$

Here $V_k$ is the set of candidate vintage dates acquired for series $k$. That
first appearance is either accepted or rejected by the lag and archive-start
rules below. The extractor does not search a later vintage after rejecting the
first appearance, because a later observation of the same period is a backfill
or revision rather than its first release. If the first appearance is eligible,
set $v_k(\rho)=\widetilde v_k(\rho)$; otherwise the feature is unavailable.

Monthly and weekly observations use a general maximum release lag of 92
calendar days. The monthly lag is measured from reference month-end; the
weekly lag is measured from `reference_date`. Define

$$
\operatorname{lag}_{k,\rho}
=\widetilde v_k(\rho)-\operatorname{anchor}(\rho),
$$

where $\operatorname{anchor}(\rho)$ is month-end for a monthly observation and
the source reference date for a weekly observation. Let $d_k^0$ be the earliest
observed release date for series $k$, and let $\rho_k^0$ be the latest
reference period present on that date. Let $d_{k,\rho}=\widetilde v_k(\rho)$.
An observation is feature-eligible exactly when

$$
0\leq\operatorname{lag}_{k,\rho}\leq92
$$

and

$$
d_{k,\rho}>d_k^0
\quad\text{or}\quad
\left(d_{k,\rho}=d_k^0\text{ and }\rho=\rho_k^0\right).
$$

The additional earliest-date rule handles the bootstrap behavior of a
real-time archive: the first visible vintage can expose a long backfilled
history all at once. Only that group's most recent reference period can seed
the feature history; its older rows remain in the normalized input audit with
`eligible_for_feature = false`. The 92-day condition still applies to the
retained bootstrap candidate. If it also fails the lag condition, the series
has no eligible bootstrap row.

After the archive-start group, the 92-day rule deliberately permits genuine
delayed publications, including multiweek claims catch-up batches following a
government shutdown. A blanket 21-day claims ceiling would incorrectly remove
those real releases.

The provider-neutral claims acquisition record contains exactly
`reference_date`, `release_date`, and `value`; the series wrapper adds
`series_id`, `provider_id`, and the credential-free `source_url`. FRED obtains
initial releases directly from output type 4. The ALFRED fallback selects the
earliest nonmissing observation across its real-time vintages and normalizes it
to the same record contract.

The monthly current and previous values must be read from that one snapshot:

$$
x_{k,m}^{(v_k(m))}
\quad\text{and}\quad
x_{k,m-1}^{(v_k(m))}.
$$

This is the same-vintage rule already used by the deterministic components. It
prevents a current first-release level from being compared with an independently
frozen previous-month level that may use a different revision or index base.

For JOLTS rates, the transformed feature is

$$
u_{k,m}
=x_{k,m}^{(v_k(m))}-x_{k,m-1}^{(v_k(m))}.
$$

For retail sales, housing, and durable-goods orders, it is

$$
u_{k,m}
=100\log\left(
\frac{x_{k,m}^{(v_k(m))}}{x_{k,m-1}^{(v_k(m))}}
\right).
$$

Order one series' transformed observations by `release_date`. Write $u_{k,e}$
for the transformed value attached to event $e$. Let
$\mathcal H_{k,e^-}$ contain the transformed observations released strictly
before event $e$. With sample standard deviation (`ddof = 1`), the monthly
event feature is

$$
z_{k,e}
=
\frac{u_{k,e}-\widehat\mu_{k,e^-}}
{\widehat\sigma_{k,e^-}}.
$$

$\widehat\mu_{k,e^-}$ and $\widehat\sigma_{k,e^-}$ are the sample mean and
sample standard deviation of the values in $\mathcal H_{k,e^-}$. The minus
sign in $e^-$ means that observations published on the current event date are
excluded; it is not subtraction.

At least 60 prior transformed observations are required. The current event
never contributes to its own mean or standard deviation, and a zero or
nonfinite prior standard deviation leaves the feature unavailable. The event
table stores $u_{k,e}$ as `transformed_value` and $z_{k,e}$ as `feature_value`.
Rows before the threshold are retained with
`feature_status = standardization_warmup`.

If more than one retained observation for one series has the same release
date, every row uses statistics from dates strictly before that date. One
catch-up or backfill observation released that morning therefore cannot enter
another same-day row's standardization history. The row records that causal
history in `standardization_prior_count`, `standardization_prior_mean`, and
`standardization_prior_std`.

There is no full-sample standardization, cross-sectional weighting, or
three-month smoothing in this evidence layer. The downstream likelihood
estimator must still estimate its regime-conditional parameters causally; it
must not renormalize the whole event table using future releases.

## 3. Weekly claims: expanding log-AR(1) innovations

Absolute initial and continued claims are strongly persistent. Feeding their
levels into the posterior every week would repeatedly count much of the same
information. Model 01 therefore treats the unexpected component of each
release as the feature.

For one claims series, order valid first-release observations first by release
date and then by weekly reference date. Index that ordered weekly sequence by
$\tau$ and define

$$
x_\tau=\log L_\tau,
$$

where $L_\tau>0$ is the published claims level. Claims observations sharing a
release date form one atomic publication group $\mathcal B_d$. Immediately
before group $d$, fit an AR(1) with an intercept using only levels published on
earlier dates:

$$
x_{\tau'}=a_d+\phi_d x_{\tau'-1}+\eta_{\tau'},
\qquad d_{\tau'}<d.
$$

Here $a_d$ is the fitted intercept, $\phi_d$ is the fitted lag-one coefficient,
and $\eta_{\tau'}$ denotes an in-sample regression error. The date
$d_{\tau'}$ is the publication date of observation $\tau'$. Both fitted
parameters use only publication dates before $d$.

The fitted $(a_d,\phi_d)$ are frozen for the entire publication. Let
$\tau_1<\cdots<\tau_K$ be the positions in the ordered weekly sequence that
belong to $\mathcal B_d$. Because the source sequence is required to have no
missing weeks, position $\tau_g-1$ is the immediately preceding seven-day
reference week. The one-step forecasts and innovations are

$$
\widehat x_{\tau_g}=a_d+\phi_d x_{\tau_g-1},
\qquad
\varepsilon_{\tau_g}=x_{\tau_g}-\widehat x_{\tau_g},
\qquad g=1,\ldots,K.
$$

$\widehat x_{\tau_g}$ is the forecast log level and
$\varepsilon_{\tau_g}$ is the new release's log-scale innovation: the observed
log level minus its forecast.

The lag $x_{\tau_g-1}$ is the actual preceding reference-week level. For the
first row in a catch-up batch it normally comes from earlier published history;
for later rows it can be the preceding level released in the same batch. Using
that same-batch lag does not refit the AR parameters and does not pretend that
one row was available earlier in the day.

The implementation requires 52 level observations published before the group
before estimating a forecast. At the first eligible group, those observations
supply 51 lag/current transition pairs. It then standardizes every innovation
in the group against valid innovations from strictly earlier release dates:

$$
z_{\tau_g}=
\frac{\varepsilon_{\tau_g}-\overline\varepsilon_{<d}}
{\widehat\sigma_{\varepsilon,<d}}.
$$

Here $\overline\varepsilon_{<d}$ and
$\widehat\sigma_{\varepsilon,<d}$ are the sample mean and sample standard
deviation (`ddof = 1`) of innovations published strictly before the current
group. At least 26 prior valid innovations are required. All rows in one
publication therefore share the same AR parameters and standardization
moments. The current release group never
contributes to its own AR parameters, innovation mean, or innovation standard
deviation. Each row's `feature_value` is $z_{\tau_g}$; the raw level, forecast,
unstandardized innovation, fitted parameters, counts, and lagged moments remain
available as audit fields.

Only after every row in $\mathcal B_d$ has been transformed are the group's
levels and valid innovations appended to history for the next release date.
Within a catch-up batch, rows are evaluated in reference-week order because the
actual level for one week is the appropriate lag for the next week. A later
row may therefore use an earlier level from the same published batch, but it
cannot refit the AR coefficients or scaling moments. This is a calculation of
weekly forecast errors inside one publication, not an assertion that the rows
arrived at different intraday times, an estimated probability density, or a
Bayesian posterior update.

There is no hidden fallback. A missing or nonpositive level is invalid. A
constant or numerically near-constant lag history does not identify both an
intercept and a slope, so its forecast and innovation stay unavailable. A zero
or nonfinite prior innovation standard deviation likewise leaves the
standardized innovation unavailable. The pipeline does not substitute a
random-walk forecast, a trailing mean, zero, or a future-fitted parameter.

The AR calculation is performed separately for `ICSA` and `CCSA`; one series
never supplies lags or parameters to the other. The AR history consists of the
sequence of frozen first-release levels. Later revisions do not rewrite prior
$L_\tau$ values, and `previous_value_as_of_release` is unused for claims.

After the archive-bootstrap and release-lag filters, one claims series must
have unique reference dates, nondecreasing publication dates, strictly
increasing reference dates across publication groups, and an unbroken
seven-day reference-date sequence. Release dates need not be unique because a
genuine catch-up publication can contain several reference weeks. A missing
week makes the build fail instead of silently redefining lag one as a two-week
change.

## 4. Reference-period and event grouping rules

Every row carries both an information date and an economic reference period:

- `release_date` is the first date on which that value became available;
- `reference_date` is the source series' observation-period date;
- `reference_month` is the calendar month containing `reference_date`.

Monthly observations therefore map directly to their source reference month,
not to the month in which the publication happens. A March retail observation
released in April remains evidence about March.

For weekly claims, `reference_date` is the source observation's reference-week
date. The row maps to the calendar month containing that date. This rule is
especially important for continued claims, whose reference week can differ
from initial claims in the same unemployment-insurance publication. The
pipeline never assigns both series to the publication month merely because they
were released together.

An `event_group_id` identifies the release block and publication date:

```text
{release_block}:{release_date YYYY-MM-DD}
```

An `event_id` additionally includes the exact `reference_date`:

```text
{release_block}:{release_date YYYY-MM-DD}:{reference_date YYYY-MM-DD}
```

For monthly observations, `reference_date` equals the normalized
`reference_month`; features from the same block, release date, and reference
month therefore share an event ID. Claims use the same rule without discarding
the exact weekly reference date. If `ICSA` and `CCSA` released on the same date
refer to different weeks, they retain a common release group but have separate
event IDs. If those weeks cross a month boundary, a later likelihood layer
must not force observations about two monthly states into a one-state vector.

This stage stores daily release ordering but not an intraday trading
assumption. If only a publication date is reliably available, a later backtest
must adopt and document a conservative next-trading-session availability rule.

## 5. Canonical long event table

The processed evidence table has one row per event feature. Its stable core
schema is:

| Field | Meaning |
|---|---|
| `event_id` | Deterministic block/release/reference-date identifier |
| `event_group_id` | Deterministic block/release-date identifier that groups one publication |
| `release_block` | One of `weekly_claims`, `jolts`, `retail_sales`, `housing`, or `durable_goods` |
| `release_date` | First publication/vintage date used as the information date |
| `reference_date` | Economic observation-period date from the source series |
| `reference_month` | Month containing `reference_date`, normalized to month start |
| `frequency` | `weekly` or `monthly` |
| `feature_name` | Stable semantic feature identifier from Section 1 |
| `series_id` | FRED/ALFRED series identifier |
| `provider_id` | Acquisition provider recorded without credentials |
| `source_url` | Public credential-free FRED series page |
| `current_value` | First-release level for the referenced period |
| `previous_value_as_of_release` | Previous period's level in the same vintage, when the transform requires it |
| `transform` | `difference`, `log_difference`, or `expanding_log_ar1_innovation` |
| `transformed_value` | Same-vintage difference/log change, or claims log innovation before standardization |
| `feature_value` | Causally standardized monthly transform or standardized claims innovation intended for later likelihood estimation |
| `feature_status` | Explicit usability/status code; never inferred only from a missing number |
| `release_lag_days` | Calendar days from month-end to release for monthly rows, or from `reference_date` to release for weekly rows |
| `is_target_defining` | Always `false` in this non-defining evidence table |
| `standardization_prior_count` | Earlier valid source features used for causal scaling |
| `standardization_prior_mean` | Mean of those earlier source features |
| `standardization_prior_std` | Sample standard deviation of those earlier source features |

Claims rows additionally expose the complete causal AR audit trail:

| Claims-only field | Meaning |
|---|---|
| `ar_prior_observation_count` | Prior level observations available to the event |
| `ar_transition_count` | Lag/current pairs in the expanding AR fit |
| `ar_intercept` | Expanding OLS intercept estimated before the event |
| `ar_lag1_coefficient` | Expanding OLS AR coefficient estimated before the event |
| `forecast_log_level` | One-step pre-event log-level forecast |
| `forecast_level` | Forecast converted back to claims units |
| `innovation_log` | Released log level minus pre-event forecast |

For monthly rows, the generic standardization audit fields describe earlier
same-vintage transformed changes. For claims rows, they describe earlier valid
log-AR(1) innovations.

The event table is long rather than one permanently wide matrix because release
blocks have different dimensions, reference periods, and availability. A later
likelihood builder can pivot only complete rows for a particular block and
historical cutoff while preserving this row-level provenance.

## 6. Output files

The processed directory is
`data/processed/m01_non_defining_release_evidence/` and contains three local
research tables:

| File | Contract |
|---|---|
| `first_release_observations.csv` | Normalized long point-in-time inputs: series ID, release block, frequency, reference date, release date, first-release value, release lag, `eligible_for_feature`, `feature_eligibility_status`, provider ID, and public source URL |
| `non_defining_release_events.csv` | Full canonical long event table from Section 5, including causal warm-up and otherwise unavailable feature rows |
| `non_defining_release_features.csv` | Available-only long subset of the canonical event rows; it has the same columns, is not pivoted, and every row has `feature_status = available` |

The tracked provenance record is
`data/manifests/m01_non_defining_release_evidence.json`. It identifies the
evidence-set configuration and the generated artifacts without embedding a
FRED API key or credential-bearing request URL. The processed CSV files remain
local research artifacts under the repository's data-ignore policy unless a
later publication review explicitly selects a derived public output.

The manifest freezes both general lag ceilings at 92 days, records
`archive_start_policy = latest_reference_period_only`, and reports eligibility
status counts globally and by series. It also records provider selection,
per-series acquisition provenance, event status/block counts, artifact hashes,
and the explicit scope statement that no likelihood or Bayesian update was
performed.

## 7. Missing values, revisions, and backfills

The evidence layer follows fail-visible rules:

1. It never forward-fills an indicator from one release date to later days.
2. It never imputes a missing current or previous value.
3. It never substitutes today's revised historical value for an unavailable
   real-time vintage.
4. It never redistributes weights or shrinks a block's dimension silently.
5. An acquired row whose feature is unavailable only because of causal warm-up
   or an unidentified AR fit remains present with an explicit `feature_status`.
6. The first genuinely contemporaneous vintage determines coverage. A series
   introduced with a long backfilled history does not create fictional release
   events for the backfilled periods.

If no admissible first-release record exists, there is no economic event to
place in the table, so no synthetic row is created. This differs from an
acquired warm-up row: the latter is retained because its value is real and may
serve as strictly prior history for later features. Monthly warm-up rows use
`standardization_warmup`; a sufficient but zero/nonfinite scaling variance uses
`standardization_unidentified`. Claims additionally distinguish `ar_warmup`
and `ar_unidentified` from those generic standardization statuses and
`available`.

The normalized observation file still preserves acquired rows rejected by the
92-day lag ceiling or the earliest-date bootstrap rule and marks them with
`eligible_for_feature = false`. They do not enter the AR history, monthly
standardization history, or canonical event table as standalone observations.
For a retained monthly bootstrap row, an older same-vintage level can still be
its `previous_value_as_of_release`; using that denominator is part of the
point-in-time monthly change and does not turn the older period into a separate
feature event.

`feature_eligibility_status` explains the Boolean decision with one of
`eligible`, `archive_bootstrap_history`, `negative_release_lag`, or
`release_lag_exceeds_max`. The manifest reports these counts globally and by
series so provider or archive behavior cannot disappear behind one aggregate
row count.

This first evidence dataset emits only each observation's initial-release event.
Later revisions are not separate evidence events yet, and a later revision does
not overwrite an already emitted first-release feature. The same-vintage
monthly transformation can nevertheless contain a revised previous-period
level because that is exactly what a user saw when the current observation was
published.

Benchmark and methodological revisions require a separate, explicitly designed
event contract. They can affect months outside the four-month smoothing window
and must not be smuggled into the initial-release table as ordinary current-month
news.

If one member of a multivariate release block is unavailable, the data table
records the available features independently. The future likelihood stage must
either require its configured complete vector or use an explicitly estimated
missing-feature marginal. It may not silently treat a smaller vector as the
same fitted release distribution.

## 8. Output interpretation

The resulting rows are transformed historical observations with publication
and reference-period provenance. They are **not** probabilities. In particular,
this stage does not:

- estimate regime-specific Student-$t$ means or a shared covariance matrix;
- learn likelihood powers or assume cross-block conditional independence;
- attach a likelihood to any coordinate of $R_{m-3:m}$;
- normalize the 256 candidate path weights;
- score calibration, Brier loss, or log loss; or
- publish a daily regime posterior.

Those operations belong to the next inference stage. This boundary makes it
possible to audit the source event history and feature causality before a
statistical likelihood can turn data errors into apparently precise posterior
probabilities.
