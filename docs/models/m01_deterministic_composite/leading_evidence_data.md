# Model 01 leading-evidence data

This stage builds the point-in-time non-defining release events consumed by
the Model 01 Bayesian filter. It does not estimate likelihoods or probabilities
and does not change deterministic regime labels. The source contract is
[`configs/models/m01_non_defining_release_evidence.yaml`](../../../configs/models/m01_non_defining_release_evidence.yaml).

## Release blocks

| Block | Series | Published features | Frequency |
|---|---|---|---|
| `weekly_claims` | `ICSA`, `CCSA` | Separate causal standardized log-AR(1) innovations | Weekly |
| `jolts` | `JTSJOR`, `JTSHIR`, `JTSQUR`, `JTSLDR` | Same-vintage rate differences, then causal z-scores | Monthly |
| `retail_sales` | `RSAFS`, `RSFSXMV` | Same-vintage log changes, then causal z-scores | Monthly |
| `housing` | `HOUST`, `PERMIT` | Same-vintage log changes, then causal z-scores | Monthly |
| `durable_goods` | `DGORDER`, `NEWORDER` | Same-vintage log changes, then causal z-scores | Monthly |

The data stage publishes 12 features. The promoted inference baseline uses 11:
it includes only `initial_claims_innovation` from the weekly block and excludes
`continued_claims_innovation`. Monthly likelihood vectors require every
configured feature in their block.

## Point-in-time rule

Let $`\rho`$ be a monthly or weekly source period and
$`x_{k,\rho}^{(v)}`$ its value in vintage $`v`$. The selected release date is
the observation's first appearance in the acquired real-time archive. It is
usable only when:

- release lag is between 0 and 92 calendar days; and
- at the first archive snapshot, it is the latest reference period visible on
  that date.

The lag anchor is month-end for monthly series and `reference_date` for weekly
series. A rejected first appearance is never replaced with a later revision.

Monthly transformations use the current and previous level from the same
selected vintage. If $`e`$ is the current event and $`\mathcal H_{k,e^-}`$
contains transformed observations released on strictly earlier dates, the
published monthly feature is

```math
z_{k,e}
=\frac{u_{k,e}-\widehat\mu_{k,e^-}}
{\widehat\sigma_{k,e^-}},
\qquad |\mathcal H_{k,e^-}|\ge60.
```

All observations published on the same date share the same prior moments.
Rows that have not completed warm-up remain in the full event table with an
explicit status rather than receiving an imputed value.

## Weekly claims

Claims use innovations rather than persistent levels. For a positive
first-release claims level $`L_\tau`$, define $`y_\tau=\log L_\tau`$. Before
publication date $`d`$, fit

```math
y_\tau=a_d+\phi_d y_{\tau-1}+\eta_\tau
```

using only levels released strictly before $`d`$. At least 52 earlier levels
are required. The innovation for a new reference week is

```math
\varepsilon_\tau
=y_\tau-\widehat a_d-\widehat\phi_d y_{\tau-1}.
```

It is standardized using at least 26 valid innovations released before $`d`$:

```math
z_\tau
=\frac{\varepsilon_\tau-\overline\varepsilon_{<d}}
{\widehat\sigma_{\varepsilon,<d}}.
```

Each same-day catch-up batch shares one frozen AR fit and one set of scaling
moments. A later row in the batch may use the preceding reference week's level
from that same batch, but no row can refit the parameters or scaling moments
for another same-day row. `ICSA` and `CCSA` are modeled separately.

## Reference periods and event identity

Every row retains both:

- `release_date`: when the observation first became available; and
- `reference_date` / `reference_month`: the economic period described.

A release is assigned to its source reference month, not its publication
month. Weekly claims use the calendar month containing the reported reference
week.

`event_group_id` identifies one block and publication date:

```text
{release_block}:{release_date}
```

`event_id` additionally includes the exact reference date:

```text
{release_block}:{release_date}:{reference_date}
```

This preserves distinct weeks within a catch-up publication and prevents
observations referring to different months from being forced into one state
vector. Dates are available, but reliable intraday timestamps are not.

## Coverage and artifacts

The published local build contains:

- 6,955 normalized first-release observations;
- 3,968 canonical event rows;
- 3,275 available feature rows; and
- event dates from 20 July 2000 through 16 July 2026.

Processed files under
`data/processed/m01_non_defining_release_evidence/` are:

| File | Contract |
|---|---|
| `first_release_observations.csv` | Acquired first appearances, eligibility, release lag, provider, and value |
| `non_defining_release_events.csv` | Complete long event table, including warm-up and unavailable rows |
| `non_defining_release_features.csv` | Available-only subset consumed by inference |

Core fields include `event_id`, `event_group_id`, `release_block`,
`release_date`, `reference_date`, `reference_month`, `feature_name`,
`series_id`, `transformed_value`, `feature_value`, `feature_status`, and causal
fit counts and moments. Claims rows also retain AR coefficients, forecasts, and
innovations.

The tracked
[`manifest`](../../../data/manifests/m01_non_defining_release_evidence.json)
records provider selection, series coverage, status counts, and artifact
hashes without credentials.

## Fail-visible rules and limitations

The stage never forward-fills, imputes missing values, rewrites a prior event
with a revision, or uses a current event in its own transform. Missing or
unidentified features retain explicit statuses. If a multivariate block is
incomplete, the inference stage skips that event rather than changing
dimension.

The indicators are correlated and do not become conditionally independent
merely because they are stored in separate blocks. Weekly claims create more
events than monthly blocks, and several series have later real-time archive
coverage. These are inference limitations, not permissions to backfill data.
