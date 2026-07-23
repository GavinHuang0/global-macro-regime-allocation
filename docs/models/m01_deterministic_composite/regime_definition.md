# Model 01 regime definition

This document defines the frozen deterministic target for
`m01_deterministic_composite`. The machine-readable source of truth is
[`configs/models/m01_deterministic_composite.yaml`](../../../configs/models/m01_deterministic_composite.yaml).

## Information and vintage rules

Let \(m\) be a reference month, \(d\) an information date, and \(k\) one of
the eight defining component series. If \(x_{k,m}^{(v)}\) is the level for
month \(m\) visible in vintage \(v\), the selected vintage \(v_k(m)\) is the
first appearance in the acquired real-time archive, provided that:

- publication is no earlier than reference month-end and no more than 92 days
  later; and
- at an archive's first snapshot, the row is the latest reference period
  visible in that snapshot.

An ineligible first appearance is not replaced with a later vintage. Both the
current and previous levels used in a transformation come from \(v_k(m)\):

\[
u_{k,m}
=h_k\!\left(x_{k,m}^{(v_k(m))},x_{k,m-1}^{(v_k(m))}\right).
\]

This same-vintage rule prevents a current first release from being compared
with a later-revised prior-month value.

## Components

| Axis | Component | Series | Transformation |
|---|---|---|---|
| Growth | Payrolls | `PAYEMS` | \(x_m-x_{m-1}\) |
| Growth | Industrial production | `INDPRO` | \(100\log(x_m/x_{m-1})\) |
| Growth | Real consumption | `PCEC96` | \(100\log(x_m/x_{m-1})\) |
| Growth | Unemployment rate | `UNRATE` | \(-(x_m-x_{m-1})\) |
| Inflation | Core CPI | `CPILFESL` | \(100\log(x_m/x_{m-1})\) |
| Inflation | Core PCE price index | `PCEPILFE` | \(100\log(x_m/x_{m-1})\) |
| Inflation | Core finished-goods PPI | `PPILFE`, then `WPSFD4131` | \(100\log(x_m/x_{m-1})\) |
| Inflation | Average hourly earnings | `AHETPI` | \(100\log(x_m/x_{m-1})\) |

Producer prices use `PPILFE` through December 2015 and `WPSFD4131` from
January 2016. Changes are computed within each source; index levels are never
spliced.

## Standardization and scores

Let \(\mathcal H_{k,m-1}\) contain valid transformed observations strictly
before \(m\), with count \(n_{k,m-1}\). Using the sample mean and sample
standard deviation of that history,

\[
z_{k,m}
=\frac{u_{k,m}-\widehat\mu_{k,m-1}}
{\widehat\sigma_{k,m-1}},
\qquad n_{k,m-1}\ge 60.
\]

The current observation never contributes to its own moments. A component is
unavailable if it lacks 60 earlier observations, its prior standard deviation
is invalid, or either same-vintage level is missing.

Let \(\mathcal G\) and \(\mathcal I\) be the four growth and four inflation
components. All four components are required on each axis:

\[
C_m^G=\frac14\sum_{k\in\mathcal G}z_{k,m},
\qquad
C_m^I=\frac14\sum_{k\in\mathcal I}z_{k,m}.
\]

Model 01 then applies a trailing, not centered, three-month average:

\[
G_m=\frac{C_m^G+C_{m-1}^G+C_{m-2}^G}{3},
\qquad
I_m=\frac{C_m^I+C_{m-1}^I+C_{m-2}^I}{3}.
\]

Missing components are never imputed and their fixed weights are never
redistributed.

## Regimes and availability

The deterministic regime \(R_m\in\mathcal R\) is:

| Condition | `regime_id` |
|---|---|
| \(G_m\ge0,\ I_m\ge0\) | `growth_up_inflation_up` |
| \(G_m<0,\ I_m\ge0\) | `growth_down_inflation_up` |
| \(G_m\ge0,\ I_m<0\) | `growth_up_inflation_down` |
| \(G_m<0,\ I_m<0\) | `growth_down_inflation_down` |

Exact zero is assigned to the corresponding `up` side. The labels describe
the signs of standardized composite scores; they do not assert literal
economic expansion, contraction, inflation, or disinflation.

Let \(T_m\) be the latest publication date among every component observation
required by the expanding transformations and trailing window for month \(m\).
It is stored as `label_available_at`. A downstream fit may use \(R_m\) only
when \(T_m\) satisfies that stage's information cutoff. If any prerequisite is
missing, both the label and its availability date remain undefined.

## Published coverage

The component panel contains 312 reference months from June 2000 through May
2026. The public regime history contains 250 rows from August 2005 through May
2026: 246 classified months and four unavailable months from October 2025
through January 2026. The gap follows missing October 2025 core-CPI and
unemployment observations and the fixed transformation and smoothing rules.

## Artifacts

- Local component history:
  `data/processed/m01_deterministic_composite/first_release_components_long.csv`
- Local score table:
  `data/processed/m01_deterministic_composite/composite_features_and_regimes.csv`
- [Published regime history](../../../results/published/m01_deterministic_composite/regime_history.csv)
- [Latest confirmed regime](../../../results/published/m01_deterministic_composite/latest_confirmed.json)
- [Manifest](../../../data/manifests/m01_deterministic_composite.json)

Field contracts are summarized in [`data_dictionary.md`](data_dictionary.md).

## Limitations

The target is a deterministic transformation of eight releases rather than a
directly observed latent state. Ordinary expanding moments retain the influence
of structural breaks and outliers, the three-month overlap mechanically
increases persistence, and several components share publication sources. The
availability data are daily rather than reliably intraday.
