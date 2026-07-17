# Model 01 regime and feature definition

## Scope

Model 01 defines a deterministic monthly target regime from point-in-time macro
releases. The target feeds the fixed first-order transition layer specified in
[`transition_model.md`](transition_model.md) and anchors the event-driven
nowcast specified in [`bayesian_filter.md`](bayesian_filter.md). The
point-in-time inputs for the non-defining release blocks are specified
separately in [`leading_evidence_data.md`](leading_evidence_data.md).

The research panel contains 312 reference months from June 2000 through May
2026. Because the standardization rule requires 60 prior observations and the
core PCE real-time archive begins in 2000, the classified regime history begins
later than the raw 26-year component panel.

## 1. First-release vintage rule

Let $x_{s,m}^{v}$ be the published level of series $s$ for reference month
$m$ as it appeared in vintage $v$. Let $V_s$ be the available candidate dates
inside the frozen acquisition window. Under the primary FRED API provider,
$V_s$ contains the series-specific initial-release dates returned by output
type 4. Under the keyless fallback, it contains the configured ALFRED release
family's calendar dates. Define the admissible candidate set

$$
A_{s,m}=\left\{v\in V_s:
x_{s,m}^{v}\text{ is available and }
0\leq v-\operatorname{end}(m)\leq92\text{ days}\right\}.
$$

The selected first-release vintage is

$$
v_s(m)=\min A_{s,m}.
$$

The transformed first-release feature is

$$
u_{s,m}=h_s\left(x_{s,m}^{v_s(m)},x_{s,m-1}^{v_s(m)}\right).
$$

The current and prior month therefore come from the same vintage. This matters
because the prior month is often revised when the current month is released,
and index base years can change. Differencing two independently frozen
first-release levels would mix vintages and can create artificial jumps.

If $A_{s,m}$ is empty, the feature is unavailable. The 92-day admissibility rule
treats later first appearances as archive backfills rather than contemporaneous
releases.

## 2. Component transformations

The growth components are

$$
u_{\text{payroll},m}
=\mathrm{PAYEMS}_m^{v(m)}-\mathrm{PAYEMS}_{m-1}^{v(m)},
$$

$$
u_{\text{IP},m}
=100\log\left(\frac{\mathrm{INDPRO}_m^{v(m)}}{\mathrm{INDPRO}_{m-1}^{v(m)}}\right),
$$

$$
u_{\text{consumption},m}
=100\log\left(\frac{\mathrm{PCEC96}_m^{v(m)}}{\mathrm{PCEC96}_{m-1}^{v(m)}}\right),
$$

$$
u_{\text{unemployment},m}
=-\left(\mathrm{UNRATE}_m^{v(m)}-\mathrm{UNRATE}_{m-1}^{v(m)}\right).
$$

The inflation components are monthly log changes:

$$
u_{s,m}
=100\log\left(\frac{x_{s,m}^{v_s(m)}}{x_{s,m-1}^{v_s(m)}}\right),
$$

for core CPI ($\mathrm{CPILFESL}$), core PCE prices ($\mathrm{PCEPILFE}$), core
finished-goods producer prices, and average hourly earnings ($\mathrm{AHETPI}$).

Producer prices use the BLS-designated transition from discontinued
$\mathrm{PPILFE}$ through December 2015 to its replacement $\mathrm{WPSFD4131}$
from January 2016. Monthly changes are calculated inside each source; index
levels are never spliced.

## 3. Strictly lagged expanding standardization

For component $k$, let $H_{k,m-1}$ be all valid transformed observations
strictly before month $m$, and let $n_{k,m-1}$ be their count. With sample
standard deviation ($\operatorname{ddof}=1$),

$$
\widehat\mu_{k,m-1}
=\frac{1}{n_{k,m-1}}\sum_{t<m}u_{k,t},
$$

$$
\widehat\sigma_{k,m-1}
=\sqrt{\frac{1}{n_{k,m-1}-1}
\sum_{t<m}(u_{k,t}-\widehat\mu_{k,m-1})^2},
$$

and

$$
z_{k,m}
=\frac{u_{k,m}-\widehat\mu_{k,m-1}}
{\widehat\sigma_{k,m-1}}.
$$

No z-score is produced until at least 60 prior valid observations exist. The
current month never contributes to its own mean or standard deviation. A zero
historical standard deviation also yields no score.

## 4. Equal-weight monthly composites

Let $G$ and $I$ be the four growth and four inflation components. The raw axis
scores are

$$
C_m^G=\frac{1}{4}\sum_{k\in G}z_{k,m},
\qquad
C_m^I=\frac{1}{4}\sum_{k\in I}z_{k,m}.
$$

All four terms are required. Missing a component makes the axis unavailable;
the implementation never redistributes its 25% weight.

## 5. Three-month trailing smoothing

$$
G_m=\frac{C_m^G+C_{m-1}^G+C_{m-2}^G}{3},
\qquad
I_m=\frac{C_m^I+C_{m-1}^I+C_{m-2}^I}{3}.
$$

This is a trailing window containing only months $m$, $m-1$, and $m-2$; it is
not a centered moving average.

## 6. Four deterministic regimes

Exact zero is assigned to the nonnegative ($\text{up}$) side.

| Condition | Stable regime ID | Human label |
|---|---|---|
| $G_m \geq 0,\ I_m \geq 0$ | `growth_up_inflation_up` | Growth composite up / inflation composite up |
| $G_m < 0,\ I_m \geq 0$ | `growth_down_inflation_up` | Growth composite down / inflation composite up |
| $G_m \geq 0,\ I_m < 0$ | `growth_up_inflation_down` | Growth composite up / inflation composite down |
| $G_m < 0,\ I_m < 0$ | `growth_down_inflation_down` | Growth composite down / inflation composite down |

“Up” means that the trailing mean of an axis's four standardized components is
nonnegative. It does not guarantee that every component is above its own mean,
nor does it necessarily imply literal economic expansion, acceleration,
slowdown, inflation, or disinflation. The deliberately literal public labels
avoid making those stronger claims.

## 7. Availability date

For a completed month,

$$
T_m=\max_{k\in G\cup I}v_k(m).
$$

$T_m$ is stored as `label_available_at`; it is the latest selected admissible
release-calendar vintage among the eight components. Any future backtest may
use the label only on or after that date. This prevents reference-month dates
from being mistaken for publication dates.

## 8. Missing-data rule

Model 01 never fills a missing defining observation, carries a score forward,
or redistributes a missing component's fixed 25% weight. The configuration
enumerates the only component-month gaps that the publication gate accepts;
any additional or unexpectedly repaired gap makes the build fail.

During the 2025 federal shutdown, BLS did not publish an October 2025 core-CPI
index or unemployment rate. October features are therefore unavailable. The
November one-month transformations are also unavailable because their required
October levels do not exist. Consequently, the consecutive three-month axis
scores and regime labels are unavailable from October 2025 through January
2026; labels resume in February 2026. Public history retains these four rows and
marks their reason instead of concealing the gap. See the
[official BLS CPI shutdown FAQ](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)
and the [December 2025 Employment Situation release](https://www.bls.gov/news.release/archives/empsit_01092026.htm).

## 9. Deviation from the original recommendation

The original proposal used real retail sales. ALFRED's
$\mathrm{RRSFS}/\mathrm{RSAFS}$ real-time archives begin in June 2001, so they
cannot honestly provide the requested 26-year first-release history. Model 01
uses real personal consumption expenditures ($\mathrm{PCEC96}$) as the closest
long-history, seasonally adjusted real consumer-activity proxy. This materially
changes the axis: real PCE includes services, is smoother, and is published
later than advance retail sales. It is also released by BEA alongside core PCE,
creating dependence between a growth and inflation component. Model 01 must
therefore be tested against an $\mathrm{RRSFS}$ sensitivity version over the
shorter common sample.

## 10. Additional frozen design choices

The 60-observation minimum is a stability choice added to the original
equal-weight proposal. It prevents very short expanding histories from creating
extreme z-scores, but it means the 312-month component panel does not produce
312 classified regimes. The exact first and last classified months are recorded
in the generated manifest.

Vintage acquisition begins in 1990 where a series permits it. This supplies a
long pre-2000 scaling history for established series without pretending that
ALFRED archived unavailable vintages. Results should later be tested with
alternative expanding-window start dates.

The 92-day release-lag ceiling is a conservative mechanical backfill filter for
monthly data. The manifest reports rows excluded by this rule. A sensitivity
test should compare 62-, 92-, and 122-day ceilings and inspect every affected
reference month.

## 11. Known limitations

- A 60-month warm-up reduces the classified sample relative to the raw panel.
- COVID observations permanently affect ordinary expanding means and standard
  deviations; a robust-scaling variant belongs in a later sensitivity test.
- Several components arrive in the same release and are dependent. This matters
  for later event likelihoods even though it does not change this deterministic
  label.
- The 2016 PPI splice is an official classification-system break.
- Although PPI index levels are never joined, monthly changes from the legacy
  and successor definitions share one expanding z-score history. An overlap
  diagnostic and separate-series sensitivity test are required.
- The current providers identify release dates but do not supply a universally
  reliable intraday timestamp contract; daily trading simulations must use a
  conservative availability convention.
