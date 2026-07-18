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

## Notation

$m$ denotes the monthly reference period being classified, and $\ell$ denotes
another historical reference month. Component index $k$ identifies one of the
eight source features. $R_m\in\mathcal R$ is the deterministic regime assigned
to month $m$, where $\mathcal R$ is the four-regime state set. A superscript
$(v)$ identifies the data vintage that was available on date $v$; it is not an
exponent. $T_m$ is the first date on which every input required to compute
$R_m$ was available. This convention separates the month an observation
describes from the later date on which it became known.

## 1. First-release vintage rule

Let $x_{k,m}^{(v)}$ be the published level of component series $k$ for reference
month $m$ as it appeared in vintage $v$. Let $V_k$ be the available candidate dates
inside the frozen acquisition window. Under the primary FRED API provider,
$V_k$ contains the series-specific initial-release dates returned by output
type 4. Under the keyless fallback, it contains the configured ALFRED release
family's calendar dates. Define the first appearance in the acquired archive:

$$
\widetilde v_k(m)=\min\left\{v\in V_k:x_{k,m}^{(v)}
\text{ is available}\right\}.
$$

The observation is eligible only if that first appearance satisfies

$$
0\leq \widetilde v_k(m)-\operatorname{end}(m)\leq92\text{ days}
$$

and the archive-start rule below. If it is eligible, set
$v_k(m)=\widetilde v_k(m)$; otherwise the feature is unavailable. Here
$\operatorname{end}(m)$ is the final calendar date of reference month $m$. The
extractor does not search later vintages for one that happens to satisfy the
lag rule. A late first appearance is archive backfill, not a contemporaneous
release.

The transformed first-release feature is

$$
u_{k,m}=h_k\!\left(x_{k,m}^{(v_k(m))},x_{k,m-1}^{(v_k(m))}\right).
$$

Here $h_k$ is the fixed transformation for component $k$ and $u_{k,m}$ is the
resulting first-release monthly feature.

The current and prior month therefore come from the same vintage. This matters
because the prior month is often revised when the current month is released,
and index base years can change. Differencing two independently frozen
first-release levels would mix vintages and can create artificial jumps.

A separate archive-start rule applies before that lag test. The earliest
selected vintage for a series can expose many older reference periods at once.
Only the latest reference period visible in that initial snapshot is eligible;
all older rows are archive-bootstrap history, not simultaneous first releases.
Seventeen raw rows previously passed the 92-day lag gate; two belonged to the
inactive pre-splice history of `WPSFD4131`, so 15 had entered active component
history. The rule is frozen as `archive_start_latest_only: true`.

## 2. Component transformations

The growth components are

$$
u_{\text{payroll},m}
=\mathrm{PAYEMS}_m^{(v_{\mathrm{PAYEMS}}(m))}
-\mathrm{PAYEMS}_{m-1}^{(v_{\mathrm{PAYEMS}}(m))},
$$

$$
u_{\text{IP},m}
=100\log\left(
\frac{\mathrm{INDPRO}_m^{(v_{\mathrm{INDPRO}}(m))}}
{\mathrm{INDPRO}_{m-1}^{(v_{\mathrm{INDPRO}}(m))}}
\right),
$$

$$
u_{\text{consumption},m}
=100\log\left(
\frac{\mathrm{PCEC96}_m^{(v_{\mathrm{PCEC96}}(m))}}
{\mathrm{PCEC96}_{m-1}^{(v_{\mathrm{PCEC96}}(m))}}
\right),
$$

$$
u_{\text{unemployment},m}
=-\left(
\mathrm{UNRATE}_m^{(v_{\mathrm{UNRATE}}(m))}
-\mathrm{UNRATE}_{m-1}^{(v_{\mathrm{UNRATE}}(m))}
\right).
$$

The inflation components are monthly log changes:

$$
u_{k,m}
=100\log\left(
\frac{x_{k,m}^{(v_k(m))}}{x_{k,m-1}^{(v_k(m))}}
\right),
$$

for core CPI ($\mathrm{CPILFESL}$), core PCE prices ($\mathrm{PCEPILFE}$), core
finished-goods producer prices, and average hourly earnings ($\mathrm{AHETPI}$).

Producer prices use the BLS-designated transition from discontinued
$\mathrm{PPILFE}$ through December 2015 to its replacement $\mathrm{WPSFD4131}$
from January 2016. Monthly changes are calculated inside each source; index
levels are never spliced.

## 3. Strictly lagged expanding standardization

For component $k$, let $\mathcal H_{k,m-1}$ be the set of valid transformed
observations strictly before month $m$, and let
$n_{k,m-1}=|\mathcal H_{k,m-1}|$ be their count. With sample
standard deviation ($\operatorname{ddof}=1$),

$$
\widehat\mu_{k,m-1}
=\frac{1}{n_{k,m-1}}
\sum_{\ell\in\mathcal H_{k,m-1}}u_{k,\ell},
$$

$$
\widehat\sigma_{k,m-1}
=\sqrt{\frac{1}{n_{k,m-1}-1}
\sum_{\ell\in\mathcal H_{k,m-1}}
(u_{k,\ell}-\widehat\mu_{k,m-1})^2},
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

Let $\mathcal G$ and $\mathcal I$ be the index sets containing the four growth
and four inflation components, respectively. The raw axis scores are

$$
C_m^G=\frac{1}{4}\sum_{k\in\mathcal G}z_{k,m},
\qquad
C_m^I=\frac{1}{4}\sum_{k\in\mathcal I}z_{k,m}.
$$

$C_m^G$ and $C_m^I$ are the unsmoothed growth and inflation composites for
month $m$. Their superscripts label the economic axis; they are not exponents.

All four terms are required. Missing a component makes the axis unavailable;
the implementation never redistributes its 25% weight.

## 5. Three-month trailing smoothing

$$
G_m=\frac{C_m^G+C_{m-1}^G+C_{m-2}^G}{3},
\qquad
I_m=\frac{C_m^I+C_{m-1}^I+C_{m-2}^I}{3}.
$$

$G_m$ and $I_m$ are the final smoothed axis scores used to classify $R_m$.

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

"Up" means that the trailing mean of an axis's four standardized components is
nonnegative. It does not guarantee that every component is above its own mean,
nor does it necessarily imply literal economic expansion, acceleration,
slowdown, inflation, or disinflation. The deliberately literal public labels
avoid making those stronger claims.

## 7. Availability date

For a completed month, the availability date must cover every prerequisite
release used by the expanding standardizations and trailing window. Let
$v_k(\ell)$ be component $k$'s selected release date for reference month
$\ell$. For a classified month $m$, define the set of required, available
component observations as

$$
\mathcal P_m=
\left\{
(k,\ell):\ell\le m,\ v_k(\ell)\text{ exists, and the corresponding observation}
\text{ is required to calculate }G_m\text{ or }I_m
\right\}.
$$

Then

$$
T_m=\max_{(k,\ell)\in\mathcal P_m}v_k(\ell).
$$

$T_m$ is stored as `label_available_at`. With normally ordered publications,
the cumulative maximum equals the latest release required by the current
three-month score. The cumulative form remains correct if a delayed or catch-up
publication makes an older prerequisite available later. Any future backtest
may use the label only on or after this date. If a required current or trailing
component is unavailable, $R_m$ and $T_m$ are left undefined rather than
manufacturing a label or availability date.

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
