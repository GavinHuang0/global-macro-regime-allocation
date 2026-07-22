# Model 02 current-baseline declaration

## 1. Decision and model hierarchy

`student_t_7_reduced_core` is the declared current Model 02 inference baseline.
This is an operational model-governance decision for prospective use, not a
confirmatory out-of-sample result.

| Publication role | Model ID | Purpose |
|---|---|---|
| Current baseline | `student_t_7_reduced_core` | Partial defining releases plus the reduced-core non-defining evidence graph |
| Major benchmark | `transition_only` | OLS VAR(1) dynamics without partial or non-defining release updates |
| Major benchmark | `partial_only` | OLS VAR(1) dynamics plus partial defining releases, without non-defining evidence |
| Frozen predecessor benchmark | `student_t_7_combined` | The baseline selected by the earlier inference-sensitivity stage, preserved exactly |
| Current sensitivity | `student_t_7_reduced_core_with_expectations` | Current baseline plus the inflation-expectations observation model |

The [sensitivity catalog](../../../results/published/m02_soft_composite/current/sensitivity_catalog.csv)
also retains 23 feature variants from the preceding feature-revision stage and
links the four older sensitivity namespaces. Their historical roles, results,
and provenance have not been rewritten.

## 2. Current baseline specification

Let

$$
\boldsymbol Z_m=(G_m,I_m)^\top
$$

denote the monthly growth and inflation composite scores. The filter maintains
a four-month joint Gaussian distribution for

$$
\boldsymbol Z_{m-3:m}
=
(\boldsymbol Z_{m-3}^\top,\ldots,\boldsymbol Z_m^\top)^\top.
$$

The monthly transition remains the causal expanding OLS VAR(1)

$$
\boldsymbol Z_{m+1}
=
\boldsymbol a
+
\boldsymbol A\boldsymbol Z_m
+
\boldsymbol\eta_{m+1},
\qquad
\boldsymbol\eta_{m+1}\sim
\mathcal N(\boldsymbol 0,\boldsymbol Q).
$$

Every fit uses only information available strictly before the release being
scored. Partial score-defining releases update the applicable monthly state as
their first releases arrive. Once every defining component for a month is
available, the completed composite is conditioned as an exact end-of-day score
observation. The quadrant map remains a reporting layer: it adds the documented
mapping uncertainty and integrates the resulting bivariate distribution over
the four growth--inflation quadrants.

Non-defining releases use robust Student-$t$ observation equations with fixed
$\nu=7$ and the previously documented approximate Gaussian moment update. The
current evidence set contains exactly three observation models:

1. `weekly_labor_stress`: the ICSA innovation, loading only on growth;
2. `consumer_real_implicit_joint`: real retail activity and its implicit-price
   coordinate, fitted jointly; and
3. `business_activity_pipeline_joint`: core capital-goods shipments activity
   and the orders-to-shipments pipeline coordinate, fitted jointly.

The baseline excludes the lagged JOLTS block, aggregate housing, legacy input
costs, import prices, vehicle sales, continued claims, and inflation
expectations. Inflation expectations is kept as the sole current sensitivity.
Only three such updates were estimable in the replay, so its incremental effect
cannot be treated as stable evidence.

## 3. Causal replay results

The primary checkpoint is `before_any_defining_release`. After the four-month
burn-in, 183 common target months from January 2011 through May 2026 are scored.

| Metric | Current baseline mean | Better direction |
|---|---:|---|
| Score-center negative log predictive density (NLPD) | 26.854150 | Lower |
| Quadrant cross-entropy | 1.535693 | Lower |
| Quadrant Brier distance | 0.150466 | Lower |
| Hard-quadrant accuracy | 0.377049 | Higher |

The following deltas are candidate minus reference. Therefore, negative values
favor the candidate for the three losses, while positive values favor the
candidate for accuracy.

| Reference | $\Delta$ NLPD | $\Delta$ cross-entropy | $\Delta$ Brier | $\Delta$ accuracy |
|---|---:|---:|---:|---:|
| `transition_only` | -0.067415 | +0.000418 | +0.000860 | +0.016393 |
| `partial_only` | -0.066712 | -0.001260 | -0.000051 | +0.016393 |
| Frozen `student_t_7_combined` | -0.000915 | -0.000240 | -0.000118 | +0.005464 |

No comparison establishes statistical superiority. Every corresponding
12-month moving-block bootstrap interval at the primary checkpoint contains
zero. In particular, the current baseline improves mean NLPD and accuracy
relative to transition-only while slightly worsening both quadrant proper
losses. Its mean changes relative to the frozen predecessor are favorable but
very small.

Adding inflation expectations changes the primary means by only
$-0.0000006$ NLPD, $-0.0000026$ cross-entropy, $-0.0000014$ Brier distance, and
$0$ accuracy. Those estimates arise from only three applied events, and their
bootstrap intervals also contain zero. The expectations model is therefore a
named sensitivity, not part of the baseline.

## 4. Latest published probabilities

The latest artifact is dated 20 July 2026. June remains a partially observed
state; July is the current-month forecast state.

| Reference month | Growth up / inflation up | Growth down / inflation up | Growth up / inflation down | Growth down / inflation down |
|---|---:|---:|---:|---:|
| June 2026 | 0.177380 | 0.161240 | 0.354790 | 0.306591 |
| July 2026 | 0.159020 | 0.306653 | 0.266124 | 0.268204 |

These are probability estimates, not hard regime declarations or investment
recommendations.

## 5. Validation scope

The reduced-core graph was chosen after inspecting block-attribution and
feature-revision results on the same causal history replayed here. There is no
untouched holdout. The current-baseline declaration consequently means:

- the specification is frozen for prospective use;
- the replay verifies implementation, causality, and benchmark continuity;
- historical deltas are descriptive diagnostics; and
- no confirmatory out-of-sample performance claim is made.

A future untouched period or separately preregistered rolling evaluation is
required before treating any performance difference as confirmatory.

## 6. Artifacts and reproduction

Primary contract and provenance:

- [configuration](../../../configs/models/m02_current_baseline.yaml);
- [manifest](../../../data/manifests/m02_current_baseline.json);
- [method summary](../../../results/published/m02_soft_composite/current/method_summary.json);
- [current model registry](../../../results/published/m02_soft_composite/current/current_registry.csv); and
- [archived sensitivity catalog](../../../results/published/m02_soft_composite/current/sensitivity_catalog.csv).

Evaluation and latest state:

- [evaluation summary](../../../results/published/m02_soft_composite/current/evaluation_summary.csv);
- [subperiod summary](../../../results/published/m02_soft_composite/current/evaluation_subperiod_summary.csv);
- [paired comparisons](../../../results/published/m02_soft_composite/current/paired_comparisons.csv);
- [paired block bootstrap](../../../results/published/m02_soft_composite/current/paired_block_bootstrap.csv);
- [baseline latest marginals](../../../results/published/m02_soft_composite/current/baseline_latest_marginals.csv); and
- [benchmark latest marginals](../../../results/published/m02_soft_composite/current/benchmark_latest_marginals.csv).

From the repository root, rebuild the current publication with:

```powershell
python -m regime_allocation.cli.build_m02_current_baseline --project-root .
```

The build is offline and reads the frozen feature-revision artifacts. It does
not read API credentials or mutate historical sensitivity namespaces.
