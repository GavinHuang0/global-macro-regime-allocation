# Model 02 promoted inference

## Status

`student_t_7_reduced_core` is the current promoted Model 02 inference baseline.
It is published for continued research use and may be revised in a future
version; Model 02 is not frozen.

The baseline was selected after inspecting the same causal history used in its
replay. It is therefore an operational research baseline, not a confirmatory
out-of-sample result.

| Role | Model ID | Definition |
|---|---|---|
| Promoted baseline | `student_t_7_reduced_core` | Partial defining updates plus the reduced-core evidence graph |
| Benchmark | `transition_only` | OLS VAR(1) dynamics only |
| Benchmark | `partial_only` | OLS VAR(1) plus partial defining updates |
| Archived predecessor benchmark | `student_t_7_combined` | Earlier, broader evidence graph |

## State, timing, and notation

Let $`m`$ be a monthly reference period, $`d`$ an information cutoff, and
$`\mathcal D_d`$ the information available at that cutoff. The completed,
unsmoothed score is

```math
\boldsymbol z_m=(G_m,I_m)^\top,
```

where $`G_m`$ and $`I_m`$ are equal-weight averages of four causally
standardized growth and inflation components. Each component uses its first
eligible release, reads the current and prior reference months from the same
vintage, and requires 60 earlier valid observations for standardization.

The filter treats the monthly score as the random state $`\boldsymbol Z_m`$ and
maintains a four-month joint Gaussian:

```math
\boldsymbol Z_{m-3:m}\mid\mathcal D_d
\sim\mathcal N(\boldsymbol\mu_d,\boldsymbol P_d).
```

The expanding monthly transition is

```math
\boldsymbol Z_{m+1}
=\boldsymbol c+\boldsymbol A\boldsymbol Z_m+\boldsymbol\eta_{m+1},
\qquad
\boldsymbol\eta_{m+1}\sim\mathcal N(\boldsymbol 0,\boldsymbol Q).
```

Every VAR and observation-model fit uses only rows available strictly before
the release being processed. On a calendar day, the replay applies a month
roll first, then the release group, then exact completed-score conditioning at
end of day.

Partial defining releases update the applicable monthly state as their first
releases arrive. When all eight defining components are available, the
completed $`\boldsymbol z_m`$ is conditioned as an exact observation. Mapping
uncertainty is added only when the Gaussian state is translated into quadrant
probabilities; it is not VAR process noise.

For event $`e`$, publication date $`d_e`$, reference month $`m(e)`$, and
observation model $`b`$, the reduced-core non-defining evidence uses a
fixed-$`\nu=7`$ Student-$`t`$ equation:

```math
\boldsymbol y_e\mid\boldsymbol Z_{m(e)}
\sim t_7\!\left(
\boldsymbol a_b
+\boldsymbol H_b\boldsymbol Z_{m(e)}
+\boldsymbol C_b\boldsymbol v_e,
\boldsymbol\Psi_b
\right).
```

The implementation uses the documented robust approximate Gaussian moment
update so the rolling joint state remains Gaussian.

## Promoted evidence graph

The baseline contains exactly three non-defining observation models:

| Model ID | Evidence |
|---|---|
| `weekly_labor_stress` | Initial-claims innovation (`ICSA`), growth loading only |
| `consumer_real_implicit_joint` | Joint real retail activity and implicit-price coordinate |
| `business_activity_pipeline_joint` | Joint core capital-goods activity and orders-to-shipments pipeline |

The promoted graph excludes JOLTS, aggregate housing, legacy input costs,
import prices, vehicle sales, continued claims, and inflation expectations.

## Quadrant probabilities

Let $`R_m\in\mathcal R`$ denote the reporting quadrant and
$`p_{m,r\mid d}=\Pr(R_m=r\mid\mathcal D_d)`$. Every table and serialized vector
uses this order:

1. `growth_up_inflation_up`;
2. `growth_down_inflation_up`;
3. `growth_up_inflation_down`;
4. `growth_down_inflation_down`.

These are soft reporting probabilities, not hard regime declarations. “Up”
means the corresponding standardized score is on the nonnegative side of zero;
it does not by itself assert economic acceleration or rising inflation.

## Causal replay result

The primary checkpoint is `before_any_defining_release`. After the four-month
burn-in, 183 common target months from January 2011 through May 2026 are scored.

| Metric | Promoted baseline mean | Better direction |
|---|---:|---|
| Score-center NLPD | 26.854150 | Lower |
| Quadrant cross-entropy | 1.535693 | Lower |
| Unscaled quadrant Brier loss | 0.150466 | Lower |
| MAP quadrant accuracy | 0.377049 | Higher |

The following values are candidate minus reference:

| Reference | $`\Delta`$ NLPD | $`\Delta`$ cross-entropy | $`\Delta`$ Brier | $`\Delta`$ accuracy |
|---|---:|---:|---:|---:|
| `transition_only` | -0.067415 | +0.000418 | +0.000860 | +0.016393 |
| `partial_only` | -0.066712 | -0.001260 | -0.000051 | +0.016393 |
| `student_t_7_combined` | -0.000915 | -0.000240 | -0.000118 | +0.005464 |

Every corresponding 12-month moving-block-bootstrap interval contains zero.
The baseline improves mean NLPD and accuracy relative to `transition_only` but
slightly worsens both quadrant probability losses. None of the comparisons
establishes statistical superiority.

## Latest published inference

The latest inference artifact is dated 20 July 2026. June is partially
observed; July is the current-month state.

| Reference month | Growth up, inflation up | Growth down, inflation up | Growth up, inflation down | Growth down, inflation down |
|---|---:|---:|---:|---:|
| June 2026 | 0.177380 | 0.161240 | 0.354790 | 0.306591 |
| July 2026 | 0.159020 | 0.306653 | 0.266124 | 0.268204 |

## Artifacts and reproduction

- [configuration](../../../configs/models/m02_current_baseline.yaml)
- [manifest](../../../data/manifests/m02_current_baseline.json)
- [method summary](../../../results/published/m02_soft_composite/current/method_summary.json)
- [model registry](../../../results/published/m02_soft_composite/current/current_registry.csv)
- [evaluation summary](../../../results/published/m02_soft_composite/current/evaluation_summary.csv)
- [paired comparisons](../../../results/published/m02_soft_composite/current/paired_comparisons.csv)
- [paired block bootstrap](../../../results/published/m02_soft_composite/current/paired_block_bootstrap.csv)
- [latest baseline marginals](../../../results/published/m02_soft_composite/current/baseline_latest_marginals.csv)

Rebuild the publication from the repository root:

```powershell
python -m regime_allocation.cli.build_m02_current_baseline --project-root .
```

The command is offline and reads the existing point-in-time Model 02 inputs.
The published probabilities are research outputs, not investment advice.
