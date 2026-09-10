# Documentation

Start with the [project overview and weekly research snapshot](../README.md).
The [research overview](architecture/research_overview.md) describes the shared
research principles, equations, and historical publication results. The model
specifications below explain the data transformations, inference, allocation,
and evaluation methods in detail.

## Shared documentation

- [Research architecture and published baselines](architecture/research_overview.md)
- [Weekly updates and automation](operations/weekly_updates.md)
- [Repository maintenance and artifact policy](architecture/repository_maintenance.md)
- [Data access and credentials](data_access.md)
- [Model versioning](architecture/model_versioning.md)
- [Security policy](../SECURITY.md)

## Model 01: frozen monthly benchmark

- [Model card and results overview](models/m01_deterministic_composite/README.md)
- [Regime definition](models/m01_deterministic_composite/regime_definition.md)
- [Data dictionary](models/m01_deterministic_composite/data_dictionary.md)
- [Leading evidence data](models/m01_deterministic_composite/leading_evidence_data.md)
- [Transition model](models/m01_deterministic_composite/transition_model.md)
- [Bayesian filter](models/m01_deterministic_composite/bayesian_filter.md)
- [Bayesian filter results](models/m01_deterministic_composite/bayesian_filter_results.md)
- [Portfolio allocation](models/m01_deterministic_composite/portfolio_allocation.md)
- [Portfolio backtest results](models/m01_deterministic_composite/portfolio_backtest_results.md)

## Model 02: promoted weekly model

- [Model card and results overview](models/m02_soft_composite/README.md)
- [Inference](models/m02_soft_composite/inference.md)
- [Portfolio allocation](models/m02_soft_composite/portfolio_allocation.md)
- [Portfolio backtest results](models/m02_soft_composite/portfolio_backtest_results.md)

## Research archives

The [research archive](archive/README.md) preserves historical diagnostics and
experiments outside the promoted model contract.

### Model 01 archive

- [Archive overview](archive/m01/README.md)
- [Model audit](archive/m01/model_audit.md)

### Model 02 archive

- [Archive overview](archive/m02/README.md)
- [Inference sensitivities](archive/m02/inference_sensitivities.md)
- [Evidence block experiments](archive/m02/evidence_block_experiments.md)
- [Existing block attribution](archive/m02/existing_block_attribution.md)
- [Feature revision](archive/m02/feature_revision.md)
- [Expanded universe backtest](archive/m02/expanded_universe_backtest.md)
- [Active optimizer and oracle diagnostic](archive/m02/active_optimizer_oracle_diagnostic.md)

## Original course project

The [legacy archive](../legacy/README.md) contains the original course manuscript,
its authorship attribution, and the methodological context for the rebuild.
