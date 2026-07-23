# Model 02 development archive

This directory preserves Model 02 experiments and diagnostics that informed
the current publication. None of these reports replaced the promoted inference
baseline, `student_t_7_reduced_core`, or the promoted weekly allocation method,
`posterior_optimized`.

The current, revisable Model 02 documentation is:

- [model card](../../models/m02_soft_composite/README.md);
- [inference contract and results](../../models/m02_soft_composite/inference.md);
- [weekly allocation contract](../../models/m02_soft_composite/portfolio_allocation.md); and
- [weekly allocation results](../../models/m02_soft_composite/portfolio_backtest_results.md).

Archived reports:

- [`inference_sensitivities.md`](inference_sensitivities.md): historical
  inference-baseline and robust-emission comparisons;
- [`evidence_block_experiments.md`](evidence_block_experiments.md): candidate
  evidence-block additions;
- [`existing_block_attribution.md`](existing_block_attribution.md): add-one and
  leave-one-out attribution of the earlier evidence graph;
- [`feature_revision.md`](feature_revision.md): reduced-core feature revision;
- [`active_optimizer_oracle_diagnostic.md`](active_optimizer_oracle_diagnostic.md):
  benchmark-relative active allocation and hindsight-regime diagnostic; and
- [`expanded_universe_backtest.md`](expanded_universe_backtest.md): all-local-ETF
  universe diagnostic.

Terms such as *locked*, *frozen*, or *hash-frozen* inside these reports refer
only to controls held fixed within the historical experiment. They do not mean
that Model 02 itself is frozen. Model 02 remains current, published, and
revisable through a future versioned change.
