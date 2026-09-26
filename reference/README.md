# Published aggregate results

These tables provide numerical targets for reproducing the submitted paper.
All values are proportions rather than percentages.

- `benchmark_metrics.csv`: 36 diagnosis evaluations and 34 response evaluations,
  including the full classification metrics and confidence intervals.
- `context_results.csv`: all 30 conditions in the PRE/POST context and lead-time analysis.
- `context_paired_differences.csv`: 36 paired AUROC comparisons for that analysis.

The benchmark metrics were recomputed from the original held-out predictions
and checked against the final experiment tables. Context results were reproduced
by fitting the released classifiers to the original feature caches. These are
reference aggregates, not newly trained neural-model results. Row-level
predictions and trained models are generated locally by the reproduction commands.
