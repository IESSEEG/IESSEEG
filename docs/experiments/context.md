# Recording context and lead time

This analysis combines embeddings from sampled 30-minute segments with qEEG
features from their complete source recordings and clinical lead time. The
pre-treatment results appear in Table 5; Appendix C.2 also reports the
post-treatment analysis.

Before running, complete [data preparation](../DATA.md) and
[activate the environment](../REPRODUCING.md#before-running).

## Extracting features

First extract whole-recording response qEEG, unless it is already available
from the [response benchmark](response.md):

```bash
python -m iesseeg extract-qeeg --task response --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Then extract segment embeddings from frozen BIOT, LaBraM, and CBraMod:

```bash
for model in biot labram cbramod; do
  python -m iesseeg extract-context --model "$model" --device cuda:0 \
    --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
done
```

The released sampling tables specify ten 30-minute segments per recording,
giving 500 segments per response task. Each segment embedding is the average
of its pretrained window embeddings. These commands use the original pretrained
weights, rather than the fine-tuned response checkpoints. They read the EDF
files directly, so diagnosis preprocessing is not required.

## Fitting the classifiers

```bash
python -m iesseeg context --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

For each training fold, embeddings are standardized and reduced to five
components using PLS supervised by treatment-response labels. The reduced
embeddings are then combined with qEEG and lead time.

| Feature | Definition |
| --- | --- |
| E | Segment embedding reduced to five PLS components |
| Q | Three response qEEG features from the complete source recording |
| L | `ln(1 + lead_time_months)`, using the participant metadata |

The experiment evaluates E, E+Q, E+L, and E+Q+L for each model, with shared Q,
L, and Q+L baselines. Logistic regression selects `C` from
`[0.001, 0.01, 0.1, 1, 10]` using validation AUROC. Each training segment has
weight 0.1, so each patient contributes total weight one. Scaling and PLS are
fitted on training patients only; the selected classifier is not refitted on
validation patients. The fixed PLS dimension was chosen during exploratory
experiments.

By default, both visits and all three models are included. For a smaller run,
pass `--visits PRE --models biot` to `context` and extract only BIOT PRE
embeddings with `extract-context --model biot --visits PRE`.

## Outputs

All fitted models and scores are saved under `$BENCHMARK_WORK/context/`.

| File | Contents |
| --- | --- |
| `results.csv` | Metrics and confidence intervals for each feature combination |
| `paired_differences.csv` | Paired AUROC differences between feature combinations |
| `<visit>_<model>_<features>_predictions.csv` | Held-out segment predictions |
| `<visit>_<model>_PLS5_fold<fold>.joblib` | Fitted embedding scaler and PLS projection |
| `<visit>_<model>_<features>_fold<fold>.joblib` | Fitted classifier and feature scaler |

The default run produces 30 result rows and 36 paired comparisons. Compare
these with [`reference/context_results.csv`](../../reference/context_results.csv)
and [`reference/context_paired_differences.csv`](../../reference/context_paired_differences.csv).
Context results are written by this command; the main benchmark's `metrics`
command exports diagnosis and full-recording response results separately.
