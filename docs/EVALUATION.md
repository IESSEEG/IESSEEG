# Evaluation and results

After running an experiment, export its classification metrics:

```bash
python -m iesseeg metrics --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

The command reads saved held-out predictions and writes two files under
`$BENCHMARK_WORK/results/`:

| File | Contents |
| --- | --- |
| `classification_point_estimates.csv` | AUROC, balanced accuracy, accuracy, sensitivity, specificity, precision, F1, and average precision |
| `classification_metrics.csv` | The same metrics with confidence intervals and sample counts |

Only experiments with saved predictions are included. A complete run has
36 diagnosis rows (18 methods/settings evaluated against two label definitions)
and 34 response rows (17 methods/settings at two visits).

## Comparing with the paper

[`reference/benchmark_metrics.csv`](../reference/benchmark_metrics.csv) contains
the main benchmark results and the full classification metrics from the
appendix. Values in the CSV files are proportions; the paper tables display
percentages.

Diagnosis neural outputs use `patient_label` and `expert_label` for the patient
diagnosis label and clinician diagnosis label, respectively. The reference
CSV and qEEG outputs use their full names, `patient_diagnosis_label` and
`clinician_diagnosis_label`. Match the task, model, training setting, and label
definition when comparing rows. Neural results use `frozen` and `finetuned`
for the two training settings.

The [recording-context experiment](experiments/context.md) writes its own
metrics to `context/results.csv` and paired AUROC comparisons to
`context/paired_differences.csv`. Reference results for that analysis are also
available under [`reference/`](../reference/README.md).

Training and numerical results can vary with GPU architecture and library
versions. The installation pins package versions in
[`requirements.txt`](../requirements.txt).

## Metrics

Diagnosis is evaluated on simulated routine EEG segments, separately against
patient diagnosis labels and clinician diagnosis labels. Response is evaluated
on complete original long recordings. For foundation models, each EEG input
receives the mean probability across its complete windows.

AUROC is the mean of the five test-fold AUROCs. Balanced accuracy and the
other classification metrics are calculated from the combined held-out
predictions. The main benchmark uses a probability threshold of 0.5.

AUROC confidence intervals use the cross-validated influence-function method
of LeDell et al. (2015), with observations grouped by patient. Balanced accuracy
and the other classification metrics use patient-cluster influence functions;
balanced accuracy uses the delta method. The implementation is in
[`iesseeg/metrics.py`](../iesseeg/metrics.py) and
[`iesseeg_paper/cv_auc_ci.py`](../iesseeg_paper/cv_auc_ci.py).
Appendix B.6 gives the statistical details.

## Patient partitions

All outer test folds are patient-disjoint. Internal validation follows the
procedures used for each experiment in Appendix B.4.

| Experiment | Training and validation |
| --- | --- |
| qEEG, frozen diagnosis heads, response models, and recording-context analysis | Fold `f` is used for testing, fold `(f+1) mod 5` for validation, and the remaining three folds for training |
| Diagnosis full fine-tuning with BIOT, CBraMod, and LaBraM | Segments from the four non-test patient folds are split into training and validation segments |
| Diagnosis full fine-tuning with EEGPT, LUNA, REVE, CodeBrain, and CSBrain | Training and validation are split by patient within the four non-test folds |
