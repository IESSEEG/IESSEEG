# Treatment-response prediction

This guide reproduces Table 4. Pre-treatment and post-treatment prediction are
separate tasks, each using the original long EEG recordings from the same
50 cases and their treatment-response labels. Each held-out recording receives
one response probability.

Before running, complete [data preparation](../DATA.md) and
[activate the environment](../REPRODUCING.md#before-running).

## qEEG baseline

Extract features from the complete case recordings, then fit the classifiers:

```bash
python -m iesseeg extract-qeeg --task response --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg qeeg --task response \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

This runs both response tasks. Logistic regression uses balanced class weights
and selects regularization by validation AUROC from `C = [0.01, 0.1, 1, 10]`.
The selected classifier is evaluated without refitting on validation patients.

## Foundation models

Run frozen-encoder training and full fine-tuning for both tasks:

```bash
python -m iesseeg response --model biot --device cuda:0 \
  --branches frozen finetuned --visits PRE POST \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Replace `biot` with `labram`, `cbramod`, `eegpt`, `luna`, `reve`, `codebrain`, or
`csbrain` to run another model. Use `--branches frozen` or
`--branches finetuned` for one training setting, and `--visits PRE` or
`--visits POST` for one task.

The response command reads the original EDF recordings directly; it does not
require the diagnosis preprocessing or embedding-extraction commands.
Frozen heads train on windows from the released 30-crop sampling coordinates.
Full fine-tuning samples windows from the training recordings. Both assign the
patient's treatment-response label to each training window.

At test time, probabilities are averaged over all complete, non-overlapping
windows of a recording. An incomplete final window is discarded. The output
contains 50 held-out recording predictions per task and training setting.

## Training settings

Full fine-tuning configurations are provided for
[BIOT, LaBraM, and CBraMod](../../configs/sustained_window_finetuning.json) and
[EEGPT, LUNA, REVE, CodeBrain, and CSBrain](../../configs/remaining_sustained_window_finetuning.json).
They specify window length, batch size, optimizer, learning-rate schedule,
and early stopping. The GPU is selected by `--device`.

The `validation_windows_per_patient` and `test_windows_per_patient` settings
control the sampled-window evaluations saved during training. The Table 4
results use the subsequent complete-recording evaluation described above.
LaBraM uses 16-second windows for response prediction; see
[model inputs](../MODELS.md#native-model-inputs) for the other models.

## Outputs

Paths below are relative to `BENCHMARK_WORK`.

| Output | Location |
| --- | --- |
| qEEG scores | `results/response_qeeg.csv` |
| qEEG held-out predictions | `predictions/response_PRE_qeeg.csv`, `predictions/response_POST_qeeg.csv` |
| Fitted qEEG classifiers | `fits/response_*_qeeg_fold*.joblib` |
| Foundation-model held-out predictions | `local/response_full_recording_20260925/<model>_<visit>/<branch>_predictions.csv` |
| Full fine-tuning logs and training summaries | `response_training/` |
| Neural checkpoints | `checkpoints/` |

Export the combined results with the [metrics command](../EVALUATION.md).
