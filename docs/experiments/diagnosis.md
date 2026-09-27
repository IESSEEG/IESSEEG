# IESS diagnosis

This guide reproduces Table 3. Models learn from pre-treatment
clinician-selected EEG segments from cases and clinician-selected EEG segments
from controls. Evaluation uses the 200 simulated routine EEG segments, with
separate scores for patient diagnosis labels and clinician diagnosis labels.

Before running, complete [data preparation](../DATA.md) and
[activate the environment](../REPRODUCING.md#before-running).

## qEEG baselines

```bash
python -m iesseeg extract-qeeg --task diagnosis --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg qeeg --task diagnosis \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

The command fits both the qEEG triplet and Expanded qEEG feature sets with
logistic regression. Regularization is selected by validation AUROC from
`C = [0.01, 0.1, 1, 10, 100]`. The selected model is evaluated without refitting
on the validation patients.

## Foundation models

The examples use BIOT. Replace `biot` with `labram`, `cbramod`, `eegpt`, `luna`,
`reve`, `codebrain`, or `csbrain` to run another model.

### Frozen encoder

Prepare the model inputs, extract pretrained embeddings, and train the
classification head:

```bash
python -m iesseeg preprocess --model biot --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg extract-diagnosis --model biot --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg diagnosis --model biot --branches frozen --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Each training window inherits its source participant's patient diagnosis label.
The encoder weights remain fixed. The head trains for up to 100 epochs, with
early stopping after 10 epochs without improvement in validation segment AUROC.

### Full fine-tuning

After preprocessing, run:

```bash
python -m iesseeg diagnosis --model biot --branches finetuned --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

This trains the encoder and classification head, then runs test inference.
Frozen embedding extraction is not needed for this setting. To run both
settings after extracting embeddings, use `--branches frozen finetuned`.

Model-specific diagnosis settings are in the `train_all.sh` scripts under
[`baselines_reference/baselines/`](../../baselines_reference/baselines).
[Model inputs](../MODELS.md#native-model-inputs) describes window lengths,
channels, and preprocessing. LaBraM, EEGPT, REVE, CodeBrain, and CSBrain share
preprocessed inputs; preprocessing one of these models also prepares the inputs
for the others.

## Outputs

Paths below are relative to `BENCHMARK_WORK`.

| Output | Location |
| --- | --- |
| qEEG scores | `results/diagnosis_qeeg.csv` |
| qEEG held-out predictions | `predictions/diagnosis_triplet.csv`, `predictions/diagnosis_expanded.csv` |
| Fitted qEEG classifiers | `fits/diagnosis_*_fold*.joblib` |
| Foundation-model held-out predictions | `local/diagnosis_probability_mean_20260924/<model>/<branch>_predictions.csv` |

Export the combined results with the [metrics command](../EVALUATION.md).
The [evaluation guide](../EVALUATION.md#patient-partitions) also documents the
training, validation, and test partitions.
