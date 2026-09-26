# Reproducing the paper

Run commands from the cloned repository after installing it in editable mode.
All commands below use the same downloaded dataset and work directory:

```bash
export DATASET_ROOT=/data/IESSEEG
export BENCHMARK_WORK=/scratch/iesseeg
export IESSEEG_PRETRAINED_DIR=/models/iesseeg
python -m iesseeg prepare --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Choose a CUDA device explicitly for signal preparation and neural experiments.
Classifier fitting and statistical summaries run on CPU. Run one model per
process because upstream packages have overlapping Python module names.

## Complete run

After configuring pretrained weights and the EEGPT interpreter in
[MODELS.md](MODELS.md), run all paper experiments serially:

```bash
bash scripts/reproduce_all.sh "$DATASET_ROOT" "$BENCHMARK_WORK" cuda:0
```

This includes neural training and full-recording inference and is a substantial
GPU workload. The commands below allow individual tasks or models to be run
separately. Published aggregate targets are in [`reference/`](../reference/README.md).

## Table 3: diagnosis

qEEG triplet and Expanded qEEG:

```bash
python -m iesseeg extract-qeeg --task diagnosis --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg qeeg --task diagnosis \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

For each foundation model, prepare its inputs and extract pretrained features:

```bash
python -m iesseeg preprocess --model biot --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg extract-diagnosis --model biot --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg diagnosis --model biot --branches frozen --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Replace `biot` with `labram`, `cbramod`, `eegpt`, `luna`, `reve`, `codebrain`, or
`csbrain`. LaBraM, EEGPT, REVE, CodeBrain, and CSBrain share a preprocessing
family, so its existing inputs can be reused. `--branches frozen finetuned`
adds model-specific full fine-tuning and test inference.

Diagnostic qEEG selects logistic regularization on a held-out patient validation
fold without refitting. Frozen heads train with inherited patient labels and
select checkpoints using validation segment AUROC. The two diagnosis targets
are applied only at evaluation.

## Training and validation partitions

All outer test folds separate patients. The training and validation procedures
are described in Appendix B.4 and implemented as follows.

| Experiment | Internal validation |
| --- | --- |
| qEEG, frozen diagnosis heads, and response models | Fold `(f+1) mod 5` for validation, fold `f` for testing, and the remaining three folds for training |
| Diagnosis full fine-tuning: BIOT, CBraMod, LaBraM | Clinical segments from the four non-test patient folds are split into training and validation segments |
| Diagnosis full fine-tuning: EEGPT, LUNA, REVE, CodeBrain, CSBrain | Training and validation are split by patient within the four non-test folds |

## Table 4: treatment response

qEEG is extracted from each entire case recording:

```bash
python -m iesseeg extract-qeeg --task response --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg qeeg --task response \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

For each foundation model:

```bash
python -m iesseeg response --model biot --device cuda:0 \
  --branches frozen finetuned --visits PRE POST \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

This trains separate response models for PRE and POST. Full fine-tuning samples
native windows from training long recordings. Frozen heads use inherited labels
on windows from the released 30-crop training coordinates. At evaluation, every
complete window of each held-out long recording contributes to its mean
probability; only an incomplete final window is dropped. Each task produces
50 held-out recording predictions.

The response configurations specify optimizer settings, batch sizes, early
stopping, and native window lengths. Response LaBraM uses 16-second windows;
diagnosis and Section 6 LaBraM embeddings use 10-second windows. Test fold `f`
and validation fold `(f+1) mod 5` are excluded from training.

## Table 5 and Appendix C.2: recording context and lead time

First run response qEEG extraction above. Then extract embeddings for the
released ten 30-minute segments per recording:

```bash
for model in biot labram cbramod; do
  python -m iesseeg extract-context --model "$model" --device cuda:0 \
    --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
done
python -m iesseeg context --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

The segment embedding is the mean of frozen window embeddings. A five-component
PLS model is fitted using training-patient response labels; the reduced
embedding is then combined with whole-recording qEEG and/or ln(1 + lead time in
months). L2 logistic regression uses validation-selected regularization. Each
training crop has weight 0.1, giving each patient total weight one.

The command evaluates E, Q, L, E+Q, E+L, Q+L, and E+Q+L for both visits. Q and L
reference models are shared across encoders. It writes the 30 conditions and
held-out segment predictions under `context/`. The main paper reports PRE;
POST results are supplementary. These experiments are exploratory, with the
reported fixed PLS dimension chosen after earlier experiments.

## Main tables and full classification metrics

```bash
python -m iesseeg metrics --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

Outputs are under `results/`. `classification_metrics.csv` contains estimates
and intervals; `classification_point_estimates.csv` provides a compact table.
Missing experiments are omitted, so check row coverage before interpreting the
export as a complete paper reproduction. Expected complete coverage is
36 diagnosis evaluations (18 methods/settings × two label definitions) and
34 response evaluations (17 methods/settings × two visits).

## Resuming and checks

Feature extraction reuses existing per-recording files. Neural kernels retain
completed fold artifacts. Use a new work directory when changing settings or
model weights. `--limit 1` is available for extraction smoke checks and does not
produce a complete benchmark cache. Exact floating-point equality is not
promised across different GPU architectures or library versions.
