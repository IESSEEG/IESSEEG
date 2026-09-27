# Reproducing the experiments

The repository provides commands for the main benchmark and the additional
analysis in Section 6. The guides below follow the experiments in the paper.

| Experiment | Methods | Paper results |
| --- | --- | --- |
| [IESS diagnosis](experiments/diagnosis.md) | Two qEEG feature sets; eight foundation models with frozen encoders and full fine-tuning | Table 3 |
| [Treatment-response prediction](experiments/response.md) | qEEG and eight foundation models, separately for pre-treatment and post-treatment EEG | Table 4 |
| [Recording context and lead time](experiments/context.md) | Logistic regression on combinations of embeddings, whole-recording qEEG, and lead time | Table 5 and Appendix C.2 |

## Before running

Complete the [installation](../README.md#installation) and
[data preparation](DATA.md). Run the commands from the repository root.
In a new terminal, activate the environment and set the two paths again:

```bash
source .iesseeg-env.sh
export DATASET_ROOT=/data/IESSEEG
export BENCHMARK_WORK=/scratch/iesseeg
```

If you installed in your own environment, activate it and source
`pretrained/env.sh` instead. The experiment guides use `cuda:0`; change this to
the GPU you want to use. Feature extraction and neural training run on GPU,
while logistic regression and metric calculations run on CPU.

## Running the complete benchmark

```bash
bash scripts/reproduce_all.sh "$DATASET_ROOT" "$BENCHMARK_WORK" cuda:0
```

The script prepares the data, runs the qEEG baselines, and trains and evaluates
all eight foundation models. It then runs the recording-context analysis and
exports the classification metrics. Models run sequentially on the selected GPU.

The [individual experiment guides](#reproducing-the-experiments) also show how
to run one model or one training setting. See [pretrained models](MODELS.md) for
checkpoint sources and input specifications, and [evaluation](EVALUATION.md)
for metric definitions and output formats.

## Continuing an interrupted run

Feature extraction reuses saved per-recording outputs. Training scripts retain
completed fold outputs. Rerun the same experiment command with the same work
directory to continue. When changing model weights or experiment settings,
choose a new work directory so that existing features and checkpoints are not
reused for a different experiment.

For a small extraction check, add `--limit 1` to `extract-qeeg`,
`extract-diagnosis`, or `extract-context`. Use a separate work directory for
this check; classifier fitting requires the complete feature set.
