# IESSEEG

Code and experiments for **IESSEEG: An Open EEG Dataset Towards Better
Understanding of Infantile Epileptic Spasms Syndrome**.

[Dataset](https://huggingface.co/datasets/Capur/IESSEEG) ·
[Experiments](docs/REPRODUCING.md) ·
[Pretrained models](docs/MODELS.md) ·
[Results](reference/README.md)

IESSEEG provides approximately 3,172 hours of EEG from 100 pediatric participants,
together with clinical annotations and metadata. This repository contains the
qEEG baselines and eight EEG foundation models evaluated on IESS diagnosis,
pre-treatment response prediction, and post-treatment response prediction.

## Installation

The code has been tested on Linux with Python 3.11 and PyTorch 2.7.1.
EEG feature extraction and foundation-model experiments require an NVIDIA GPU
with CUDA support.

```bash
git clone https://github.com/IESSEEG/IESSEEG.git
cd IESSEEG
bash scripts/setup.sh --device cuda:0
source .iesseeg-env.sh
```

The setup script creates a virtual environment, installs the dependencies in
[`requirements.txt`](requirements.txt), and downloads the pretrained models.
All eight models run in the same environment. To choose a different location
for the environment or weights, add `--venv /path/to/env` or
`--models-dir /path/to/models` to the setup command.

<details>
<summary>Installing in an existing Python 3.11 environment</summary>

```bash
pip install -r requirements.txt
python scripts/prepare_models.py --output pretrained
source pretrained/env.sh
python scripts/check_environment.py --device cuda:0
```

</details>

## Getting started

Download [IESSEEG v1.1](https://huggingface.co/datasets/Capur/IESSEEG) and prepare
a directory for experiment outputs. Replace the two paths below with your data
and output locations. The dataset download is approximately 119 GB; extracted
features and model checkpoints need additional space.

```bash
export DATASET_ROOT=/data/IESSEEG
export BENCHMARK_WORK=/scratch/iesseeg

hf download Capur/IESSEEG --repo-type dataset --local-dir "$DATASET_ROOT"
python -m iesseeg validate --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg prepare --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

For a first experiment, run the diagnostic qEEG baselines. These commands
extract the features, fit logistic regression, and evaluate both diagnosis
label definitions across the five test folds.

```bash
python -m iesseeg extract-qeeg --task diagnosis --device cuda:0 \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg qeeg --task diagnosis \
  --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

The scores are printed in the terminal and saved to
`$BENCHMARK_WORK/results/diagnosis_qeeg.csv`. See [data preparation](docs/DATA.md)
for the dataset layout and annotation tables.

## Reproducing the experiments

Each guide includes the data used, commands, training settings, and output files.

| Experiment | Paper results | Guide |
| --- | --- | --- |
| IESS diagnosis | Table 3 | [qEEG, frozen encoders, and full fine-tuning](docs/experiments/diagnosis.md) |
| Treatment-response prediction | Table 4 | [Pre-treatment and post-treatment EEG](docs/experiments/response.md) |
| Recording context and clinical metadata | Table 5 and Appendix C.2 | [Embeddings, whole-recording qEEG, and lead time](docs/experiments/context.md) |

To run all experiments sequentially on one GPU:

```bash
bash scripts/reproduce_all.sh "$DATASET_ROOT" "$BENCHMARK_WORK" cuda:0
```

This trains all eight foundation models and evaluates the complete recordings.
For individual models or tasks, follow the guides above. The
[evaluation guide](docs/EVALUATION.md) explains how to export metrics and compare
them with the [paper results](reference/README.md).

## Code structure

| Directory | Contents |
| --- | --- |
| `iesseeg/` | Command-line interface, dataset loading, qEEG classifiers, and metadata experiments |
| `iesseeg_paper/` | qEEG feature extraction, patient splits, and metric calculations |
| `encoders/` | Pretrained EEG encoders and embedding extraction |
| `baselines_reference/` | Foundation-model implementations and diagnosis training |
| `experiments/` | Response training and recording-level evaluation |
| `preprocessing_reference/` | Model-specific EEG preprocessing |
| `configs/` | Response fine-tuning configurations |
| `scripts/` | Installation, model downloads, and the complete experiment runner |
| `docs/` | Data, model, and experiment documentation |
| `reference/` | Results reported in the paper |
| `tests/` | Tests for data handling, features, inference, and metrics |

Run the tests with `pytest -q` from the repository root.

## License and acknowledgments

The benchmark code is released under the [MIT License](LICENSE), and the
[dataset](https://huggingface.co/datasets/Capur/IESSEEG) under CC BY 4.0.
We thank the authors of the EEG foundation models for releasing their code and
weights. Their licenses and attribution are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
