# IESSEEG

Code for **IESSEEG: An Open EEG Dataset Towards Better Understanding of
Infantile Epileptic Spasms Syndrome**.

[Dataset](https://huggingface.co/datasets/Capur/IESSEEG) ·
[Reproduction guide](docs/REPRODUCING.md) ·
[Model weights](docs/MODELS.md) ·
[Reference results](reference/README.md)

IESSEEG contains approximately 3,172 hours of original EEG from 100 pediatric
participants, with clinician-selected EEG segments, simulated routine EEG
segments, annotations, and clinical metadata. The benchmark evaluates IESS
diagnosis and pre-treatment and post-treatment response prediction using qEEG
and eight EEG foundation models.

The code accepts the downloaded v1.1 dataset directory. It preserves the
released patient assignments, joins segments to their source recordings, and
writes generated features, checkpoints, and predictions to a separate work
directory. EDF files are referenced through symbolic links and are not copied
or modified by the benchmark.

## Install

Use Linux, Python 3.11, and an NVIDIA GPU.

```bash
git clone https://github.com/IESSEEG/IESSEEG.git
cd IESSEEG
bash scripts/setup.sh --device cuda:0
source .iesseeg-env.sh
```

Setup creates a virtual environment, installs `requirements.txt`, downloads the
pretrained models, and configures their paths. All eight models use this environment.
Use `--venv /path/to/env` and `--models-dir /path/to/models` to choose where files
are stored. Downloads are reused when setup is rerun.

If you already have a Python 3.11 environment:

```bash
pip install -r requirements.txt
python scripts/prepare_models.py --output pretrained
source pretrained/env.sh
```

## Prepare the dataset

```bash
hf download Capur/IESSEEG --repo-type dataset --local-dir /data/IESSEEG
python -m iesseeg validate --data /data/IESSEEG --work /scratch/iesseeg
python -m iesseeg prepare --data /data/IESSEEG --work /scratch/iesseeg
```

The dataset passes the official BIDS validator with zero errors; remaining
metadata warnings are listed in the [dataset validation summary](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/bids_validation.json).
The CLI `validate` command checks benchmark data consistency, separately from BIDS.

The full EEG download is approximately 119 GB. `validate --metadata-only` can
check the tables before the signals finish downloading. Generated model inputs
and checkpoints require additional space in the work directory.

## Run an experiment

For example, extract and fit the two diagnostic qEEG feature sets:

```bash
python -m iesseeg extract-qeeg --task diagnosis \
  --data /data/IESSEEG --work /scratch/iesseeg --device cuda:0
python -m iesseeg qeeg --task diagnosis \
  --data /data/IESSEEG --work /scratch/iesseeg
```

Commands for all benchmark tasks, frozen probes, fine-tuning, and the
Section 6 PLS5 analysis are in the [reproduction guide](docs/REPRODUCING.md).

## Evaluation

Diagnosis predictions are evaluated on simulated routine EEG segments against
patient diagnosis labels and clinician diagnosis labels. Response predictions
are evaluated on complete original recordings. Foundation-model probabilities
are averaged over all complete input windows. The main results use a probability
threshold of 0.5.

AUROC is the mean across five test folds. Confidence intervals follow the
cross-validated influence-function approach of LeDell et al. (2015), with
patient clustering. Balanced accuracy and the additional classification metrics
use the implemented patient-cluster influence functions. The Section 6 analysis
uses the released sampled-segment coordinates and reports segment-level metrics.

```bash
python -m iesseeg metrics --data /data/IESSEEG --work /scratch/iesseeg
pytest -q
```

The metrics command exports AUROC, balanced accuracy, accuracy, sensitivity,
specificity, precision, F1, and average precision, with confidence intervals,
from the available held-out predictions. It does not retrain models.

## Repository layout

| Directory | Contents |
| --- | --- |
| `iesseeg/` | Public CLI, released-data adapter, qEEG/context experiments, metrics |
| `iesseeg_paper/` | Signal features, fixed folds, statistical implementations |
| `experiments/` | Training and inference kernels used by the CLI |
| `baselines_reference/` | Model-specific adapters and attributed upstream implementations |
| `preprocessing_reference/` | Native model input preparation |
| `encoders/` | Frozen foundation-model feature extraction |
| `configs/` | Response training recipes |
| `tests/` | Feature, grouping, inference, and uncertainty tests |

## Attribution

Please cite the IESSEEG paper when using the dataset or benchmark. Author
citation metadata will be added after anonymous review. The dataset is released
separately under CC BY 4.0. Software and upstream model notices are described in
[THIRD_PARTY.md](THIRD_PARTY.md); no pretrained or fine-tuned weights are included.
