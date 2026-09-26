# IESSEEG

Code for **IESSEEG: An Open EEG Dataset Towards Better Understanding of
Infantile Epileptic Spasms Syndrome**.

[Dataset](https://huggingface.co/datasets/Capur/IESSEEG) ·
[Reproduction guide](docs/REPRODUCING.md) ·
[Model weights](docs/MODELS.md) ·
[Validation status](docs/VALIDATION.md) ·
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

Python 3.11 was used for the reported experiments. Install a CUDA-compatible
PyTorch in your environment, then install this checkout in editable mode:

```bash
git clone https://github.com/IESSEEG/IESSEEG.git
cd IESSEEG
pip install -e '.[eeg,test]'
pip install --no-deps pyhealth==1.1.6
```

PyHealth is installed separately because its package metadata pins pandas below
2; the benchmark uses only its metrics functions, verified with pandas 2.3.1.

The exact versions of the principal packages used for the experiments are
recorded in `requirements-tested.txt`. EEGPT uses a separate interpreter with
Braindecode 1.7.0; see [model setup](docs/MODELS.md). For model-specific diagnosis full fine-tuning, also install
`pip install -r requirements-models.txt`. Pretrained model weights
are downloaded separately from their upstream providers.

## Prepare the dataset

```bash
hf download Capur/IESSEEG --repo-type dataset --local-dir /data/IESSEEG
python -m iesseeg validate --data /data/IESSEEG --work /scratch/iesseeg
python -m iesseeg prepare --data /data/IESSEEG --work /scratch/iesseeg
```

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

Commands for all current-paper tasks, frozen probes, fine-tuning, and the
Section 6 PLS5 analysis are in the [reproduction guide](docs/REPRODUCING.md).
The [validation page](docs/VALIDATION.md) records the numerical checks and
model-specific internal validation procedures.

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
| `legacy/analysis/` | Shared implementation dependencies retained for numerical continuity |
| `configs/` | Response training recipes |
| `tests/` | Feature, grouping, inference, and uncertainty tests |

## Attribution

Please cite the IESSEEG paper when using the dataset or benchmark. Author
citation metadata will be added after anonymous review. The dataset is released
separately under CC BY 4.0. Software and upstream model notices are described in
[THIRD_PARTY.md](THIRD_PARTY.md); no pretrained or fine-tuned weights are included.
