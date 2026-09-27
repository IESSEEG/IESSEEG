# Data preparation

Download the [IESSEEG v1.1 release](https://huggingface.co/datasets/Capur/IESSEEG)
after [installing the benchmark](../README.md#installation):

```bash
export DATASET_ROOT=/data/IESSEEG
export BENCHMARK_WORK=/scratch/iesseeg
hf download Capur/IESSEEG --repo-type dataset --local-dir "$DATASET_ROOT"
```

The full download is approximately 119 GB. `DATASET_ROOT` should contain
`participants.tsv` and `recordings.tsv`, rather than point to an individual
participant's directory.

## Reading a recording before the full download

The [dataset card](https://huggingface.co/datasets/Capur/IESSEEG#read-an-eeg-recording)
includes a Python example that downloads one clinician-selected EEG segment
and reads its first ten seconds with MNE. This requires only `huggingface_hub`,
`pandas`, and `mne`; pretrained models and a GPU are not needed to read the data.

The [HF preview](https://huggingface.co/datasets/Capur/IESSEEG) lets you browse
the recording index, participant metadata, and BASED ratings. The preview's
`records` split contains metadata rows, not a benchmark training partition.
The five-fold assignments are stored separately in `splits/`.

## EEG and annotations

| EEG subset | Files | Use in the benchmark |
| --- | ---: | --- |
| Original long EEG recordings | 150 | Pre-treatment and post-treatment response prediction |
| Clinician-selected EEG segments | 600 | Diagnosis training and validation using case pre-treatment and control segments |
| Simulated routine EEG segments | 200 | Diagnosis evaluation |

The recordings come from 50 cases and 50 controls. Each case has a pre-treatment
and a post-treatment long recording; each control has one long recording.

`recordings.tsv` lists EEG file paths and links each segment to its source
recording and participant. `participants.tsv` contains demographics and clinical
metadata. Field definitions are maintained with the data in
[recordings.json](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/recordings.json)
and [participants.json](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/participants.json).
Use `recording_id` for recording joins and `patient_id` for the released fold
assignments. The dataset card includes a
[join example](https://huggingface.co/datasets/Capur/IESSEEG#files-and-identifiers).

| Directory | Contents |
| --- | --- |
| `annotations/` | Patient diagnosis labels, treatment-response labels, sleep/awake labels, clinician diagnosis labels, and BASED scores |
| `metadata/` | Segment-to-recording mappings |
| `splits/` | Five-fold patient assignments and diagnosis recording assignments |
| `sampling/` | Segment coordinates for response-head training and the recording-context analysis |

Sleep/awake labels are available for clinician-selected EEG segments, and
clinician diagnosis labels for simulated routine EEG segments. The 140 BASED
ratings cover a subset of 20 original long recordings. See the
[dataset documentation](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/README.md)
for the full annotation descriptions.

## Preparing the experiment directory

```bash
python -m iesseeg validate --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
python -m iesseeg prepare --data "$DATASET_ROOT" --work "$BENCHMARK_WORK"
```

`validate` checks the cohort counts, participant mappings, fold tables, and
presence of the EEG files. Add `--metadata-only` to check the tables while EEG
files are still downloading. The separate
[BIDS validation report](https://huggingface.co/datasets/Capur/IESSEEG/blob/main/bids_validation.json)
is included with the dataset.

`prepare` creates the input manifests and patient splits used by the experiment
scripts. It links to the downloaded EDF files without copying or modifying them.
Features, fitted models, and predictions are written under `BENCHMARK_WORK`.
Keep the dataset at the same location while using that work directory.

Next, choose an experiment from the [reproduction guide](REPRODUCING.md).
