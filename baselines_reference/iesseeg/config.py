"""Path and protocol configuration for the IESSEEG benchmark.

Every path the benchmark needs is resolved here, from environment
variables with sensible defaults, so that no script in the repository
carries a machine-specific absolute path. Users set the roots once:

    export IESSEEG_DATA_ROOT=/path/to/iesseeg          # preprocessed .npz trees
    export IESSEEG_SPLIT_ROOT=/path/to/splits          # optional, defaults to ./splits

Preprocessed data is expected in the per-model layout produced by the
the original IESSEEG preprocessing layout:

    $IESSEEG_DATA_ROOT/
        scalp_eeg_data_200HZ_np_format/          # 22-ch bipolar, GBDT + CNN/ViT
        scalp_eeg_data_200HZ_np_format_biot/     # 18-ch BIOT montage
        scalp_eeg_data_200HZ_np_format_labram/   # 19-ch TUEG montage
        scalp_eeg_data_200HZ_np_format_cbramod/  # 19-ch TUEG montage
        baseline_test/                           # Routine-Clip test recordings
        final_test.csv                           # expert clip-level labels
"""

import os

TASKS = ("case_control", "immediate_responder", "meaningful_responder")

# Column in the split CSVs carrying each task's binary target.
TASK_LABEL_COLUMN = {
    "case_control": "case_control_label",
    "immediate_responder": "immediate_responder",
    "meaningful_responder": "meaningful_responder",
}

# Human-readable task names, used in generated tables and logs.
TASK_DISPLAY = {
    "case_control": "Infantile Spasm Diagnosis",
    "immediate_responder": "Immediate Treatment Response Prediction",
    "meaningful_responder": "Sustained Treatment Response Prediction",
}

# Each model consumes its own montage/preprocessing tree, for both the
# Clinical-Clip training pool and the Routine-Clip evaluation pool.
MODEL_DATA_SUBDIR = {
    "handcrafted": "scalp_eeg_data_200HZ_np_format",
    "cnn_resnet": "scalp_eeg_data_200HZ_np_format",
    "cnn_vit": "scalp_eeg_data_200HZ_np_format",
    "biot": "scalp_eeg_data_200HZ_np_format_biot",
    "labram": "scalp_eeg_data_200HZ_np_format_labram",
    "cbramod": "scalp_eeg_data_200HZ_np_format_cbramod",
    # LUNA consumes the same 22-channel bipolar tree as the in-house
    # baselines; its montage reordering and 200->256 Hz resampling happen
    # at load time rather than in a separate preprocessing pass.
    "luna": "scalp_eeg_data_200HZ_np_format",
    # EEGPT wants a referential 10-20 montage, which is the tree the
    # TUEG-style models already use.
    "eegpt": "scalp_eeg_data_200HZ_np_format_labram",
    # REVE is trained at 200 Hz, our native rate, and takes electrode
    # positions explicitly, so it consumes the referential tree as-is.
    "reve": "scalp_eeg_data_200HZ_np_format_labram",
    "codebrain": "scalp_eeg_data_200HZ_np_format_labram",
    "csbrain": "scalp_eeg_data_200HZ_np_format_labram",
}

# Which baselines/ directory holds each model's runner scripts. Two
# models can share a directory when one implementation serves both
# (the CNN directory hosts the ResNet and the ViT).
MODEL_BASELINE_DIR = {
    "handcrafted": "handcrafted",
    "cnn_resnet": "cnn",
    "cnn_vit": "cnn",
    "biot": "biot",
    "labram": "labram",
    "cbramod": "cbramod",
    "luna": "luna",
    "eegpt": "eegpt",
    "reve": "reve",
    "codebrain": "codebrain",
    "csbrain": "csbrain",
}

MODEL_TEST_SUBDIR = {
    "handcrafted": "baseline_test",
    "cnn_resnet": "baseline_test",
    "cnn_vit": "baseline_test",
    "biot": "biot_test",
    "labram": "labram_test",
    "cbramod": "cbramod_test",
    "luna": "baseline_test",
    "eegpt": "labram_test",
    "reve": "labram_test",
    "codebrain": "labram_test",
    "csbrain": "labram_test",
}

N_FOLDS = 5

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root():
    return _REPO_ROOT


def data_root():
    """Root of the preprocessed IESSEEG trees."""
    root = os.environ.get("IESSEEG_DATA_ROOT")
    if not root:
        raise RuntimeError(
            "IESSEEG_DATA_ROOT is not set. Point it at the directory holding the "
            "preprocessed data trees (see iesseeg/config.py for the expected layout)."
        )
    return root


def split_root():
    """Root of the fixed fold manifests shipped with this repository."""
    return os.environ.get("IESSEEG_SPLIT_ROOT", os.path.join(_REPO_ROOT, "splits"))


def model_data_dir(model):
    """Preprocessed training data directory for one model."""
    if model not in MODEL_DATA_SUBDIR:
        raise KeyError(f"Unknown model '{model}'. Known: {sorted(MODEL_DATA_SUBDIR)}")
    return os.path.join(data_root(), MODEL_DATA_SUBDIR[model])


def test_data_dir(model):
    """Routine-Clip recordings for one model's montage, used at evaluation."""
    if model not in MODEL_TEST_SUBDIR:
        raise KeyError(f"Unknown model '{model}'. Known: {sorted(MODEL_TEST_SUBDIR)}")
    return os.path.join(data_root(), MODEL_TEST_SUBDIR[model])


def human_label_meta():
    """Clip-level expert consensus labels, the ground truth for Task 1."""
    return os.path.join(data_root(), "final_test.csv")


def fold_csv(task, fold, split):
    """Path to one fold's train or test manifest."""
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Known: {list(TASKS)}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    return os.path.join(split_root(), task, f"fold_{fold}", f"{split}.csv")


def fold_manifest(task):
    """Path to a task's patient-level fold assignment table."""
    return os.path.join(split_root(), task, "fold_manifest.csv")
