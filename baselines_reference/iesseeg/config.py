"""Paths and label columns used by the model-specific diagnosis adapters.

The public CLI sets IESSEEG_DATA_ROOT, IESSEEG_SPLIT_ROOT, and output paths
from the requested dataset and work directories.
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
# clinician-selected training segments and simulated routine test segments.
MODEL_DATA_SUBDIR = {
    "biot": "scalp_eeg_data_200HZ_np_format_biot",
    "labram": "scalp_eeg_data_200HZ_np_format_labram",
    "cbramod": "scalp_eeg_data_200HZ_np_format_cbramod",
    # LUNA montage reordering and 200->256 Hz resampling happen
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

# Directory containing each model's runner scripts.
MODEL_BASELINE_DIR = {
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
    """Clinician diagnosis labels for simulated routine EEG segments."""
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
