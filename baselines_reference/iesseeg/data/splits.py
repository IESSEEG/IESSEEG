"""Loading the fixed subject-wise fold manifests.

The benchmark's evaluation protocol is stratified five-fold subject-wise
cross-validation. The fold assignment is a released artifact (shipped in
splits/), not something each user regenerates, so that numbers reported by
different groups are computed on identical partitions.

Within a fold, training rows come from the Clinical-Clip pool and test rows
from the Routine-Clip pool, and no patient appears on both sides.
"""

import pandas as pd

from .. import config


def encode_labels(meta_df, label_key):
    """Map a task's categorical label column to binary 0/1 targets."""
    if label_key == "case_control_label":
        return meta_df[label_key].apply(lambda x: 1 if x == "CASE" else 0).values
    if label_key in ("immediate_responder", "meaningful_responder"):
        return meta_df[label_key].apply(lambda x: 1 if x == "Responder" else 0).values
    raise ValueError(f"Unknown label key: {label_key}")


def load_fold(task, fold, split):
    """Read one fold manifest and attach encoded binary labels.

    Returns the dataframe with an added `label` column.
    """
    path = config.fold_csv(task, fold, split)
    df = pd.read_csv(path)
    df["label"] = encode_labels(df, config.TASK_LABEL_COLUMN[task])
    return df


def fold_info_list(task, fold, split, recording_id_col="short_recording_id"):
    """Return a fold as the (patient_id, recording_id, label) tuples the
    datasets consume."""
    df = load_fold(task, fold, split)
    if recording_id_col not in df.columns:
        candidates = [c for c in df.columns if "recording_id" in c]
        if not candidates:
            raise KeyError(
                f"No recording id column in {config.fold_csv(task, fold, split)}; "
                f"columns are {list(df.columns)}"
            )
        recording_id_col = candidates[0]
    return list(zip(df["patient_id"], df[recording_id_col].astype(str), df["label"]))


def assert_no_patient_leakage(task, fold):
    """Fail loudly if a patient appears in both sides of a fold.

    Cheap to check and the single most damaging thing that can silently go
    wrong in a subject-wise benchmark, so callers are encouraged to run it
    before training.
    """
    train_patients = set(load_fold(task, fold, "train")["patient_id"])
    test_patients = set(load_fold(task, fold, "test")["patient_id"])
    overlap = train_patients & test_patients
    if overlap:
        raise AssertionError(
            f"Patient leakage in {task} fold {fold}: {sorted(overlap)} appear in "
            f"both train and test."
        )
