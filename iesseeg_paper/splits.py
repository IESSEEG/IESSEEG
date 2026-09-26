"""The frozen, task-specific patient assignments for the IESSEEG benchmark."""

from pathlib import Path
import pandas as pd

TASK_NAMES = {
    "diagnosis": "case_control",
    "immediate": "immediate_responder",
    "sustained": "meaningful_responder",
}


def load_fold_manifest(task):
    """Load published assignments; never generate a new random partition."""
    task = TASK_NAMES.get(task, task)
    if task not in TASK_NAMES.values():
        raise ValueError(f"Unknown benchmark task: {task}")
    path = Path(__file__).parent / "fixed_folds" / f"{task}.csv"
    frame = pd.read_csv(path)
    expected = 100 if task == "case_control" else 50
    if (list(frame.columns) != ["patient_id", "fold"]
            or len(frame) != expected or frame.patient_id.duplicated().any()
            or frame.isna().any().any() or set(frame.fold) != set(range(5))):
        raise ValueError(f"Invalid fixed patient assignments: {path}")
    return frame


def validate_patient_folds(frame, task, require_complete=False):
    """Reject unknown patients and fold drift in metadata or saved predictions."""
    expected = load_fold_manifest(task).set_index("patient_id").fold
    assigned = frame.patient_id.map(expected)
    if assigned.isna().any() or not assigned.eq(frame.fold).all():
        raise ValueError(f"Patient/fold assignments differ from the fixed {task} manifest")
    if require_complete and set(frame.patient_id) != set(expected.index):
        raise ValueError(f"Incomplete patient coverage for {task}")
