"""Patient labels and fixed folds for the released selected-segment metadata."""
import pandas as pd
from .config import ENDPOINTS
from .splits import load_fold_manifest

def normalized(value):
    return str(value).strip().removesuffix(".0").upper()


def metadata(c):
    df = pd.read_csv(
        c["workspace"] / "data/final_short_merged.csv",
        dtype={"short_recording_id": str},
    )
    df["recording_id"] = df.short_recording_id.map(normalized)
    if (
        len(df) != 600
        or df.recording_id.nunique() != 600
        or df.patient_id.nunique() != 100
    ):
        raise ValueError("Expected the complete 600 clinician-selected EEG segments from 100 patients")
    if (
        not df.groupby(["patient_id", "pre_post_treatment_label", "sleep_awake_label"])
        .size()
        .eq(2)
        .all()
    ):
        raise ValueError("Expected two clips per patient, treatment stage and state")
    return df.sort_values(
        ["patient_id", "pre_post_treatment_label", "sleep_awake_label", "recording_id"]
    ).reset_index(drop=True)


def patient_info(c, task):
    df = metadata(c)
    if task != "diagnosis":
        df = df[df.case_control_label.eq("CASE")]
    column = ENDPOINTS[task] if task != "diagnosis" else "case_control_label"
    if not df.groupby("patient_id")[column].nunique().eq(1).all():
        raise ValueError("Inconsistent patient labels")
    pt = df.groupby("patient_id", sort=True).first().reset_index()
    mapping = {"Responder": 1, "Non-responder": 0, "CASE": 1, "CONTROL": 0}
    pt["label"] = pt[column].map(mapping)
    if pt.label.isna().any():
        raise ValueError("Unrecognized target value")
    folds = load_fold_manifest(task)
    pt = pt.merge(
        folds[["patient_id", "fold"]],
        on="patient_id",
        validate="one_to_one",
    )
    expected = 100 if task == "diagnosis" else 50
    if len(pt) != expected or set(pt.fold) != set(range(5)):
        raise ValueError("Incomplete patient/fold mapping")
    return pt[["patient_id", "label", "fold"]]
