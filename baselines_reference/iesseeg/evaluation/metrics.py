"""Clip-level metrics for one model x task x fold run.

Consumes an inference CSV (one row per evaluated clip, carrying at least a
recording id, a predicted label and a predicted probability) and writes
metrics.json plus a confusion-matrix figure beside it.

Task 1 is scored against the clip-level expert consensus labels
(`human_label` in the released test metadata) rather than the subject-level
diagnosis, because a randomly positioned Routine Clip from a confirmed IESS
patient need not itself display hypsarrhythmia. Tasks 2 and 3 have no
clip-level counterpart and are scored against the subject-level outcome.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, recall_score, roc_auc_score,
)

TASK_FROM_DIRNAME = (
    ("case_control", lambda name: "case" in name and "control" in name),
    ("immediate_responder", lambda name: "immediate_responder" in name),
    ("meaningful_responder", lambda name: "meaningful_responder" in name),
)


def infer_task_from_path(prediction_csv):
    """Identify the task from the run directory name.

    Accepts both the k-fold naming ("case_control_fold0") and the legacy
    two-fold naming ("A_case_vs_control").
    """
    dirname = os.path.basename(os.path.dirname(prediction_csv))
    for task, predicate in TASK_FROM_DIRNAME:
        if predicate(dirname):
            return task
    raise ValueError(
        f"Could not determine task from result directory '{dirname}'. Expected the "
        f"name to contain 'case'+'control', 'immediate_responder', or "
        f"'meaningful_responder'."
    )


def evaluate_run(prediction_csv, output_folder, human_label_meta, task=None):
    """Score one run and persist metrics.json + confusion_matrix.png."""
    task = task or infer_task_from_path(prediction_csv)

    df = pd.read_csv(prediction_csv)
    recording_id_cols = [c for c in df.columns if "recording_id" in c]
    if not recording_id_cols:
        raise KeyError(f"No recording id column in {prediction_csv}")
    df = df.rename(columns={recording_id_cols[0]: "short_recording_id"})
    df["short_recording_id"] = df["short_recording_id"].astype(str)

    if task == "case_control":
        human_df = pd.read_csv(human_label_meta)[["short_recording_id", "human_label"]]
        human_df["short_recording_id"] = human_df["short_recording_id"].astype(str)
        df = df.merge(human_df, on="short_recording_id", how="left")
        y_true = df["human_label"]
    else:
        y_true = df["known_label"]

    keep = y_true.notna()
    if not keep.all():
        print(f"[warn] dropping {(~keep).sum()} rows without a ground-truth label")
    df, y_true = df[keep], y_true[keep]

    y_true = y_true.astype(int)
    y_pred = df["pred_label"].astype(int)
    y_prob = df["pred_prob"].astype(float) if "pred_prob" in df.columns else None

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    # A fold whose held-out subjects happen to share one label has no
    # defined AUROC; record it as missing rather than failing the run.
    auroc = None
    if y_prob is not None and y_true.nunique() > 1:
        auroc = roc_auc_score(y_true, y_prob)

    print(f"\nTask: {task}   (n = {len(y_true)})")
    print(f"AUROC             = {auroc:.4f}" if auroc is not None else "AUROC             = N/A")
    print(f"Balanced Accuracy = {bal_acc:.4f}")
    print(f"Sensitivity       = {sensitivity:.4f}")
    print(f"Specificity       = {specificity:.4f}")
    print(f"F1-score          = {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    os.makedirs(output_folder, exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title(f"Confusion Matrix ({task})")
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)

    fmt = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    metrics = dict(
        task=task,
        n=int(len(y_true)),
        auroc=fmt(auroc),
        balanced_accuracy=fmt(bal_acc),
        sensitivity=fmt(sensitivity),
        specificity=fmt(specificity),
        f1=fmt(f1),
    )
    with open(os.path.join(output_folder, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nAll outputs written to: {os.path.abspath(output_folder)}")
    return metrics
