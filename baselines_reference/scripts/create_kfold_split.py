#!/usr/bin/env python
"""
create_kfold_split.py

Replaces the fixed 2-fold (Set A / Set B) train-test split used in
preprocessing_release/meta_data_preprocess/ with a stratified k-fold
split, to reduce the cross-fold performance variance observed with
the original 2-fold protocol (small per-fold N -> a handful of
subjects flipping swings balanced accuracy by several points).

Lives in kfold_split/ (separate from preprocessing_release/) so the
existing release pipeline and its outputs are untouched while this is
being evaluated.

For each task, patients are split into `n_folds` stratified,
patient-disjoint folds (StratifiedKFold on the task's binary label,
patient-level, no patient ever appears in both train and test within
the same fold). For fold i:
  - train = all "PRE" Clinical-Clip rows (final_short_merged.csv)
            belonging to patients NOT in fold i
  - test  = all Routine-Clip rows (final_test.csv) belonging to
            patients IN fold i

This mirrors the original scripts' train/test *source* convention
(train draws from the Clinical-Clip pool, test draws from the
Routine-Clip pool) while replacing the split *mechanism* (2-fold ->
k-fold).

Outputs, per task, under --out_dir:
  fold_manifest.csv          patient_id,label,fold   (one row per patient)
  fold_{i}/train.csv         Clinical-Clip rows for fold i's training patients
  fold_{i}/test.csv          Routine-Clip rows for fold i's test patients

A leakage assertion (no patient_id appears in both a fold's train and
test set) is run automatically before anything is written.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TASK_TO_LABEL_COLUMN = {
    "case_control": "case_control_label",
    "immediate_responder": "immediate_responder",
    "meaningful_responder": "meaningful_responder",
}


def load_task_frames(merged_csv, test_csv, task):
    label_column = TASK_TO_LABEL_COLUMN[task]

    df_merged = pd.read_csv(merged_csv)
    df_merged = df_merged[df_merged["pre_post_treatment_label"] == "PRE"].copy()

    df_test = pd.read_csv(test_csv)

    if label_column not in df_merged.columns:
        raise ValueError(f"merged_csv must contain '{label_column}' column.")
    if "patient_id" not in df_merged.columns or "patient_id" not in df_test.columns:
        raise ValueError("Both merged_csv and test_csv must contain 'patient_id'.")

    if task != "case_control":
        # treatment-response tasks are restricted to case subjects with a
        # definitive Responder / Non-responder outcome
        df_merged = df_merged[df_merged[label_column].isin(["Responder", "Non-responder"])].copy()
        df_test = df_test[df_test[label_column].isin(["Responder", "Non-responder"])].copy()

    return df_merged, df_test, label_column


def build_patient_label_table(df_merged, label_column):
    patient_label_df = df_merged[["patient_id", label_column]].drop_duplicates()
    if patient_label_df["patient_id"].duplicated().any():
        dup = patient_label_df[patient_label_df["patient_id"].duplicated(keep=False)]
        raise ValueError(
            f"Some patient_id have more than one distinct '{label_column}' value, "
            f"labels must be a per-patient attribute:\n{dup}"
        )
    return patient_label_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_csv", type=str, required=True,
                         help="Path to final_short_merged.csv (Clinical-Clip / train pool).")
    parser.add_argument("--test_csv", type=str, required=True,
                         help="Path to final_test.csv (Routine-Clip / test pool).")
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Output directory for this task's fold files.")
    parser.add_argument("--task", type=str, required=True,
                         choices=list(TASK_TO_LABEL_COLUMN.keys()),
                         help="Which benchmark task to split.")
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of stratified folds (default: 5).")
    parser.add_argument("--random_state", type=int, default=42,
                         help="Random seed for reproducibility.")
    args = parser.parse_args()

    np.random.seed(args.random_state)

    df_merged, df_test, label_column = load_task_frames(args.merged_csv, args.test_csv, args.task)
    patient_label_df = build_patient_label_table(df_merged, label_column)

    patient_ids = patient_label_df["patient_id"].values
    labels = patient_label_df[label_column].values

    if len(patient_ids) < args.n_folds:
        raise ValueError(
            f"n_folds={args.n_folds} exceeds patient count={len(patient_ids)} for task={args.task}."
        )

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)

    os.makedirs(args.out_dir, exist_ok=True)

    manifest_rows = []
    fold_summaries = []

    for fold_idx, (train_pos, test_pos) in enumerate(skf.split(patient_ids, labels)):
        train_patient_ids = set(patient_ids[train_pos])
        test_patient_ids = set(patient_ids[test_pos])

        # leakage assertion: a patient must never be in both sides of a fold
        overlap = train_patient_ids & test_patient_ids
        assert not overlap, f"fold {fold_idx}: patient overlap between train/test: {overlap}"

        train_df = df_merged[df_merged["patient_id"].isin(train_patient_ids)].copy().reset_index(drop=True)
        test_df = df_test[df_test["patient_id"].isin(test_patient_ids)].copy().reset_index(drop=True)

        fold_dir = os.path.join(args.out_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_out = os.path.join(fold_dir, "train.csv")
        test_out = os.path.join(fold_dir, "test.csv")
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        for pid in test_patient_ids:
            lbl = patient_label_df.loc[patient_label_df["patient_id"] == pid, label_column].iloc[0]
            manifest_rows.append({"patient_id": pid, "label": lbl, "fold": fold_idx})

        train_counts = train_df.drop_duplicates("patient_id")[label_column].value_counts().to_dict()
        test_counts = test_df.drop_duplicates("patient_id")[label_column].value_counts().to_dict()
        fold_summaries.append(
            f"fold {fold_idx}: train patients={len(train_patient_ids)} {train_counts} "
            f"| test patients={len(test_patient_ids)} {test_counts} "
            f"| train rows={train_df.shape[0]} test rows={test_df.shape[0]}"
        )
        print(f"fold_{fold_idx} => {train_out} (train, {train_df.shape}), {test_out} (test, {test_df.shape})")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["fold", "patient_id"]).reset_index(drop=True)
    manifest_path = os.path.join(args.out_dir, "fold_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nmanifest => {manifest_path}, shape={manifest_df.shape}")

    # sanity: every patient assigned to exactly one test fold, all patients covered
    assert set(manifest_df["patient_id"]) == set(patient_ids), "manifest is missing some patients"
    assert manifest_df["patient_id"].duplicated().sum() == 0, "a patient was assigned to more than one test fold"

    print(f"\n=== {args.task} ({label_column}), n_folds={args.n_folds}, seed={args.random_state} ===")
    for line in fold_summaries:
        print(" ", line)


if __name__ == "__main__":
    main()
