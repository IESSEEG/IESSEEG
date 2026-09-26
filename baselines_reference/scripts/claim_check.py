#!/usr/bin/env python
"""Adjudication test for the treatment-response challenge tasks.

The response tasks ship as open challenges rather than a leaderboard: no
current baseline separates from chance, and at 50 subjects the fold
variance is large relative to plausible method differences, so ranking by
mean AUROC would reward noise. This script implements the pre-specified
criterion a method must clear to claim predictive signal on Task 2 or 3:

  Pool the subject-level scores from the five released held-out folds
  (each subject is scored exactly once, by the model that never saw it;
  subject score = mean predicted probability over the subject's Routine
  Clips). Compute the pooled AUROC over the 50 case subjects. Compare it
  to a null built by permuting subject labels WITHIN each released fold
  (10,000 permutations; within-fold permutation respects the stratified
  fold design). Claim predictive signal only if one-sided p < 0.05.

Superiority over another method additionally requires a paired test on
the same subjects; this script reports the signal criterion only.

Usage (against this repo's result-tree layout):
  python scripts/claim_check.py --results_root <dir> [--task immediate_responder]

or against a single pooled predictions CSV with columns
  subject_id, fold, score, label   (label in {0,1}):
  python scripts/claim_check.py --predictions my_method.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iesseeg.evaluation.aggregate import MODEL_RESULT_SUBDIR  # canonical
TASKS = ("immediate_responder", "meaningful_responder")
N_PERM = 10_000


def load_model_task(results_root, model, task, labels_csv):
    meta = pd.read_csv(labels_csv)[
        ["short_recording_id", "patient_id",
         "immediate_responder", "meaningful_responder"]]
    frames = []
    for fold in range(5):
        d = os.path.join(results_root, MODEL_RESULT_SUBDIR[model],
                         f"{task}_fold{fold}")
        hits = [h for h in glob.glob(os.path.join(d, "*inference_results.csv"))
                if not h.endswith("_window.csv")]
        if len(hits) != 1:
            raise SystemExit(f"{model} {task} fold{fold}: expected exactly "
                             f"one inference CSV in {d}, found {len(hits)}")
        f = pd.read_csv(hits[0])
        rid = [c for c in f.columns if "recording_id" in c][0]
        f = f.rename(columns={rid: "short_recording_id"})
        f["fold"] = fold
        frames.append(f[["short_recording_id", "pred_prob", "fold"]])
    df = pd.concat(frames).merge(meta, on="short_recording_id")
    col = {"immediate_responder": "immediate_responder",
           "meaningful_responder": "meaningful_responder"}[task]
    subj = df.groupby("patient_id").agg(
        score=("pred_prob", "mean"), fold=("fold", "first"),
        label=(col, "first"))
    subj["label"] = (subj.label == "Responder").astype(int)
    return subj.reset_index()[["patient_id", "fold", "score", "label"]]


def permutation_test(subj, n_perm=N_PERM, seed=0):
    """One-sided within-fold label-permutation test on pooled AUROC."""
    rng = np.random.default_rng(seed)
    y, s, folds = subj.label.values, subj.score.values, subj.fold.values
    obs = roc_auc_score(y, s)
    idx_by_fold = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    null = np.empty(n_perm)
    yp = y.copy()
    for i in range(n_perm):
        for idx in idx_by_fold:
            yp[idx] = rng.permutation(yp[idx])
        null[i] = roc_auc_score(yp, s)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    bar = float(np.quantile(null, 0.95))
    return obs, p, bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root")
    ap.add_argument("--labels_csv")
    ap.add_argument("--predictions",
                    help="single CSV: subject_id,fold,score,label")
    ap.add_argument("--task", choices=TASKS)
    args = ap.parse_args()
    if not args.predictions and not (args.results_root and args.labels_csv):
        ap.error("provide --predictions, or both --results_root and --labels_csv")

    if args.predictions:
        subj = pd.read_csv(args.predictions).rename(
            columns={"subject_id": "patient_id"})
        obs, p, bar = permutation_test(subj)
        verdict = "PASS" if p < 0.05 else "no claim"
        print(f"pooled AUROC {obs:.3f}  perm p={p:.4f}  "
              f"(null 95th pct {bar:.3f})  -> {verdict}")
        return

    tasks = [args.task] if args.task else list(TASKS)
    rows = []
    for task in tasks:
        for model in MODEL_RESULT_SUBDIR:
            subj = load_model_task(args.results_root, model, task,
                                   args.labels_csv)
            obs, p, bar = permutation_test(subj)
            rows.append(dict(task=task, model=model, pooled_auroc=obs,
                             perm_p=p, null_q95=bar,
                             verdict="PASS" if p < 0.05 else "no claim"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = os.environ.get("IESSEEG_OUT")
    if out:
        df.to_csv(os.path.join(out, "claim_check_baselines.csv"), index=False)


if __name__ == "__main__":
    main()
