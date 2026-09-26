"""Cross-fold aggregation and LaTeX table generation.

Collects the per-fold metrics.json files written by
`iesseeg.evaluation.metrics` and produces (i) a tidy CSV of every
fold-level measurement, (ii) a mean/std summary, and (iii) the result
tables included by the paper.

The paper's tables were once transcribed by hand from scattered result
directories, which is how numbers drifted between a rerun and the
manuscript. Generating them keeps the two in sync: rerun this after any
change to the underlying runs, then recompile the paper.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from .. import config

# Where each model's per-fold result directories live, relative to the
# results root. cnn/ hosts two architectures under one tree, so it
# contributes two entries.
MODEL_RESULT_SUBDIR = {
    "handcrafted": "handcrafted/result/inference",
    "cnn_resnet": "cnn/result/cnn/inference",
    "cnn_vit": "cnn/result/vit/inference",
    "biot": "biot/result/inference",
    "labram": "labram/result/inference",
    "cbramod": "cbramod/result/inference",
    "luna": "luna/result/inference",
    "eegpt": "eegpt/result/inference",
    "reve": "reve/result/inference",
    "codebrain": "codebrain/result/inference",
    "csbrain": "csbrain/result/inference",
}

MODEL_DISPLAY = {
    "handcrafted": "GBDT + Clinical Prior",
    "cnn_resnet": "3D ResNet-18",
    "cnn_vit": "3D ViT",
    "biot": "BIOT",
    "labram": "LaBraM",
    "cbramod": "CBraMod",
    "luna": "LUNA",
    "eegpt": "EEGPT",
    "reve": "REVE",
    "codebrain": "CodeBrain",
    "csbrain": "CSBrain",
}

# Row order used in the paper's tables: interpretable baselines first,
# then from-scratch deep models, then pre-trained foundation models.
MODEL_ORDER = [
    "handcrafted", "cnn_resnet", "cnn_vit", "biot", "labram", "cbramod",
    "luna", "eegpt", "reve", "codebrain", "csbrain",
]

TABLE_SPECS = {
    "case_control": dict(
        latex_file="experiment_results_case_vs_control.tex",
        label="tab:casecontrol",
        caption="Task 1 --- Infantile Spasm Diagnosis Benchmark. "
                "Mean $\\pm$ standard deviation over five subject-wise folds.",
    ),
    "immediate_responder": dict(
        latex_file="experiment_results_immediate.tex",
        label="tab:immediate",
        caption="Task 2 --- Immediate Treatment Response Prediction Benchmark. "
                "Mean $\\pm$ standard deviation over five subject-wise folds.",
    ),
    "meaningful_responder": dict(
        latex_file="experiment_results_sustained.tex",
        label="tab:sustained",
        caption="Task 3 --- Sustained Treatment Response Prediction Benchmark. "
                "Mean $\\pm$ standard deviation over five subject-wise folds.",
    ),
}

REPORTED_METRICS = [
    ("balanced_accuracy", "Acc"),
    ("f1", "F1"),
    ("auroc", "AUROC"),
]

FOLD_DIR_RE = re.compile(
    r"^(?P<task>case_control|immediate_responder|meaningful_responder)_fold(?P<fold>\d+)$"
)


def collect(results_root, models):
    """Read every per-fold metrics.json under the given models' result trees."""
    rows = []
    for model in models:
        result_dir = os.path.join(results_root, MODEL_RESULT_SUBDIR[model])
        if not os.path.isdir(result_dir):
            print(f"  [skip] {model}: no {MODEL_RESULT_SUBDIR[model]}")
            continue

        found = 0
        for entry in sorted(os.listdir(result_dir)):
            match = FOLD_DIR_RE.match(entry)
            if not match:
                continue
            metrics_path = os.path.join(result_dir, entry, "metrics.json")
            if not os.path.isfile(metrics_path):
                print(f"  [warn] {model}/{entry}: no metrics.json (eval not run?)")
                continue

            with open(metrics_path) as handle:
                metrics = json.load(handle)
            row = dict(model=model, task=match["task"], fold=int(match["fold"]))
            for key, _ in REPORTED_METRICS:
                value = metrics.get(key)
                # Metrics are stored pre-rounded as strings; AUROC is null
                # when a fold's test set turned out single-class.
                row[key] = float(value) if value is not None else np.nan
            rows.append(row)
            found += 1
        print(f"  {model}: {found} fold results")

    return pd.DataFrame(rows)


def summarize(df):
    """Mean/std/count per model x task x metric."""
    metric_keys = [key for key, _ in REPORTED_METRICS]
    summary = df.groupby(["model", "task"])[metric_keys].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def format_cell(mean, std):
    """Value with its standard deviation as a subscript.

    An inline "0.900 $\\pm$ 0.112" is wide enough that with nine models and
    three metrics the columns wrap mid-cell. The subscript form is roughly
    half the width, keeps each cell on one line, and is standard in the
    ML literature.
    """
    if np.isnan(mean):
        return "--"
    if np.isnan(std):
        return f"${mean:.3f}$"
    return f"${mean:.3f}_{{\\pm {std:.3f}}}$"


def render_latex_table(summary, task, spec, models):
    headers = " & ".join(label for _, label in REPORTED_METRICS)
    # A plain tabular sizes columns to their content; tabularx would
    # stretch them to a fixed total width and wrap the cells instead.
    column_spec = "l" + "c" * len(REPORTED_METRICS)

    lines = [
        "% ---------------------------------------------------------------",
        f"% {spec['label']}",
        "% GENERATED by iesseeg.evaluation.aggregate -- do not edit by hand;",
        "% rerun the aggregation instead.",
        "% ---------------------------------------------------------------",
        "\\begin{table}[!htbp]",
        "  \\centering \\small",
        f"  \\caption{{{spec['caption']}}}",
        f"  \\label{{{spec['label']}}}",
        f"  \\begin{{tabular}}{{{column_spec}}}",
        "    \\toprule",
        f"    \\textbf{{Model}} & {headers} \\\\ \\midrule",
    ]

    for model in models:
        row = summary[(summary.model == model) & (summary.task == task)]
        if row.empty:
            continue
        row = row.iloc[0]
        cells = [format_cell(row[f"{k}_mean"], row[f"{k}_std"]) for k, _ in REPORTED_METRICS]
        lines.append(f"    {MODEL_DISPLAY[model]:<22} & " + " & ".join(cells) + " \\\\")

    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results_root", default=os.path.join(config.repo_root(), "baselines"),
                        help="Directory holding the per-model result trees.")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--out_dir", default=os.path.join(config.repo_root(), "results"),
                        help="Where to write the aggregated CSVs.")
    parser.add_argument("--latex_dir", default=None,
                        help="If set, write the paper's LaTeX result tables here.")
    args = parser.parse_args()

    print("Collecting per-fold metrics...")
    df = collect(args.results_root, args.models)
    if df.empty:
        raise SystemExit("No metrics found. Run the inference and eval stages first.")

    os.makedirs(args.out_dir, exist_ok=True)
    df = df.sort_values(["task", "model", "fold"])
    per_fold_csv = os.path.join(args.out_dir, "results_all_folds.csv")
    summary_csv = os.path.join(args.out_dir, "results_summary.csv")

    df.to_csv(per_fold_csv, index=False)
    print(f"\nPer-fold metrics -> {per_fold_csv} ({len(df)} rows)")

    summary = summarize(df)
    summary.to_csv(summary_csv, index=False)
    print(f"Summary          -> {summary_csv}")

    # Flag any model x task missing folds, so a partial rerun never
    # silently becomes a headline number.
    count_key = f"{REPORTED_METRICS[0][0]}_count"
    for _, row in summary.iterrows():
        if row[count_key] != config.N_FOLDS:
            print(f"  [warn] {row['model']}/{row['task']}: "
                  f"{int(row[count_key])} folds, expected {config.N_FOLDS}")

    for task in TABLE_SPECS:
        print(f"\n  {config.TASK_DISPLAY[task]}")
        for model in args.models:
            row = summary[(summary.model == model) & (summary.task == task)]
            if row.empty:
                continue
            row = row.iloc[0]
            print(f"    {MODEL_DISPLAY[model]:<22} "
                  f"Acc {row['balanced_accuracy_mean']:.3f} +/- {row['balanced_accuracy_std']:.3f}   "
                  f"F1 {row['f1_mean']:.3f}   AUROC {row['auroc_mean']:.3f}")

    if args.latex_dir:
        os.makedirs(args.latex_dir, exist_ok=True)
        for task, spec in TABLE_SPECS.items():
            path = os.path.join(args.latex_dir, spec["latex_file"])
            with open(path, "w") as handle:
                handle.write(render_latex_table(summary, task, spec, args.models))
            print(f"LaTeX table      -> {path}")


if __name__ == "__main__":
    main()
