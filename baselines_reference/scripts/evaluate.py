#!/usr/bin/env python
"""Score benchmark runs and aggregate them across folds.

Walks the per-model result trees, scores every inference CSV that does not
yet have metrics, then writes the aggregated CSVs and (optionally) the
paper's LaTeX tables.

Usage:
  python scripts/evaluate.py                                  # score + aggregate
  python scripts/evaluate.py --latex_dir ../paper/sections/tables
  python scripts/evaluate.py --force                          # rescore everything
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iesseeg import config
from iesseeg.evaluation import evaluate_run
from iesseeg.evaluation.aggregate import (
    MODEL_ORDER, MODEL_RESULT_SUBDIR, TABLE_SPECS, collect, render_latex_table, summarize,
)


def score_pending(results_root, models, force):
    """Run per-fold scoring for every inference CSV lacking metrics.json."""
    scored = skipped = 0
    for model in models:
        result_dir = os.path.join(results_root, MODEL_RESULT_SUBDIR[model])
        if not os.path.isdir(result_dir):
            continue

        for dirpath, _, filenames in os.walk(result_dir):
            for filename in filenames:
                # Matches both "inference_results.csv" and the CNN
                # baselines' "<tag>_inference_results.csv", while excluding
                # the "_window.csv" dumps, which are per-window diagnostics
                # rather than the clip-level predictions being scored.
                if not filename.endswith("inference_results.csv"):
                    continue

                prediction_csv = os.path.join(dirpath, filename)
                if os.path.isfile(os.path.join(dirpath, "metrics.json")) and not force:
                    skipped += 1
                    continue

                print(f"\n=== Scoring {os.path.relpath(prediction_csv, results_root)} ===")
                evaluate_run(
                    prediction_csv=prediction_csv,
                    output_folder=dirpath,
                    human_label_meta=config.human_label_meta(),
                )
                scored += 1

    print(f"\nScored {scored} run(s); {skipped} already had metrics.")


def main():
    repo_root = config.repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results_root", default=os.environ.get("IESSEEG_OUTPUT_ROOT"))
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    parser.add_argument("--out_dir", default=os.path.join(repo_root, "results"))
    parser.add_argument("--latex_dir", default=None,
                        help="If set, write the paper's LaTeX result tables here.")
    parser.add_argument("--force", action="store_true",
                        help="Rescore runs that already have metrics.json.")
    parser.add_argument("--skip_scoring", action="store_true",
                        help="Only aggregate existing metrics.json files.")
    args = parser.parse_args()

    if not args.skip_scoring:
        score_pending(args.results_root, args.models, args.force)

    print("\nAggregating across folds...")
    df = collect(args.results_root, args.models)
    if df.empty:
        raise SystemExit("No metrics found. Run training and inference first.")

    os.makedirs(args.out_dir, exist_ok=True)
    df = df.sort_values(["task", "model", "fold"])
    df.to_csv(os.path.join(args.out_dir, "results_all_folds.csv"), index=False)

    summary = summarize(df)
    summary.to_csv(os.path.join(args.out_dir, "results_summary.csv"), index=False)
    print(f"Wrote aggregated results to {args.out_dir}")

    count_key = "balanced_accuracy_count"
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
            print(f"    {row['model']:<14} Acc {row['balanced_accuracy_mean']:.3f} "
                  f"+/- {row['balanced_accuracy_std']:.3f}   "
                  f"F1 {row['f1_mean']:.3f}   AUROC {row['auroc_mean']:.3f}")

    if args.latex_dir:
        os.makedirs(args.latex_dir, exist_ok=True)
        for task, spec in TABLE_SPECS.items():
            path = os.path.join(args.latex_dir, spec["latex_file"])
            with open(path, "w") as handle:
                handle.write(render_latex_table(summary, task, spec, args.models))
            print(f"LaTeX table -> {path}")


if __name__ == "__main__":
    main()
