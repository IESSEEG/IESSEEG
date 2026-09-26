#!/usr/bin/env python
"""Test a complete PRE/POST by awake/sleep qEEG feature grid.

The source-paper analysis selected four condition/state/feature cells. This
script evaluates all twelve cells formed by two conditions, two vigilance
states, and three qEEG quantities. It reports both individual-clip values and
the mean of the two matching clips from each patient. Response labels are
permuted across patients, never across clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd

from analyze_response_feature_associations import (
    ENDPOINTS,
    cluster_permutation_p_values,
    holm_adjust,
    response_auc,
)
from rajaraman2024_contract import PLI_CONNECTIVITY_POLICY, PLI_EPOCH_POLICY
from reproduce_rajaraman2024 import normalize_recording_id, response_value


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

CONDITIONS = ("PRE", "POST")
STATES = ("AWAKE", "SLEEP")
FEATURES = (
    ("beta_dfa_intercept_mean", "beta DFA intercept"),
    ("beta_entropy_mean", "beta Shannon entropy"),
    ("connectivity_percent_raw_pli", "delta PLI connectivity"),
)
SOURCE_SELECTED = {
    ("PRE", "AWAKE", "beta_dfa_intercept_mean"),
    ("PRE", "AWAKE", "connectivity_percent_raw_pli"),
    ("POST", "AWAKE", "beta_dfa_intercept_mean"),
    ("POST", "SLEEP", "beta_entropy_mean"),
}


def table_name(condition: str, state: str, feature: str, aggregation: str) -> str:
    return "__".join((condition.lower(), state.lower(), feature, aggregation))


def load_feature_grid(
    features_dir: Path,
    metadata_csv: Path,
    label_column: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    metadata["recording_id"] = metadata.short_recording_id.map(normalize_recording_id)
    metadata["label"] = metadata[label_column].map(response_value)
    if len(metadata) != 400 or metadata.patient_id.nunique() != 50:
        raise RuntimeError("Expected 400 Clinical Clips from 50 case patients")

    values: dict[str, list[dict[str, object]]] = {}
    designs: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        for state in STATES:
            for feature, feature_label in FEATURES:
                single_name = table_name(condition, state, feature, "single_clip")
                mean_name = table_name(condition, state, feature, "patient_average")
                selected = (condition, state, feature) in SOURCE_SELECTED
                values[single_name] = []
                designs[single_name] = {
                    "analysis_family": "single_clip",
                    "input": f"{condition} {state.lower()}",
                    "aggregation": "one clip",
                    "quantity": feature_label,
                    "source_selected_cell": selected,
                }
                designs[mean_name] = {
                    "analysis_family": "patient_average",
                    "input": f"{condition} {state.lower()}",
                    "aggregation": f"mean of two {state.lower()} clips",
                    "quantity": feature_label,
                    "source_selected_cell": selected,
                }

    for row in metadata.itertuples(index=False):
        path = features_dir / f"{row.recording_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = [feature for feature, _ in FEATURES if feature not in payload]
        if missing:
            raise RuntimeError(f"{path.name} is missing full-grid features: {missing}")
        if payload.get("pli_epoch_policy") != PLI_EPOCH_POLICY:
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete epoch policy"
            )
        if payload.get("pli_connectivity_policy") != PLI_CONNECTIVITY_POLICY:
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete connectivity policy"
            )
        condition = str(row.pre_post_treatment_label)
        state = str(row.sleep_awake_label)
        common = {
            "patient_id": int(row.patient_id),
            "recording_id": row.recording_id,
            "duration_category": float(row.LeadtimeUKISS),
            "label": int(row.label),
        }
        for feature, _ in FEATURES:
            name = table_name(condition, state, feature, "single_clip")
            values[name].append({**common, "value": float(payload[feature])})

    tables: dict[str, pd.DataFrame] = {}
    for name, rows in values.items():
        table = pd.DataFrame(rows)
        counts = table.groupby("patient_id").size()
        if len(table) != 100 or len(counts) != 50 or not counts.eq(2).all():
            raise RuntimeError(f"{name}: expected two clips from each of 50 patients")
        if table.groupby("patient_id").label.nunique().ne(1).any():
            raise RuntimeError(f"{name}: response labels differ within patient")
        tables[name] = table
        mean_name = name.replace("__single_clip", "__patient_average")
        tables[mean_name] = table.groupby("patient_id", as_index=False).agg(
            label=("label", "first"),
            duration_category=("duration_category", "first"),
            value=("value", "mean"),
        )
    return tables, designs


def summarize_endpoint(
    endpoint: str,
    tables: dict[str, pd.DataFrame],
    designs: dict[str, dict[str, object]],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    p_values = cluster_permutation_p_values(tables, n_permutations, seed)
    rows = []
    for name, table in tables.items():
        design = designs[name]
        responders = table.loc[table.label.eq(1), "value"].to_numpy(float)
        nonresponders = table.loc[table.label.eq(0), "value"].to_numpy(float)
        raw_auc = response_auc(table)
        rows.append(
            {
                "endpoint": endpoint,
                **design,
                "n_patients": table.patient_id.nunique(),
                "n_values": len(table),
                "values_per_patient": int(table.groupby("patient_id").size().iloc[0]),
                "responder_patients": table.loc[
                    table.label.eq(1), "patient_id"
                ].nunique(),
                "nonresponder_patients": table.loc[
                    table.label.eq(0), "patient_id"
                ].nunique(),
                "responder_median": float(np.median(responders)),
                "nonresponder_median": float(np.median(nonresponders)),
                "responder_higher_auc": raw_auc,
                "separation_auc": max(raw_auc, 1.0 - raw_auc),
                "responder_direction": "higher" if raw_auc >= 0.5 else "lower",
                "patient_clustered_permutation_p": p_values[name],
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p_across_24_grid_tests"] = holm_adjust(
        result.patient_clustered_permutation_p.tolist()
    )
    result["significant_after_holm_0_05"] = result.holm_p_across_24_grid_tests.lt(0.05)
    result["permutations"] = n_permutations
    result["seed"] = seed
    return result


def run_analysis(
    metadata_csv: Path,
    raw_features_dir: Path,
    n_permutations: int = 1_000_000,
    seed: int = 20260912,
) -> pd.DataFrame:
    outputs = []
    for endpoint_index, (endpoint, label_column) in enumerate(ENDPOINTS.items()):
        tables, designs = load_feature_grid(
            raw_features_dir, metadata_csv, label_column
        )
        outputs.append(
            summarize_endpoint(
                endpoint,
                tables,
                designs,
                n_permutations,
                seed + endpoint_index,
            )
        )
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--raw-features-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS / "full_qeeg_response_grid.csv",
    )
    args = parser.parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")
    result = run_analysis(
        args.metadata_csv,
        args.raw_features_dir,
        args.permutations,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved aggregate results to {args.output}")


if __name__ == "__main__":
    main()
