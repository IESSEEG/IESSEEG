#!/usr/bin/env python
"""Test published EEG quantities against immediate and sustained response.

The analysis separates three questions: individual clips, a POST awake--sleep
combination, and the source study's patient averaging. Response labels are
always permuted across patients, so repeated observations from one patient
remain together and are never treated as independent patients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from reproduce_rajaraman2024 import normalize_recording_id, response_value
from rajaraman2024_contract import PLI_CONNECTIVITY_POLICY, PLI_EPOCH_POLICY


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

ENDPOINTS = {
    "immediate": "immediate_responder",
    "sustained": "meaningful_responder",
}

DESIGNS = {
    "pre_awake_dfa_intercept_beta": (
        "single_clip",
        "PRE awake",
        "one clip",
        "beta DFA intercept",
    ),
    "pre_awake_connectivity_percent": (
        "single_clip",
        "PRE awake",
        "one clip",
        "delta PLI connectivity",
    ),
    "pre_awake_r0": (
        "single_clip",
        "PRE awake",
        "one clip",
        "R0 from that clip's DFA and PLI",
    ),
    "post_sleep_entropy_beta": (
        "single_clip",
        "POST sleep",
        "one clip",
        "beta Shannon entropy",
    ),
    "post_awake_dfa_intercept_beta": (
        "single_clip",
        "POST awake",
        "one clip",
        "beta DFA intercept",
    ),
    "post_awake_sleep_r1_all_pairs": (
        "awake_sleep",
        "POST awake + POST sleep",
        "one awake clip paired with one sleep clip; all four within-patient pairs",
        "R1 from awake DFA and sleep entropy",
    ),
    "pre_awake_r0_two_clip_mean": (
        "patient_average",
        "PRE awake",
        "mean of two awake clips",
        "R0 after within-patient averaging",
    ),
    "post_awake_sleep_r1_patient_mean": (
        "patient_average",
        "POST awake + POST sleep",
        "mean of two awake and two sleep clips",
        "R1 after within-patient averaging",
    ),
}


def load_single_clip_quantities(
    features_dir: Path, metadata_csv: Path, label_column: str
) -> dict[str, pd.DataFrame]:
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    metadata["recording_id"] = metadata.short_recording_id.map(normalize_recording_id)
    metadata["label"] = metadata[label_column].map(response_value)
    if len(metadata) != 400 or metadata.patient_id.nunique() != 50:
        raise RuntimeError("Expected 400 Clinical Clips from 50 case patients")

    rows: dict[str, list[dict[str, object]]] = {
        name: [] for name, design in DESIGNS.items() if design[0] == "single_clip"
    }
    for row in metadata.itertuples(index=False):
        pre_awake = (
            row.pre_post_treatment_label == "PRE"
            and row.sleep_awake_label == "AWAKE"
        )
        post_sleep = (
            row.pre_post_treatment_label == "POST"
            and row.sleep_awake_label == "SLEEP"
        )
        post_awake = (
            row.pre_post_treatment_label == "POST"
            and row.sleep_awake_label == "AWAKE"
        )
        if not (pre_awake or post_sleep or post_awake):
            continue
        path = features_dir / f"{row.recording_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if pre_awake and payload.get("pli_epoch_policy") != PLI_EPOCH_POLICY:
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete epoch policy; "
                "rerun extract_rajaraman2024_raw_features.py --features pli"
            )
        if pre_awake and payload.get("pli_connectivity_policy") != PLI_CONNECTIVITY_POLICY:
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete connectivity policy; "
                "rerun extract_rajaraman2024_raw_features.py --features pli"
            )
        common = {"patient_id": int(row.patient_id), "label": int(row.label)}
        if pre_awake:
            dfa = float(payload["beta_dfa_intercept_mean"])
            connectivity = float(payload["connectivity_percent_raw_pli"])
            rows["pre_awake_dfa_intercept_beta"].append({**common, "value": dfa})
            rows["pre_awake_connectivity_percent"].append(
                {**common, "value": connectivity}
            )
            rows["pre_awake_r0"].append(
                {**common, "value": -2.361 * dfa - 0.051 * connectivity}
            )
        if post_sleep:
            rows["post_sleep_entropy_beta"].append(
                {**common, "value": float(payload["beta_entropy_mean"])}
            )
        if post_awake:
            rows["post_awake_dfa_intercept_beta"].append(
                {**common, "value": float(payload["beta_dfa_intercept_mean"])}
            )

    tables = {name: pd.DataFrame(values) for name, values in rows.items()}
    for name, table in tables.items():
        counts = table.groupby("patient_id").size()
        if len(table) != 100 or len(counts) != 50 or not counts.eq(2).all():
            raise RuntimeError(f"{name}: expected two clips from each of 50 patients")
        if table.groupby("patient_id").label.nunique().ne(1).any():
            raise RuntimeError(f"{name}: response labels differ within patient")
    return tables


def add_awake_sleep_and_patient_averages(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output = dict(tables)
    pre = tables["pre_awake_r0"]
    output["pre_awake_r0_two_clip_mean"] = (
        pre.groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), value=("value", "mean"))
    )

    awake = tables["post_awake_dfa_intercept_beta"].rename(
        columns={"value": "awake_dfa"}
    )
    sleep = tables["post_sleep_entropy_beta"].rename(
        columns={"value": "sleep_entropy"}
    )
    pairs = awake.merge(sleep, on=["patient_id", "label"], validate="many_to_many")
    pairs["value"] = 4.765 * pairs.sleep_entropy - 7.786 * pairs.awake_dfa
    output["post_awake_sleep_r1_all_pairs"] = pairs[
        ["patient_id", "label", "value"]
    ]
    output["post_awake_sleep_r1_patient_mean"] = (
        pairs.groupby("patient_id", as_index=False)
        .agg(label=("label", "first"), value=("value", "mean"))
    )
    return output


def response_auc(table: pd.DataFrame) -> float:
    ranks = rankdata(table.value.to_numpy(float), method="average")
    labels = table.label.to_numpy(int)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    u_statistic = ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def cluster_permutation_p_values(
    tables: dict[str, pd.DataFrame], n_permutations: int, seed: int
) -> dict[str, float]:
    grouped: dict[int, dict[str, pd.DataFrame]] = {}
    for name, table in tables.items():
        counts = table.groupby("patient_id").size()
        if counts.nunique() != 1:
            raise RuntimeError(f"{name}: unequal observation counts across patients")
        grouped.setdefault(int(counts.iloc[0]), {})[name] = table

    rng = np.random.default_rng(seed)
    p_values = {}
    for observations_per_patient, group in grouped.items():
        names = list(group)
        patients = np.sort(group[names[0]].patient_id.unique())
        labels = (
            group[names[0]]
            .groupby("patient_id")
            .label.first()
            .reindex(patients)
            .to_numpy(int)
        )
        n_positive_patients = int(labels.sum())
        n_positive = observations_per_patient * n_positive_patients
        n_negative = observations_per_patient * (
            len(patients) - n_positive_patients
        )

        patient_rank_sums = []
        observed_distance = []
        for name in names:
            table = group[name].copy()
            table["rank_value"] = rankdata(
                table.value.to_numpy(float), method="average"
            )
            rank_sums = (
                table.groupby("patient_id")["rank_value"]
                .sum()
                .reindex(patients)
                .to_numpy(float)
            )
            patient_rank_sums.append(rank_sums)
            observed_distance.append(abs(response_auc(table) - 0.5))
        rank_sum_matrix = np.stack(patient_rank_sums, axis=1)
        observed_distance_array = np.asarray(observed_distance)

        extreme = np.zeros(len(names), dtype=np.int64)
        completed = 0
        while completed < n_permutations:
            batch_size = min(10_000, n_permutations - completed)
            order = np.argsort(rng.random((batch_size, len(patients))), axis=1)
            permuted_labels = np.zeros((batch_size, len(patients)), dtype=float)
            batch_rows = np.arange(batch_size)[:, None]
            permuted_labels[batch_rows, order[:, :n_positive_patients]] = 1.0
            positive_rank_sum = permuted_labels @ rank_sum_matrix
            u_statistic = (
                positive_rank_sum - n_positive * (n_positive + 1) / 2.0
            )
            permuted_auc = u_statistic / (n_positive * n_negative)
            extreme += (
                np.abs(permuted_auc - 0.5) >= observed_distance_array
            ).sum(axis=0)
            completed += batch_size
        for name, count in zip(names, extreme, strict=True):
            p_values[name] = float((count + 1) / (n_permutations + 1))
    return p_values


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize_endpoint(
    endpoint: str,
    tables: dict[str, pd.DataFrame],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    p_values = cluster_permutation_p_values(tables, n_permutations, seed)
    rows = []
    for name, table in tables.items():
        responders = table.loc[table.label.eq(1), "value"].to_numpy(float)
        nonresponders = table.loc[table.label.eq(0), "value"].to_numpy(float)
        raw_auc = response_auc(table)
        analysis_family, input_data, aggregation, quantity = DESIGNS[name]
        rows.append(
            {
                "endpoint": endpoint,
                "analysis_family": analysis_family,
                "input": input_data,
                "aggregation": aggregation,
                "quantity": quantity,
                "n_patients": table.patient_id.nunique(),
                "n_values": len(table),
                "values_per_patient": int(
                    table.groupby("patient_id").size().iloc[0]
                ),
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
    result["holm_p_within_endpoint"] = holm_adjust(
        result.patient_clustered_permutation_p.tolist()
    )
    result["significant_after_holm_0_05"] = result.holm_p_within_endpoint.lt(
        0.05
    )
    result["permutations"] = n_permutations
    result["seed"] = seed
    return result


def run_analysis(
    metadata_csv: Path,
    raw_features_dir: Path,
    n_permutations: int = 1_000_000,
    seed: int = 20260910,
) -> pd.DataFrame:
    outputs = []
    for endpoint_index, (endpoint, label_column) in enumerate(ENDPOINTS.items()):
        single = load_single_clip_quantities(
            raw_features_dir, metadata_csv, label_column
        )
        tables = add_awake_sleep_and_patient_averages(single)
        outputs.append(
            summarize_endpoint(
                endpoint,
                tables,
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
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS / "clip_and_state_response_associations.csv",
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
