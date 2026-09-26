#!/usr/bin/env python
"""Evaluate every response-relevant derived metric defined in Rajaraman et al.

The table includes the published EEG scores R0 and R1, their duration-adjusted
response probabilities, the three PRE-to-POST feature changes available from
the reproduced qEEG quantities, and the published relapse score rho. Rho is
included for completeness but is marked as a relapse-time metric; applying it
to a binary response endpoint is exploratory and is not a reproduction of the
paper's survival analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.special import expit

from analyze_full_qeeg_response_grid import load_feature_grid, table_name
from analyze_response_feature_associations import (
    ENDPOINTS,
    cluster_permutation_p_values,
    holm_adjust,
    response_auc,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

DFA = "beta_dfa_intercept_mean"
ENTROPY = "beta_entropy_mean"
PLI = "connectivity_percent_raw_pli"


def feature_table(
    base: dict[str, pd.DataFrame],
    condition: str,
    state: str,
    feature: str,
    aggregation: str,
) -> pd.DataFrame:
    return base[table_name(condition, state, feature, aggregation)].copy()


def patient_feature(
    base: dict[str, pd.DataFrame], condition: str, state: str, feature: str
) -> pd.DataFrame:
    return feature_table(base, condition, state, feature, "patient_average")


def register(
    tables: dict[str, pd.DataFrame],
    designs: dict[str, dict[str, object]],
    name: str,
    table: pd.DataFrame,
    *,
    metric: str,
    input_data: str,
    aggregation: str,
    metric_family: str,
    source_target: str,
    source_formula: str,
    matches_source_patient_construction: bool,
) -> None:
    tables[name] = table[["patient_id", "label", "value"]].copy()
    designs[name] = {
        "analysis_family": metric_family,
        "input": input_data,
        "aggregation": aggregation,
        "quantity": metric,
        "source_target": source_target,
        "source_formula": source_formula,
        "matches_source_patient_construction": matches_source_patient_construction,
    }


def same_clip(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(
        right,
        on=["patient_id", "recording_id", "duration_category", "label"],
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )


def within_patient_pairs(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(
        right,
        on=["patient_id", "duration_category", "label"],
        how="inner",
        validate="many_to_many",
        suffixes=("_left", "_right"),
    )


def build_derived_tables(
    base: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    tables: dict[str, pd.DataFrame] = {}
    designs: dict[str, dict[str, object]] = {}

    pre_dfa_clip = feature_table(base, "PRE", "AWAKE", DFA, "single_clip")
    pre_pli_clip = feature_table(base, "PRE", "AWAKE", PLI, "single_clip")
    pre_clip = same_clip(pre_dfa_clip, pre_pli_clip)
    pre_clip["r0"] = -2.361 * pre_clip.value_left - 0.051 * pre_clip.value_right
    pre_clip["p0"] = expit(2.010 - 0.455 * pre_clip.duration_category + pre_clip.r0)

    pre_dfa_patient = patient_feature(base, "PRE", "AWAKE", DFA)
    pre_pli_patient = patient_feature(base, "PRE", "AWAKE", PLI)
    pre_patient = pre_dfa_patient.merge(
        pre_pli_patient,
        on=["patient_id", "duration_category", "label"],
        validate="one_to_one",
        suffixes=("_dfa", "_pli"),
    )
    pre_patient["r0"] = -2.361 * pre_patient.value_dfa - 0.051 * pre_patient.value_pli
    pre_patient["p0"] = expit(
        2.010 - 0.455 * pre_patient.duration_category + pre_patient.r0
    )

    for suffix, frame, aggregation, matches_source in (
        ("clip", pre_clip, "one PRE awake clip", False),
        (
            "patient",
            pre_patient,
            "mean of two PRE awake clips",
            True,
        ),
    ):
        for metric_name, column, family, formula in (
            (
                "R0",
                "r0",
                "published EEG response score",
                "-2.361*PRE-awake DFA - 0.051*PRE-awake PLI",
            ),
            (
                "P0",
                "p0",
                "published duration-adjusted response probability",
                "sigmoid(2.010 - 0.455*duration + R0)",
            ),
        ):
            table = frame.assign(value=frame[column])
            register(
                tables,
                designs,
                f"{metric_name.lower()}_{suffix}",
                table,
                metric=metric_name,
                input_data="PRE awake",
                aggregation=aggregation,
                metric_family=family,
                source_target="sustained response",
                source_formula=formula,
                matches_source_patient_construction=matches_source,
            )

    post_dfa_clip = feature_table(base, "POST", "AWAKE", DFA, "single_clip")
    post_entropy_clip = feature_table(base, "POST", "SLEEP", ENTROPY, "single_clip")
    post_pairs = within_patient_pairs(post_dfa_clip, post_entropy_clip)
    post_pairs["r1"] = 4.765 * post_pairs.value_right - 7.786 * post_pairs.value_left
    post_pairs["p1"] = expit(
        -37.433
        - 0.571 * post_pairs.duration_category
        + 4.765 * post_pairs.value_right
        - 7.781 * post_pairs.value_left
    )
    post_pairs["rho"] = 4.008 * post_pairs.value_left - 1.661 * post_pairs.value_right

    post_dfa_patient = patient_feature(base, "POST", "AWAKE", DFA)
    post_entropy_patient = patient_feature(base, "POST", "SLEEP", ENTROPY)
    post_patient = post_dfa_patient.merge(
        post_entropy_patient,
        on=["patient_id", "duration_category", "label"],
        validate="one_to_one",
        suffixes=("_dfa", "_entropy"),
    )
    post_patient["r1"] = (
        4.765 * post_patient.value_entropy - 7.786 * post_patient.value_dfa
    )
    post_patient["p1"] = expit(
        -37.433
        - 0.571 * post_patient.duration_category
        + 4.765 * post_patient.value_entropy
        - 7.781 * post_patient.value_dfa
    )
    post_patient["rho"] = (
        4.008 * post_patient.value_dfa - 1.661 * post_patient.value_entropy
    )

    for suffix, frame, aggregation, matches_source in (
        (
            "pairs",
            post_pairs,
            "all four within-patient POST awake/sleep clip pairs",
            False,
        ),
        (
            "patient",
            post_patient,
            "mean of two POST awake and two POST sleep clips",
            True,
        ),
    ):
        for metric_name, column, family, target, formula in (
            (
                "R1",
                "r1",
                "published EEG response score",
                "sustained response",
                "4.765*POST-sleep entropy - 7.786*POST-awake DFA",
            ),
            (
                "P1",
                "p1",
                "published duration-adjusted response probability",
                "sustained response",
                "sigmoid(-37.433 - 0.571*duration + 4.765*entropy - 7.781*DFA)",
            ),
            (
                "rho",
                "rho",
                "published relapse metric",
                "relapse time among immediate responders",
                "4.008*POST-awake DFA - 1.661*POST-sleep entropy",
            ),
        ):
            table = frame.assign(value=frame[column])
            register(
                tables,
                designs,
                f"{metric_name.lower()}_{suffix}",
                table,
                metric=metric_name,
                input_data="POST awake + POST sleep",
                aggregation=aggregation,
                metric_family=family,
                source_target=target,
                source_formula=formula,
                matches_source_patient_construction=matches_source,
            )

    change_specs = (
        ("awake DFA change", "AWAKE", DFA),
        ("awake PLI change", "AWAKE", PLI),
        ("sleep entropy change", "SLEEP", ENTROPY),
    )
    for metric_name, state, feature in change_specs:
        pre_clip_feature = feature_table(base, "PRE", state, feature, "single_clip")
        post_clip_feature = feature_table(base, "POST", state, feature, "single_clip")
        pairs = within_patient_pairs(post_clip_feature, pre_clip_feature)
        pairs["value"] = pairs.value_left - pairs.value_right
        key = metric_name.replace(" ", "_")
        register(
            tables,
            designs,
            f"{key}_pairs",
            pairs,
            metric=f"POST - PRE {metric_name.removesuffix(' change')}",
            input_data=f"PRE + POST {state.lower()}",
            aggregation="all four within-patient PRE/POST clip pairs",
            metric_family="published interval-change analysis",
            source_target="sustained response",
            source_formula="POST feature - PRE feature",
            matches_source_patient_construction=False,
        )

        pre_patient_feature = patient_feature(base, "PRE", state, feature)
        post_patient_feature = patient_feature(base, "POST", state, feature)
        patient = post_patient_feature.merge(
            pre_patient_feature,
            on=["patient_id", "duration_category", "label"],
            validate="one_to_one",
            suffixes=("_post", "_pre"),
        )
        patient["value"] = patient.value_post - patient.value_pre
        register(
            tables,
            designs,
            f"{key}_patient",
            patient,
            metric=f"POST - PRE {metric_name.removesuffix(' change')}",
            input_data=f"PRE + POST {state.lower()}",
            aggregation="difference between patient-level study means",
            metric_family="published interval-change analysis",
            source_target="sustained response",
            source_formula="POST feature - PRE feature",
            matches_source_patient_construction=True,
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
        responders = table.loc[table.label.eq(1), "value"].to_numpy(float)
        nonresponders = table.loc[table.label.eq(0), "value"].to_numpy(float)
        raw_auc = response_auc(table)
        rows.append(
            {
                "endpoint_tested_here": endpoint,
                **designs[name],
                "n_patients": table.patient_id.nunique(),
                "n_values": len(table),
                "values_per_patient": int(table.groupby("patient_id").size().iloc[0]),
                "responder_median": float(np.median(responders)),
                "nonresponder_median": float(np.median(nonresponders)),
                "responder_higher_auc": raw_auc,
                "separation_auc": max(raw_auc, 1.0 - raw_auc),
                "responder_direction": "higher" if raw_auc >= 0.5 else "lower",
                "patient_clustered_permutation_p": p_values[name],
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p_across_16_derived_tests"] = holm_adjust(
        result.patient_clustered_permutation_p.tolist()
    )
    result["significant_after_holm_0_05"] = result.holm_p_across_16_derived_tests.lt(
        0.05
    )
    result["permutations"] = n_permutations
    result["seed"] = seed
    return result


def run_analysis(
    metadata_csv: Path,
    raw_features_dir: Path,
    n_permutations: int = 1_000_000,
    seed: int = 20260914,
) -> pd.DataFrame:
    outputs = []
    for endpoint_index, (endpoint, label_column) in enumerate(ENDPOINTS.items()):
        base, _ = load_feature_grid(raw_features_dir, metadata_csv, label_column)
        tables, designs = build_derived_tables(base)
        if len(tables) != 16:
            raise RuntimeError(f"Expected 16 derived tests, found {len(tables)}")
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


def build_complete_catalog(
    feature_grid: pd.DataFrame, derived: pd.DataFrame
) -> pd.DataFrame:
    """Place individual and paper-derived metrics in one explicit catalog."""
    base = feature_grid.rename(
        columns={
            "endpoint": "endpoint_tested_here",
            "patient_clustered_permutation_p": "raw_p",
            "holm_p_across_24_grid_tests": "holm_adjusted_p",
        }
    ).copy()
    base["metric_category"] = "individual qEEG feature"
    base["source_target"] = "response"
    base["source_formula"] = ""
    base["source_selected_or_defined"] = base.source_selected_cell
    base["matches_source_patient_construction"] = (
        base.source_selected_cell & base.analysis_family.eq("patient_average")
    )
    base["calculation_level"] = np.where(
        base.analysis_family.eq("single_clip"), "clip-based", "patient-based"
    )
    base["adjustment_family"] = "24 individual-feature tests within endpoint"

    combined = derived.rename(
        columns={
            "analysis_family": "metric_category",
            "patient_clustered_permutation_p": "raw_p",
            "holm_p_across_16_derived_tests": "holm_adjusted_p",
        }
    ).copy()
    combined["source_selected_or_defined"] = True
    combined["calculation_level"] = np.where(
        combined.matches_source_patient_construction,
        "patient-based",
        "clip-based",
    )
    combined["adjustment_family"] = "16 paper-derived tests within endpoint"

    columns = [
        "endpoint_tested_here",
        "metric_category",
        "source_target",
        "input",
        "quantity",
        "calculation_level",
        "aggregation",
        "source_selected_or_defined",
        "matches_source_patient_construction",
        "source_formula",
        "n_patients",
        "n_values",
        "values_per_patient",
        "responder_median",
        "nonresponder_median",
        "responder_higher_auc",
        "separation_auc",
        "responder_direction",
        "raw_p",
        "holm_adjusted_p",
        "adjustment_family",
        "significant_after_holm_0_05",
        "permutations",
        "seed",
    ]
    catalog = pd.concat([base[columns], combined[columns]], ignore_index=True)
    return catalog.sort_values(
        [
            "endpoint_tested_here",
            "calculation_level",
            "metric_category",
            "input",
            "quantity",
            "aggregation",
        ],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--raw-features-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--feature-grid-csv",
        type=Path,
        default=LOCAL_RESULTS / "full_qeeg_response_grid.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS / "paper_derived_response_metrics.csv",
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=LOCAL_RESULTS / "complete_response_association_catalog.csv",
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
    if not args.feature_grid_csv.is_file():
        raise FileNotFoundError(args.feature_grid_csv)
    catalog = build_complete_catalog(pd.read_csv(args.feature_grid_csv), result)
    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.catalog_output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved aggregate results to {args.output}")
    print(f"Saved complete catalog to {args.catalog_output}")


if __name__ == "__main__":
    main()
