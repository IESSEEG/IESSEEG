#!/usr/bin/env python
"""Quantify multivariate qEEG association with sustained response.

The Rajaraman et al. reproduction yields three qEEG quantities for every
PRE/POST by awake/sleep cell.  This script averages the two matching Clinical
Clips from each patient, constructs complete and source-selected feature sets,
and asks three different questions:

1. How strongly can a linear combination correlate with the binary response
   label in the same 50 patients?  With one response variable this is the CCA
   coefficient, equivalently the ordinary multiple correlation coefficient.
2. Is there any multivariate dependence, including a nonlinear relationship?
   This is summarized by distance correlation.
3. Does a prespecified ridge-logistic model improve predictions for held-out
   patients?  This is measured with the project's fixed five patient folds,
   AUROC, and held-out log-likelihood gain in bits per patient.

Permutation tests shuffle labels among patients within each fixed fold.  For
the predictive statistic, the logistic model is refitted after every shuffle.
The source-selected feature sets were selected on this same cohort in the
source article, so their P-values remain descriptive and do not constitute an
independent validation of the selection step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analyze_full_qeeg_response_grid import (
    CONDITIONS,
    FEATURES,
    STATES,
    load_feature_grid,
    table_name,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = Path(os.environ.get("IESSEEG_WORKSPACE", "local/input_workspace"))
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

DEFAULT_METADATA = WORKSPACE / "data" / "final_short_merged.csv"
DEFAULT_FEATURES = LOCAL_RESULTS / "raw_features"
DEFAULT_FOLDS = WORKSPACE / "kfold_split" / "data" / "meaningful_responder" / "fold_manifest.csv"
DEFAULT_OUTPUT = LOCAL_RESULTS / "multivariate_qeeg_response_association.csv"
DEFAULT_METADATA_OUTPUT = LOCAL_RESULTS / "multivariate_qeeg_response_association.json"

ENDPOINT_COLUMN = "meaningful_responder"
RIDGE_C = 1.0
EPSILON = 1e-12


def feature_column(condition: str, state: str, feature: str) -> str:
    return f"{condition.lower()}_{state.lower()}_{feature}"


def build_patient_table(features_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    """Return one row per patient with all twelve reproduced qEEG features."""
    tables, _ = load_feature_grid(features_dir, metadata_csv, ENDPOINT_COLUMN)
    patient_table: pd.DataFrame | None = None
    keys = ["patient_id", "label", "duration_category"]
    for condition in CONDITIONS:
        for state in STATES:
            for feature, _ in FEATURES:
                column = feature_column(condition, state, feature)
                current = tables[
                    table_name(condition, state, feature, "patient_average")
                ][keys + ["value"]].rename(columns={"value": column})
                if patient_table is None:
                    patient_table = current
                else:
                    patient_table = patient_table.merge(
                        current, on=keys, validate="one_to_one"
                    )
    if patient_table is None:
        raise RuntimeError("No qEEG features were loaded")
    if len(patient_table) != 50 or patient_table.patient_id.nunique() != 50:
        raise RuntimeError("Expected exactly one row for each of 50 case patients")
    if patient_table.label.value_counts().to_dict() != {1: 28, 0: 22}:
        raise RuntimeError("Expected 28 sustained responders and 22 nonresponders")
    if patient_table.isna().any().any():
        raise RuntimeError("Patient-level qEEG table contains missing values")
    return patient_table.sort_values("patient_id").reset_index(drop=True)


def feature_sets(patient_table: pd.DataFrame) -> dict[str, list[str]]:
    q_eeg = patient_table.columns[3:].tolist()
    pre = [column for column in q_eeg if column.startswith("pre_")]
    post = [column for column in q_eeg if column.startswith("post_")]
    pre_selected = [
        "pre_awake_beta_dfa_intercept_mean",
        "pre_awake_connectivity_percent_raw_pli",
    ]
    post_selected = [
        "post_sleep_beta_entropy_mean",
        "post_awake_beta_dfa_intercept_mean",
    ]
    return {
        "PRE selected": pre_selected,
        "PRE all": pre,
        "POST selected": post_selected,
        "POST all": post,
        "PRE+POST selected": pre_selected + post_selected,
        "PRE+POST all": q_eeg,
    }


def load_folds(patient_table: pd.DataFrame, fold_manifest: Path) -> np.ndarray:
    manifest = pd.read_csv(fold_manifest)
    required = {"patient_id", "fold"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Fold manifest must contain {sorted(required)}")
    manifest = manifest[["patient_id", "fold"]].drop_duplicates()
    if len(manifest) != 50 or manifest.patient_id.nunique() != 50:
        raise RuntimeError("Expected one fixed fold assignment for each patient")
    merged = patient_table[["patient_id"]].merge(
        manifest, on="patient_id", how="left", validate="one_to_one"
    )
    if merged.fold.isna().any() or set(merged.fold.astype(int)) != set(range(5)):
        raise RuntimeError("Fold manifest does not cover the 50 response patients")
    return merged.fold.to_numpy(int)


def standardize_full(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x, dtype=float))


def multiple_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """CCA with a single outcome, computed as the OLS multiple correlation."""
    x_centered = standardize_full(x)
    y_centered = np.asarray(y, dtype=float) - np.mean(y)
    q, _ = np.linalg.qr(x_centered, mode="reduced")
    denominator = float(y_centered @ y_centered)
    if denominator <= 0:
        return float("nan")
    explained = float(np.sum((q.T @ y_centered) ** 2))
    return float(np.sqrt(np.clip(explained / denominator, 0.0, 1.0)))


def centered_distance_matrix(values: np.ndarray) -> np.ndarray:
    distances = squareform(pdist(np.asarray(values, dtype=float), metric="euclidean"))
    return (
        distances
        - distances.mean(axis=0, keepdims=True)
        - distances.mean(axis=1, keepdims=True)
        + distances.mean()
    )


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Biased sample distance correlation on standardized qEEG features."""
    a = centered_distance_matrix(standardize_full(x))
    b = centered_distance_matrix(np.asarray(y, dtype=float).reshape(-1, 1))
    covariance_squared = float(np.mean(a * b))
    variance_x_squared = float(np.mean(a * a))
    variance_y_squared = float(np.mean(b * b))
    denominator = np.sqrt(variance_x_squared * variance_y_squared)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(np.clip(covariance_squared / denominator, 0.0, 1.0)))


def fit_fold_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    duration: np.ndarray | None = None,
    duration_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-five-fold model and intercept-only probabilities."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if duration is not None:
        duration = np.asarray(duration, dtype=float).reshape(-1, 1)
    probabilities = np.full(len(y), np.nan, dtype=float)
    intercept_probabilities = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        y_train = y[train]
        if np.unique(y_train).size != 2:
            raise RuntimeError(f"Training fold {fold} has only one response class")
        intercept_probabilities[test] = np.mean(y_train)
        if duration_only:
            if duration is None:
                raise ValueError("duration_only requires duration")
            design = duration
        elif duration is not None:
            design = np.column_stack([duration, x])
        else:
            design = x
        scaler = StandardScaler().fit(design[train])
        model = LogisticRegression(
            C=RIDGE_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            random_state=0,
        )
        model.fit(scaler.transform(design[train]), y_train)
        probabilities[test] = model.predict_proba(scaler.transform(design[test]))[:, 1]
    if np.isnan(probabilities).any() or np.isnan(intercept_probabilities).any():
        raise RuntimeError("Out-of-fold prediction is incomplete")
    return probabilities, intercept_probabilities


def mean_log_likelihood_bits(y: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    y = np.asarray(y, dtype=int)
    return float(
        np.mean(y * np.log2(probabilities) + (1 - y) * np.log2(1 - probabilities))
    )


def information_gain_bits(
    y: np.ndarray, probabilities: np.ndarray, reference_probabilities: np.ndarray
) -> float:
    return mean_log_likelihood_bits(y, probabilities) - mean_log_likelihood_bits(
        y, reference_probabilities
    )


def predictive_summary(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, duration: np.ndarray
) -> dict[str, float]:
    probabilities, intercept_probabilities = fit_fold_probabilities(x, y, folds)
    duration_probabilities, _ = fit_fold_probabilities(
        x, y, folds, duration=duration, duration_only=True
    )
    combined_probabilities, _ = fit_fold_probabilities(
        x, y, folds, duration=duration
    )
    return {
        "cv_auroc": float(roc_auc_score(y, probabilities)),
        "cv_log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "cv_brier": float(brier_score_loss(y, probabilities)),
        "intercept_only_cv_log_loss": float(
            log_loss(y, intercept_probabilities, labels=[0, 1])
        ),
        "intercept_only_cv_brier": float(brier_score_loss(y, intercept_probabilities)),
        "cv_information_gain_bits_per_patient": information_gain_bits(
            y, probabilities, intercept_probabilities
        ),
        "duration_only_cv_auroc": float(roc_auc_score(y, duration_probabilities)),
        "q_eeg_plus_duration_cv_auroc": float(
            roc_auc_score(y, combined_probabilities)
        ),
        "incremental_bits_over_duration": information_gain_bits(
            y, combined_probabilities, duration_probabilities
        ),
    }


def permute_within_folds(y: np.ndarray, folds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = np.asarray(y, dtype=int).copy()
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def association_permutation_tests(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed_cca = multiple_correlation(x, y)
    observed_dcor = distance_correlation(x, y)
    cca_null = np.empty(n_permutations, dtype=float)
    dcor_null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        permuted = permute_within_folds(y, folds, rng)
        cca_null[index] = multiple_correlation(x, permuted)
        dcor_null[index] = distance_correlation(x, permuted)
    return permutation_p(observed_cca, cca_null), permutation_p(observed_dcor, dcor_null)


def predictive_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    observed: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        permuted = permute_within_folds(y, folds, rng)
        probabilities, intercept_probabilities = fit_fold_probabilities(
            x, permuted, folds
        )
        null[index] = information_gain_bits(
            permuted, probabilities, intercept_probabilities
        )
    return permutation_p(observed, null)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def run_analysis(
    metadata_csv: Path,
    features_dir: Path,
    fold_manifest: Path,
    association_permutations: int,
    predictive_permutations: int,
    seed: int,
) -> pd.DataFrame:
    patient_table = build_patient_table(features_dir, metadata_csv)
    folds = load_folds(patient_table, fold_manifest)
    y = patient_table.label.to_numpy(int)
    duration = patient_table.duration_category.to_numpy(float)
    rows: list[dict[str, object]] = []
    for set_index, (name, columns) in enumerate(feature_sets(patient_table).items()):
        x = patient_table[columns].to_numpy(float)
        summary = predictive_summary(x, y, folds, duration)
        association_rng = np.random.default_rng(seed + 10_000 * set_index)
        predictive_rng = np.random.default_rng(seed + 10_000 * set_index + 1)
        cca_p, dcor_p = association_permutation_tests(
            x,
            y,
            folds,
            association_permutations,
            association_rng,
        )
        predictive_p = predictive_permutation_test(
            x,
            y,
            folds,
            summary["cv_information_gain_bits_per_patient"],
            predictive_permutations,
            predictive_rng,
        )
        rows.append(
            {
                "endpoint": "sustained response composite",
                "feature_set": name,
                "time_input": name.split()[0],
                "feature_scope": "source-selected" if "selected" in name else "all reproduced",
                "selection_status": (
                    "selected using this cohort in the source article"
                    if "selected" in name
                    else "complete condition/state grid fixed before this analysis"
                ),
                "n_patients": len(patient_table),
                "responders": int(y.sum()),
                "nonresponders": int(len(y) - y.sum()),
                "n_qeeg_features": len(columns),
                "q_eeg_features": ";".join(columns),
                "cca_multiple_r_in_sample": multiple_correlation(x, y),
                "cca_multiple_r_squared_in_sample": multiple_correlation(x, y) ** 2,
                "cca_fixed_set_permutation_p": cca_p,
                "distance_correlation_in_sample": distance_correlation(x, y),
                "distance_correlation_fixed_set_permutation_p": dcor_p,
                **summary,
                "cv_information_gain_full_refit_permutation_p": predictive_p,
                "association_permutations": association_permutations,
                "predictive_full_refit_permutations": predictive_permutations,
                "ridge_C": RIDGE_C,
                "fold_source": str(fold_manifest),
                "seed": seed,
            }
        )
    result = pd.DataFrame(rows)
    for source, target in (
        ("cca_fixed_set_permutation_p", "cca_holm_p_across_six_sets"),
        (
            "distance_correlation_fixed_set_permutation_p",
            "distance_correlation_holm_p_across_six_sets",
        ),
        (
            "cv_information_gain_full_refit_permutation_p",
            "cv_information_gain_holm_p_across_six_sets",
        ),
    ):
        result[target] = holm_adjust(result[source].tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--association-permutations", type=int, default=9_999)
    parser.add_argument("--predictive-permutations", type=int, default=1_999)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    args = parser.parse_args()
    if args.association_permutations < 1 or args.predictive_permutations < 1:
        raise ValueError("Permutation counts must be positive")
    result = run_analysis(
        args.metadata_csv,
        args.features_dir,
        args.fold_manifest,
        args.association_permutations,
        args.predictive_permutations,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    method_metadata = {
        "analysis_unit": "one patient after averaging two matching Clinical Clips",
        "endpoint": "meaningful_responder; 28 positive and 22 negative patients",
        "all_qeeg_definition": "PRE/POST x awake/sleep x beta DFA intercept, beta Shannon entropy, and delta PLI connectivity",
        "selected_qeeg_definition": "Rajaraman et al. source-selected PRE and POST EEG variables",
        "cca_interpretation": "with one binary outcome, CCA equals the in-sample OLS multiple correlation and is descriptive",
        "predictive_model": "StandardScaler fit on training patients followed by L2 logistic regression with C=1",
        "folds": str(args.fold_manifest),
        "permutation_scheme": "shuffle patient labels within fixed folds; refit the predictive model for every shuffled assignment",
        "conditional_duration_note": "incremental bits over duration is descriptive; no conditional-independence P-value is claimed",
        "selection_note": "source-selected feature sets were selected on this same cohort and are not an independent feature-selection validation",
        "seed": args.seed,
        "association_permutations": args.association_permutations,
        "predictive_full_refit_permutations": args.predictive_permutations,
    }
    args.metadata_output.write_text(
        json.dumps(method_metadata, indent=2) + "\n", encoding="utf-8"
    )
    display_columns = [
        "feature_set",
        "n_qeeg_features",
        "cca_multiple_r_in_sample",
        "cca_fixed_set_permutation_p",
        "distance_correlation_in_sample",
        "distance_correlation_fixed_set_permutation_p",
        "cv_auroc",
        "cv_information_gain_bits_per_patient",
        "cv_information_gain_full_refit_permutation_p",
        "q_eeg_plus_duration_cv_auroc",
        "incremental_bits_over_duration",
    ]
    print(result[display_columns].to_string(index=False))
    print(f"\nSaved aggregate results to {args.output}")
    print(f"Saved method metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
