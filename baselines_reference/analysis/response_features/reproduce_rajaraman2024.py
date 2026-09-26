#!/usr/bin/env python
"""Reconstruct the patient-level Rajaraman et al. (2024) response metrics.

Two inputs are supported.  ``raw`` uses the independent raw-EDF re-extraction
implemented in ``extract_rajaraman2024_raw_features.py``.  ``legacy-cache`` is
the earlier compatibility audit of the local Python feature cache; it is kept
only to document why that cache cannot reproduce the paper.

The source study averages the two clips from the same patient, treatment
condition, and vigilance state.  The sustained-response endpoint is therefore
represented by one row per patient, not by 200 nominally independent clips.
Row-level output is written only under the ignored ``local_results`` tree
because it joins local patient identifiers to restricted response labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rajaraman2024_contract import PLI_CONNECTIVITY_POLICY, PLI_EPOCH_POLICY


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

PUBLISHED = {
    "pre_awake_dfa_intercept_beta": {
        "responder_median": -0.20,
        "responder_q1": -0.41,
        "responder_q3": 0.09,
        "nonresponder_median": -0.05,
        "nonresponder_q1": -0.14,
        "nonresponder_q3": 0.15,
        "p_value": 0.046,
    },
    "pre_awake_connectivity_percent": {
        "responder_median": 6.7,
        "responder_q1": 1.6,
        "responder_q3": 13.5,
        "nonresponder_median": 12.7,
        "nonresponder_q1": 4.4,
        "nonresponder_q3": 23.7,
        "p_value": 0.039,
    },
    "pre_awake_connectivity_frequency_sensitivity": {
        "responder_median": np.nan,
        "responder_q1": np.nan,
        "responder_q3": np.nan,
        "nonresponder_median": np.nan,
        "nonresponder_q1": np.nan,
        "nonresponder_q3": np.nan,
        "p_value": np.nan,
    },
    "post_sleep_entropy_beta": {
        "responder_median": 8.06,
        "responder_q1": 7.79,
        "responder_q3": 8.19,
        "nonresponder_median": 7.50,
        "nonresponder_q1": 7.15,
        "nonresponder_q3": 7.81,
        "p_value": 0.006,
    },
    "post_awake_dfa_intercept_beta": {
        "responder_median": -0.43,
        "responder_q1": -0.70,
        "responder_q3": -0.26,
        "nonresponder_median": -0.11,
        "nonresponder_q1": -0.29,
        "nonresponder_q3": 0.04,
        "p_value": 0.006,
    },
    "r0_distribution": {
        "responder_median": 0.07,
        "responder_q1": -0.64,
        "responder_q3": 0.54,
        "nonresponder_median": -0.92,
        "nonresponder_q1": -1.79,
        "nonresponder_q3": 0.08,
        "p_value": 0.002,
    },
    "r1_distribution": {
        "responder_median": 41.3,
        "responder_q1": 40.2,
        "responder_q3": 44.1,
        "nonresponder_median": 36.9,
        "nonresponder_q1": 34.3,
        "nonresponder_q3": 38.7,
        # The article reports P < 0.0001, so this is an upper bound rather
        # than an equality target.
        "p_value_upper_bound": 0.0001,
    },
    "r0": {"auroc": 0.75, "auroc_ci_low": 0.61, "auroc_ci_high": 0.89, "loocv_auroc": 0.69},
    "r1": {"auroc": 0.93, "auroc_ci_low": 0.85, "auroc_ci_high": 1.00, "loocv_auroc": 0.91},
}


def normalize_recording_id(value: object) -> str:
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".0") else text


def response_value(value: object) -> int:
    if str(value) == "Responder":
        return 1
    if str(value) == "Non-responder":
        return 0
    raise ValueError(f"Unexpected sustained-response value: {value!r}")


def mean_feature(payload: dict[str, object], key: str, band_index: int) -> float:
    values = np.asarray(payload[key], dtype=float)
    if values.ndim != 2 or values.shape[0] <= band_index:
        raise ValueError(f"Unexpected {key} shape: {values.shape}")
    return float(np.nanmean(values[band_index]))


def load_clip_features(features_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    clips = pd.read_csv(metadata_csv)
    clips = clips.loc[clips.case_control_label.eq("CASE")].copy()
    clips["recording_id"] = clips.short_recording_id.map(normalize_recording_id)
    clips["label"] = clips.meaningful_responder.map(response_value)
    if len(clips) != 400 or clips.patient_id.nunique() != 50:
        raise RuntimeError("Expected 400 Clinical Clips from 50 case patients")

    rows: list[dict[str, object]] = []
    for row in clips.itertuples(index=False):
        path = features_dir / f"{row.recording_id}_features.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "patient_id": int(row.patient_id),
                "recording_id": row.recording_id,
                "condition": row.pre_post_treatment_label,
                "state": row.sleep_awake_label,
                "label": int(row.label),
                "duration_category": float(row.LeadtimeUKISS),
                "dfa_intercept_beta": mean_feature(payload, "DFAint", 3),
                "entropy_beta": mean_feature(payload, "Ent", 3),
                # The cached Python implementation stores the mean raw PLI
                # across channel pairs on [0, 1].  Multiplication by 100 only
                # puts it on the percentage scale used in the article; it does
                # not make it the article's thresholded connectivity measure.
                "cached_mean_pli_percent": 100.0 * float(payload["PLI_delta"]),
            }
        )
    return pd.DataFrame(rows)


def load_raw_patient_features(features_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    """Load raw-EDF features and average the two matching clips per patient."""
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    metadata["recording_id"] = metadata.short_recording_id.map(normalize_recording_id)
    metadata["label"] = metadata.meaningful_responder.map(response_value)
    if len(metadata) != 400 or metadata.patient_id.nunique() != 50:
        raise RuntimeError("Expected 400 Clinical Clips from 50 case patients")

    rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        target = (
            (row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE")
            or row.pre_post_treatment_label == "POST"
        )
        if not target:
            continue
        path = features_dir / f"{row.recording_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = []
        if row.sleep_awake_label == "AWAKE":
            required.append("beta_dfa_intercept_mean")
        if row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "SLEEP":
            required.append("beta_entropy_mean")
        if row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE":
            required.extend(
                [
                    "connectivity_percent_retained_pli",
                    "connectivity_percent_significance_frequency",
                    "connectivity_percent_raw_pli",
                ]
            )
        missing = [name for name in required if name not in payload]
        if missing:
            raise RuntimeError(f"{path.name} is incomplete: {missing}")
        if (
            row.pre_post_treatment_label == "PRE"
            and row.sleep_awake_label == "AWAKE"
            and payload.get("pli_epoch_policy") != PLI_EPOCH_POLICY
        ):
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete epoch policy; "
                "rerun extract_rajaraman2024_raw_features.py --features pli"
            )
        if (
            row.pre_post_treatment_label == "PRE"
            and row.sleep_awake_label == "AWAKE"
            and payload.get("pli_connectivity_policy") != PLI_CONNECTIVITY_POLICY
        ):
            raise RuntimeError(
                f"{path.name} contains PLI from an obsolete connectivity policy; "
                "rerun extract_rajaraman2024_raw_features.py --features pli"
            )
        rows.append(
            {
                "patient_id": int(row.patient_id),
                "recording_id": row.recording_id,
                "condition": row.pre_post_treatment_label,
                "state": row.sleep_awake_label,
                "label": int(row.label),
                "duration_category": float(row.LeadtimeUKISS),
                "dfa_intercept_beta": payload.get("beta_dfa_intercept_mean", np.nan),
                "entropy_beta": payload.get("beta_entropy_mean", np.nan),
                "connectivity_percent_retained_pli": payload.get(
                    "connectivity_percent_retained_pli", np.nan
                ),
                "connectivity_percent_raw_pli": payload.get(
                    "connectivity_percent_raw_pli", np.nan
                ),
                "connectivity_percent_significance_frequency": payload.get(
                    "connectivity_percent_significance_frequency", np.nan
                ),
                "clean_fraction": float(payload["clean_fraction"]),
            }
        )
    clips = pd.DataFrame(rows)
    expected = {("PRE", "AWAKE"): 100, ("POST", "AWAKE"): 100, ("POST", "SLEEP"): 100}
    observed = clips.groupby(["condition", "state"]).size().to_dict()
    if observed != expected:
        raise RuntimeError(f"Unexpected raw feature cells: {observed}")

    stable = clips.groupby("patient_id").agg(
        label_n=("label", "nunique"), duration_n=("duration_category", "nunique")
    )
    if not stable.eq(1).all().all():
        raise RuntimeError("Patient labels or duration categories are inconsistent")
    counts = clips.groupby(["patient_id", "condition", "state"]).size()
    if not counts.eq(2).all():
        raise RuntimeError("Expected two clips per patient/condition/state")

    base = clips.groupby("patient_id", as_index=False).agg(
        label=("label", "first"),
        duration_category=("duration_category", "first"),
        clean_fraction_mean=("clean_fraction", "mean"),
    )
    cells = {
        "dfa_intercept_beta_pre_awake": ("PRE", "AWAKE", "dfa_intercept_beta"),
        "connectivity_percent_pre_awake": (
            "PRE",
            "AWAKE",
            "connectivity_percent_raw_pli",
        ),
        "connectivity_frequency_sensitivity_pre_awake": (
            "PRE",
            "AWAKE",
            "connectivity_percent_significance_frequency",
        ),
        "dfa_intercept_beta_post_awake": ("POST", "AWAKE", "dfa_intercept_beta"),
        "entropy_beta_post_sleep": ("POST", "SLEEP", "entropy_beta"),
    }
    for output_name, (condition, state, feature) in cells.items():
        values = (
            clips.loc[clips.condition.eq(condition) & clips.state.eq(state)]
            .groupby("patient_id")[feature]
            .mean()
            .rename(output_name)
        )
        base = base.merge(values, on="patient_id", validate="one_to_one")
    if base.isna().any().any():
        raise RuntimeError("Raw patient feature table contains missing values")
    if base.label.value_counts().to_dict() != {1: 28, 0: 22}:
        raise RuntimeError("Expected 28 sustained responders and 22 nonresponders")
    base["r0_raw"] = (
        -2.361 * base["dfa_intercept_beta_pre_awake"]
        - 0.051 * base["connectivity_percent_pre_awake"]
    )
    base["r0_frequency_sensitivity"] = (
        -2.361 * base["dfa_intercept_beta_pre_awake"]
        - 0.051 * base["connectivity_frequency_sensitivity_pre_awake"]
    )
    base["r1_raw"] = (
        4.765 * base["entropy_beta_post_sleep"]
        - 7.786 * base["dfa_intercept_beta_post_awake"]
    )
    return base


def raw_preprocessing_summary(features_dir: Path, metadata_csv: Path) -> pd.DataFrame:
    """Summarize excluded-data fractions without releasing patient-level rows."""
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    metadata["recording_id"] = metadata.short_recording_id.map(normalize_recording_id)
    metadata["label"] = metadata.meaningful_responder.map(response_value)
    rows = []
    for row in metadata.itertuples(index=False):
        path = features_dir / f"{row.recording_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "clean_fraction" not in payload:
            continue
        rows.append(
            {
                "condition": row.pre_post_treatment_label,
                "state": row.sleep_awake_label,
                "label": int(row.label),
                "excluded_percent": 100.0 * (1.0 - float(payload["clean_fraction"])),
            }
        )
    clips = pd.DataFrame(rows)
    definitions = [
        ("all_extracted_awake", clips.state.eq("AWAKE"), 8.9, 4.3, 12.3),
        ("post_sleep_responders", clips.state.eq("SLEEP") & clips.condition.eq("POST") & clips.label.eq(1), 0.7, 0.1, 1.1),
        ("post_sleep_nonresponders", clips.state.eq("SLEEP") & clips.condition.eq("POST") & clips.label.eq(0), 1.7, 0.7, 4.4),
    ]
    output = []
    for name, mask, paper_median, paper_q1, paper_q3 in definitions:
        values = clips.loc[mask, "excluded_percent"].to_numpy(float)
        output.append(
            {
                "group": name,
                "n_clips": len(values),
                "local_excluded_percent_median": float(np.median(values)),
                "local_excluded_percent_q1": float(np.quantile(values, 0.25)),
                "local_excluded_percent_q3": float(np.quantile(values, 0.75)),
                "published_excluded_percent_median": paper_median,
                "published_excluded_percent_q1": paper_q1,
                "published_excluded_percent_q3": paper_q3,
            }
        )
    return pd.DataFrame(output)


def patient_table(clips: pd.DataFrame) -> pd.DataFrame:
    counts = clips.groupby(["patient_id", "condition", "state"]).size()
    if not counts.eq(2).all():
        raise RuntimeError("Expected exactly two clips per patient/condition/state")
    stable = clips.groupby("patient_id").agg(
        label_n=("label", "nunique"),
        duration_n=("duration_category", "nunique"),
    )
    if not stable.eq(1).all().all():
        raise RuntimeError("Patient labels or duration categories are inconsistent")

    averaged = (
        clips.groupby(["patient_id", "condition", "state"], as_index=False)
        .agg(
            label=("label", "first"),
            duration_category=("duration_category", "first"),
            dfa_intercept_beta=("dfa_intercept_beta", "mean"),
            entropy_beta=("entropy_beta", "mean"),
            cached_mean_pli_percent=("cached_mean_pli_percent", "mean"),
        )
    )
    wide = averaged.pivot(
        index=["patient_id", "label", "duration_category"],
        columns=["condition", "state"],
        values=["dfa_intercept_beta", "entropy_beta", "cached_mean_pli_percent"],
    )
    wide.columns = ["_".join((metric, condition, state)).lower() for metric, condition, state in wide.columns]
    wide = wide.reset_index()
    if len(wide) != 50 or wide.label.value_counts().to_dict() != {1: 28, 0: 22}:
        raise RuntimeError("Expected 28 sustained responders and 22 nonresponders")

    wide["r0_cached"] = (
        -2.361 * wide["dfa_intercept_beta_pre_awake"]
        - 0.051 * wide["cached_mean_pli_percent_pre_awake"]
    )
    wide["r1_cached"] = (
        4.765 * wide["entropy_beta_post_sleep"]
        - 7.786 * wide["dfa_intercept_beta_post_awake"]
    )
    return wide


def distribution_row(
    patients: pd.DataFrame,
    feature: str,
    published_key: str,
) -> dict[str, object]:
    positive = patients.loc[patients.label.eq(1), feature].to_numpy(float)
    negative = patients.loc[patients.label.eq(0), feature].to_numpy(float)
    p_value = float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
    result: dict[str, object] = {
        "quantity": published_key,
        "local_feature": feature,
        "n_responders": len(positive),
        "n_nonresponders": len(negative),
        "local_responder_median": float(np.median(positive)),
        "local_responder_q1": float(np.quantile(positive, 0.25)),
        "local_responder_q3": float(np.quantile(positive, 0.75)),
        "local_nonresponder_median": float(np.median(negative)),
        "local_nonresponder_q1": float(np.quantile(negative, 0.25)),
        "local_nonresponder_q3": float(np.quantile(negative, 0.75)),
        "local_mann_whitney_p": p_value,
    }
    result.update({f"published_{key}": value for key, value in PUBLISHED[published_key].items()})
    return result


def leave_one_patient_out_auc(patients: pd.DataFrame, features: list[str]) -> float:
    probabilities = np.empty(len(patients), dtype=float)
    x = patients[features].to_numpy(float)
    y = patients.label.to_numpy(int)
    for test_index in range(len(patients)):
        train = np.arange(len(patients)) != test_index
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
        model.fit(x[train], y[train])
        probabilities[test_index] = model.predict_proba(x[[test_index]])[0, 1]
    return float(roc_auc_score(y, probabilities))


def evaluate(patients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distributions = pd.DataFrame(
        [
            distribution_row(
                patients,
                "dfa_intercept_beta_pre_awake",
                "pre_awake_dfa_intercept_beta",
            ),
            distribution_row(
                patients,
                "cached_mean_pli_percent_pre_awake",
                "pre_awake_connectivity_percent",
            ),
            distribution_row(
                patients,
                "entropy_beta_post_sleep",
                "post_sleep_entropy_beta",
            ),
            distribution_row(
                patients,
                "dfa_intercept_beta_post_awake",
                "post_awake_dfa_intercept_beta",
            ),
        ]
    )

    score_rows = []
    for score, published_key, loocv_features in (
        (
            "r0_cached",
            "r0",
            [
                "duration_category",
                "dfa_intercept_beta_pre_awake",
                "cached_mean_pli_percent_pre_awake",
            ],
        ),
        (
            "r1_cached",
            "r1",
            [
                "duration_category",
                "entropy_beta_post_sleep",
                "dfa_intercept_beta_post_awake",
            ],
        ),
    ):
        score_rows.append(
            {
                "score": score,
                "n_patients": len(patients),
                "local_published_formula_auroc": float(
                    roc_auc_score(patients.label, patients[score])
                ),
                "published_formula_auroc": PUBLISHED[published_key]["auroc"],
                "local_fixed_feature_loocv_auroc": leave_one_patient_out_auc(
                    patients, loocv_features
                ),
                "published_loocv_auroc": PUBLISHED[published_key]["loocv_auroc"],
                "loocv_note": (
                    "Local LOOCV refits an unpenalized-equivalent logistic model with "
                    "the article's final EEG features plus duration; the article does "
                    "not disclose whether feature selection was repeated in each n-1 fold."
                ),
            }
        )
    return distributions, pd.DataFrame(score_rows)


def evaluate_raw(patients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    distributions = pd.DataFrame(
        [
            distribution_row(
                patients, "dfa_intercept_beta_pre_awake", "pre_awake_dfa_intercept_beta"
            ),
            distribution_row(
                patients, "connectivity_percent_pre_awake", "pre_awake_connectivity_percent"
            ),
            distribution_row(
                patients,
                "connectivity_frequency_sensitivity_pre_awake",
                "pre_awake_connectivity_frequency_sensitivity",
            ),
            distribution_row(
                patients, "entropy_beta_post_sleep", "post_sleep_entropy_beta"
            ),
            distribution_row(
                patients, "dfa_intercept_beta_post_awake", "post_awake_dfa_intercept_beta"
            ),
            distribution_row(patients, "r0_raw", "r0_distribution"),
            distribution_row(patients, "r1_raw", "r1_distribution"),
        ]
    )
    rows = []
    for score, key, features in (
        (
            "r0_raw",
            "r0",
            [
                "duration_category",
                "dfa_intercept_beta_pre_awake",
                "connectivity_percent_pre_awake",
            ],
        ),
        (
            "r1_raw",
            "r1",
            [
                "duration_category",
                "entropy_beta_post_sleep",
                "dfa_intercept_beta_post_awake",
            ],
        ),
    ):
        rows.append(
            {
                "score": score,
                "n_patients": len(patients),
                "local_published_formula_auroc": float(roc_auc_score(patients.label, patients[score])),
                "published_formula_auroc": PUBLISHED[key]["auroc"],
                "published_formula_auroc_ci_low": PUBLISHED[key]["auroc_ci_low"],
                "published_formula_auroc_ci_high": PUBLISHED[key]["auroc_ci_high"],
                "local_fixed_feature_loocv_auroc": leave_one_patient_out_auc(patients, features),
                "published_loocv_auroc": PUBLISHED[key]["loocv_auroc"],
            }
        )
    rows.append(
        {
            "score": "r0_frequency_sensitivity",
            "n_patients": len(patients),
            "local_published_formula_auroc": float(
                roc_auc_score(patients.label, patients.r0_frequency_sensitivity)
            ),
            "published_formula_auroc": np.nan,
            "published_formula_auroc_ci_low": np.nan,
            "published_formula_auroc_ci_high": np.nan,
            "local_fixed_feature_loocv_auroc": leave_one_patient_out_auc(
                patients,
                [
                    "duration_category",
                    "dfa_intercept_beta_pre_awake",
                    "connectivity_frequency_sensitivity_pre_awake",
                ],
            ),
            "published_loocv_auroc": np.nan,
        }
    )
    return distributions, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["raw", "legacy-cache"], default="raw")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help=(
            "Local clip metadata containing the governed meaningful_responder "
            "field. This file is not distributed."
        ),
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Legacy cached-feature directory; required only with --source legacy-cache.",
    )
    parser.add_argument(
        "--raw-features-dir",
        type=Path,
        default=LOCAL_RESULTS / "raw_features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(args.metadata_csv)
    if args.source == "legacy-cache" and args.features_dir is None:
        parser.error("--features-dir is required with --source legacy-cache")
    if args.output_dir is None:
        suffix = "raw_reproduction" if args.source == "raw" else "cache_audit"
        args.output_dir = LOCAL_RESULTS / suffix
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "raw":
        patients = load_raw_patient_features(args.raw_features_dir, args.metadata_csv)
        distributions, scores = evaluate_raw(patients)
        preprocessing = raw_preprocessing_summary(args.raw_features_dir, args.metadata_csv)
    else:
        clips = load_clip_features(args.features_dir, args.metadata_csv)
        patients = patient_table(clips)
        distributions, scores = evaluate(patients)

    patients.to_csv(args.output_dir / "patient_level_local.csv", index=False)
    distributions.to_csv(args.output_dir / "published_distribution_comparison.csv", index=False)
    scores.to_csv(args.output_dir / "published_score_comparison.csv", index=False)
    if args.source == "raw":
        preprocessing.to_csv(args.output_dir / "preprocessing_comparison.csv", index=False)
    metadata = {
        "source_article": "Rajaraman et al., Clinical Neurophysiology, 2024",
        "doi": "10.1016/j.clinph.2024.03.035",
        "stage": (
            "independent raw-EDF reproduction"
            if args.source == "raw"
            else "compatibility audit of legacy Python feature cache"
        ),
        "claim": (
            "method reproduction with disclosed approximations; not a bitwise source-code reproduction"
            if args.source == "raw"
            else "not an exact reproduction of the source MATLAB feature extraction"
        ),
        "patient_unit": "mean of two clips for each condition and vigilance state",
        "n_patients": 50,
        "response_counts": {"responder": 28, "nonresponder": 22},
        "published_equations": {
            "R0": "-2.361 * pre_awake_beta_DFA_intercept - 0.051 * pre_awake_connectivity",
            "R1": "4.765 * post_sleep_beta_Shannon_entropy - 7.786 * post_awake_beta_DFA_intercept",
        },
        "limitations": (
            [
                "clinician-marked sleep artifacts used by the source study are unavailable",
                "SciPy uses the nearest odd-tap FIR least-squares design for MATLAB compatibility",
                "PLI uses non-overlapping eight-second epochs within contiguous clean runs",
                "C0 is the fraction of electrode pairs whose across-epoch mean raw PLI exceeds 0.20; surrogate-thresholded variants are retained only as diagnostics",
            ]
            if args.source == "raw"
            else [
                "cached PLI is mean raw PLI across pairs rather than the article's thresholded connectivity measure",
                "legacy features were extracted in volts although the artifact detector expects microvolts",
            ]
        ),
    }
    if args.source == "raw":
        metadata["pli_epoch_policy"] = PLI_EPOCH_POLICY
        metadata["pli_connectivity_policy"] = PLI_CONNECTIVITY_POLICY
    (args.output_dir / "method_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nPatient-level distribution comparison")
    print(distributions.to_string(index=False))
    print("\nPublished-score comparison")
    print(scores.to_string(index=False))
    if args.source == "raw":
        print("\nArtifact-exclusion comparison")
        print(preprocessing.to_string(index=False))


if __name__ == "__main__":
    main()
