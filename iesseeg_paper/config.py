"""Explicit, external data and artifact locations; no source data mutations."""

import json
from pathlib import Path

NEURAL_PERMUTATIONS = 99
NEURAL_SELECTED_SYSTEMS = (
    ("eegpt", "top10", "PRE", "immediate"),
    ("biot", "millet_instance", "PRE", "sustained"),
    ("reve", "psmil", "POST", "immediate"),
    ("labram", "millet_additive", "POST", "sustained"),
)

ENCODERS = (
    "biot",
    "labram",
    "cbramod",
    "eegpt",
    "luna",
    "reve",
    "codebrain",
    "csbrain",
)
WINDOW_SECONDS = dict.fromkeys(ENCODERS, 30)
WINDOW_SECONDS.update(labram=10, eegpt=4, handcrafted=30)
ENDPOINTS = {
    "immediate": "immediate_responder",
    "sustained": "meaningful_responder",
    "diagnosis": "case_control",
}
QEEG_FIELDS = (
    "beta_dfa_intercept_mean",
    "beta_entropy_mean",
    "connectivity_percent_raw_pli",
)
CLINICAL_FIELDS = ("AgeAtEEG1y", "AOOmo", "LeadtimeMo", "LeadtimeUKISS")


def load_config(path):
    c = json.loads(Path(path).read_text())
    for key in ("workspace", "previous_paper", "output_root", "qeeg_features"):
        c[key] = Path(c[key]).expanduser().resolve()
    source_roots = [
        c["workspace"] / "data",
        c["workspace"] / "kfold_split",
        c["previous_paper"],
    ]
    if any(c["output_root"].is_relative_to(x) for x in source_roots):
        raise ValueError(
            "Output root must be separate from the source data and repositories"
        )
    for name in (
        "feature_cache",
        "checkpoints",
        "predictions",
        "runs",
        "results",
        "logs",
        "tmp",
    ):
        (c["output_root"] / name).mkdir(parents=True, exist_ok=True)
    return c
