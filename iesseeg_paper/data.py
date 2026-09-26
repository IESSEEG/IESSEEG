"""Patient joins and matched observation-budget views of existing features."""

import json
import numpy as np
import pandas as pd
from .config import ENCODERS, WINDOW_SECONDS, ENDPOINTS, QEEG_FIELDS, CLINICAL_FIELDS
from .splits import load_fold_manifest


def normalized(value):
    return str(value).strip().removesuffix(".0").upper()


def metadata(c):
    df = pd.read_csv(
        c["workspace"] / "data/final_short_merged.csv",
        dtype={"short_recording_id": str},
    )
    df["recording_id"] = df.short_recording_id.map(normalized)
    if (
        len(df) != 600
        or df.recording_id.nunique() != 600
        or df.patient_id.nunique() != 100
    ):
        raise ValueError("Expected the complete 600 Clinical Clips from 100 patients")
    if (
        not df.groupby(["patient_id", "pre_post_treatment_label", "sleep_awake_label"])
        .size()
        .eq(2)
        .all()
    ):
        raise ValueError("Expected two clips per patient, treatment stage and state")
    return df.sort_values(
        ["patient_id", "pre_post_treatment_label", "sleep_awake_label", "recording_id"]
    ).reset_index(drop=True)


def patient_info(c, task):
    df = metadata(c)
    if task != "diagnosis":
        df = df[df.case_control_label.eq("CASE")]
    column = ENDPOINTS[task] if task != "diagnosis" else "case_control_label"
    if not df.groupby("patient_id")[column].nunique().eq(1).all():
        raise ValueError("Inconsistent patient labels")
    pt = df.groupby("patient_id", sort=True).first().reset_index()
    mapping = {"Responder": 1, "Non-responder": 0, "CASE": 1, "CONTROL": 0}
    pt["label"] = pt[column].map(mapping)
    if pt.label.isna().any():
        raise ValueError("Unrecognized target value")
    folds = load_fold_manifest(task)
    pt = pt.merge(
        folds[["patient_id", "fold"]],
        on="patient_id",
        validate="one_to_one",
    )
    expected = 100 if task == "diagnosis" else 50
    if len(pt) != expected or set(pt.fold) != set(range(5)):
        raise ValueError("Incomplete patient/fold mapping")
    return pt[["patient_id", "label", "fold"]]


def clinical_features(c, patients):
    # Benchmark patient_id was assigned from the real long-short mapping table,
    # not from the independently shuffled clinical metadata table.
    mapping = pd.read_csv(c["workspace"] / "data/raw_data/long_short_mappings.csv")
    key = mapping[["PatientID"]].reset_index(names="patient_id")
    clinical = pd.read_csv(c["workspace"] / "data/raw_data/MetaDataForMingjian.csv")
    joined = key.merge(
        clinical[["PatientID", *CLINICAL_FIELDS]], on="PatientID", validate="one_to_one"
    )
    return (
        patients[["patient_id"]]
        .merge(joined, on="patient_id", validate="one_to_one")[list(CLINICAL_FIELDS)]
        .to_numpy(float)
    )


def window_path_index(c, model):
    if model == "handcrafted":
        root = (
            c["workspace"] / "kfold_split/baselines/handcrafted/feature_cache/regional"
        )
        index = {
            normalized(p.name.split(".npz__")[0]): p
            for p in root.glob("*regional_asym1_pli1.npy")
        }
        post = (
            c["previous_paper"]
            / "analysis/generated/clip_response_screen/baselines/qeeg_30s/post"
        )
        index.update({normalized(p.stem): p for p in post.glob("*.npy")})
    else:
        index = {}
        for stage in ("long", "post"):
            root = (
                c["previous_paper"]
                / f"analysis/generated/{stage}_frozen_representations"
                / model
            )
            index.update({normalized(p.stem): p for p in root.glob("*.npy")})
    return index


def prepare_cache(c):
    """Small 60-second block means; original full feature arrays stay in place."""
    df = metadata(c)
    dest = c["output_root"] / "feature_cache/patient_inputs"
    dest.mkdir(parents=True, exist_ok=True)
    for model in (*ENCODERS, "handcrafted", "clinical_qeeg"):
        target = dest / f"{model}.npz"
        if target.exists():
            continue
        arrays = {}
        index = window_path_index(c, model) if model != "clinical_qeeg" else None
        for row in df.itertuples():
            rid = row.recording_id
            if model == "clinical_qeeg":
                if row.case_control_label != "CASE":
                    continue
                path = c["qeeg_features"] / f"{rid}.json"
                if not path.exists():
                    matches = [
                        p
                        for p in c["qeeg_features"].glob("*.json")
                        if normalized(p.stem) == rid
                    ]
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"No unique clinical qEEG input for {rid}"
                        )
                    path = matches[0]
                payload = json.loads(path.read_text())
                values = np.array([payload[k] for k in QEEG_FIELDS], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite clinical qEEG features: {rid}")
                arrays[rid + "__mean"] = values
                continue
            if rid not in index:
                raise FileNotFoundError(f"Missing {model} cache: {rid}")
            x = np.load(index[rid], mmap_mode="r")
            if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
                raise ValueError(f"Invalid feature matrix: {index[rid]}")
            n = 60 // WINDOW_SECONDS[model]
            blocks = len(x) // n
            arrays[rid + "__blocks"] = (
                np.asarray(x[: blocks * n]).reshape(blocks, n, -1).mean(1)
            )
            arrays[rid + "__mean"] = np.asarray(x).mean(0)
        np.savez_compressed(target, **arrays)
        print(f"Prepared {model}: {len(arrays)} arrays", flush=True)
    # Use a common block count across representations so all models receive
    # the exact same 60-second positions for each fixed-budget comparison.
    minimum = {}
    for model in (*ENCODERS, "handcrafted"):
        with np.load(dest / f"{model}.npz") as arrays:
            for key in arrays.files:
                if key.endswith("__blocks"):
                    rid = key.removesuffix("__blocks")
                    minimum[rid] = min(minimum.get(rid, 10**9), len(arrays[key]))
    (dest / "common_blocks.json").write_text(json.dumps(minimum))
    df.to_csv(dest / "clinical_clip_metadata.csv", index=False)


class Features:
    def __init__(self, c, model):
        self.c, self.model = c, model
        base = c["output_root"] / "feature_cache/patient_inputs"
        with np.load(base / f"{model}.npz") as z:
            self.arrays = {k: z[k] for k in z.files}
        self.minimum = json.loads((base / "common_blocks.json").read_text())
        self.meta = metadata(c)
        self.dimension = next(
            v.size for k, v in self.arrays.items() if k.endswith("__mean")
        )
        self.ordinal = {rid: i for i, rid in enumerate(sorted(self.meta.recording_id))}

    def matrix(
        self, patients, condition, state="both", budget=0, draw=0, seed=20260920
    ):
        selected_states = ("AWAKE", "SLEEP") if state == "both" else (state.upper(),)
        count_clips = 2 * len(selected_states)
        if budget and (budget % (60 * count_clips)):
            raise ValueError(
                "Budget must allocate complete common 60-second blocks equally to clips"
            )
        if budget and self.model == "clinical_qeeg":
            raise ValueError(
                "Whole-clip clinical qEEG cannot be reused as a duration-limited feature"
            )
        result = []
        for pid in patients.patient_id:
            blocks = []
            rows = self.meta[
                (self.meta.patient_id == pid)
                & (self.meta.pre_post_treatment_label == condition)
            ]
            for st in ("AWAKE", "SLEEP"):
                if st not in selected_states:
                    blocks.append(np.zeros(self.dimension, dtype=np.float32))
                    continue
                clips = rows[rows.sleep_awake_label == st]
                if len(clips) != 2:
                    raise ValueError(f"Missing paired {st} clips")
                values = []
                for rid in clips.recording_id:
                    if budget:
                        n = budget // (60 * count_clips)
                        available = self.minimum[rid]
                        if n > available:
                            raise ValueError(
                                f"Budget {budget}s exceeds available input in {rid}"
                            )
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, draw, self.ordinal[rid]])
                        )
                        ix = rng.permutation(available)[:n]
                        values.append(self.arrays[rid + "__blocks"][ix].mean(0))
                    else:
                        values.append(self.arrays[rid + "__mean"])
                blocks.append(np.mean(values, axis=0))
            result.append(np.concatenate(blocks))
        return np.asarray(result, dtype=np.float64)
