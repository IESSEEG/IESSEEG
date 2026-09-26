#!/usr/bin/env python
"""Extract native-window frozen representations for the long-EEG benchmark.

Run one encoder per CUDA process.  Only PRE Clinical Clips and independently
sampled Routine Clips are extracted because those are the development and
evaluation views used by the patient-level weak-supervision tasks.
"""

from __future__ import annotations
import legacy_paths as _paths

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frozen_encoders import MODEL_IO  # noqa: E402


DATA_DIRS = {
    "biot": (
        "data/scalp_eeg_data_200HZ_np_format_biot",
        "data/biot_test",
    ),
    "cbramod": (
        "data/scalp_eeg_data_200HZ_np_format_cbramod",
        "data/cbramod_test",
    ),
    "labram": (
        "data/scalp_eeg_data_200HZ_np_format_labram",
        "data/labram_test",
    ),
    "luna": (
        "data/scalp_eeg_data_200HZ_np_format",
        "data/baseline_test",
    ),
    "eegpt": (
        "data/scalp_eeg_data_200HZ_np_format_labram",
        "data/labram_test",
    ),
    "reve": (
        "data/scalp_eeg_data_200HZ_np_format_labram",
        "data/labram_test",
    ),
    "codebrain": (
        "data/scalp_eeg_data_200HZ_np_format_labram",
        "data/labram_test",
    ),
    "csbrain": (
        "data/scalp_eeg_data_200HZ_np_format_labram",
        "data/labram_test",
    ),
}


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def main() -> None:
    paper = _paths.runtime()
    project = _paths.workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_IO), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-root", type=Path,
        default=paper / "analysis/generated/long_frozen_representations",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen representation extraction requires explicit CUDA")
    torch.cuda.set_device(device)

    clinical = pd.read_csv(
        project / "data/final_short_merged.csv", dtype={"short_recording_id": str}
    )
    clinical = clinical[clinical.pre_post_treatment_label.eq("PRE")].copy()
    clinical["short_recording_id"] = normalize_id(clinical["short_recording_id"])
    clinical["view"] = "clinical"
    routine = pd.read_csv(
        project / "data/final_test.csv", dtype={"short_recording_id": str}
    )
    routine["short_recording_id"] = normalize_id(routine["short_recording_id"])
    routine["view"] = "routine"
    metadata = pd.concat([clinical, routine], ignore_index=True)
    if len(metadata) != 600 or metadata.short_recording_id.nunique() != 600:
        raise RuntimeError("Expected 400 PRE Clinical Clips and 200 Routine Clips")
    if args.limit is not None:
        metadata = metadata.iloc[: args.limit].copy()

    loader, runner = MODEL_IO[args.model]
    model = loader(project, device)
    clinical_dir, routine_dir = [project / path for path in DATA_DIRS[args.model]]
    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        recording_id = str(row.short_recording_id)
        data_dir = clinical_dir if row.view == "clinical" else routine_dir
        path = data_dir / f"{recording_id}.npz"
        cached = output_dir / f"{recording_id}.npy"
        if cached.exists():
            matrix = np.load(cached, allow_pickle=False)
        else:
            with np.load(path, allow_pickle=False) as payload:
                data = payload["data"].astype(np.float32, copy=False)
                channels = payload["channel"] if "channel" in payload.files else None
            matrix = runner(
                model, data, device, args.batch_size, channels, return_windows=True
            ).astype(np.float32, copy=False)
        if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid {args.model} representation for {recording_id}: {matrix.shape}")
        np.save(output_dir / f"{recording_id}.npy", matrix)
        rows.append({
            "recording_id": recording_id,
            "patient_id": int(row.patient_id),
            "view": row.view,
            "n_windows": len(matrix),
            "embedding_dim": matrix.shape[1],
        })
        if index % 25 == 0 or index == len(metadata):
            print(f"[{args.model}] {index}/{len(metadata)}", flush=True)

    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)
    print(
        f"Wrote {len(rows)} recordings for {args.model} to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
