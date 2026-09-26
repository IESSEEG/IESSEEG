#!/usr/bin/env python
"""Clip-level inference for a fine-tuned EEGPT model.

Tiles each held-out Routine Clip with non-overlapping windows, averages
the window probabilities into one clip-level score, and thresholds at
0.5 -- the same aggregation every other baseline uses.
"""

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from iesseeg.data.raw_dataset import InMemoryRandomDataset
from iesseeg.data.splits import encode_labels
from eegpt_data import EEGPT_CHANNELS, EEGPT_SAMPLE_RATE, make_recording_transform

N_CHANNELS = len(EEGPT_CHANNELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--label_key", required=True)
    parser.add_argument("--epoch_length", type=int, default=4)
    parser.add_argument("--source_sfreq", type=int, default=200)
    parser.add_argument("--source_channels", type=int, default=19)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()

    from braindecode.models import EEGPT

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] device={device} task={args.label_key}")

    df = pd.read_csv(args.inference_csv)
    df["label"] = encode_labels(df, args.label_key)
    recording_col = [c for c in df.columns if "recording_id" in c][0]
    info = list(zip(df["patient_id"], df[recording_col].astype(str), df["label"]))
    print(f"[DATA] {len(info)} clips from {df['patient_id'].nunique()} patients")

    dataset = InMemoryRandomDataset(
        data_dir=args.data_root, info_list=info, mode="test",
        sample_rate=EEGPT_SAMPLE_RATE,
        window_sec=args.epoch_length, step_sec=args.epoch_length,
        n_channels=args.source_channels, scale_factor=1.0,
        window_transform="none",
        recording_transform=make_recording_transform(
            source_rate=args.source_sfreq, target_rate=EEGPT_SAMPLE_RATE
        ),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=InMemoryRandomDataset.collate_fn)

    checkpoint = torch.load(args.model_file, map_location="cpu", weights_only=False)
    n_times = args.epoch_length * EEGPT_SAMPLE_RATE
    model = EEGPT(n_outputs=2, n_chans=N_CHANNELS, n_times=n_times,
                  sfreq=EEGPT_SAMPLE_RATE, chan_proj_type="conv1d_constraint",
                  n_chans_target=N_CHANNELS)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    print(f"[MODEL] {args.model_file} (epoch {checkpoint.get('epoch')}, "
          f"val_loss {checkpoint.get('val_loss'):.4f})")

    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["waveform"].to(device).float()
            probs = torch.softmax(model(x).float(), dim=-1)[:, 1].cpu().numpy()
            for i, prob in enumerate(probs):
                rows.append({
                    "patient_id": batch["patient_id"][i],
                    "recording_id": batch["recording_id"][i],
                    "start_ind": batch["start_ind"][i],
                    "end_ind": batch["end_ind"][i],
                    "pred_prob": float(prob),
                    "pred_label": int(prob >= 0.5),
                    "known_label": int(batch["label"][i]),
                })

    window_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    window_path = args.out_csv.replace(".csv", "_window.csv")
    window_df.to_csv(window_path, index=False)
    print(f"[DONE] {len(window_df)} window predictions -> {window_path}")

    clip_df = window_df.groupby("recording_id").agg(
        patient_id=("patient_id", "first"),
        pred_prob=("pred_prob", "mean"),
        known_label=("known_label", "first"),
    ).reset_index()
    clip_df["pred_label"] = (clip_df["pred_prob"] >= 0.5).astype(int)
    clip_df = clip_df[["patient_id", "recording_id", "pred_prob", "pred_label", "known_label"]]
    clip_df.to_csv(args.out_csv, index=False)
    print(f"[DONE] {len(clip_df)} clip predictions -> {args.out_csv}")


if __name__ == "__main__":
    main()
