#!/usr/bin/env python
"""
run_inference.py

Example script for running inference with your BIOT-finetuned (or any
Lightning-based) model on a new CSV listing recordings.

Usage:
  python run_inference.py \
    --inference_csv /path/to/test_inference_list.csv \
    --data_root    /path/to/biot_npz \
    --model_ckpt   ./ckpts/best.ckpt \
    --out_csv      inference_results.csv \
    --epoch_length 10 \
    --sfreq        200 \
    --batch_size   64 \
    --cuda         0
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Reuse your code:
from model_zoo import build_model
from run_binary_supervised import LitModel_finetune  # The same script/class used in training
from inmem_raw_dataset import InMemoryRandomDataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_csv", type=str, required=True,
                        help="CSV listing short_recording_id, patient_id, etc.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with preprocessed .npz files.")
    parser.add_argument("--model_ckpt", type=str, required=True,
                        help="Path to the .ckpt from training (the best checkpoint).")
    parser.add_argument("--out_csv", type=str, default="inference_results.csv",
                        help="Where to save final aggregated predictions.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                        help="Seconds used in training for each window.")
    parser.add_argument("--sfreq", type=float, default=200.0,
                        help="Sampling frequency used in training.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cuda", type=int, default=0,
                        help="Which GPU to use (0-based) if available.")

    # If you need them for building your model:
    parser.add_argument("--model", type=str, default="BIOT",
                        help="Architecture name, same as training")
    parser.add_argument("--n_classes", type=int, default=1,
                        help="1 => binary classification in your training code.")
    parser.add_argument(
        "--in_channels", type=int, default=18, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )

    parser.add_argument("--label_key", type=str, default="case_control_label",
                help="Key for the label in the CSV file")

    args = parser.parse_args()
    print(args)

    device = torch.device(f"cuda:{args.cuda}" if (torch.cuda.is_available() and args.cuda >= 0) else "cpu")

    # 1) Load the inference CSV
    df_infer = pd.read_csv(args.inference_csv)
    df_infer["label"] = create_label_from_meta_csv(df_infer, args.label_key)

    rec_ids = df_infer["short_recording_id"].astype(str).values
    labels  = df_infer["label"].values
    if "patient_id" in df_infer.columns:
        pt_ids = df_infer["patient_id"].astype(str).values
    else:
        pt_ids = ["UNKNOWN"] * len(df_infer)

    # Build a list of (patient_id, recording_id, label) for the dataset constructor
    info_list = [(pt_ids[i], rec_ids[i], labels[i]) for i in range(len(df_infer))]

    # 2) Create the dataset in 'test' mode
    test_ds = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=info_list,
        mode='test',
        sample_rate=args.sfreq,
        window_sec=args.epoch_length,
        step_sec=args.epoch_length,
        n_channels=args.in_channels,
        scale_factor=1.0,
        train_iterations=1,  # not used in test mode, so any small int
        verbose=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=test_ds.collate_fn
    )

    # 3) Recreate the same model architecture as training.
    # We'll do it just like your training script does:
    #    backbone = build_model(args.model, args)
    #    lit_model = LitModel_finetune(args, backbone)
    # Then we'll load the checkpoint state.
    # But with Lightning, we can do `LitModel_finetune.load_from_checkpoint(...)`.
    # We'll do the direct approach:
    print(f"Loading checkpoint from: {args.model_ckpt}")
    lit_model = LitModel_finetune.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        args=args,  # if your constructor needs `args`
        model=build_model(args.model, args)
    )
    lit_model.to(device)
    lit_model.eval()

    # 4) Inference loop
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x = batch["waveform"].to(device).float()  # shape: (B,C,seq,200)
            logits = lit_model.model(x).view(-1)      # shape => (B,)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)

            for i in range(len(x)):
                row = {
                    "patient_id":   batch["patient_id"][i],
                    "recording_id": batch["recording_id"][i],
                    "start_ind":    int(batch["start_ind"][i]),
                    "end_ind":      int(batch["end_ind"][i]),
                    "pred_prob":    float(probs[i]),
                    "pred_label":   int(preds[i]),
                    "known_label":   int(batch["label"][i])  # might be -1 if unknown
                }
                results.append(row)

    df_out = pd.DataFrame(results)
    window_save_path = args.out_csv.replace("inference_results", "inference_results_window")
    df_out.to_csv(window_save_path, index=False)
    print(f"[DONE] Wrote {len(df_out)} window-level predictions to {window_save_path}.")

    # If you want to group by recording_id and average across windows:
    grouped = df_out.groupby("recording_id")
    final_rows = []
    for rid, group in grouped:
        mean_prob = group["pred_prob"].mean()
        pred_label = 1 if mean_prob >= 0.5 else 0
        final_rows.append({
            "patient_id": group["patient_id"].iloc[0],
            "recording_id": rid,
            "pred_prob": mean_prob,
            "pred_label": pred_label,
            "known_label": int(group["known_label"].iloc[0])
        })
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote {len(df_final)} recording-level predictions to {args.out_csv}.")

if __name__ == "__main__":
    main()
