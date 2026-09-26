#!/usr/bin/env python
"""
Example: Inference script for a CBraMod-based model on scalp EEG.
This script:
  - loads a CSV with short_recording_id, plus optional patient_id, case_control_label
  - builds a dataset that loads each .npz
  - runs the model on each window
  - aggregates predictions per recording_id

Usage:
  python inference_cbramod.py \
    --inference_csv "some_inference_list.csv" \
    --data_root "/path/to/eeg_npz_files" \
    --model_file "./ckpts/final_model.pth" \
    --out_csv "inference_result.csv" \
    --epoch_length 30 \
    --sfreq 200
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

# Suppose we have an InMemory or step-based dataset akin to your training "InMemoryRandomDataset":
from inmem_raw_dataset import InMemoryRandomDataset

# Suppose your finetuned CBraMod-based model is in models/adapted_model.py
from models.adapted_model import Model

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class InferenceCBraMod:
    """
    A simple inference pipeline for your CBraMod-based model.
    """
    def __init__(self, args):
        args.use_pretrained_weights = False
        args.num_of_classes = 2
        args.dropout = 0.1

        self.args = args
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

        # 1) Read the CSV of recordings
        df_infer = pd.read_csv(args.inference_csv)

        labels = create_label_from_meta_csv(df_infer, args.label_key)
        df_infer["binary_label"] = labels

        self.short_ids   = df_infer["short_recording_id"].astype(str).values
        self.labels      = df_infer["binary_label"].astype(int).values
        if "patient_id" in df_infer.columns:
            self.patient_ids = df_infer["patient_id"].astype(str).values
        else:
            self.patient_ids = np.zeros_like(self.labels)

        print(f"Found {len(self.short_ids)} total recordings in {args.inference_csv}.")

        # 2) Load model
        print(f"Loading model from {args.model_file}")
        # Build the same shaped model (like in training)
        self.model = Model(args).to(self.device)
        state_dict = torch.load(args.model_file, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def build_concat_dataset(self):
        """
        We'll build up a ConcatDataset of all windows from each short_recording_id
        by re-using e.g. 'InMemoryRandomDataset' in 'test' mode, or a simple stepping approach.
        """
        final_ds = InMemoryRandomDataset(
                data_dir=self.args.data_root,
                info_list=list(zip(self.patient_ids, self.short_ids, self.labels)),
                mode='test',
                sample_rate=self.args.sfreq,
                window_sec=self.args.epoch_length,
                step_sec=self.args.epoch_length,
                n_channels=19,
                scale_factor=1.0/100.0,
                verbose=False
            )
        return final_ds

    def run_inference(self):
        dataset = self.build_concat_dataset()
        loader = DataLoader(
            dataset,
            batch_size=64,  # or whatever
            shuffle=True,
            num_workers=4,
            drop_last=False
        )
        # We'll store each window's prediction
        # We'll store: short_recording_id, prob, pred, known_label
        per_window_rows = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=True):
                x = batch["waveform"].to(self.device).float()  # shape (B,19,seq_len,200)
                # print(f"[Inference] x.shape={x.shape}, x.mean={x.mean(dim=(1,2,3))}, x.min={x.min():.3f}, x.max={x.max():.3f}")

                # we flatten in the same forward pass as training
                logits = self.model(x).view(-1)
                # assume binary => logits => shape (B,)
                probs  = torch.sigmoid(logits).cpu().numpy()
                preds  = (probs >= 0.5).astype(int)

                known = batch["label"].cpu().numpy()
                recids= batch["recording_id"]  # a list of str
                start_inds  = batch["start_ind"]
                end_inds    = batch["end_ind"]

                for i, (rid, st, ed, prob, pred_label, lbl) in enumerate(zip(recids, start_inds,
                                                                    end_inds, probs,
                                                                    preds, known)):
                    per_window_rows.append({
                        "short_recording_id": rid,
                        "start_ind": int(st),
                        "end_ind": int(ed),
                        "pred_prob": float(prob),
                        "pred_label": int(pred_label),
                        "known_label": (None if lbl<0 else int(lbl))
                    })

        # Convert to dataframe
        df_window = pd.DataFrame(per_window_rows)
        print(f"Got {len(df_window)} window-level predictions.")
        df_window.to_csv(self.args.out_csv.replace("inference_results", "inference_results_window"), index=False)

        # Optionally group by short_recording_id and average
        grouped = df_window.groupby("short_recording_id")
        final_rows = []
        for sid, group in grouped:
            mprob = group["pred_prob"].mean()
            mpred = int(mprob >= 0.5)
            # assume the known_label is the same across all windows
            known_lbl = group["known_label"].iloc[0]
            final_rows.append({
                "short_recording_id": sid,
                "pred_prob": mprob,
                "pred_label": mpred,
                "known_label": known_lbl
            })
        df_final = pd.DataFrame(final_rows)

        # Save
        os.makedirs(os.path.dirname(self.args.out_csv), exist_ok=True)
        df_final.to_csv(self.args.out_csv, index=False)
        print(f"[Done] wrote final results => {self.args.out_csv}")
        print(df_final.head(10))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_csv", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--model_file", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="inference_cbramod_out.csv")
    p.add_argument("--epoch_length", type=float, default=30.0)
    p.add_argument("--sfreq", type=float, default=200.0)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--label_key", type=str, default="case_control_label",
                   help="Key for the label in the CSV file")
    args = p.parse_args()

    inferer = InferenceCBraMod(args)
    inferer.run_inference()


if __name__=="__main__":
    main()
