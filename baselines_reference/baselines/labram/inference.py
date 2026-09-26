#!/usr/bin/env python
"""
Example: Inference script for a LaBraM-based model on scalp EEG.

This script:
  - Loads a CSV with short_recording_id (plus optional patient_id, case_control_label).
  - Builds a dataset that loads each .npz from 'data_root'.
  - Runs the finetuned LaBraM model on each window.
  - Aggregates window-level predictions per recording_id.

Usage:
  python inference_labram.py \
    --inference_csv "some_inference_list.csv" \
    --data_root "/path/to/eeg_npz_files" \
    --model_ckpt "./finetune_output/checkpoint-best.pth" \
    --out_csv "inference_result.csv" \
    --epoch_length 10 \
    --sfreq 200 \
    --nb_classes 2 \
    --device cuda
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import your InMemoryRandomDataset from the code you provided
from inmem_raw_dataset import InMemoryRandomDataset

# Import your utils and timm create_model from the LaBraM code
import utils
from timm.models import create_model
from einops import rearrange
import modeling_finetune

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

class InferenceLaBraM:
    """
    A simple inference pipeline for your LaBraM-based model.
    """
    def __init__(self, args):
        self.args = args

        self.ch_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'FZ', 'CZ', 'PZ'
        ]

        # Decide device
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # 1) Read the CSV of recordings
        df_infer = pd.read_csv(args.inference_csv)
        df_infer["binary_label"] = create_label_from_meta_csv(df_infer, args.label_key)

        self.short_ids = df_infer["short_recording_id"].astype(str).values
        self.labels    = df_infer["binary_label"].astype(int).values
        if "patient_id" in df_infer.columns:
            self.patient_ids = df_infer["patient_id"].astype(str).values
        else:
            self.patient_ids = np.zeros_like(self.labels).astype(str)

        print(f"Found {len(self.short_ids)} total recordings in {args.inference_csv}.")

        # 2) Load the LaBraM model
        #    Make sure you pass the same argument set as in your fine-tuning code
        print(f"Creating model {args.model_name} with {args.nb_classes} classes.")
        self.model = create_model(
            args.model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
        )
        # print(f"""created with param:
        #     args.model = {args.model_name}
        #     pretrained=False,
        #     num_classes = {args.nb_classes},
        #     drop_rate = {args.drop},
        #     drop_path_rate = {args.drop_path},
        #     attn_drop_rate = {args.attn_drop_rate},
        #     drop_block_rate=None,
        #     use_mean_pooling = {args.use_mean_pooling},
        #     init_scale = {args.init_scale},
        #     use_rel_pos_bias = {args.rel_pos_bias},
        #     use_abs_pos_emb = {args.abs_pos_emb},
        #     init_values = {args.layer_scale_init_value},
        #     qkv_bias    = {args.qkv_bias},
        # """)
        # exit()
        self.model.to(self.device)

        # 3) Load checkpoint (best .pth)
        print(f"Loading checkpoint from {args.model_ckpt}")
        checkpoint = torch.load(args.model_ckpt, map_location='cpu', weights_only=False)
        # The key inside checkpoint might be "model"
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint
        # Use the existing LaBraM “utils.load_state_dict” or do a direct load_state_dict
        utils.load_state_dict(self.model, checkpoint_model, prefix="")  # or prefix=''

        self.model.eval()
        print("Model loaded and set to eval().")

    def build_dataset(self):
        """
        We'll build a single 'test' dataset for all recordings in CSV.
        """
        info_list = []
        for pid, sid, lbl in zip(self.patient_ids, self.short_ids, self.labels):
            info_list.append((pid, sid, lbl))

        # Build the dataset in 'test' mode => step-based windows
        ds = InMemoryRandomDataset(
            data_dir=self.args.data_root,
            info_list=info_list,
            mode='test',
            sample_rate=self.args.sfreq,
            window_sec=self.args.epoch_length,
            step_sec=self.args.epoch_length,
            n_channels=19,         # or 23, etc. match your training
            scale_factor=1.0/100,  # same scaling as your training
            verbose=False,
            return_dict=True       # so we can read "recording_id" etc. in each batch
        )
        return ds

    def run_inference(self):
        dataset = self.build_dataset()
        # print(f"Got {len(dataset)} recordings in dataset.")
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=dataset.collate_fn,
        )

        # print(f"Got {len(loader)} batches of size {self.args.batch_size}.")

        # We'll store each window's prediction
        per_window_rows = []
        is_binary = (self.args.nb_classes <= 2)

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=True):
                wave_data = batch["waveform"].to(self.device)  # shape (B, C, frames)
                wave_data = wave_data.float()  # already scaled in dataset
                # Re-arrange to (B, N, A, T)
                wave_data = rearrange(wave_data, 'B N (A T) -> B N A T', T=200) / 100

                input_ch = utils.get_input_chans(self.ch_names)
                logits = self.model(wave_data, input_chans=input_ch)  # shape (B, nb_classes)
                if is_binary:
                    # For binary with nb_classes=1 => shape (B,1); or nb_classes=2 => (B,2)
                    if self.args.nb_classes == 1:
                        # shape = (B,1)
                        probs = torch.sigmoid(logits).squeeze(dim=-1).cpu().numpy()
                        preds = (probs >= 0.5).astype(int)
                    else:
                        # shape = (B,2)
                        softm = F.softmax(logits, dim=-1).cpu().numpy()
                        probs = softm[:, 1]  # probability for class 1
                        preds = np.argmax(softm, axis=-1)
                else:
                    # multi-class => do softmax
                    softm = F.softmax(logits, dim=-1).cpu().numpy()
                    probs = np.max(softm, axis=-1)      # confidence
                    preds = np.argmax(softm, axis=-1)   # predicted class

                known = batch["label"].cpu().numpy()
                recids= batch["recording_id"]  # a list[str]
                start_inds = batch["start_ind"]
                end_inds   = batch["end_ind"]

                # Save row per window
                for i, (rid, st, ed, pconf, pcls, klbl) in enumerate(zip(
                    recids, start_inds, end_inds, probs, preds, known
                )):
                    per_window_rows.append({
                        "short_recording_id": rid,
                        "start_ind": int(st),
                        "end_ind": int(ed),
                        "pred_prob": float(pconf),
                        "pred_label": int(pcls),
                        "known_label": (None if klbl < 0 else int(klbl))
                    })

        # Convert to dataframe
        df_window = pd.DataFrame(per_window_rows)
        # print(f"Got {len(df_window)} window-level predictions.")

        # Save the window-level predictions
        window_out_csv = self.args.out_csv.replace(".csv", "_window.csv")
        df_window.to_csv(window_out_csv, index=False)

        # 5) Optionally group by 'short_recording_id' => average probability
        #    Then do a final classification
        grouped = df_window.groupby("short_recording_id")
        final_rows = []
        for sid, group in grouped:
            mean_prob = group["pred_prob"].mean()
            # For binary => threshold at 0.5
            # For multiclass => threshold doesn't make sense, but you might pick largest mean prob
            # Here we assume binary classes => do something simple
            pred_lab = 1 if mean_prob >= 0.5 else 0
            known_lbl = group["known_label"].iloc[0]  # same label for entire recording
            final_rows.append({
                "short_recording_id": sid,
                "pred_prob": mean_prob,
                "pred_label": pred_lab,
                "known_label": known_lbl
            })
        df_final = pd.DataFrame(final_rows)
        df_final.to_csv(self.args.out_csv, index=False)
        print(f"[Done] wrote final results => {self.args.out_csv}")
        print(df_final.head(10))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_csv", type=str, required=True,
                   help="CSV with columns [short_recording_id, optional: case_control_label, patient_id].")
    parser.add_argument("--data_root", type=str, required=True,
                   help="Folder with {recording_id}.npz for EEG data.")
    parser.add_argument("--model_ckpt", type=str, required=True,
                   help="Path to 'checkpoint-best.pth' from fine-tuning.")
    parser.add_argument("--out_csv", type=str, default="inference_labram_out.csv",
                   help="Where to save the final aggregated CSV with predictions.")
    parser.add_argument("--epoch_length", type=float, default=10.0,
                   help="Window size in seconds (matching training).")
    parser.add_argument("--sfreq", type=float, default=200,
                   help="Sampling frequency of EEG.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda",
                   help="Which device to run on (e.g. 'cuda' or 'cpu').")
    parser.add_argument("--model_name", type=str, default="labram_base_patch200_200",
                   help="LaBraM variant name in timm registry, e.g. 'labram_base_patch200_200'.")
    parser.add_argument("--nb_classes", type=int, default=2,
                   help="Number of classes for classification head. 1 => single-output binary, 2 => standard 2-class, etc.")
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--label_key', type=str, default="case_control_label",
                        help="Key to use for label column in inference CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    inferer = InferenceLaBraM(args)
    inferer.run_inference()

if __name__=="__main__":
    main()
