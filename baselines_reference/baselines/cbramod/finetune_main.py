#!/usr/bin/env python
"""
finetune_main.py

Example: High-level code for finetuning the CBraMod model.
It uses a Trainer class from finetune_trainer.py to handle
the training loops and evaluation.

Usage example:
  python finetune_main.py \
    --train_csv "final_open_source_metadata.csv" \
    --data_root "../data/scalp_eeg_200hz_npz" \
    --epoch_length 30 \
    --sfreq 200 \
    --test_size 0.2 \
    --model_out "finetuned_cbramod.pth" \
    --confusion_fig "cm.png" \
    --metrics_out "finetune_metrics.txt" \
    --num_of_classes 2 \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3 \
    --model_dir "./finetune_ckpts" \
    --cuda 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

# 1) Import your new Trainer
from finetune_trainer import Trainer

# 2) Import your adapted CBraMod-based model
from models.adapted_model import Model

# 3) Import your dataset
from inmem_raw_dataset import InMemoryRandomDataset  # or whichever dataset class

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

def setup_seed(seed):
    """Simple function to fix the random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Finetune CBraMod model on custom dataset")

    # Foundation model
    parser.add_argument('--use_pretrained_weights', action='store_true',
                    help='Start from released CBraMod weights if set (default: False)')
    parser.add_argument('--foundation_dir', type=str,
                    default='pretrained_weights/pretrained_weights.pth',
                    help='Path to the released CBraMod .pth file')

    # Basic training hyperparams
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='which cuda device to use')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='[AdamW, SGD]')
    parser.add_argument('--clip_value', type=float, default=1.0, help='clip grad norm')
    parser.add_argument('--dropout', type=float, default=0.1, help='model dropout')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing (for multi-class CE).")
    parser.add_argument("--multi_lr", action="store_true",
                        help="Use different LR for backbone vs rest if set.")
    parser.add_argument("--frozen", action="store_true",
                        help="Freeze backbone layers if set.")
    parser.add_argument("--debug", action="store_true", help="If set, print debug info each batch.")


    # Dataset and data path
    parser.add_argument("--train_csv", type=str, required=True,
                        help="CSV with e.g. columns => [patient_id, short_recording_id, case_control_label]")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with {recording_id}.npz for EEG data.")
    parser.add_argument("--epoch_length", type=float, default=30.0,
                        help="Seconds per EEG window (default=30).")
    parser.add_argument("--sfreq", type=int, default=200,
                        help="Sampling frequency in your data (default=200).")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction for test split (default=0.2).")
    parser.add_argument("--model_out", type=str, default="finetuned_cbramod.pth",
                        help="Where to save final model checkpoint.")
    parser.add_argument("--num_of_classes", type=int, default=2,
                        help="2 => binary, >2 => multi-class, else => regression.")
    parser.add_argument("--model_dir", type=str, default="./finetune_ckpts",
                        help="Directory to save best model weights.")
    parser.add_argument("--save_preds_dir",
                        type=str,
                        default="./epoch_preds",
                        help="Folder in which to write per-epoch prediction CSVs")
    parser.add_argument(
        "--label_key", type=str, default="case_control_label", help="Key for the label in the CSV file"
    )


    args = parser.parse_args()

    print(f"Running with args: {args}")

    # 1) Setup random seed
    setup_seed(args.seed)

    # 2) parse CSV
    df = pd.read_csv(args.train_csv)
    rec_ids = df["short_recording_id"].values
    labels  = create_label_from_meta_csv(df, args.label_key)
    pt_ids  = df["patient_id"].values

    labels_names, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(labels_names, counts))}")

    # 3) train-test split
    train_idx, test_idx = train_test_split(
        np.arange(len(rec_ids)),
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed
    )
    print(f"Split => train={len(train_idx)}, test={len(test_idx)}")

    # We'll store train_list = list of (patient_id, rec_id, lbl)
    train_list = [(pt_ids[i], rec_ids[i], labels[i]) for i in train_idx]
    test_list  = [(pt_ids[i], rec_ids[i], labels[i]) for i in test_idx]

    # 4) Build Dataset => We'll pass mode='train' or 'test'
    train_ds = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=train_list,
        mode='train',
        sample_rate=args.sfreq,
        window_sec=args.epoch_length,
        step_sec=args.epoch_length,
        n_channels=19,  # TUEG subset
        scale_factor=1/100.0,
        train_iterations=5000,
        verbose=False
    )

    test_ds = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=test_list,
        mode='test',
        sample_rate=args.sfreq,
        window_sec=args.epoch_length,
        step_sec=args.epoch_length,
        n_channels=19,
        scale_factor=1/100.0,
        train_iterations=5000,
        verbose=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=train_ds.collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=test_ds.collate_fn
    )

    # We'll treat test as val & test for this example
    data_loader_dict = {
        "train": train_loader,
        "val":   test_loader,
        "test":  test_loader
    }

    # print dataset sample
    # print(f"Train dataset sample: {train_ds[0]}")

    # 5) Build model
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model = Model(args).to(device)
    print("Model created. #Params:", sum(p.numel() for p in model.parameters()))

    # 6) Build trainer
    trainer = Trainer(params=args, data_loader=data_loader_dict, model=model)

    # 7) Decide training function based on #classes
    if args.num_of_classes > 2:
        trainer.train_for_multiclass()
    elif args.num_of_classes == 2:
        trainer.train_for_binaryclass()
    else:
        trainer.train_for_regression()

    # 8) (Optional) Save final model to `args.model_out`
    # torch.save(model.state_dict(), args.model_out)
    print(f"Finetune done. Model saved => {args.model_out}")

if __name__ == "__main__":
    main()
