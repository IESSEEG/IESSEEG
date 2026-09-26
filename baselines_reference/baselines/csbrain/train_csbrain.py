#!/usr/bin/env python
"""Fine-tune CSBrain on one patient-disjoint IESSEEG task and fold."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from csbrain_model import build_model
from iesseeg.foundation_transfer import train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--save_preds_dir", default=None)
    parser.add_argument("--label_key", required=True)
    parser.add_argument("--model_name", default="CSBrain")
    parser.add_argument("--epoch_length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train_iterations", type=int, default=2500)
    parser.add_argument("--val_step_sec", type=int, default=240)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", required=True)
    train_model(parser.parse_args(), build_model)


if __name__ == "__main__":
    main()
