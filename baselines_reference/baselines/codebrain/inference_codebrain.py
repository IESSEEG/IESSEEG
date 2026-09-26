#!/usr/bin/env python
"""Run clip-level inference with a fine-tuned CodeBrain model."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from codebrain_model import build_model
from iesseeg.foundation_transfer import infer_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--label_key", required=True)
    parser.add_argument("--epoch_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cuda", required=True)
    infer_model(parser.parse_args(), build_model)


if __name__ == "__main__":
    main()
