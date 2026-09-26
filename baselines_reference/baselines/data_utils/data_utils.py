"""Compatibility shim for the per-model training and inference scripts.

Label encoding lives in `iesseeg.data.splits.encode_labels`, so that the
mapping from the CSV's categorical labels to binary targets has exactly
one definition. The upstream model scripts call it under its original
name; this keeps that call site working without giving the benchmark two
implementations that could disagree about what counts as a responder.
"""

import os
import sys

import pandas as pd

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from iesseeg.data.splits import encode_labels


def create_label_from_meta_csv(meta_csv: pd.DataFrame, label_key: str):
    """Binary targets for one task, given a split dataframe.

    Args:
        meta_csv: A loaded split CSV (despite the name, a dataframe).
        label_key: The task's label column, e.g. "case_control_label".

    Returns:
        numpy array of 0/1 labels, one per row.
    """
    return encode_labels(meta_csv, label_key)
