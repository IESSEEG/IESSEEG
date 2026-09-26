"""BIOT window loading, bound to the shared dataset implementation.

The windowing, in-memory preloading and batching live in
`iesseeg.data.raw_dataset`. This module exists so the BIOT scripts keep
importing `InMemoryRandomDataset` from their own package while there is
only one implementation to keep correct.

BIOT's window policy is `quantile`: each channel is divided by its own 95th-percentile amplitude, which is
robust to the electrode pops and movement transients common in infant EEG.
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from iesseeg.data.raw_dataset import (  # noqa: F401
    WINDOW_TRANSFORMS, build_dataloaders, load_recording,
)
from iesseeg.data.raw_dataset import InMemoryRandomDataset as _SharedDataset

WINDOW_TRANSFORM = "quantile"

# Kept under its historical name for backwards compatibility with scripts
# and notebooks that imported it from here.
load_data_by_recording_id = load_recording


class InMemoryRandomDataset(_SharedDataset):
    """Shared dataset with BIOT's window transform applied by default."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("window_transform", WINDOW_TRANSFORM)
        super().__init__(*args, **kwargs)
