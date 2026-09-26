"""CBraMod window loading, bound to the shared dataset implementation.

The windowing, in-memory preloading and batching live in
`iesseeg.data.raw_dataset`. This module exists so the CBraMod scripts keep
importing `InMemoryRandomDataset` from their own package while there is
only one implementation to keep correct.

CBraMod's window policy is `patch`: the window is reshaped into one-second patches, (C, T) -> (C, seconds,
samples per second), which is the tokenisation its encoder expects.
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

WINDOW_TRANSFORM = "patch"

# Kept under its historical name for backwards compatibility with scripts
# and notebooks that imported it from here.
load_data_by_recording_id = load_recording


class InMemoryRandomDataset(_SharedDataset):
    """Shared dataset with CBraMod's window transform applied by default."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("window_transform", WINDOW_TRANSFORM)
        super().__init__(*args, **kwargs)
