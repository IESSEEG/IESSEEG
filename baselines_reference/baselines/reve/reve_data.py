"""Adapting IESSEEG recordings to REVE's expected input.

REVE is the least intrusive of the pre-trained baselines to adapt: it is
trained at **200 Hz**, which is IESSEEG's native rate, and it takes
electrode positions explicitly rather than assuming a fixed montage. So
unlike LUNA (montage reorder + polarity fix + resample) and EEGPT
(rename + resample), REVE needs only:

1. Channel naming. Our referential tree suffixes channels with `-REF`
   (`FP1-REF`); REVE's position bank is keyed on the bare electrode name.
2. Positions. Taken from the official `brain-bzh/reve-positions` bank
   rather than derived locally, so the coordinates are exactly the ones
   the model was pre-trained against.

No resampling is applied, so the signal reaching the model is the
released data itself.
"""

import numpy as np

# The 19-channel referential montage, in the order our tree stores it.
REVE_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
]

# REVE is pre-trained at 200 Hz, which is what IESSEEG already stores.
REVE_SAMPLE_RATE = 200
SOURCE_SAMPLE_RATE = 200

POSITION_BANK_REPO = "brain-bzh/reve-positions"


def normalize_channel_name(name):
    """`FP1-REF` -> `FP1`; leaves already-bare names alone."""
    name = str(name).upper().replace(" ", "")
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_channel_map(source_channels):
    """Index of each REVE channel within the source recording."""
    lookup = {}
    for index, name in enumerate(source_channels):
        lookup.setdefault(normalize_channel_name(name), index)

    missing = [name for name in REVE_CHANNELS if name not in lookup]
    if missing:
        raise ValueError(
            f"Recording montage is missing REVE channels {missing}. "
            f"Available: {[normalize_channel_name(c) for c in source_channels]}"
        )
    return np.asarray([lookup[name] for name in REVE_CHANNELS])


def channel_positions(channel_names=None):
    """(C, 3) electrode coordinates from REVE's official position bank.

    Using the released bank rather than deriving coordinates from a
    standard montage keeps the positions identical to the ones the model
    saw in pre-training, which matters because REVE encodes them directly.
    """
    from transformers import AutoModel

    names = list(channel_names or REVE_CHANNELS)
    bank = AutoModel.from_pretrained(POSITION_BANK_REPO, revision="befa5b57a455b77cf302daf610c2e9ed8140bace", trust_remote_code=True)
    positions = bank(names)

    if positions.shape[0] != len(names):
        raise ValueError(
            f"Position bank resolved {positions.shape[0]} of {len(names)} channels; "
            f"a channel name is not in REVE's vocabulary."
        )
    return positions.detach().float()


def make_recording_transform(source_rate=SOURCE_SAMPLE_RATE, target_rate=REVE_SAMPLE_RATE):
    """Recording-level transform: select and order channels.

    Included for symmetry with the other adapters. Because REVE's rate
    matches ours, this resamples only if a caller supplies a different
    source rate, and is otherwise a pure channel selection.
    """
    from math import gcd

    divisor = gcd(target_rate, source_rate)
    up, down = target_rate // divisor, source_rate // divisor

    def transform(raw_data, source_channels):
        selected = raw_data[build_channel_map(source_channels), :]
        if up == down:
            return selected
        from scipy.signal import resample_poly

        return resample_poly(selected, up, down, axis=-1)

    return transform
