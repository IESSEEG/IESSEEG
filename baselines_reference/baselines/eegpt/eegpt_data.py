"""Adapting IESSEEG recordings to EEGPT's expected input.

EEGPT is pre-trained on referential 10-20 style montages at 250 Hz and
consumes short windows (4 s by default). IESSEEG's
`scalp_eeg_data_200HZ_np_format_labram` tree is already the 19-channel
referential 10-20 montage, so only two adjustments are needed:

1. Channel naming. Our tree suffixes every channel with `-REF`
   (`FP1-REF`); EEGPT's channel vocabulary uses the bare electrode name.
   The set and order otherwise match the standard 19, so this is a rename
   rather than a remap.
2. Sample rate. Recordings are resampled 200 -> 250 Hz so that EEGPT's
   patch spans the same interval it did in pre-training. Exactly 5/4, so
   polyphase resampling is exact.

Amplitudes are standardised with braindecode's exponential moving
standardization, which is what EEGPT's own downstream pipeline applies.
"""

import numpy as np
from scipy.signal import resample_poly

# The 19-channel referential montage EEGPT's channel vocabulary expects,
# in the order our preprocessed tree stores them.
EEGPT_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
]

EEGPT_SAMPLE_RATE = 250
SOURCE_SAMPLE_RATE = 200


def normalize_channel_name(name):
    """`FP1-REF` -> `FP1`; leaves already-bare names alone."""
    name = str(name).upper().replace(" ", "")
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def build_channel_map(source_channels):
    """Index of each EEGPT channel within the source recording.

    Raises if any is absent, so a montage mismatch fails at startup
    instead of silently training on the wrong channels.
    """
    lookup = {}
    for index, name in enumerate(source_channels):
        lookup.setdefault(normalize_channel_name(name), index)

    missing = [name for name in EEGPT_CHANNELS if name not in lookup]
    if missing:
        raise ValueError(
            f"Recording montage is missing EEGPT channels {missing}. "
            f"Available: {[normalize_channel_name(c) for c in source_channels]}"
        )
    return np.asarray([lookup[name] for name in EEGPT_CHANNELS])


def make_recording_transform(source_rate=SOURCE_SAMPLE_RATE,
                             target_rate=EEGPT_SAMPLE_RATE,
                             standardize=True):
    """Recording-level transform: select channels, resample, standardise.

    Suitable for `InMemoryRandomDataset(recording_transform=...)`, which
    applies it once per recording at load time. Standardisation is done
    on the continuous recording rather than per window, matching how a
    running standardiser is normally used.
    """
    from math import gcd

    divisor = gcd(target_rate, source_rate)
    up, down = target_rate // divisor, source_rate // divisor

    def transform(raw_data, source_channels):
        indices = build_channel_map(source_channels)
        selected = raw_data[indices, :]

        if up != down:
            selected = resample_poly(selected, up, down, axis=-1)

        if standardize:
            from braindecode.preprocessing import exponential_moving_standardize

            selected = exponential_moving_standardize(
                np.ascontiguousarray(selected, dtype=np.float64)
            )
        return selected

    return transform
