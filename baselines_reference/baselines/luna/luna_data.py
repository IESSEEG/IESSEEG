"""Adapting IESSEEG recordings to LUNA's expected input.

LUNA was pre-trained on TUH EEG in the 22-channel TCP bipolar montage at
256 Hz, and it consumes 3D electrode coordinates alongside the signal.
IESSEEG's `scalp_eeg_data_200HZ_np_format` tree is already the same 22
bipolar derivations at 200 Hz, so no new preprocessing tree is needed --
three adjustments bridge the two:

1. Channel order. Our tree lists the derivations in a different order
   than LUNA's TCP montage, and six of the central-chain derivations are
   written with the opposite polarity (we store `A2-T4`, LUNA expects
   `T4-A2`). Reversing a bipolar derivation negates its signal, so those
   channels are reordered and sign-flipped rather than dropped.
2. Sample rate. Recordings are resampled 200 -> 256 Hz so that LUNA's
   40-sample patch spans the same 156 ms of signal it did in
   pre-training. Done once per recording at load time.
3. Electrode coordinates. Each bipolar derivation is located at the
   midpoint of its two electrodes in the standard_1005 montage, matching
   how the upstream code derives positions for TUH channels.
"""

import numpy as np
from scipy.signal import resample_poly

# LUNA's TCP bipolar montage, in the order its pre-training used.
TCP_MONTAGE = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

LUNA_SAMPLE_RATE = 256
SOURCE_SAMPLE_RATE = 200


def build_channel_map(source_channels):
    """Map our channel order onto LUNA's TCP order.

    Returns (indices, signs): source channel index and polarity (+1/-1)
    for each of LUNA's 22 derivations. Raises if any is unavailable, so a
    montage mismatch fails loudly at startup instead of silently training
    on misaligned channels.
    """
    lookup = {}
    for index, name in enumerate(source_channels):
        name = str(name).upper().replace(" ", "")
        if "-" not in name:
            continue
        anode, cathode = name.split("-", 1)
        lookup[(anode, cathode)] = (index, 1.0)
        # The reversed derivation is the same pair with inverted sign.
        lookup.setdefault((cathode, anode), (index, -1.0))

    indices, signs = [], []
    missing = []
    for target in TCP_MONTAGE:
        anode, cathode = target.split("-")
        if (anode, cathode) not in lookup:
            missing.append(target)
            continue
        index, sign = lookup[(anode, cathode)]
        indices.append(index)
        signs.append(sign)

    if missing:
        raise ValueError(
            f"Recording montage is missing LUNA channels {missing}. "
            f"Available: {list(source_channels)}"
        )

    return np.asarray(indices), np.asarray(signs, dtype=np.float32)


def channel_locations():
    """3D coordinates for LUNA's 22 derivations, as (22, 3) float32.

    Each derivation sits at the midpoint of its two electrodes. Imported
    lazily so that merely importing this module does not pull in MNE.
    """
    import mne

    electrodes = sorted({part for name in TCP_MONTAGE for part in name.split("-")})
    info = mne.create_info(ch_names=electrodes, sfreq=LUNA_SAMPLE_RATE, ch_types="eeg")
    info = info.set_montage(
        mne.channels.make_standard_montage("standard_1005"), match_case=False
    )
    positions = info.get_montage().get_positions()["ch_pos"]

    locations = []
    for name in TCP_MONTAGE:
        anode, cathode = name.split("-")
        locations.append((positions[anode] + positions[cathode]) / 2.0)
    return np.stack(locations).astype(np.float32)


def make_recording_transform(source_rate=SOURCE_SAMPLE_RATE, target_rate=LUNA_SAMPLE_RATE):
    """Recording-level transform: reorder/flip channels, then resample.

    Suitable for `InMemoryRandomDataset(recording_transform=...)`, which
    applies it once per recording at load time.
    """
    # 200 -> 256 Hz reduces to 32/25, so polyphase resampling is exact
    # rather than an interpolation onto an irrational grid.
    from math import gcd
    divisor = gcd(target_rate, source_rate)
    up, down = target_rate // divisor, source_rate // divisor

    def transform(raw_data, source_channels):
        indices, signs = build_channel_map(source_channels)
        reordered = raw_data[indices, :] * signs[:, None]
        if up == down:
            return reordered
        return resample_poly(reordered, up, down, axis=-1)

    return transform


def normalize_window(window, eps=1e-8):
    """Channel-wise z-score over time, as LUNA's fine-tuning task does."""
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    return (window - mean) / (std + eps)
