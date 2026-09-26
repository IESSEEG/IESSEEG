"""Signal-processing primitives adapted from Smith (2021) and Rajaraman (2024)."""
import math
import re
import numpy as np
from scipy.signal import butter, lfilter, firls, hilbert

SCALP = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "FZ",
    "CZ",
    "PZ",
]

def canonical_channel(name: str) -> str:
    """Return the electrode token from an EDF channel label."""
    upper = name.upper().replace("-REF", "")
    tokens = re.findall(r"[A-Z]+\d*", upper)
    return tokens[-1] if tokens else upper


def find_channel(ch_names: list[str], target: str) -> int:
    matches = [
        i for i, name in enumerate(ch_names) if canonical_channel(name) == target
    ]
    eeg_matches = [i for i in matches if ch_names[i].upper().startswith("EEG ")]
    if len(eeg_matches) == 1:
        return eeg_matches[0]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {target} channel, found {[ch_names[i] for i in matches]}"
        )
    return matches[0]


def matlab_firls_approximation(kind: str, fs: float) -> np.ndarray:
    """Approximate MATLAB ``firls(order, ...)`` with an odd-tap SciPy design.

    MATLAB accepts the even number of taps produced for beta at 200 Hz,
    whereas SciPy's ``firls`` requires an odd number.  The nearest odd length
    is used and recorded in every output file.
    """
    nyquist = fs / 2.0
    if kind == "beta":
        bands = np.array([0, 11, 14, 30, 35, nyquist], dtype=float) / nyquist
        desired = np.array([0, 0, 1, 1, 0, 0], dtype=float)
        lowest = 14.0
    elif kind == "delta":
        bands = np.array([0, 0.2, 1, 4, 6, nyquist], dtype=float) / nyquist
        desired = np.array([0, 0, 1, 1, 0, 0], dtype=float)
        lowest = 1.0
    else:
        raise ValueError(kind)
    matlab_order = int(math.ceil(2.0 * fs / lowest))
    matlab_taps = matlab_order + 1
    scipy_taps = matlab_taps if matlab_taps % 2 else matlab_taps + 1
    return firls(scipy_taps, bands, desired)


def automated_clean_seconds(
    scalp_uv: np.ndarray, ears_uv: np.ndarray, fs: int
) -> np.ndarray:
    """Reproduce the published automatic extreme-value artifact detector."""
    ear_mean = ears_uv.mean(axis=0, keepdims=True)
    linked_ear_21 = np.concatenate([scalp_uv, ears_uv], axis=0) - ear_mean

    cutoff = 1.0 / (2.0 * np.pi * 0.1)
    b_high, a_high = butter(1, cutoff / (fs / 2.0), btype="high")
    b_low, a_low = butter(3, 40.0 / (fs / 2.0), btype="low")
    viewed = lfilter(b_high, a_high, linked_ear_21, axis=1)
    viewed = -lfilter(b_low, a_low, viewed, axis=1)

    means = viewed.mean(axis=1, keepdims=True)
    stds = viewed.std(axis=1, ddof=1, keepdims=True)
    stds = np.maximum(stds, 200.0 / 7.5)
    extreme = np.any(np.abs(viewed - means) > 7.5 * stds, axis=0)

    # The MATLAB code pads every detected sample by 0.9 seconds on both sides.
    pad = int(round(0.9 * fs))
    if extreme.any():
        hits = np.flatnonzero(extreme)
        difference = np.zeros(extreme.size + 1, dtype=np.int32)
        np.add.at(difference, np.maximum(0, hits - pad), 1)
        np.add.at(difference, np.minimum(extreme.size, hits + pad + 1), -1)
        extreme = np.cumsum(difference[:-1]) > 0

    # Impedance checks are runs where more than eight channels are unchanged.
    unchanged = np.sum(np.diff(linked_ear_21, axis=1) == 0, axis=0) > 8
    impedance = np.zeros(extreme.size, dtype=bool)
    impedance[1:] = unchanged
    artifact_samples = extreme | impedance

    n_seconds = scalp_uv.shape[1] // fs
    artifact_seconds = (
        artifact_samples[: n_seconds * fs].reshape(n_seconds, fs).any(axis=1)
    )
    return ~artifact_seconds


def concatenate_clean_seconds(
    data: np.ndarray, clean_seconds: np.ndarray, fs: int
) -> np.ndarray:
    n_seconds = len(clean_seconds)
    blocks = data[:, : n_seconds * fs].reshape(data.shape[0], n_seconds, fs)
    return blocks[:, clean_seconds].reshape(data.shape[0], -1)


def contiguous_clean_epochs(
    data: np.ndarray,
    clean_seconds: np.ndarray,
    fs: int,
    epoch_seconds: int = 8,
) -> np.ndarray:
    """Cut complete epochs without joining EEG across artifact gaps.

    Artifact detection marks whole one-second blocks.  Each run of consecutive
    clean blocks is tiled independently, so two samples that were separated by
    an excluded block can never become neighbors in an epoch.
    """
    clean_seconds = np.asarray(clean_seconds, dtype=bool)
    epoch_samples = epoch_seconds * fs
    padded = np.r_[False, clean_seconds, False]
    run_starts = np.flatnonzero(~padded[:-1] & padded[1:])
    run_stops = np.flatnonzero(padded[:-1] & ~padded[1:])

    epochs = []
    for start_second, stop_second in zip(run_starts, run_stops):
        n_epochs = (stop_second - start_second) // epoch_seconds
        if n_epochs == 0:
            continue
        start_sample = start_second * fs
        stop_sample = start_sample + n_epochs * epoch_samples
        run = data[:, start_sample:stop_sample]
        run_epochs = run.reshape(data.shape[0], n_epochs, epoch_samples)
        epochs.append(np.moveaxis(run_epochs, 1, 0))

    if not epochs:
        return np.empty((0, data.shape[0], epoch_samples), dtype=data.dtype)
    return np.concatenate(epochs, axis=0)


def matlab_hist_entropy(signal: np.ndarray, bins: int = 350) -> np.ndarray:
    """Shannon entropy using MATLAB ``hist(x, bins)``-style bin centers."""
    values = np.empty(signal.shape[0], dtype=float)
    for channel, x in enumerate(signal):
        lo, hi = float(np.min(x)), float(np.max(x))
        if lo == hi:
            values[channel] = 0.0
            continue
        centers = np.linspace(lo, hi, bins)
        edges = np.r_[-np.inf, (centers[:-1] + centers[1:]) / 2.0, np.inf]
        counts = np.histogram(x, bins=edges)[0].astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        values[channel] = -np.sum(probabilities * np.log2(probabilities))
    return values


def fluctuation_function(profile: np.ndarray, window_sizes: np.ndarray) -> np.ndarray:
    centered_profile = np.cumsum(profile - np.mean(profile))
    length = centered_profile.size
    sample_index = np.arange(length, dtype=float)
    prefix_y = np.r_[0.0, np.cumsum(centered_profile)]
    prefix_y2 = np.r_[0.0, np.cumsum(centered_profile * centered_profile)]
    prefix_ty = np.r_[0.0, np.cumsum(sample_index * centered_profile)]
    result = np.full(window_sizes.size, np.nan, dtype=float)
    for index, width in enumerate(window_sizes):
        step = max(1, int(round(width * 0.5)))
        starts = np.arange(0, length - width, step, dtype=int)
        if starts.size == 0:
            continue
        stops = starts + width + 1
        count = float(width + 1)
        sum_y = prefix_y[stops] - prefix_y[starts]
        sum_y2 = prefix_y2[stops] - prefix_y2[starts]
        # Convert the global sample index to the local 0..width index used by
        # MATLAB's linspace.  Scaling x does not change the regression SSE.
        sum_xy = prefix_ty[stops] - prefix_ty[starts] - starts * sum_y
        mean_x = width / 2.0
        sxx = width * (width + 1.0) * (width + 2.0) / 12.0
        syy = np.maximum(sum_y2 - sum_y * sum_y / count, 0.0)
        sxy = sum_xy - mean_x * sum_y
        sse = np.maximum(syy - sxy * sxy / sxx, 0.0)
        deviations = np.sqrt(sse / count)
        result[index] = np.median(deviations)
    return result


def dfa_intercept(signal: np.ndarray, fs: int) -> np.ndarray:
    envelope = np.abs(hilbert(signal, axis=1))
    high = min(envelope.shape[1] / 10.0, 120.0 * fs)
    low = float(fs)
    if high <= low:
        raise ValueError("Not enough clean EEG for DFA")
    windows = np.round(np.logspace(np.log10(low), np.log10(high), 20)).astype(int)
    output = np.empty(signal.shape[0], dtype=float)
    for channel in range(signal.shape[0]):
        fluctuation = fluctuation_function(envelope[channel], windows)
        valid = np.isfinite(fluctuation) & (fluctuation > 0)
        output[channel] = np.polyfit(
            np.log10(windows[valid]), np.log10(fluctuation[valid]), 1
        )[1]
    return output


def analytic_phase_torch(signal, torch):
    """Hilbert analytic-signal phase along the final axis."""
    n = signal.shape[-1]
    multiplier = torch.zeros(n, device=signal.device, dtype=signal.dtype)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[1 : n // 2] = 2.0
        multiplier[n // 2] = 1.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    analytic = torch.fft.ifft(torch.fft.fft(signal, dim=-1) * multiplier, dim=-1)
    return torch.angle(analytic)
