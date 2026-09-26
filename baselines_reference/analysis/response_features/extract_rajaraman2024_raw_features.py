#!/usr/bin/env python
"""Re-extract the Rajaraman et al. (2024) EEG response features.

This is an independent implementation from the published methods and the
public Smith et al. (2021) MATLAB routines. It reads
the original EDF clips in physical microvolts, applies the published montage,
filter, and automated artifact rules, and extracts only the four quantities
used by the two published response scores:

* PRE awake beta-band DFA intercept;
* PRE awake delta-band PLI connectivity;
* POST sleep beta-band Shannon entropy; and
* POST awake beta-band DFA intercept.

The source study also removed clinician-marked sleep artifacts.  Those marks
are not present in the released project data, so this script can reproduce the
automatic portion of preprocessing but cannot be an exact bitwise recreation
of the source analysis.

With ``--cell-scope all``, the same three feature definitions are computed for
all PRE/POST and awake/sleep cells as an exploratory complete grid. PLI is
computed on CUDA and the program exits if CUDA is unavailable. DFA and entropy
can be extracted separately on CPU with ``--features beta``. Each clip is
cached as a small JSON file so interrupted runs resume without recomputing
completed clips.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
import os

# MNE imports Numba functions with caching enabled.  The environment's package
# directory is read-only, so direct its cache to a writable local temporary
# directory before importing MNE.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/iesseeg_numba")

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, lfilter, firls, resample_poly

from rajaraman2024_contract import PLI_CONNECTIVITY_POLICY, PLI_EPOCH_POLICY


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = Path(os.environ.get("IESSEEG_QEEG_OUTPUT", "local/qeeg"))

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


def normalize_recording_id(value: object) -> str:
    text = str(value).strip().upper()
    return text[:-2] if text.endswith(".0") else text


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


def resolve_edf(edf_dir: Path, recording_id: str) -> Path:
    direct = edf_dir / f"{recording_id}.edf"
    if direct.exists():
        return direct
    matches = [
        path
        for path in edf_dir.glob("*.edf")
        if path.stem.casefold() == recording_id.casefold()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one case-insensitive EDF match for {recording_id}: {matches}"
        )
    return matches[0]


def load_edf_microvolts(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    source_fs = float(raw.info["sfreq"])
    scalp_indices = [find_channel(raw.ch_names, name) for name in SCALP]
    ear_indices = [find_channel(raw.ch_names, "A1"), find_channel(raw.ch_names, "A2")]
    # MNE exposes EDF physical values in volts.  The source MATLAB artifact
    # detector and the published absolute threshold are in microvolts.
    values_uv = raw.get_data(picks=scalp_indices + ear_indices) * 1e6
    scalp_uv = values_uv[: len(SCALP)].astype(np.float64, copy=False)
    ears_uv = values_uv[len(SCALP) :].astype(np.float64, copy=False)
    analysis_fs = 200.0
    if not math.isclose(source_fs, analysis_fs):
        source_int = int(round(source_fs))
        analysis_int = int(round(analysis_fs))
        if not math.isclose(source_fs, source_int):
            raise ValueError(
                f"{path.name}: unsupported non-integer sampling rate {source_fs}"
            )
        divisor = math.gcd(source_int, analysis_int)
        up, down = analysis_int // divisor, source_int // divisor
        scalp_uv = resample_poly(scalp_uv, up, down, axis=1)
        ears_uv = resample_poly(ears_uv, up, down, axis=1)
    return scalp_uv, ears_uv, analysis_fs, source_fs


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


def pli_matrix_from_phase_torch(phase, torch):
    """PLI matrices for phase shaped (batch, channels, time)."""
    batch, channels, _ = phase.shape
    output = torch.zeros(
        (batch, channels, channels), device=phase.device, dtype=torch.float32
    )
    for first in range(channels):
        differences = phase[:, first : first + 1, :] - phase[:, first + 1 :, :]
        values = torch.abs(torch.sign(torch.sin(differences)).mean(dim=-1))
        output[:, first, first + 1 :] = values
        output[:, first + 1 :, first] = values
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


def gpu_pli_connectivity(
    clean_epochs: np.ndarray,
    fs: int,
    n_surrogates: int,
    seed: int,
) -> dict[str, float | int]:
    """Compute the source clip metric and two surrogate-threshold diagnostics."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PLI extraction requires CUDA; refusing CPU fallback")
    device = torch.device("cuda")
    epoch_samples = 8 * fs
    if clean_epochs.ndim != 3 or clean_epochs.shape[1:] != (19, epoch_samples):
        raise ValueError(
            "Expected clean PLI epochs shaped "
            f"(n_epochs, 19, {epoch_samples}), got {clean_epochs.shape}"
        )
    n_epochs = clean_epochs.shape[0]
    if n_epochs < 1:
        raise ValueError("No complete clean 8-second epoch for PLI")
    epochs = clean_epochs.astype(np.float32, copy=False)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    raw_sum = torch.zeros((19, 19), device=device)
    retained_sum = torch.zeros((19, 19), device=device) if n_surrogates > 0 else None
    significant_sum = torch.zeros((19, 19), device=device) if n_surrogates > 0 else None

    for epoch_np in epochs:
        epoch = torch.as_tensor(epoch_np, device=device)
        observed_phase = analytic_phase_torch(epoch, torch)[None, ...]
        observed = pli_matrix_from_phase_torch(observed_phase, torch)[0]
        raw_sum += observed

        if n_surrogates > 0:
            spectrum = torch.fft.rfft(epoch, dim=-1)
            magnitude = torch.abs(spectrum)[None, ...]
            random_phase = (
                2.0
                * torch.pi
                * torch.rand(
                    (n_surrogates, 19, spectrum.shape[-1]),
                    generator=generator,
                    device=device,
                )
            )
            random_phase[..., 0] = torch.angle(spectrum[:, 0])[None, :]
            if epoch_samples % 2 == 0:
                random_phase[..., -1] = torch.angle(spectrum[:, -1])[None, :]
            randomized = torch.fft.irfft(
                magnitude * torch.exp(1j * random_phase),
                n=epoch_samples,
                dim=-1,
            )
            surrogate = pli_matrix_from_phase_torch(
                analytic_phase_torch(randomized, torch), torch
            )
            threshold = torch.quantile(surrogate, 0.95, dim=0)
            significant = observed > threshold
            retained_sum += observed * significant
            significant_sum += significant

    pair = torch.triu_indices(19, 19, offset=1, device=device)
    raw_network = raw_sum / n_epochs
    result = {
        "pli_n_clean_8s_epochs": int(n_epochs),
        # Rajaraman et al.'s C0: average epoch PLI for every electrode pair,
        # then report the percentage of pair means above 0.20.  This definition
        # reproduces the article's reported group distribution.
        "connectivity_percent_raw_pli": float(
            100.0 * (raw_network[pair[0], pair[1]] > 0.20).float().mean().item()
        ),
    }
    if n_surrogates > 0:
        retained_network = retained_sum / n_epochs
        frequency_network = significant_sum / n_epochs
        result.update(
            {
                # Smith et al.'s surrogate-threshold wording, retained as a diagnostic.
                "connectivity_percent_retained_pli": float(
                    100.0
                    * (retained_network[pair[0], pair[1]] > 0.20).float().mean().item()
                ),
                # Sensitivity reading: each significant edge is binary before the
                # epoch average, making the network value a detection frequency.
                "connectivity_percent_significance_frequency": float(
                    100.0
                    * (frequency_network[pair[0], pair[1]] > 0.20).float().mean().item()
                ),
            }
        )
    return result


def extract_one(
    row,
    edf_dir: Path,
    feature_set: str,
    n_surrogates: int,
    seed: int,
    cell_scope: str = "source",
) -> dict[str, object]:
    recording_id = normalize_recording_id(row.short_recording_id)
    edf_path = resolve_edf(edf_dir, recording_id)
    scalp_uv, ears_uv, fs_float, source_fs = load_edf_microvolts(edf_path)
    fs = int(round(fs_float))
    clean_seconds = automated_clean_seconds(scalp_uv, ears_uv, fs)
    ear_mean = ears_uv.mean(axis=0, keepdims=True)
    linked_ear = scalp_uv - ear_mean
    output: dict[str, object] = {
        "recording_id": recording_id,
        "patient_id": int(row.patient_id),
        "condition": row.pre_post_treatment_label,
        "state": row.sleep_awake_label,
        "n_total_seconds": int(len(clean_seconds)),
        "n_clean_seconds": int(clean_seconds.sum()),
        "clean_fraction": float(clean_seconds.mean()),
        "input_unit": "microvolts",
        "source_sampling_rate_hz": source_fs,
        "analysis_sampling_rate_hz": fs_float,
        "resampling_note": (
            "none"
            if math.isclose(source_fs, fs_float)
            else "polyphase anti-alias resampling to the published 200 Hz analysis rate"
        ),
        "beta_filter_note": "nearest odd-tap SciPy firls approximation to MATLAB firls",
        "clinician_sleep_artifacts_available": False,
    }

    needs_beta = (
        cell_scope == "all"
        or (row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE")
        or (
            row.pre_post_treatment_label == "POST"
            and row.sleep_awake_label in {"AWAKE", "SLEEP"}
        )
    )
    if feature_set in {"beta", "all"} and needs_beta:
        beta = filtfilt(
            matlab_firls_approximation("beta", fs), [1.0], linked_ear, axis=1
        )
        clean_beta = concatenate_clean_seconds(beta, clean_seconds, fs)
        if cell_scope == "all" or row.sleep_awake_label == "AWAKE":
            channel_dfa = dfa_intercept(clean_beta, fs)
            output["beta_dfa_intercept_channels"] = channel_dfa.tolist()
            output["beta_dfa_intercept_mean"] = float(np.mean(channel_dfa))
        if cell_scope == "all" or (
            row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "SLEEP"
        ):
            channel_entropy = matlab_hist_entropy(clean_beta, bins=350)
            output["beta_entropy_channels"] = channel_entropy.tolist()
            output["beta_entropy_mean"] = float(np.mean(channel_entropy))

    needs_pli = cell_scope == "all" or (
        row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE"
    )
    if feature_set in {"pli", "all"} and needs_pli:
        car = scalp_uv - scalp_uv.mean(axis=0, keepdims=True)
        delta = filtfilt(matlab_firls_approximation("delta", fs), [1.0], car, axis=1)
        clean_epochs = contiguous_clean_epochs(
            delta, clean_seconds, fs, epoch_seconds=8
        )
        output.update(gpu_pli_connectivity(clean_epochs, fs, n_surrogates, seed))
        if n_surrogates > 0:
            output["pli_surrogates_per_epoch"] = int(n_surrogates)
        output["pli_epoch_policy"] = PLI_EPOCH_POLICY
        output["pli_connectivity_policy"] = PLI_CONNECTIVITY_POLICY
    return output


def target_rows(
    metadata: pd.DataFrame, feature_set: str, cell_scope: str = "source"
) -> pd.DataFrame:
    cases = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    if cell_scope == "all":
        return cases
    beta = (
        cases.pre_post_treatment_label.eq("PRE") & cases.sleep_awake_label.eq("AWAKE")
    ) | cases.pre_post_treatment_label.eq("POST")
    pli = cases.pre_post_treatment_label.eq("PRE") & cases.sleep_awake_label.eq("AWAKE")
    if feature_set == "beta":
        return cases.loc[beta]
    if feature_set == "pli":
        return cases.loc[pli]
    return cases.loc[beta | pli]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=["beta", "pli", "all"], default="all")
    parser.add_argument(
        "--cell-scope",
        choices=["source", "all"],
        default="source",
        help="Use source-selected cells or compute a complete PRE/POST by awake/sleep grid.",
    )
    parser.add_argument("--n-surrogates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202409)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help=(
            "Local clip metadata containing patient_id, short_recording_id, "
            "case_control_label, pre_post_treatment_label, sleep_awake_label, "
            "and meaningful_responder. This governed file is not distributed."
        ),
    )
    parser.add_argument(
        "--edf-dir",
        type=Path,
        required=True,
        help="Directory containing the source Clinical Clip EDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_RESULTS / "raw_features",
    )
    args = parser.parse_args()
    if args.n_surrogates < 0:
        raise ValueError("--n-surrogates must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.metadata_csv.is_file():
        raise FileNotFoundError(args.metadata_csv)
    if not args.edf_dir.is_dir():
        raise NotADirectoryError(args.edf_dir)

    metadata = pd.read_csv(args.metadata_csv)
    rows = target_rows(metadata, args.features, args.cell_scope)
    if args.limit is not None:
        rows = rows.iloc[: args.limit]
    print(
        f"Target clips: {len(rows)} ({args.features}); output: {args.output_dir}",
        flush=True,
    )

    for position, row in enumerate(rows.itertuples(index=False), start=1):
        recording_id = normalize_recording_id(row.short_recording_id)
        path = args.output_dir / f"{recording_id}.json"
        existing: dict[str, object] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        requires_dfa = args.cell_scope == "all" or row.sleep_awake_label == "AWAKE"
        requires_entropy = args.cell_scope == "all" or (
            row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "SLEEP"
        )
        beta_done = (not requires_dfa or "beta_dfa_intercept_mean" in existing) and (
            not requires_entropy or "beta_entropy_mean" in existing
        )
        requires_pli = args.cell_scope == "all" or (
            row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE"
        )
        pli_done = not requires_pli or (
            "connectivity_percent_raw_pli" in existing
            and existing.get("pli_epoch_policy") == PLI_EPOCH_POLICY
            and existing.get("pli_connectivity_policy") == PLI_CONNECTIVITY_POLICY
            and (
                args.n_surrogates == 0
                or (
                    "connectivity_percent_retained_pli" in existing
                    and "connectivity_percent_significance_frequency" in existing
                )
            )
        )
        if (
            (args.features == "beta" and beta_done)
            or (args.features == "pli" and pli_done)
            or (args.features == "all" and beta_done and pli_done)
        ):
            print(f"[{position}/{len(rows)}] {recording_id}: cached", flush=True)
            continue
        print(f"[{position}/{len(rows)}] {recording_id}: extracting", flush=True)
        result = extract_one(
            row,
            args.edf_dir,
            args.features,
            args.n_surrogates,
            args.seed + position,
            args.cell_scope,
        )
        existing.update(result)
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
