"""Continuous-input studies: source coordinates, published measures and encoders.

Source EEG is read without modification. All filtering is confined to the stated
input interval. The diagnostic panel implements Smith et al. (2021) measures;
the response panel preserves the earlier Rajaraman-derived definitions.
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/iesseeg_numba")
import mne
import numpy as np
import torch
import pyedflib
from scipy.signal import butter, sosfiltfilt, filtfilt, hilbert, resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "legacy/analysis"),
    str(ROOT / "baselines_reference/analysis/response_features")]
from extract_rajaraman2024_raw_features import (
    SCALP, find_channel, automated_clean_seconds, concatenate_clean_seconds,
    contiguous_clean_epochs, matlab_firls_approximation, matlab_hist_entropy,
    dfa_intercept, fluctuation_function, analytic_phase_torch)

SEED = 20260924
FS = 200
TRIPLET = ("smith_pli_retained_mean", "smith_beta_entropy_fd", "beta_dfa_intercept_mean")
RESPONSE = ("beta_dfa_intercept_mean", "beta_entropy_350", "raw_pli_percent_over_020")
DIAGNOSTIC = ("amplitude_median_uv", "cz_amplitude_median_uv", "power_delta_log10",
    "power_theta_log10", "power_alpha_log10", "power_beta_log10", "sef95_hz",
    "smith_beta_entropy_fd", "beta_permutation_entropy_cz", "beta_dfa_intercept_mean",
    "beta_dfa_exponent_mean", "smith_pli_retained_mean")
PAIRS = [("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),
    ("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
    ("FP1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),
    ("FP2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
    ("FZ","CZ"),("CZ","PZ")]


def resolve_edf(root, rid):
    matches = [p for p in Path(root).iterdir()
        if p.suffix.lower() == ".edf" and p.stem.casefold() == str(rid).casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact case-insensitive source for {rid}; got {matches}")
    return matches[0]


def read_interval_mne(path, start_sec=0., duration_sec=None):
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    native_fs = float(raw.info["sfreq"])
    start = int(round(start_sec * native_fs))
    stop = raw.n_times if duration_sec is None else start + int(round(duration_sec * native_fs))
    if start < 0 or stop > raw.n_times:
        raise ValueError(f"Input exceeds EDF duration: {path.name}, {start_sec}, {duration_sec}")
    picks = [find_channel(raw.ch_names, ch) for ch in [*SCALP, "A1", "A2"]]
    x = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
    if not native_fs.is_integer():
        raise ValueError("Noninteger native sampling rate")
    if native_fs != FS:
        common = math.gcd(int(native_fs), FS)
        x = resample_poly(x, FS // common, int(native_fs) // common, axis=1)
    if duration_sec is not None:
        x = x[:, :int(round(duration_sec * FS))]
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite raw EEG: {path}")
    return x.astype(np.float64, copy=False)


def read_interval(path, start_sec=0., duration_sec=None):
    """Read only the requested signal samples; skip unused EDF annotations.

    EDF physical units are converted explicitly to microvolts. This reader is
    checked against MNE on real inputs in tests/test_input_studies.py.
    """
    try:
        reader=pyedflib.EdfReader(str(path), annotations_mode=0)
    except OSError as error:
        if 'discontinuous' not in str(error):raise
        return read_interval_mne(path,start_sec,duration_sec)
    with reader as raw:
        names=raw.getSignalLabels()
        picks=[find_channel(names,ch) for ch in [*SCALP,'A1','A2']]
        signals=[]
        for pick in picks:
            fs=float(raw.getSampleFrequency(pick))
            start=int(round(start_sec*fs))
            count=int(raw.getNSamples()[pick])-start if duration_sec is None else int(round(duration_sec*fs))
            if start<0 or start+count>raw.getNSamples()[pick]:
                raise ValueError(f'Input exceeds EDF duration: {path}')
            factor={'uv':1.,'µv':1.,'μv':1.,'mv':1e3,'v':1e6}.get(raw.getPhysicalDimension(pick).strip().lower())
            if factor is None:raise ValueError('Unrecognized EDF physical unit')
            x=raw.readSignal(pick,start=start,n=count)*factor
            if not fs.is_integer():raise ValueError('Noninteger native sampling rate')
            if fs!=FS:
                common=math.gcd(int(fs),FS);x=resample_poly(x,FS//common,int(fs)//common)
            signals.append(x)
    result=np.stack(signals)
    if not np.isfinite(result).all():raise ValueError('Nonfinite raw EEG')
    return result


def dfa_both(beta):
    envelope = np.abs(hilbert(beta, axis=1))
    high = min(envelope.shape[1] / 10., 120. * FS)
    if high <= FS:
        return np.nan, np.nan, high / FS
    scales = np.round(np.logspace(np.log10(FS), np.log10(high), 20)).astype(int)
    coefficients = []
    for channel in envelope:
        values = fluctuation_function(channel, scales)
        valid = np.isfinite(values) & (values > 0)
        coefficients.append(np.polyfit(np.log10(scales[valid]), np.log10(values[valid]), 1))
    return float(np.mean(coefficients, axis=0)[1]), float(np.mean(coefficients, axis=0)[0]), high / FS


def phase_pairs(phases):
    """Return pairwise PLI; chunk pairs to bound temporary CUDA memory."""
    pair = torch.triu_indices(19, 19, 1, device=phases.device)
    values = []
    for start in range(0, pair.shape[1], 16):
        i, j = pair[:, start:start+16]
        diff = phases[:, i] - phases[:, j]
        values.append(torch.sin(diff).sign().mean(-1).abs())
    return torch.cat(values, dim=1)


def pli_features(epochs, device, surrogates=100, seed=SEED):
    if len(epochs) == 0:
        return dict(raw_pli_percent_over_020=np.nan, smith_pli_retained_mean=np.nan,
            smith_pli_retained_percent_over_020=np.nan, pli_clean_epochs=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    raw, retained = [], []
    with torch.inference_mode():
        for start in range(0, len(epochs), 4):
            x = torch.tensor(epochs[start:start+4], dtype=torch.float32, device=device)
            obs = phase_pairs(analytic_phase_torch(x, torch))
            raw.append(obs)
            if surrogates:
                spec = torch.fft.rfft(x)
                angle = 2 * torch.pi * torch.rand((len(x), surrogates, 19, spec.shape[-1]), device=device, generator=generator)
                angle[..., 0] = torch.angle(spec[..., 0])[:, None]
                angle[..., -1] = torch.angle(spec[..., -1])[:, None]
                y = torch.fft.irfft(spec.abs()[:, None] * torch.exp(1j * angle), n=x.shape[-1])
                null = phase_pairs(analytic_phase_torch(y.flatten(0, 1), torch)).reshape(len(x), surrogates, 171)
                retained.append(obs * (obs > torch.quantile(null, .95, dim=1)))
    avg = torch.cat(raw).mean(0)
    ret = torch.cat(retained).mean(0) if retained else torch.full_like(avg, float("nan"))
    return dict(raw_pli_percent_over_020=float((avg > .20).float().mean().item()*100),
        smith_pli_retained_mean=float(ret.mean().item()),
        smith_pli_retained_percent_over_020=float((ret > .20).float().mean().item()*100) if retained else np.nan,
        pli_clean_epochs=len(epochs))


def qeeg_features(x, device, diagnostic=True):
    scalp, ears = x[:19], x[19:]
    clean = automated_clean_seconds(scalp, ears, FS)
    out = dict(duration_seconds=x.shape[1]/FS, clean_seconds=int(clean.sum()), clean_fraction=float(clean.mean()))
    if clean.sum() <= 10:
        out.update({k: np.nan for k in set(RESPONSE+DIAGNOSTIC)})
        return out
    linked = scalp - ears.mean(0, keepdims=True)
    beta = filtfilt(matlab_firls_approximation("beta", FS), [1.], linked, axis=1)
    cb = concatenate_clean_seconds(beta, clean, FS)
    dfa_i, dfa_e, high = dfa_both(cb)
    out.update(beta_dfa_intercept_mean=dfa_i, beta_dfa_exponent_mean=dfa_e,
        dfa_max_scale_seconds=high, beta_entropy_350=float(matlab_hist_entropy(cb).mean()))
    car = scalp - scalp.mean(0, keepdims=True)
    delta = filtfilt(matlab_firls_approximation("delta", FS), [1.], car, axis=1)
    epochs = contiguous_clean_epochs(delta, clean, FS)
    out.update(pli_features(epochs, device, 100 if diagnostic else 0))
    if not diagnostic:
        return out
    entropies = []
    for channel in cb:
        counts = np.histogram(channel, bins="fd")[0]
        probs = counts[counts > 0] / counts.sum()
        entropies.append(float(-np.sum(probs*np.log2(probs))))
    out["smith_beta_entropy_fd"] = float(np.mean(entropies))
    # Count ordinal patterns within clean contiguous segments, not across gaps.
    orders = []
    padded = np.r_[False, clean, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, stop in zip(starts, stops):
        sig = beta[SCALP.index("CZ"), start*FS:stop*FS]
        if len(sig) >= 4:
            idx = np.argsort(np.lib.stride_tricks.sliding_window_view(sig, 4), axis=1, kind="stable")
            orders.append((idx * np.array([1,4,16,64])).sum(1))
    counts = np.unique(np.concatenate(orders), return_counts=True)[1]
    probs = counts/counts.sum()
    out["beta_permutation_entropy_cz"] = float(-np.sum(probs*np.log2(probs)))
    broad = sosfiltfilt(butter(4, [.5,55.], btype="bandpass", fs=FS, output="sos"), linked, axis=1)
    blocks = broad[:, :len(clean)*FS].reshape(19, len(clean), FS)[:, clean]
    amps = np.ptp(blocks, axis=-1)
    out["amplitude_median_uv"] = float(np.median(amps, axis=1).mean())
    out["cz_amplitude_median_uv"] = float(np.median(amps[SCALP.index("CZ")]))
    ep5 = contiguous_clean_epochs(broad, clean, FS, 5)
    if len(ep5):
        power = (np.abs(np.fft.rfft(ep5, axis=-1))**2 / (FS*ep5.shape[-1])).mean(0)
        freq = np.fft.rfftfreq(ep5.shape[-1], 1/FS)
        for name, lo, hi in [("delta",1,4),("theta",4,8),("alpha",8,13),("beta",13,30)]:
            out[f"power_{name}_log10"] = float(np.log10(np.maximum(power[:, (freq>=lo)&(freq<hi)].sum(1), 1e-30)).mean())
        keep = (freq >= .5) & (freq <= 55)
        cum = np.cumsum(power[:, keep], axis=1)
        edges = np.argmax(cum >= .95*cum[:, -1:], axis=1)
        out["sef95_hz"] = float(freq[keep][edges].mean())
    return out


def prepare_encoder_input(x, model):
    if model == "biot":
        return np.stack([x[SCALP.index(a)]-x[SCALP.index(b)] for a,b in PAIRS]).astype(np.float32)
    # Retain each encoder's band limits, line-frequency notch and montage.
    scalp = mne.filter.filter_data(x[:19], FS, .1 if model == "labram" else .3, 75., verbose="ERROR")
    scalp = mne.filter.notch_filter(scalp, FS, [60.], verbose="ERROR")
    return (scalp-scalp.mean(0, keepdims=True)).astype(np.float32)


def encode(model, model_name, prepared, length, device, batch_size=16):
    samples = int(length*FS)
    n = prepared.shape[-1]//samples
    windows = prepared[:, :n*samples].reshape(prepared.shape[0], n, samples).transpose(1,0,2).copy()
    if model_name == "biot":
        windows /= np.quantile(np.abs(windows), .95, axis=-1, keepdims=True)+1e-8
    else:
        windows = windows.reshape(n,19,int(length),FS)/100.
    if model_name == "labram":
        sys.path.insert(0, str(ROOT / "baselines_reference/baselines/labram"))
        import utils
        input_chans = utils.get_input_chans(SCALP)
    result=[]
    with torch.inference_mode():
        for start in range(0,n,batch_size):
            z = torch.tensor(windows[start:start+batch_size], device=device, dtype=torch.float32)
            if model_name == "labram":
                y = model.forward_features(z, input_chans=input_chans)
            else:
                y = model(z)
                if model_name == "cbramod":
                    y = y.mean((1,2))
            result.append(y.float().cpu().numpy())
    y = np.concatenate(result)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite encoder output")
    return y
