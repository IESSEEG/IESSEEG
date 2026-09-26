#!/usr/bin/env python
"""Native-window frozen encoder loaders. Used by PRE and POST extractors."""

from __future__ import annotations
import legacy_paths as _paths

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
]


def batches(array: np.ndarray, batch_size: int):
    for start in range(0, len(array), batch_size):
        yield array[start : start + batch_size]


def load_biot(project: Path, device: torch.device):
    code = _paths.baselines() / "biot"
    sys.path.insert(0, str(code))
    from model.biot import BIOTEncoder  # type: ignore

    model = BIOTEncoder(n_channels=18, n_fft=200, hop_length=100).to(device)
    state = torch.load(
        _paths.pretrained(project / "baselines/biot/pretrained-models/EEG-six-datasets-18-channels.ckpt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


def run_biot(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    samples = 30 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(18, n_windows, samples).transpose(1, 0, 2).copy()
    scale = np.quantile(np.abs(windows), 0.95, axis=-1, keepdims=True)
    windows /= scale + 1e-8
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            output.append(model(torch.from_numpy(batch).float().to(device)).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_cbramod(project: Path, device: torch.device):
    code = _paths.baselines() / "cbramod"
    sys.path.insert(0, str(code))
    from models.cbramod import CBraMod  # type: ignore

    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
        seq_len=30, n_layer=12, nhead=8,
    ).to(device)
    state = torch.load(
        _paths.pretrained(project / "baselines/cbramod/pretrained_weights/pretrained_weights.pth"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state, strict=True)
    model.proj_out = torch.nn.Identity()
    return model.eval()


def run_cbramod(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    samples = 30 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(19, n_windows, 30, 200).transpose(1, 0, 2, 3).copy()
    windows /= 100.0
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            features = model(torch.from_numpy(batch).float().to(device))
            output.append(features.mean(dim=(1, 2)).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_labram(project: Path, device: torch.device):
    code = _paths.baselines() / "labram"
    sys.path.insert(0, str(code))
    import modeling_finetune  # type: ignore

    model = modeling_finetune.labram_base_patch200_200(
        pretrained=False,
        num_classes=0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        use_mean_pooling=False,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
    payload = torch.load(
        _paths.pretrained(project / "baselines/labram/checkpoints/labram-base.pth"),
        map_location="cpu",
        weights_only=False,
    )["model"]
    state = {
        key.removeprefix("student."): value
        for key, value in payload.items()
        if key.startswith("student.") and "relative_position_index" not in key
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    permitted_unexpected = {
        "mask_token", "lm_head.weight", "lm_head.bias"
    }
    material_unexpected = [key for key in unexpected if key not in permitted_unexpected]
    if material_unexpected:
        raise RuntimeError(f"Unexpected LaBraM tensors: {material_unexpected[:8]}")
    material_missing = [key for key in missing if not key.startswith("head.")]
    if material_missing:
        raise RuntimeError(f"Missing LaBraM backbone tensors: {material_missing[:8]}")
    return model.to(device).eval()


def run_labram(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    code = _paths.baselines() / "labram"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import utils as labram_utils  # type: ignore

    samples = 10 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(19, n_windows, 10, 200).transpose(1, 0, 2, 3).copy()
    windows /= 100.0
    input_chans = labram_utils.get_input_chans(CHANNELS)
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            features = model.forward_features(
                torch.from_numpy(batch).float().to(device), input_chans=input_chans
            )
            output.append(features.cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_luna(project: Path, device: torch.device):
    code = _paths.baselines() / "luna"
    sys.path.insert(0, str(code))
    from models.LUNA import LUNA  # type: ignore
    from luna_data import channel_locations  # type: ignore
    from safetensors.torch import load_file

    model = LUNA(
        patch_size=40, embed_dim=64, num_heads=2, depth=8,
        num_queries=4, drop_path=0.0, num_classes=2,
    )
    state = load_file(
        _paths.pretrained(project / "baselines_release/pretrained-models/LUNA_base.safetensors")
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    material_missing = [key for key in missing if not key.startswith("classifier.")]
    material_unexpected = [
        key for key in unexpected
        if not key.startswith(("decoder_head.", "channel_emb."))
    ]
    if material_missing or material_unexpected:
        raise RuntimeError(
            f"LUNA checkpoint mismatch; missing={material_missing[:8]}, "
            f"unexpected={material_unexpected[:8]}"
        )
    model = model.to(device).eval()
    positions = torch.from_numpy(channel_locations()).float().to(device)
    return model, positions


def run_luna(
    bundle, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    code = _paths.baselines() / "luna"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from einops import rearrange
    from luna_data import make_recording_transform, normalize_window  # type: ignore

    model, positions = bundle
    # The stored BASED bipolar montage already has the channel order expected
    # by the adapter; the transform resamples 200 Hz to LUNA's 256 Hz.
    transform = make_recording_transform(source_rate=200, target_rate=256)
    resampled = transform(data, channels)
    samples = 30 * 256
    n_windows = resampled.shape[-1] // samples
    windows = resampled[:, : n_windows * samples].reshape(22, n_windows, samples).transpose(1, 0, 2).copy()
    windows = normalize_window(windows)
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            x = torch.from_numpy(batch).float().to(device)
            pos = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
            tokens, _ = model.prepare_tokens(x, pos, mask=None)
            latent, _ = model.cross_attn(tokens)
            latent = rearrange(
                latent, "(b t) q d -> b t (q d)", b=x.shape[0]
            )
            for block in model.blocks:
                latent = block(latent)
            output.append(model.norm(latent).mean(dim=1).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_eegpt(project: Path, device: torch.device):
    """Load only learned upstream EEGPT tensors with deterministic channel IDs."""
    import mne
    from braindecode.models import EEGPT
    from safetensors.torch import load_file

    # The upstream 62-channel vocabulary uses the modern aliases below.
    mapped_names = [
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
    ]
    chs_info = mne.create_info(mapped_names, 250, "eeg")["chs"]
    model = EEGPT(
        n_outputs=1,
        n_chans=19,
        n_times=1000,
        sfreq=250,
        chs_info=chs_info,
        chan_proj_type="none",
        return_encoder_output=True,
    )
    upstream = Path(
        os.environ.get(
            "IESSEEG_EEGPT_UPSTREAM",
            project / "kfold_split/tmp/upstream/eegpt-pretrained",
        )
    )
    state = load_file(upstream / "model.safetensors")
    state.pop("chans_id")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["chans_id"] or unexpected:
        raise RuntimeError(
            f"EEGPT checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    if not isinstance(model.chan_proj, torch.nn.Identity):
        raise RuntimeError("EEGPT frozen probe must not use a random channel projection")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def run_eegpt(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    from braindecode.preprocessing import exponential_moving_standardize
    from scipy.signal import resample_poly

    resampled = resample_poly(data, 5, 4, axis=-1)
    resampled = exponential_moving_standardize(
        np.ascontiguousarray(resampled, dtype=np.float64),
        factor_new=0.001,
        init_block_size=None,
        eps=1e-4,
    )
    samples = 4 * 250
    n_windows = resampled.shape[-1] // samples
    windows = resampled[:, : n_windows * samples].reshape(19, n_windows, samples).transpose(1, 0, 2).copy()
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            tokens = model(torch.from_numpy(batch).float().to(device))
            output.append(tokens.mean(dim=(1, 2)).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_reve(project: Path, device: torch.device):
    from transformers import AutoModel

    cache_dir = Path(os.environ.get("IESSEEG_HF_CACHE", str(_paths.runtime() / "hf_cache")))
    def local_snapshot(repo_id: str) -> Path:
        repository = cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}"
        revisions = {"brain-bzh/reve-base": "fa9a2163a4b7c0a42c8e28b56077ef9c368944dc",
                     "brain-bzh/reve-positions": "befa5b57a455b77cf302daf610c2e9ed8140bace"}
        snapshot = repository / "snapshots" / revisions[repo_id]
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Missing pinned REVE snapshot {snapshot}. See docs/MODELS.md.")
        return snapshot

    model = AutoModel.from_pretrained(
        local_snapshot("brain-bzh/reve-base"),
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    bank = AutoModel.from_pretrained(
        local_snapshot("brain-bzh/reve-positions"),
        trust_remote_code=True,
        local_files_only=True,
    )
    positions = bank(CHANNELS).detach().float().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, positions


def run_reve(
    bundle, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    model, positions = bundle
    samples = 30 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(19, n_windows, samples).transpose(1, 0, 2).copy()
    windows = (
        windows - windows.mean(axis=-1, keepdims=True)
    ) / (windows.std(axis=-1, keepdims=True) + 1e-8)
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            x = torch.from_numpy(batch).float().to(device)
            pos = positions.unsqueeze(0).expand(x.shape[0], -1, -1)
            tokens = model(x, pos)
            output.append(model.attention_pooling(tokens).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_codebrain(project: Path, device: torch.device):
    code = _paths.baselines() / "codebrain"
    sys.path.insert(0, str(code))
    from codebrain_model import build_model  # type: ignore

    model = build_model(
        _paths.pretrained(project / "baselines_release/pretrained-models/CodeBrain.pth"),
        dropout=0.1,
    )
    return model.backbone.to(device).eval()


def run_codebrain(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    samples = 30 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(19, n_windows, 30, 200).transpose(1, 0, 2, 3).copy()
    windows /= 100.0
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            features = model(torch.from_numpy(batch).float().to(device))
            if features.ndim == 3:
                features = features.unsqueeze(0)
            output.append(features.mean(dim=(1, 2)).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


def load_csbrain(project: Path, device: torch.device):
    code = _paths.baselines() / "csbrain"
    sys.path.insert(0, str(code))
    from csbrain_model import build_model  # type: ignore

    model = build_model(
        _paths.pretrained(project / "baselines_release/pretrained-models/CSBrain.pth"),
        dropout=0.1,
    )
    return model.backbone.to(device).eval()


def run_csbrain(
    model, data: np.ndarray, device: torch.device, batch_size: int, channels,
    return_windows: bool = False,
):
    samples = 30 * 200
    n_windows = data.shape[-1] // samples
    windows = data[:, : n_windows * samples].reshape(19, n_windows, 30, 200).transpose(1, 0, 2, 3).copy()
    windows /= 100.0
    output = []
    with torch.inference_mode():
        for batch in batches(windows, batch_size):
            features = model(torch.from_numpy(batch).float().to(device))
            output.append(features.mean(dim=(1, 2)).cpu())
    matrix = torch.cat(output).numpy()
    return matrix if return_windows else matrix.mean(0)


# All loaders operate on the released native-window encoder contract.
MODEL_IO = {name: (globals()[f"load_{name}"], globals()[f"run_{name}"])
            for name in ("biot", "cbramod", "labram", "luna", "eegpt", "reve", "codebrain", "csbrain")}
