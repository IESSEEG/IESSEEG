"""In-memory windowed EEG dataset shared by every deep baseline.

This replaces the three near-identical `inmem_raw_dataset.py` copies that
previously lived under biot/, labram/ and cbramod/. Those copies differed
only in how a window is post-processed before it leaves __getitem__, so
that step is now a named per-model policy (`window_transform`) and
everything else -- loading, window indexing, batching -- is shared.

Window transforms:
  "none"        window returned as (C, window_frames).           [LaBraM]
  "quantile"    scaled by its own 95th percentile amplitude.     [BIOT]
  "patch"       reshaped to (C, window_sec, sample_rate).        [CBraMod]
  "zscore"      per-channel standardised over time.               [LUNA]
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_recording(data_dir, recording_id, n_channels=19, scale_factor=1.0, verbose=False):
    """Load one preprocessed recording: {recording_id}.npz -> (C, T) array."""
    path = os.path.join(data_dir, f"{recording_id}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    channel_arr = loaded["channel"][:n_channels]
    raw_data = loaded["data"][:n_channels, :] * scale_factor

    if verbose:
        print(f"Loaded: {recording_id}, shape={raw_data.shape}, channels={channel_arr.shape}")

    return raw_data, channel_arr


def _transform_none(window, n_channels, window_sec, sample_rate):
    return window


def _transform_quantile(window, n_channels, window_sec, sample_rate):
    # Per-channel amplitude normalisation; the 95th percentile is robust to
    # the electrode pops and movement transients common in infant EEG,
    # which a max-abs scaling would let dominate the window.
    return window / (np.quantile(np.abs(window), 0.95, axis=-1, keepdims=True) + 1e-8)


def _transform_patch(window, n_channels, window_sec, sample_rate):
    return window.reshape(n_channels, window_sec, sample_rate)


def _transform_zscore(window, n_channels, window_sec, sample_rate):
    # Per-channel standardisation over time, which is what LUNA's
    # fine-tuning task applies to its inputs.
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    return (window - mean) / (std + 1e-8)


WINDOW_TRANSFORMS = {
    "none": _transform_none,
    "quantile": _transform_quantile,
    "patch": _transform_patch,
    "zscore": _transform_zscore,
}


class InMemoryRandomDataset(Dataset):
    """Preloads recordings, then serves fixed-length windows.

    In "train" mode each __getitem__ draws a random recording and a random
    window offset, so an epoch is a fixed number of random draws
    (`train_iterations`) rather than a pass over a fixed index. In
    "val"/"test" mode the recordings are tiled with non-overlapping windows
    in a deterministic order, which is what makes clip-level probability
    averaging at inference reproducible.
    """

    def __init__(
        self,
        data_dir,
        info_list,          # list of (patient_id, recording_id, label)
        mode="train",
        sample_rate=200,
        window_sec=30,
        step_sec=30,
        n_channels=19,
        scale_factor=1.0,
        train_iterations=10000,
        window_transform="none",
        recording_transform=None,
        return_dict=True,
        verbose=False,
    ):
        super().__init__()
        if window_transform not in WINDOW_TRANSFORMS:
            raise ValueError(
                f"Unknown window_transform '{window_transform}'. "
                f"Known: {sorted(WINDOW_TRANSFORMS)}"
            )

        self.data_dir = data_dir
        self.recording_list = info_list
        self.mode = mode
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_frames = int(window_sec * sample_rate)
        self.step_frames = int(step_sec * sample_rate)
        self.n_channels = n_channels
        self.scale_factor = scale_factor
        self.train_iterations = train_iterations
        self.window_transform = window_transform
        self._transform = WINDOW_TRANSFORMS[window_transform]
        self.recording_transform = recording_transform
        self.return_dict = return_dict
        self.verbose = verbose

        self.memory = []
        for (patient_id, recording_id, label) in tqdm(
            info_list, total=len(info_list), desc=f"[{mode}] Loading data"
        ):
            try:
                raw_data, channel_names = load_recording(
                    data_dir, recording_id,
                    n_channels=n_channels, scale_factor=scale_factor, verbose=False,
                )
            except FileNotFoundError:
                if verbose:
                    print(f"[WARN] Missing {recording_id}, skipping.")
                continue

            # Applied once per recording rather than per window, for
            # whole-recording operations that would be wasteful to repeat:
            # resampling to a model's pre-training rate, or reordering
            # channels into the montage order a model expects. It runs
            # before the window index is built, so window_sec/step_sec are
            # interpreted at the post-transform sample rate.
            if recording_transform is not None:
                raw_data = recording_transform(raw_data, channel_names)

            self.memory.append({
                "patient_id": patient_id,
                "recording_id": recording_id,
                "label": label,
                "raw_data": raw_data,
                "T": raw_data.shape[1],
            })

        if self.mode == "train":
            self._length = train_iterations
        else:
            self.index_map = []
            for mem_idx, info in enumerate(self.memory):
                if info["T"] < self.window_frames:
                    continue
                n_steps = (info["T"] - self.window_frames) // self.step_frames + 1
                for step in range(n_steps):
                    self.index_map.append((mem_idx, step * self.step_frames))
            self._length = len(self.index_map)
            if verbose:
                print(f"[{mode}] total windows => {self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self.mode == "train":
            mem_idx = np.random.randint(0, len(self.memory))
            rec = self.memory[mem_idx]
            if rec["T"] < self.window_frames:
                return self.__getitem__(idx)
            start_ind = np.random.randint(0, rec["T"] - self.window_frames + 1)
        else:
            mem_idx, start_ind = self.index_map[idx]
            rec = self.memory[mem_idx]

        end_ind = start_ind + self.window_frames
        window = rec["raw_data"][:, start_ind:end_ind]
        window = self._transform(window, self.n_channels, self.window_sec, self.sample_rate)

        if not self.return_dict:
            return window, rec["label"]

        return {
            "patient_id": rec["patient_id"],
            "recording_id": rec["recording_id"],
            "start_ind": start_ind,
            "end_ind": end_ind,
            "label": rec["label"],
            "waveform": window,
        }

    @staticmethod
    def collate_fn(batch_list):
        """Stack waveforms and labels; keep identifying fields as lists."""
        batch_out = {
            "waveform": torch.from_numpy(
                np.stack([d["waveform"] for d in batch_list], axis=0)
            ).float(),
            "label": torch.tensor([d["label"] for d in batch_list], dtype=torch.long),
            "patient_id": [d["patient_id"] for d in batch_list],
            "recording_id": [d["recording_id"] for d in batch_list],
            "start_ind": [d["start_ind"] for d in batch_list],
            "end_ind": [d["end_ind"] for d in batch_list],
        }
        return batch_out


def build_dataloaders(
    data_dir,
    train_list,
    val_list,
    test_list,
    batch_size,
    sample_rate=200,
    window_sec=30,
    step_sec=30,
    n_channels=19,
    scale_factor=1.0 / 100,
    train_iterations=10000,
    window_transform="none",
    num_workers=4,
    verbose=False,
):
    """Construct the train/val/test loaders for one fold."""
    common = dict(
        data_dir=data_dir, sample_rate=sample_rate, window_sec=window_sec,
        step_sec=step_sec, n_channels=n_channels, scale_factor=scale_factor,
        window_transform=window_transform, verbose=verbose,
    )

    datasets = {
        "train": InMemoryRandomDataset(
            info_list=train_list, mode="train",
            train_iterations=train_iterations, **common
        ),
        "val": InMemoryRandomDataset(info_list=val_list, mode="val", **common),
        "test": InMemoryRandomDataset(info_list=test_list, mode="test", **common),
    }

    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
            collate_fn=ds.collate_fn,
        )
        for name, ds in datasets.items()
    }

    print(f"[DATA] " + "  ".join(f"{n}={len(ds)}" for n, ds in datasets.items()))
    return loaders
