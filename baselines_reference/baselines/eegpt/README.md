# EEGPT baseline

[EEGPT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
(Wang et al., NeurIPS 2024) is a pre-trained EEG transformer trained with
a dual objective: spatio-temporal representation alignment plus
mask-based reconstruction. We fine-tune the released encoder (25M
parameters as configured here).

## Separate environment

EEGPT is loaded through [braindecode](https://braindecode.org)'s
implementation, which needs braindecode >= 1.3. The rest of the benchmark
pins an older release, and upgrading it in place would change the
environment that produced every other baseline's numbers. EEGPT therefore
runs from its own interpreter, sharing the same PyTorch:

```bash
python -m venv --system-site-packages /path/to/venv_eegpt
/path/to/venv_eegpt/bin/pip install --no-deps braindecode==1.7.0
export IESSEEG_PYTHON_EEGPT=/path/to/venv_eegpt/bin/python
```

`--system-site-packages` reuses the already-installed torch, and
`--no-deps` stops braindecode from pulling its own pinned torch/numpy,
so both environments run identical PyTorch. The runner scripts fall back
to `PYTHON_BIN` when `IESSEEG_PYTHON_EEGPT` is unset.

## Weights

Fetched automatically from
[braindecode/eegpt-pretrained](https://huggingface.co/braindecode/eegpt-pretrained)
(ungated) on first run, cached under `HF_HOME`. The original authors
distribute the checkpoint via figshare; the braindecode mirror is the
same encoder in a directly loadable format.

## Bridging IESSEEG to EEGPT's expected input

EEGPT expects a referential 10-20 montage, which is the
`scalp_eeg_data_200HZ_np_format_labram` tree the TUEG-style models
already use, so no new preprocessing pass is needed. `eegpt_data.py`
handles the rest at recording-load time:

| Difference | Handling |
|---|---|
| Channels named `FP1-REF` | Suffix stripped to `FP1`; the set and order otherwise match the standard 19 |
| 200 Hz vs 250 Hz | Polyphase resample (exactly 5/4) |
| Amplitude scale | `exponential_moving_standardize`, the standardiser EEGPT's own downstream pipeline uses, applied to the continuous recording |

Windows are 4 seconds (1000 samples at 250 Hz), matching EEGPT's
pre-training context. That is much shorter than the 30 s the other
baselines use, so a clip yields proportionally more windows; clip-level
aggregation is unchanged, averaging window probabilities.

## Channel projection and the `chans_id` buffer

braindecode's EEGPT adds a learned channel projection
(`chan_proj_type="conv1d_constraint"`) mapping an arbitrary montage into
the channel space the encoder was pre-trained on. In that mode it selects
the standard 19-channel vocabulary, so its `chans_id` buffer has 19
entries, while the released checkpoint stores the 62 entries of its own
pre-training montage. We load every pre-trained tensor **except**
`chans_id`, which keeps the whole encoder and lets braindecode index the
standard 19 channels. The load is checked: any unexpected tensor raises
rather than being silently dropped.

## Files

```
eegpt_data.py        montage naming, resampling, standardisation
train_eegpt.py       fine-tuning (AdamW + layer-wise LR decay + cosine)
inference_eegpt.py   clip-level inference
train_all.sh         all tasks x folds
inference_all.sh     all tasks x folds
```
