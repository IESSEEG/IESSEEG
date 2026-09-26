# LUNA baseline

[LUNA](https://arxiv.org/abs/2510.22257) (Döner et al., NeurIPS 2025) is a
topology-agnostic EEG foundation model. It compresses any channel layout
into a fixed-size latent with learned queries and cross-attention, so
compute scales linearly rather than quadratically in channel count, and
runs temporal self-attention on that latent. We fine-tune **LUNA-Base**
(7M parameters), pre-trained with masked-patch reconstruction on TUEG and
Siena (>21,000 hours).

## Licensing

The model **code** under `models/` is vendored from
[pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation) under
Apache 2.0 (see `LICENSE.upstream`).

The **weights** are CC BY-ND 4.0. That permits fine-tuning for internal
use and publishing code, configs and results, but forbids redistributing
modified weights. This repository ships no checkpoints and `.gitignore`
excludes them, so running the benchmark stays within the licence — but do
not publish the fine-tuned checkpoints it produces.

Download `LUNA_base.safetensors` from
[PulpBio/LUNA](https://huggingface.co/PulpBio/LUNA) into
`IESSEEG_PRETRAINED_DIR` (default `../../pretrained-models`).

## Bridging IESSEEG to LUNA's expected input

LUNA was pre-trained on the 22-channel TCP bipolar montage at 256 Hz, and
consumes 3D electrode coordinates alongside the signal. IESSEEG's
`scalp_eeg_data_200HZ_np_format` tree already holds the same 22 bipolar
derivations, so no new preprocessing tree is needed. `luna_data.py`
reconciles the remaining three differences, all at recording-load time:

| Difference | Handling |
|---|---|
| Channel order differs | Reordered into LUNA's TCP order |
| Six central-chain derivations stored reversed (`A2-T4` vs `T4-A2`) | Sign-flipped, since reversing a bipolar derivation negates it |
| 200 Hz vs 256 Hz | Polyphase resample (exactly 32/25), so a 40-sample patch spans the same 156 ms as in pre-training |

Electrode coordinates are the midpoint of each derivation's two
electrodes in the `standard_1005` montage, matching how the upstream code
derives positions for TUH channels. Windows are then channel-wise
z-scored, as LUNA's own fine-tuning task does.

A 30-second window at 256 Hz is 7680 samples, exactly 192 patches of 40,
so nothing is dropped at the tail.

## Deviations from the upstream recipe, and why

- **Batch size 64**, not upstream's 256. Matches the other foundation
  models in this benchmark, so the comparison is not confounded by batch
  size. Learning rate, layer-wise decay (0.75), warmup and early-stopping
  patience follow upstream.
- **Coarse validation stride** (`--val_step_sec 120`). Validation only
  selects which epoch to keep; tiling every validation clip exhaustively
  tripled epoch time for no measurable change in the selection signal.
  Test inference always tiles fully with a non-overlapping stride.
- **`mask=None` at inference.** Upstream passes an all-false mask during
  fine-tuning, which leaves the signal intact but jitters the electrode
  coordinates slightly as augmentation. We keep that during training and
  drop it at evaluation so clip-level scores are deterministic.

## Files

```
luna_data.py         montage mapping, resampling, electrode coordinates
train_luna.py        fine-tuning: AdamW + layer-wise LR decay + cosine
inference_luna.py    clip-level inference (window probabilities averaged)
models/              vendored LUNA architecture (Apache 2.0)
train_all.sh         all tasks x folds
inference_all.sh     all tasks x folds
```
