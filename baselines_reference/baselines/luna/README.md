# LUNA adapter

LUNA uses 30-second windows from 22 bipolar channels, resampled from 200 to
256 Hz. The adapter reorders channels into TCP order and reverses the sign of
six central-chain derivations to match that montage. Electrode positions are
the midpoints of each derivation's electrodes in the `standard_1005` montage.
Each window is standardized channel by channel.

Diagnosis fine-tuning uses AdamW, layer-wise learning-rate decay, cosine
scheduling, and early stopping. Validation uses a 120-second stride; test
inference uses non-overlapping windows. Training retains upstream coordinate
augmentation; evaluation passes `mask=None` for deterministic positions.

Download `LUNA_base.safetensors` into `IESSEEG_PRETRAINED_DIR` following the
[model setup guide](../../../docs/MODELS.md). The model code is from
[pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation), with the
Apache 2.0 notice in `LICENSE.upstream`. Weights are downloaded separately under
the terms published by [PulpBio/LUNA](https://huggingface.co/PulpBio/LUNA).
