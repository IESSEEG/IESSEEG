"""Fine-tuning pieces shared by the foundation-model baselines.

The models differ in what their forward pass needs -- LUNA takes
electrode coordinates and a mask, EEGPT takes only the signal -- but the
surrounding machinery is the same for all of them, and getting any of it
subtly wrong (a clip-level split instead of a patient-level one, say)
would quietly inflate results. It lives here so there is one version to
review.
"""

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def patient_level_split(info, val_size=0.2, seed=42):
    """Split (patient_id, recording_id, label) triples by patient.

    Clips from one infant resemble each other far more than they
    resemble another infant's, so splitting at the clip level would leak
    across the boundary and inflate validation scores. Stratified on the
    patient's label so both sides keep the class balance.
    """
    patients = sorted({patient for patient, _, _ in info})
    patient_label = {
        patient: max(label for other, _, label in info if other == patient)
        for patient in patients
    }
    stratify = [patient_label[p] for p in patients]

    train_patients, val_patients = train_test_split(
        patients, test_size=val_size, random_state=seed,
        stratify=stratify if len(set(stratify)) > 1 else None,
    )
    train_patients, val_patients = set(train_patients), set(val_patients)

    train_info = [t for t in info if t[0] in train_patients]
    val_info = [t for t in info if t[0] in val_patients]
    return train_info, val_info


def clip_level_scores(patient_id, recording_id, prob, label):
    """Average window probabilities within a clip and threshold at 0.5.

    The same aggregation the benchmark uses at test time, applied to
    validation so that model selection optimises the quantity actually
    being reported.
    """
    df = pd.DataFrame({
        "patient_id": patient_id,
        "recording_id": recording_id,
        "pred_prob": prob,
        "known_label": label,
    })
    clip = df.groupby("recording_id").agg(
        patient_id=("patient_id", "first"),
        pred_prob=("pred_prob", "mean"),
        known_label=("known_label", "first"),
    ).reset_index()
    clip["pred_label"] = (clip["pred_prob"] >= 0.5).astype(int)
    return clip


def layer_decay_param_groups(model, layer_of, n_layers, base_lr, weight_decay, decay=0.75):
    """Parameter groups with layer-wise learning-rate decay.

    Layers nearest the input keep most of their pre-trained behaviour and
    train slowest; the new head trains at the full rate. `layer_of` maps a
    parameter name to its depth index in [0, n_layers).
    """
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer = layer_of(name)
        # Biases and norm scales are conventionally exempt from decay.
        no_decay = param.ndim <= 1
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * (decay ** (n_layers - 1 - layer)),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(param)
    return list(groups.values())


class CosineWithWarmup:
    """Linear warmup then cosine decay, stepped once per optimizer step."""

    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def _scale(self):
        if self.step_count < self.warmup_steps:
            return self.step_count / max(self.warmup_steps, 1)
        progress = (self.step_count - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def step(self):
        self.step_count += 1
        scale = self._scale()
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = max(base * scale, self.min_lr)


class EarlyStopping:
    """Track the best validation loss and when to give up.

    Returns True from `update` when the score improved, so the caller can
    save a checkpoint at exactly the epochs that matter.
    """

    def __init__(self, patience=8, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.since_improvement = 0

    def update(self, value, epoch):
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.since_improvement = value, epoch, 0
            return True
        self.since_improvement += 1
        return False

    @property
    def should_stop(self):
        return self.since_improvement >= self.patience
