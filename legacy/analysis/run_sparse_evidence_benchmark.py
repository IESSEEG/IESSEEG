#!/usr/bin/env python
"""Weakly supervised long-EEG evidence benchmark.

Each patient is a bag, each released 30-second window is an unlabeled
instance, and supervision is available only for the patient.  Models train on
the four Clinical Clips of development patients and are evaluated on the two
separately sampled Routine Clips of held-out patients.

The script also runs a semi-synthetic recovery test.  A known contiguous motif
is injected into real EEG feature sequences while patient identities and outer
folds remain fixed.  This test asks whether a method can recover sparse local
evidence at all; it does not turn clinical attention maps into annotations.
"""

from __future__ import annotations
import legacy_paths as _paths

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


SEED_BASE = 20260907
TASK_COLUMN = {
    "case_control": "case_control_label",
    "immediate_responder": "immediate_responder",
    "meaningful_responder": "meaningful_responder",
}
POSITIVE = {"CASE", "Responder"}
METHODS = (
    "mean",
    "max",
    "top10",
    "attention",
    "temporal_attention",
    "multiscale",
)
SCALES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)


@dataclass
class PatientBag:
    patient_id: int
    label: int
    clips: list[np.ndarray]
    recording_ids: list[str]


class SparseEvidenceModel(torch.nn.Module):
    def __init__(self, input_dim: int, method: str, dropout: float = 0.20):
        super().__init__()
        self.method = method
        hidden = 64
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
        )
        self.temporal = torch.nn.Conv1d(hidden, hidden, 5, padding=2)
        self.instance_head = torch.nn.Linear(hidden, 1)
        self.attention_v = torch.nn.Linear(hidden, 32)
        self.attention_u = torch.nn.Linear(hidden, 32)
        self.attention_w = torch.nn.Linear(32, 1, bias=False)
        self.scale_logits = torch.nn.Parameter(torch.zeros(len(SCALES)))
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def encode(self, clips: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        encoded, lengths = [], []
        for clip in clips:
            h = self.encoder(clip)
            if self.method == "temporal_attention":
                residual = h
                h = torch.nn.functional.gelu(
                    self.temporal(h.transpose(0, 1).unsqueeze(0))
                    .squeeze(0).transpose(0, 1)
                ) + residual
            encoded.append(h)
            lengths.append(len(h))
        return torch.cat(encoded, dim=0), lengths

    def forward(
        self, clips: list[torch.Tensor], return_local: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, lengths = self.encode(clips)
        local = self.instance_head(h).squeeze(-1)
        contributions, all_weights, clip_logits = [], [], []
        n_clips = len(lengths)
        for clip_h, clip_local in zip(torch.split(h, lengths), torch.split(local, lengths)):
            weights = torch.full_like(clip_local, 1.0 / len(clip_local))
            if self.method == "mean":
                contribution = clip_local / len(clip_local)
                clip_logit = contribution.sum()
            elif self.method == "max":
                index = torch.argmax(clip_local)
                contribution = torch.zeros_like(clip_local)
                contribution[index] = clip_local[index]
                weights = torch.zeros_like(clip_local)
                weights[index] = 1.0
                clip_logit = clip_local[index]
            elif self.method == "top10":
                k = max(1, math.ceil(0.10 * len(clip_local)))
                values, indices = torch.topk(clip_local, k)
                contribution = torch.zeros_like(clip_local)
                contribution[indices] = values / k
                weights = torch.zeros_like(clip_local)
                weights[indices] = 1.0 / k
                clip_logit = values.mean()
            elif self.method in {"attention", "temporal_attention"}:
                attention = self.attention_w(
                    torch.tanh(self.attention_v(clip_h))
                    * torch.sigmoid(self.attention_u(clip_h))
                ).squeeze(-1)
                weights = torch.softmax(attention, dim=0)
                contribution = weights * clip_local
                clip_logit = contribution.sum()
            elif self.method == "multiscale":
                order = torch.argsort(clip_local, descending=True)
                scale_weights = torch.softmax(self.scale_logits, dim=0)
                contribution = torch.zeros_like(clip_local)
                weights = torch.zeros_like(clip_local)
                pooled = []
                for mixture_weight, fraction in zip(scale_weights, SCALES):
                    k = max(1, math.ceil(fraction * len(clip_local)))
                    chosen = order[:k]
                    pooled.append(clip_local[chosen].mean())
                    contribution[chosen] += mixture_weight * clip_local[chosen] / k
                    weights[chosen] += mixture_weight / k
                clip_logit = torch.stack(pooled).dot(scale_weights)
            else:
                raise KeyError(self.method)
            contributions.append(contribution / n_clips)
            all_weights.append(weights / n_clips)
            clip_logits.append(clip_logit)

        contribution = torch.cat(contributions)
        weights = torch.cat(all_weights)
        bag_logit = torch.stack(clip_logits).mean() + self.bias

        if return_local:
            return bag_logit, contribution, weights
        return bag_logit


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def label_value(value: str) -> int:
    return int(value in POSITIVE)


def cache_index(workspace: Path) -> dict[str, Path]:
    cache = workspace / "kfold_split/baselines/handcrafted/feature_cache/regional"
    paths = list(cache.glob("*regional_asym1_pli1.npy"))
    index = {path.name.split(".npz__")[0]: path for path in paths}
    if len(index) != 600:
        raise RuntimeError(f"Expected 600 full feature caches, found {len(index)}")
    return index


def load_bags(
    frame: pd.DataFrame,
    index: dict[str, Path],
    task: str,
) -> dict[int, PatientBag]:
    column = TASK_COLUMN[task]
    bags: dict[int, PatientBag] = {}
    for patient_id, group in frame.groupby("patient_id", sort=True):
        values = group[column].dropna().astype(str).unique()
        if len(values) != 1 or values[0] == "UNKNOWN":
            continue
        clips, recording_ids = [], []
        for recording_id in normalize_id(group["short_recording_id"]):
            if recording_id not in index:
                raise FileNotFoundError(f"No feature cache for {recording_id}")
            features = np.load(index[recording_id]).astype(np.float32)
            if features.ndim != 2 or features.shape[1] != 122:
                raise RuntimeError(f"Unexpected feature shape for {recording_id}: {features.shape}")
            if not np.isfinite(features).all():
                raise RuntimeError(f"Non-finite features for {recording_id}")
            clips.append(features)
            recording_ids.append(recording_id)
        bags[int(patient_id)] = PatientBag(
            patient_id=int(patient_id),
            label=label_value(values[0]),
            clips=clips,
            recording_ids=recording_ids,
        )
    return bags


def diagnosis_cohort_ids(workspace: Path, cohort: str) -> set[int] | None:
    """Return a prespecified diagnosis sensitivity cohort."""
    if cohort == "all":
        return None
    raw = workspace / "data/raw_data"
    order = pd.read_csv(raw / "long_short_mappings.csv")[["PatientID"]].reset_index(
        names="patient_id"
    )
    metadata = order.merge(
        pd.read_csv(raw / "MetaDataForMingjian.csv"),
        on="PatientID",
        validate="one_to_one",
    )
    masks = {
        "clinical_mimic": metadata.CASE.eq(1) | metadata.CONTROL_IS.eq(1),
        "flagged_subset": metadata.CASE40.eq(1) | metadata.CONTROL20.eq(1),
        "onset_le24_full_controls": metadata.CASE.ne(1) | metadata.AOOmo.le(24),
        "onset_le24_clinical_mimic": (
            (metadata.CASE.eq(1) & metadata.AOOmo.le(24))
            | metadata.CONTROL_IS.eq(1)
        ),
    }
    if cohort not in masks:
        raise KeyError(cohort)
    selected = metadata.loc[masks[cohort], "patient_id"].astype(int)
    return set(selected)


def fit_scaler(bags: list[PatientBag]) -> tuple[np.ndarray, np.ndarray]:
    # Give every patient equal weight, every clip within a patient equal weight,
    # and every window within a clip equal weight.  Clinical Clips vary from
    # 14 to 598 windows, so concatenating them would let duration determine the
    # training distribution.
    patient_mean = np.stack([
        np.stack([clip.mean(axis=0) for clip in bag.clips]).mean(axis=0)
        for bag in bags
    ])
    patient_second = np.stack([
        np.stack([(clip.astype(np.float64) ** 2).mean(axis=0) for clip in bag.clips])
        .mean(axis=0)
        for bag in bags
    ])
    center = patient_mean.mean(axis=0, dtype=np.float64)
    variance = patient_second.mean(axis=0) - center.astype(np.float64) ** 2
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform_bag(bag: PatientBag, center: np.ndarray, scale: np.ndarray) -> PatientBag:
    return PatientBag(
        bag.patient_id,
        bag.label,
        [np.clip((clip - center) / scale, -10, 10).astype(np.float32) for clip in bag.clips],
        bag.recording_ids,
    )


def tensor_clips(bag: PatientBag, device: torch.device) -> list[torch.Tensor]:
    return [
        clip.to(device) if isinstance(clip, torch.Tensor) else torch.from_numpy(clip).to(device)
        for clip in bag.clips
    ]


def device_bags(bags: list[PatientBag], device: torch.device) -> list[PatientBag]:
    return [
        PatientBag(
            bag.patient_id,
            bag.label,
            [torch.from_numpy(clip).to(device) for clip in bag.clips],
            bag.recording_ids,
        )
        for bag in bags
    ]


def auc_or_half(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    labels = np.asarray(labels)
    return 0.5 if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))


def fold_comparable_auc(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    folds: list[int] | np.ndarray,
) -> float:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    folds = np.asarray(folds, int)
    concordant, pairs = 0.0, 0
    for fold in np.unique(folds):
        positive = scores[(folds == fold) & (labels == 1)]
        negative = scores[(folds == fold) & (labels == 0)]
        difference = positive[:, None] - negative[None, :]
        concordant += (difference > 0).sum() + 0.5 * (difference == 0).sum()
        pairs += difference.size
    if not pairs:
        raise ValueError("no within-fold positive-negative pairs")
    return float(concordant / pairs)


def predict(
    model: SparseEvidenceModel,
    bags: list[PatientBag],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for bag in bags:
            logit = model(tensor_clips(bag, device))
            output.append(torch.sigmoid(logit).item())
    return np.asarray(output)


def train_fixed_epochs(
    model: SparseEvidenceModel,
    bags: list[PatientBag],
    device: torch.device,
    epochs: int,
    seed: int,
) -> None:
    bags = device_bags(bags, device)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    labels = np.asarray([bag.label for bag in bags])
    pos_weight = torch.tensor(
        (labels == 0).sum() / max(1, (labels == 1).sum()),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(bags))
        for start in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + 8]:
                bag = bags[int(index)]
                target = torch.tensor(float(bag.label), device=device)
                losses.append(criterion(model(tensor_clips(bag, device)), target))
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


def select_epoch(
    method: str,
    input_dim: int,
    training: list[PatientBag],
    validation: list[PatientBag],
    device: torch.device,
    seed: int,
    max_epochs: int,
) -> int:
    training = device_bags(training, device)
    validation = device_bags(validation, device)
    torch.manual_seed(seed)
    model = SparseEvidenceModel(input_dim, method).to(device)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    labels = np.asarray([bag.label for bag in training])
    pos_weight = torch.tensor(
        (labels == 0).sum() / max(1, (labels == 1).sum()),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_epoch, best_auc, stale = 1, -np.inf, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(training))
        for start in range(0, len(order), 8):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + 8]:
                bag = training[int(index)]
                target = torch.tensor(float(bag.label), device=device)
                losses.append(criterion(model(tensor_clips(bag, device)), target))
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 2 == 0:
            score = predict(model, validation, device)
            auc = auc_or_half([bag.label for bag in validation], score)
            if auc > best_auc + 1e-5:
                best_auc, best_epoch, stale = auc, epoch, 0
            else:
                stale += 2
            if stale >= 20:
                break
    return best_epoch


def fit_outer_model(
    method: str,
    input_dim: int,
    outer_training: list[PatientBag],
    inner_training_ids: set[int],
    validation_ids: set[int],
    device: torch.device,
    seed: int,
    max_epochs: int,
) -> tuple[SparseEvidenceModel, int, np.ndarray, np.ndarray]:
    inner_raw = [bag for bag in outer_training if bag.patient_id in inner_training_ids]
    validation_raw = [bag for bag in outer_training if bag.patient_id in validation_ids]
    center_inner, scale_inner = fit_scaler(inner_raw)
    inner = [transform_bag(bag, center_inner, scale_inner) for bag in inner_raw]
    validation = [transform_bag(bag, center_inner, scale_inner) for bag in validation_raw]
    best_epoch = select_epoch(
        method, input_dim, inner, validation, device, seed, max_epochs
    )

    center, scale = fit_scaler(outer_training)
    all_training = [transform_bag(bag, center, scale) for bag in outer_training]
    torch.manual_seed(seed)
    final_model = SparseEvidenceModel(input_dim, method).to(device)
    train_fixed_epochs(final_model, all_training, device, best_epoch, seed)
    return final_model, best_epoch, center, scale


def split_clip_bags(bag: PatientBag) -> list[PatientBag]:
    return [
        PatientBag(bag.patient_id, bag.label, [clip], [recording_id])
        for clip, recording_id in zip(bag.clips, bag.recording_ids)
    ]


def ensemble_attribution(
    models: list[SparseEvidenceModel],
    bags: list[PatientBag],
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, set[int]]:
    """Return ensemble probability and a scale-balanced window attribution.

    The reported classifier averages probabilities over random initializations.
    Evidence audits therefore operate on that ensemble.  We align each member's
    exact additive logit contributions to the ensemble-predicted class, divide
    by its L1 mass so that one member cannot dominate only through logit scale,
    and then average.  ``active`` records the union of non-zero supports; for
    maximum pooling this is the only defensible selected set.
    """
    if len(models) != len(bags) or not models:
        raise ValueError("models and bags must be non-empty and aligned")
    probabilities, aligned, local_probabilities = [], [], []
    active: set[int] = set()
    with torch.inference_mode():
        for model, bag in zip(models, bags):
            clips = tensor_clips(bag, device)
            logit, contribution, weights = model(clips, return_local=True)
            probabilities.append(torch.sigmoid(logit).item())
            encoded, _ = model.encode(clips)
            local_probabilities.append(
                torch.sigmoid(model.instance_head(encoded).squeeze(-1) + model.bias)
                .detach().cpu().numpy()
            )
            active.update(np.flatnonzero(weights.detach().cpu().numpy() > 0).tolist())
            aligned.append(contribution.detach().cpu().numpy())
    probability = float(np.mean(probabilities))
    sign = 1.0 if probability >= 0.5 else -1.0
    normalized = []
    for contribution in aligned:
        contribution = sign * contribution
        normalized.append(contribution / (np.abs(contribution).sum() + 1e-12))
    return (
        probability,
        np.mean(normalized, axis=0),
        np.mean(local_probabilities, axis=0),
        active,
    )


def ensemble_probability(
    models: list[SparseEvidenceModel],
    clips_by_model: list[list[torch.Tensor]],
) -> float:
    with torch.inference_mode():
        return float(np.mean([
            torch.sigmoid(model(clips)).item()
            for model, clips in zip(models, clips_by_model)
        ]))


def evidence_statistics(
    models: list[SparseEvidenceModel],
    bags: list[PatientBag],
    method: str,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict]:
    for model in models:
        model.eval()
    base_prob, contribution, local_probability, active = ensemble_attribution(
        models, bags, device
    )
    predicted_sign = 1.0 if base_prob >= 0.5 else -1.0
    order = np.argsort(-contribution, kind="stable")
    normalized = np.abs(contribution)
    normalized = normalized / (normalized.sum() + 1e-12)
    concentration = {
        "effective_window_fraction": float(1.0 / np.sum(normalized ** 2) / len(normalized)),
        "top10_contribution_mass": float(np.sort(normalized)[-max(1, math.ceil(.10 * len(normalized))):].sum()),
        "fraction_windows_on_patient_side": float(np.mean(
            local_probability >= 0.5 if bags[0].label else local_probability < 0.5
        )),
        "window_probability_sd": float(np.std(local_probability)),
        "window_probability_iqr": float(
            np.quantile(local_probability, .75) - np.quantile(local_probability, .25)
        ),
    }

    deletion_rows, sufficiency_rows = [], []
    clips_by_model = [tensor_clips(bag, device) for bag in bags]
    lengths = [len(clip) for clip in bags[0].clips]
    base_class_conf = base_prob if predicted_sign > 0 else 1.0 - base_prob

    def rebuild(
        clips: list[torch.Tensor], mask: np.ndarray, retain: bool
    ) -> list[torch.Tensor]:
        altered, start = [], 0
        for clip, length in zip(clips, lengths):
            local_mask = mask[start : start + length]
            changed = clip.clone()
            if retain:
                changed[~torch.from_numpy(local_mask).to(device)] = 0.0
            else:
                changed[torch.from_numpy(local_mask).to(device)] = 0.0
            altered.append(changed)
            start += length
        return altered

    # Maximum pooling has no ranking beyond the window selected by a member.
    # Its natural ensemble support is the union of the members' argmax windows.
    requested_fractions = (
        (0.10,) if method in {"max", "top10"} else (0.05, 0.10, 0.25, 0.50)
    )
    with torch.inference_mode():
        for fraction in requested_fractions:
            k = (
                max(1, len(active))
                if method in {"max", "top10"}
                else max(1, math.ceil(fraction * len(order)))
            )
            top_mask = np.zeros(len(order), dtype=bool)
            if method in {"max", "top10"}:
                top_mask[np.asarray(sorted(active), dtype=int)] = True
            else:
                top_mask[order[:k]] = True
            for analysis, retain, output in (
                ("deletion", False, deletion_rows),
                ("sufficiency", True, sufficiency_rows),
            ):
                probability = ensemble_probability(
                    models,
                    [rebuild(clips, top_mask, retain) for clips in clips_by_model],
                )
                class_conf = probability if predicted_sign > 0 else 1.0 - probability
                random_conf = []
                for _ in range(20):
                    random_mask = np.zeros(len(order), dtype=bool)
                    random_mask[rng.choice(len(order), k, replace=False)] = True
                    random_probability = ensemble_probability(
                        models,
                        [rebuild(clips, random_mask, retain) for clips in clips_by_model],
                    )
                    random_conf.append(
                        random_probability if predicted_sign > 0 else 1.0 - random_probability
                    )
                output.append({
                    "analysis": analysis,
                    "fraction": fraction,
                    "selected_windows": k,
                    "selected_fraction": k / len(order),
                    "base_confidence": base_class_conf,
                    "selected_confidence": class_conf,
                    "random_confidence": float(np.mean(random_conf)),
                    "selected_minus_random_drop": (
                        (base_class_conf - class_conf)
                        - (base_class_conf - float(np.mean(random_conf)))
                    ),
                })
    return deletion_rows, sufficiency_rows, concentration


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def evidence_centroid(
    models: list[SparseEvidenceModel],
    bags: list[PatientBag],
    method: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    _, score, _, active = ensemble_attribution(models, bags, device)
    if method in {"max", "top10"}:
        selected = np.asarray(sorted(active), dtype=int)
        k = len(selected)
    else:
        k = max(1, math.ceil(0.10 * len(score)))
        selected = np.argsort(-score, kind="stable")[:k]
    matrix = np.concatenate(bags[0].clips, axis=0)
    return matrix[selected].mean(axis=0), matrix, k


def random_subset_cross_view(
    first: tuple[np.ndarray, np.ndarray, int],
    second: tuple[np.ndarray, np.ndarray, int],
    rng: np.random.Generator,
    draws: int = 100,
) -> float:
    """Mean cross-view cosine for random subsets matched to selected-set size."""
    first_matrix, first_k = first[1], first[2]
    second_matrix, second_k = second[1], second[2]
    values = []
    for _ in range(draws):
        first_centroid = first_matrix[
            rng.choice(len(first_matrix), first_k, replace=False)
        ].mean(0)
        second_centroid = second_matrix[
            rng.choice(len(second_matrix), second_k, replace=False)
        ].mean(0)
        values.append(cosine(first_centroid, second_centroid))
    return float(np.mean(values))


def clinical_benchmark(
    workspace: Path,
    device: torch.device,
    methods: tuple[str, ...],
    tasks: tuple[str, ...],
    seeds: int,
    max_epochs: int,
    diagnosis_cohort: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = cache_index(workspace)
    clinical = pd.read_csv(workspace / "data/final_short_merged.csv", dtype={"short_recording_id": str})
    clinical = clinical[clinical.pre_post_treatment_label.eq("PRE")].copy()
    routine = pd.read_csv(workspace / "data/final_test.csv", dtype={"short_recording_id": str})
    result_rows, perturb_rows, concentration_rows, scale_rows = [], [], [], []

    for task in tasks:
        development = load_bags(clinical, index, task)
        evaluation = load_bags(routine, index, task)
        manifest = pd.read_csv(workspace / f"kfold_split/data/{task}/fold_manifest.csv")
        if task == "case_control":
            keep = diagnosis_cohort_ids(workspace, diagnosis_cohort)
            if keep is not None:
                development = {pid: bag for pid, bag in development.items() if pid in keep}
                evaluation = {pid: bag for pid, bag in evaluation.items() if pid in keep}
                manifest = manifest[manifest.patient_id.astype(int).isin(keep)].copy()
        fold_of = manifest.set_index("patient_id")["fold"].astype(int).to_dict()
        if set(development) != set(fold_of) or set(evaluation) != set(fold_of):
            raise RuntimeError(f"Patient/fold mismatch for {task}")

        for method in methods:
            for fold in range(5):
                test_ids = {pid for pid, assigned in fold_of.items() if assigned == fold}
                validation_fold = (fold + 1) % 5
                validation_ids = {pid for pid, assigned in fold_of.items() if assigned == validation_fold}
                outer_ids = set(fold_of) - test_ids
                inner_ids = outer_ids - validation_ids
                outer = [development[pid] for pid in sorted(outer_ids)]
                raw_test = [evaluation[pid] for pid in sorted(test_ids)]

                seed_predictions, seed_clip_predictions = [], []
                seed_models = []
                seed_epochs = []
                seed_transformed_test = []
                for seed_index in range(seeds):
                    seed = SEED_BASE + 1000 * seed_index + 100 * fold + len(method)
                    model, epoch, center, scale = fit_outer_model(
                        method, 122, outer, inner_ids, validation_ids, device, seed, max_epochs
                    )
                    test = [transform_bag(bag, center, scale) for bag in raw_test]
                    seed_predictions.append(predict(model, test, device))
                    clip_predictions = {
                        bag.patient_id: predict(model, split_clip_bags(bag), device)
                        for bag in test
                    }
                    seed_clip_predictions.append(clip_predictions)
                    seed_models.append(model)
                    seed_epochs.append(epoch)
                    seed_transformed_test.append(test)

                    if method == "multiscale":
                        weights = torch.softmax(model.scale_logits, dim=0).detach().cpu().numpy()
                        for fraction, weight in zip(SCALES, weights):
                            scale_rows.append({
                                "task": task, "method": method, "fold": fold,
                                "seed": seed_index, "fraction": fraction,
                                "weight": float(weight), "selected_epoch": epoch,
                            })

                ensemble = np.mean(seed_predictions, axis=0)
                labels = np.asarray([bag.label for bag in raw_test])
                for patient_index, (bag, label, score) in enumerate(
                    zip(raw_test, labels, ensemble)
                ):
                    clip_scores = np.mean(
                        [item[bag.patient_id] for item in seed_clip_predictions], axis=0
                    )
                    result_rows.append({
                        "task": task, "cohort": diagnosis_cohort,
                        "method": method, "fold": fold,
                        "patient_id": bag.patient_id, "label": int(label),
                        "patient_probability": float(score),
                        "routine_clip_1_probability": float(clip_scores[0]),
                        "routine_clip_2_probability": float(clip_scores[1]),
                        "selected_epoch_mean": float(np.mean(seed_epochs)),
                        "seed_probability_sd": float(
                            np.std([p[patient_index] for p in seed_predictions], ddof=1)
                        ) if len(seed_predictions) > 1 else 0.0,
                        **{
                            f"seed_{seed_index}_probability": float(values[patient_index])
                            for seed_index, values in enumerate(seed_predictions)
                        },
                    })

                # Audit the same probability ensemble reported for classification.
                rng = np.random.default_rng(SEED_BASE + fold * 31 + len(method))
                test_by_patient = [
                    {bag.patient_id: bag for bag in test}
                    for test in seed_transformed_test
                ]
                centroids = {
                    bag.patient_id: [
                        evidence_centroid(
                            seed_models,
                            [split_clip_bags(test[bag.patient_id])[clip_index]
                             for test in test_by_patient],
                            method,
                            device,
                        )
                        for clip_index in range(2)
                    ]
                    for bag in raw_test
                }
                same_label_ids = {
                    label: [bag.patient_id for bag in raw_test if bag.label == label]
                    for label in (0, 1)
                }
                for bag in raw_test:
                    aligned_bags = [test[bag.patient_id] for test in test_by_patient]
                    deletion, sufficiency, concentration = evidence_statistics(
                        seed_models, aligned_bags, method, device, rng
                    )
                    for row in deletion + sufficiency:
                        perturb_rows.append({
                            "task": task, "cohort": diagnosis_cohort,
                            "method": method, "fold": fold,
                            "patient_id": bag.patient_id, "label": bag.label, **row,
                        })
                    first, second = centroids[bag.patient_id]
                    concentration_rows.append({
                        "task": task, "cohort": diagnosis_cohort,
                        "method": method, "fold": fold,
                        "patient_id": bag.patient_id, "label": bag.label,
                        **concentration,
                        "selected_windows_view_1": first[2],
                        "selected_windows_view_2": second[2],
                        "matched_view_cosine": cosine(first[0], second[0]),
                        "random_matched_view_cosine": random_subset_cross_view(
                            first, second, rng
                        ),
                        "all_window_matched_view_cosine": cosine(
                            first[1].mean(0), second[1].mean(0)
                        ),
                        "unmatched_same_label_cosine": np.nan,
                    })
                    candidates = [
                        pid for pid in same_label_ids[bag.label] if pid != bag.patient_id
                    ]
                    if candidates:
                        other = int(rng.choice(candidates))
                        concentration_rows[-1]["unmatched_same_label_cosine"] = cosine(
                            first[0], centroids[other][1][0]
                        )
                print(f"clinical {task} {method} fold {fold} complete", flush=True)

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(perturb_rows),
        pd.DataFrame(concentration_rows),
        pd.DataFrame(scale_rows),
    )


def inject_motif(
    bag: PatientBag,
    label: int,
    fraction: float,
    rng: np.random.Generator,
    amplitude: float = 2.0,
) -> tuple[PatientBag, list[np.ndarray]]:
    # Standardized feature-space motif: higher delta power and delta/beta ratio,
    # lower spectral edge frequency, repeated across the five anatomical regions.
    direction = np.zeros(122, dtype=np.float32)
    for start in (0, 16, 32, 48, 64):
        direction[start + 0] = 1.0
        direction[start + 6] = 0.75
        direction[start + 15] = -0.75
    direction /= np.linalg.norm(direction)
    direction *= amplitude * math.sqrt(np.count_nonzero(direction))
    clips = [clip.copy() for clip in bag.clips]
    masks = [np.zeros(len(clip), dtype=bool) for clip in bag.clips]
    if label == 1 and fraction > 0:
        # Fraction is defined over the complete patient bag.  Allocate one
        # contiguous segment at a time, so a low-prevalence motif can occur in
        # only one recording rather than being repeated in every clip.
        remaining = max(1, math.ceil(fraction * sum(map(len, clips))))
        order = rng.permutation(len(clips))
        for clip_index in order:
            if remaining == 0:
                break
            width = min(remaining, len(clips[clip_index]))
            start = int(rng.integers(0, len(clips[clip_index]) - width + 1))
            masks[clip_index][start : start + width] = True
            clips[clip_index][masks[clip_index]] += direction
            remaining -= width
        if remaining:
            raise RuntimeError("Could not allocate the requested synthetic motif")
    return PatientBag(bag.patient_id, label, clips, bag.recording_ids), masks


def balanced_synthetic_labels(manifest: pd.DataFrame, assignment: int) -> dict[int, int]:
    labels = {}
    rng = np.random.default_rng(SEED_BASE + 7000 + assignment)
    for _, group in manifest.groupby("fold"):
        ids = group.patient_id.astype(int).to_numpy().copy()
        rng.shuffle(ids)
        for position, patient_id in enumerate(ids):
            labels[int(patient_id)] = int(position % 2 == 0)
    return labels


def tie_aware_top_k_recall(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Expected recall when the kth score is tied."""
    if k <= 0:
        return np.nan
    cutoff = np.partition(score, len(score) - k)[len(score) - k]
    above = score > cutoff
    tied = score == cutoff
    remaining = k - int(above.sum())
    expected_hits = float(truth[above].sum())
    if remaining:
        expected_hits += remaining * float(truth[tied].mean())
    return expected_hits / k


def synthetic_benchmark(
    workspace: Path,
    device: torch.device,
    methods: tuple[str, ...],
    max_epochs: int,
    assignment_start: int,
    assignments: int,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    index = cache_index(workspace)
    clinical = pd.read_csv(workspace / "data/final_short_merged.csv", dtype={"short_recording_id": str})
    clinical = clinical[clinical.pre_post_treatment_label.eq("PRE")].copy()
    routine = pd.read_csv(workspace / "data/final_test.csv", dtype={"short_recording_id": str})
    # Case/control loading is used only to construct bags; original labels are discarded.
    development_raw = load_bags(clinical, index, "case_control")
    evaluation_raw = load_bags(routine, index, "case_control")
    manifest = pd.read_csv(workspace / "kfold_split/data/case_control/fold_manifest.csv")
    fold_of = manifest.set_index("patient_id")["fold"].astype(int).to_dict()
    rows = []

    for fraction in fractions:
        for assignment in range(assignment_start, assignment_start + assignments):
            synthetic_label = balanced_synthetic_labels(manifest, assignment)
            accumulated = {
                method: {
                    "labels": [], "scores": [], "folds": [],
                    "aps": [], "recalls": [],
                    "fold_aurocs": [], "epochs": [],
                }
                for method in methods
            }
            for fold in range(5):
                test_ids = {pid for pid, assigned in fold_of.items() if assigned == fold}
                validation_fold = (fold + 1) % 5
                validation_ids = {pid for pid, assigned in fold_of.items() if assigned == validation_fold}
                outer_ids = set(fold_of) - test_ids
                inner_ids = outer_ids - validation_ids
                raw_outer = [development_raw[pid] for pid in sorted(outer_ids)]

                def prepare(
                    bags: list[PatientBag], center: np.ndarray, scale: np.ndarray
                ) -> tuple[list[PatientBag], dict[int, np.ndarray]]:
                    prepared, masks_by_patient = [], {}
                    for bag in bags:
                        # A joint assignment defines one fixed motif location
                        # per patient across every outer-fold model.
                        rng = np.random.default_rng(
                            SEED_BASE + 10000 * assignment
                            + int(fraction * 100) + 1000000 * bag.patient_id
                        )
                        scaled = transform_bag(bag, center, scale)
                        injected, masks = inject_motif(
                            scaled, synthetic_label[bag.patient_id], fraction, rng
                        )
                        prepared.append(injected)
                        masks_by_patient[bag.patient_id] = np.concatenate(masks)
                    return prepared, masks_by_patient

                # Epoch selection sees scaling statistics from inner-training
                # patients only.  The final fit then uses all outer-training
                # patients, exactly as in the clinical benchmark.
                raw_inner = [bag for bag in raw_outer if bag.patient_id in inner_ids]
                raw_validation = [
                    bag for bag in raw_outer if bag.patient_id in validation_ids
                ]
                center_inner, scale_inner = fit_scaler(raw_inner)
                inner, _ = prepare(raw_inner, center_inner, scale_inner)
                validation, _ = prepare(
                    raw_validation, center_inner, scale_inner
                )

                center, scale = fit_scaler(raw_outer)
                outer, _ = prepare(raw_outer, center, scale)
                raw_test = [evaluation_raw[pid] for pid in sorted(test_ids)]
                test, truth_masks = prepare(raw_test, center, scale)

                for method in methods:
                    seed = SEED_BASE + 50000 + 1000 * assignment + 100 * fold + len(method)
                    epoch = select_epoch(
                        method, 122, inner, validation, device, seed, max_epochs
                    )
                    torch.manual_seed(seed)
                    model = SparseEvidenceModel(122, method).to(device)
                    train_fixed_epochs(model, outer, device, epoch, seed)
                    scores = predict(model, test, device)
                    labels = np.asarray([bag.label for bag in test])
                    bucket = accumulated[method]
                    bucket["labels"].extend(labels.tolist())
                    bucket["scores"].extend(scores.tolist())
                    bucket["folds"].extend([fold] * len(labels))
                    bucket["fold_aurocs"].append(auc_or_half(labels, scores))
                    bucket["epochs"].append(epoch)
                    model.eval()
                    with torch.inference_mode():
                        for bag in test:
                            if bag.label != 1:
                                continue
                            _, contribution, _ = model(
                                tensor_clips(bag, device), return_local=True
                            )
                            evidence = contribution.detach().cpu().numpy()
                            truth = truth_masks[bag.patient_id].astype(int)
                            k = int(truth.sum())
                            if k and k < len(truth):
                                bucket["aps"].append(
                                    average_precision_score(truth, evidence)
                                )
                                bucket["recalls"].append(
                                    tie_aware_top_k_recall(truth, evidence, k)
                                )
                    print(
                        f"synthetic f={fraction:.2f} assignment={assignment} "
                        f"fold={fold} {method} complete", flush=True
                    )
            for method, bucket in accumulated.items():
                rows.append({
                    "fraction": fraction, "assignment": assignment,
                    "method": method,
                    "bag_auroc": fold_comparable_auc(
                        bucket["labels"], bucket["scores"], bucket["folds"]
                    ),
                    "pooled_oof_bag_auroc": auc_or_half(
                        bucket["labels"], bucket["scores"]
                    ),
                    "fold_auroc_mean": float(np.mean(bucket["fold_aurocs"])),
                    "interval_average_precision": (
                        float(np.mean(bucket["aps"])) if bucket["aps"] else np.nan
                    ),
                    "top_k_recall": (
                        float(np.mean(bucket["recalls"]))
                        if bucket["recalls"] else np.nan
                    ),
                    "selected_epoch_mean": float(np.mean(bucket["epochs"])),
                })
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, method), group in predictions.groupby(["task", "method"]):
        fold_aucs = [
            auc_or_half(fold_group.label, fold_group.patient_probability)
            for _, fold_group in group.groupby("fold")
        ]
        rows.append({
            "task": task,
            "method": method,
            "n_patients": group.patient_id.nunique(),
            "fold_comparable_auroc": fold_comparable_auc(
                group.label, group.patient_probability, group.fold
            ),
            "pooled_oof_auroc": auc_or_half(group.label, group.patient_probability),
            "fold_auroc_mean": float(np.mean(fold_aucs)),
            "fold_auroc_sd": float(np.std(fold_aucs, ddof=1)),
            "routine_view_probability_mae": float(np.mean(np.abs(
                group.routine_clip_1_probability - group.routine_clip_2_probability
            ))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    paper = _paths.runtime()
    workspace = _paths.workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mode", choices=("clinical", "synthetic", "all"), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASK_COLUMN), default=list(TASK_COLUMN)
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--diagnosis-cohort",
        choices=(
            "all", "clinical_mimic", "flagged_subset",
            "onset_le24_full_controls", "onset_le24_clinical_mimic",
        ),
        default="all",
        help="Retrain and evaluate diagnosis within a prespecified sensitivity cohort.",
    )
    parser.add_argument("--synthetic-assignments", type=int, default=3)
    parser.add_argument("--synthetic-assignment-start", type=int, default=0)
    parser.add_argument(
        "--synthetic-fractions", nargs="+", type=float,
        default=[0.00, 0.02, 0.10, 0.25, 1.00],
    )
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument(
        "--output-dir", type=Path,
        default=paper / "analysis/generated/sparse_evidence",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Sparse-evidence experiments require explicit CUDA")
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(args.methods)

    if args.mode in {"clinical", "all"}:
        predictions, perturbations, concentration, scales = clinical_benchmark(
            workspace, device, methods, tuple(args.tasks), args.seeds, args.max_epochs,
            args.diagnosis_cohort,
        )
        predictions.to_csv(args.output_dir / "clinical_patient_predictions.csv", index=False)
        summarize_predictions(predictions).to_csv(
            args.output_dir / "clinical_summary.csv", index=False
        )
        perturbations.to_csv(args.output_dir / "clinical_evidence_perturbation.csv", index=False)
        concentration.to_csv(args.output_dir / "clinical_evidence_concentration.csv", index=False)
        scales.to_csv(args.output_dir / "clinical_multiscale_weights.csv", index=False)

    if args.mode in {"synthetic", "all"}:
        synthetic = synthetic_benchmark(
            workspace, device, methods, args.max_epochs,
            args.synthetic_assignment_start, args.synthetic_assignments,
            tuple(args.synthetic_fractions),
        )
        synthetic.to_csv(args.output_dir / "synthetic_recovery.csv", index=False)
        synthetic.groupby(["fraction", "method"], as_index=False).agg(
            bag_auroc_mean=("bag_auroc", "mean"),
            bag_auroc_sd=("bag_auroc", "std"),
            interval_ap_mean=("interval_average_precision", "mean"),
            interval_ap_sd=("interval_average_precision", "std"),
            top_k_recall_mean=("top_k_recall", "mean"),
            top_k_recall_sd=("top_k_recall", "std"),
        ).to_csv(args.output_dir / "synthetic_recovery_summary.csv", index=False)


if __name__ == "__main__":
    main()
