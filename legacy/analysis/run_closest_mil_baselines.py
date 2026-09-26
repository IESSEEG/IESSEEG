#!/usr/bin/env python
"""Run MILLET pooling heads and a TimeMIL-style baseline on qEEG bags.

The unit of supervision is a patient.  Each of the four development clips is
treated as an ordered bag of 30-second qEEG windows; clip logits are averaged
so that recording duration cannot determine a patient's weight.  Evaluation
uses the two separately sampled Routine Clips from held-out patients.

The four MILLET heads implement Eqs. 2--5 and the released ICLR 2024 code:
sigmoid (not softmax) attention, mean pooling, and the distinct order of
attention and classification in Attention, Additive, and Conjunctive pooling.
TimeMIL retains the released model's three learnable Mexican-hat wavelet
positional bases, two pre-normalized Nystrom multi-head self-attention
layers, class token, and warm-up blend.  Both methods are adapted to precomputed
qEEG window vectors and multiple clips; these adaptations are recorded in the
output metadata, so this script does not claim an exact end-to-end reproduction
of either paper's original raw-time-series backbone or training recipe.

Patient-level outputs can contain governed response labels and therefore belong
under an ignored local analysis directory.  Only reviewed aggregates should be
committed.
"""

from __future__ import annotations
import legacy_paths as _paths

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sparse_evidence_benchmark import (  # noqa: E402
    PatientBag,
    SEED_BASE,
    TASK_COLUMN,
    cache_index,
    fit_scaler,
    load_bags,
    split_clip_bags,
    transform_bag,
)


METHODS = ("instance", "attention", "additive", "conjunctive", "timemil")
MILLET_METHODS = set(METHODS[:-1])
SOURCE_URLS = {
    "millet_paper": (
        "https://openreview.net/pdf?id=xriGRsoAza"
    ),
    "millet_code": "https://github.com/JAEarly/MILTimeSeriesClassification",
    "timemil_paper": "https://proceedings.mlr.press/v235/chen24af.html",
    "timemil_code": "https://github.com/xiwenc1/TimeMIL",
}


class SinusoidalPosition(torch.nn.Module):
    """MILLET's optional sinusoidal encoding, applied before pooling."""

    def __init__(self, dimension: int, max_length: int = 4096):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(1, max_length, dimension)
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds positional limit "
                f"{self.encoding.shape[1]}"
            )
        return x + self.encoding[:, : x.shape[1]]


class MILLETPooling(torch.nn.Module):
    """Binary forms of the four official MILLET pooling equations."""

    def __init__(
        self,
        dimension: int,
        method: str,
        attention_dimension: int = 8,
        dropout: float = 0.10,
        positional_encoding: bool = True,
    ):
        super().__init__()
        if method not in MILLET_METHODS:
            raise ValueError(method)
        self.method = method
        self.position = (
            SinusoidalPosition(dimension) if positional_encoding else torch.nn.Identity()
        )
        self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()
        self.classifier = torch.nn.Linear(dimension, 1)
        if method != "instance":
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(dimension, attention_dimension),
                torch.nn.Tanh(),
                torch.nn.Linear(attention_dimension, 1),
                torch.nn.Sigmoid(),
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return one clip logit from shape ``[1, windows, dimension]``."""
        z = self.dropout(self.position(embeddings))
        if self.method == "instance":
            return self.classifier(z).mean(dim=1).squeeze()

        attention = self.attention(z)
        if self.method == "attention":
            # MILLET Eq. 2: classify the mean of sigmoid-weighted embeddings.
            return self.classifier((attention * z).mean(dim=1)).squeeze()
        if self.method == "additive":
            # MILLET Eq. 4: weight embeddings before instance classification.
            return self.classifier(attention * z).mean(dim=1).squeeze()
        if self.method == "conjunctive":
            # MILLET Eq. 5: classify first, then weight instance logits.
            return (attention * self.classifier(z)).mean(dim=1).squeeze()
        raise AssertionError(self.method)


def iterative_pseudoinverse(matrix: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    """Moore--Penrose approximation used by the released TimeMIL attention."""
    absolute = matrix.abs()
    column = absolute.sum(dim=-1)
    row = absolute.sum(dim=-2)
    denominator = (column.amax() * row.amax()).clamp_min(1e-8)
    estimate = matrix.transpose(-1, -2) / denominator
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    for _ in range(iterations):
        product = matrix @ estimate
        estimate = 0.25 * estimate @ (
            13 * identity
            - product @ (15 * identity - product @ (7 * identity - product))
        )
    return estimate


class NystromSelfAttention(torch.nn.Module):
    """Nystrom MHSA configured as in the released TimeMIL model."""

    def __init__(
        self,
        dimension: int = 128,
        heads: int = 8,
        landmarks: int = 64,
        inverse_iterations: int = 6,
        residual_kernel: int = 33,
        dropout: float = 0.20,
    ):
        super().__init__()
        if dimension % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.landmarks = landmarks
        self.inverse_iterations = inverse_iterations
        self.scale = self.head_dimension ** -0.5
        self.to_qkv = torch.nn.Linear(dimension, 3 * dimension, bias=False)
        self.to_output = torch.nn.Sequential(
            torch.nn.Linear(dimension, dimension), torch.nn.Dropout(dropout)
        )
        padding = residual_kernel // 2
        self.value_residual = torch.nn.Conv2d(
            heads,
            heads,
            (residual_kernel, 1),
            padding=(padding, 0),
            groups=heads,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, original_length, dimension = x.shape
        remainder = original_length % self.landmarks
        if remainder:
            x = torch.nn.functional.pad(
                x, (0, 0, self.landmarks - remainder, 0), value=0.0
            )
        length = x.shape[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, length, self.heads, self.head_dimension
            ).permute(0, 2, 1, 3)

        q, k, v = map(heads, (q, k, v))
        q = q * self.scale
        windows_per_landmark = length // self.landmarks
        q_landmarks = q.reshape(
            batch, self.heads, self.landmarks, windows_per_landmark,
            self.head_dimension,
        ).mean(dim=3)
        k_landmarks = k.reshape(
            batch, self.heads, self.landmarks, windows_per_landmark,
            self.head_dimension,
        ).mean(dim=3)
        similarity_1 = torch.einsum("bhid,bhjd->bhij", q, k_landmarks)
        similarity_2 = torch.einsum(
            "bhid,bhjd->bhij", q_landmarks, k_landmarks
        )
        similarity_3 = torch.einsum("bhid,bhjd->bhij", q_landmarks, k)
        attention_1 = similarity_1.softmax(dim=-1)
        attention_2 = similarity_2.softmax(dim=-1)
        attention_3 = similarity_3.softmax(dim=-1)
        output = (
            attention_1
            @ iterative_pseudoinverse(attention_2, self.inverse_iterations)
            @ (attention_3 @ v)
        )
        output = output + self.value_residual(v)
        output = output.permute(0, 2, 1, 3).reshape(batch, length, dimension)
        output = self.to_output(output)
        return output[:, -original_length:]


class TimeMILTransformerLayer(torch.nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dimension)
        self.attention = NystromSelfAttention(
            dimension=dimension,
            heads=8,
            landmarks=dimension // 2,
            inverse_iterations=6,
            residual_kernel=33,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.norm(x))


def mexican_hat_kernels(
    scale: torch.Tensor, shift: torch.Tensor, kernel_size: int
) -> torch.Tensor:
    """Return one learnable Mexican-hat depthwise kernel per feature."""
    coordinate = torch.linspace(
        -(kernel_size - 1) // 2,
        (kernel_size - 1) // 2,
        kernel_size,
        device=scale.device,
        dtype=scale.dtype,
    )[None, :]
    # The public implementation learns unconstrained scales.  A tiny signed
    # floor only prevents an undefined division if optimization reaches zero.
    signed_scale = torch.where(
        scale >= 0, scale.clamp_min(1e-4), scale.clamp_max(-1e-4)
    )
    normalized = (coordinate - shift[:, None]) / signed_scale[:, None]
    constant = 2.0 / (math.sqrt(3.0) * math.pi ** 0.25)
    kernel = (
        constant
        * (1.0 - normalized.square())
        * torch.exp(-0.5 * normalized.square())
        / signed_scale.abs().sqrt()[:, None]
    )
    return kernel


class WaveletPosition(torch.nn.Module):
    """Three learnable Mexican-hat bases from TimeMIL Eqs. 10--11."""

    def __init__(self, dimension: int, kernel_size: int = 19):
        super().__init__()
        self.kernel_size = kernel_size
        self.scales = torch.nn.Parameter(
            torch.ones(3, dimension) + torch.randn(3, dimension)
        )
        self.shifts = torch.nn.Parameter(torch.zeros(3, dimension))
        self.projection = torch.nn.Linear(dimension, dimension)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        class_token, features = tokens[:, :1], tokens[:, 1:]
        channel_first = features.transpose(1, 2)
        encoded = torch.zeros_like(channel_first)
        for index in range(3):
            kernel = mexican_hat_kernels(
                self.scales[index], self.shifts[index], self.kernel_size
            )
            encoded = encoded + torch.nn.functional.conv1d(
                channel_first,
                kernel[:, None, :],
                groups=channel_first.shape[1],
                padding="same",
            )
        features = features + self.projection(encoded.transpose(1, 2))
        return torch.cat((class_token, features), dim=1)


class TimeMILPooling(torch.nn.Module):
    """TimeMIL aggregation after an externally supplied window encoder."""

    def __init__(self, dimension: int = 128, dropout: float = 0.20):
        super().__init__()
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, dimension))
        self.wavelet_1 = WaveletPosition(dimension)
        self.layer_1 = TimeMILTransformerLayer(dimension, dropout)
        self.wavelet_2 = WaveletPosition(dimension)
        self.layer_2 = TimeMILTransformerLayer(dimension, dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dimension, dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dimension, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Matches TimeMIL's explicit Xavier initialization of linear layers.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)

    def forward(self, embeddings: torch.Tensor, warmup: bool = False) -> torch.Tensor:
        global_token = embeddings.mean(dim=1)
        class_token = self.class_token.expand(embeddings.shape[0], -1, -1)
        tokens = torch.cat((class_token, embeddings), dim=1)
        tokens = self.layer_1(self.wavelet_1(tokens))
        tokens = self.layer_2(self.wavelet_2(tokens))
        representation = tokens[:, 0]
        if warmup:
            # This intentionally follows the released 0.1 / 0.99 blend.
            representation = 0.1 * representation + 0.99 * global_token
        return self.classifier(representation).squeeze()


class QEEGMILModel(torch.nn.Module):
    """Shared qEEG window encoder with clip-equal hierarchical aggregation."""

    def __init__(
        self,
        input_dimension: int,
        method: str,
        embedding_dimension: int = 128,
        encoder_dropout: float = 0.20,
        millet_dropout: float = 0.10,
        timemil_dropout: float = 0.20,
        millet_positional_encoding: bool = True,
        timemil_mask_ratio: float = 0.50,
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        if embedding_dimension % 8:
            raise ValueError("Embedding dimension must be divisible by 8")
        self.method = method
        self.timemil_mask_ratio = timemil_mask_ratio
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, embedding_dimension),
            torch.nn.LayerNorm(embedding_dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(encoder_dropout),
            torch.nn.Linear(embedding_dimension, embedding_dimension),
            torch.nn.LayerNorm(embedding_dimension),
            torch.nn.GELU(),
        )
        if method in MILLET_METHODS:
            self.pool = MILLETPooling(
                embedding_dimension,
                method,
                attention_dimension=8,
                dropout=millet_dropout,
                positional_encoding=millet_positional_encoding,
            )
        else:
            self.pool = TimeMILPooling(embedding_dimension, timemil_dropout)

    def mask_timemil_windows(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply TimeMIL's training-only decile masking to one clip."""
        if self.method != "timemil" or self.timemil_mask_ratio <= 0:
            return clip
        number_of_deciles = 10
        selected = int(self.timemil_mask_ratio * number_of_deciles)
        if selected <= 0:
            return clip
        interval = len(clip) // number_of_deciles
        if interval == 0:
            return clip
        masked = clip.clone()
        for index in torch.randperm(number_of_deciles, device=clip.device)[:selected]:
            start = int(index) * interval
            masked[start : start + interval] = torch.randn(
                (), device=clip.device, dtype=clip.dtype
            )
        return masked

    def clip_logits(
        self,
        clips: list[torch.Tensor],
        warmup: bool = False,
        patch_mask: bool = False,
    ) -> torch.Tensor:
        logits = []
        for clip in clips:
            if patch_mask:
                clip = self.mask_timemil_windows(clip)
            embeddings = self.encoder(clip).unsqueeze(0)
            if self.method == "timemil":
                logit = self.pool(embeddings, warmup=warmup)
            else:
                logit = self.pool(embeddings)
            logits.append(logit.reshape(()))
        return torch.stack(logits)

    def forward(
        self,
        clips: list[torch.Tensor],
        warmup: bool = False,
        patch_mask: bool = False,
    ) -> torch.Tensor:
        return self.clip_logits(clips, warmup=warmup, patch_mask=patch_mask).mean()


def tensor_clips(bag: PatientBag, device: torch.device) -> list[torch.Tensor]:
    return [
        clip if isinstance(clip, torch.Tensor) else torch.from_numpy(clip)
        for clip in bag.clips
    ] if all(isinstance(clip, torch.Tensor) and clip.device == device for clip in bag.clips) else [
        clip.to(device) if isinstance(clip, torch.Tensor) else torch.from_numpy(clip).to(device)
        for clip in bag.clips
    ]


def device_bags(bags: Iterable[PatientBag], device: torch.device) -> list[PatientBag]:
    return [
        PatientBag(
            bag.patient_id,
            bag.label,
            tensor_clips(bag, device),
            bag.recording_ids,
        )
        for bag in bags
    ]


def auc_or_half(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(scores), dtype=float)
    return 0.5 if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))


def predict(
    model: QEEGMILModel,
    bags: list[PatientBag],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for bag in bags:
            logit = model(tensor_clips(bag, device))
            probabilities.append(torch.sigmoid(logit).item())
    return np.asarray(probabilities)


def shuffled_patient_probabilities(
    models: list[QEEGMILModel],
    transformed_bags: list[dict[int, PatientBag]],
    patient_id: int,
    device: torch.device,
    task_index: int,
    fold: int,
    draws: int,
) -> np.ndarray:
    """Shuffle each Routine Clip independently with fixed permutations.

    A permutation is shared across the fitted random-initialization ensemble so
    that the control changes temporal order, not the windows presented to one
    seed versus another.
    """
    reference = transformed_bags[0][patient_id]
    output = []
    for draw in range(draws):
        generator = np.random.default_rng(
            SEED_BASE
            + 170000
            + 10000 * task_index
            + 1000 * fold
            + 10 * patient_id
            + draw
        )
        permutations = [generator.permutation(len(clip)) for clip in reference.clips]
        seed_probabilities = []
        for model, by_patient in zip(models, transformed_bags):
            bag = by_patient[patient_id]
            shuffled = PatientBag(
                patient_id=bag.patient_id,
                label=bag.label,
                clips=[
                    clip[permutation].copy()
                    for clip, permutation in zip(bag.clips, permutations)
                ],
                recording_ids=bag.recording_ids,
            )
            seed_probabilities.append(predict(model, [shuffled], device)[0])
        output.append(float(np.mean(seed_probabilities)))
    return np.asarray(output)


def train_epochs(
    model: QEEGMILModel,
    bags: list[PatientBag],
    device: torch.device,
    epochs: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    timemil_warmup_epochs: int,
) -> None:
    bags = device_bags(bags, device)
    labels = np.asarray([bag.label for bag in bags])
    positive_weight = torch.tensor(
        (labels == 0).sum() / max(1, (labels == 1).sum()),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(bags))
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + batch_size]:
                bag = bags[int(index)]
                target = torch.tensor(float(bag.label), device=device)
                losses.append(
                    criterion(
                        model(
                            tensor_clips(bag, device),
                            warmup=(
                                model.method == "timemil"
                                and epoch <= timemil_warmup_epochs
                            ),
                            patch_mask=model.method == "timemil",
                        ),
                        target,
                    )
                )
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()


def make_model(args: argparse.Namespace, method: str, device: torch.device) -> QEEGMILModel:
    return QEEGMILModel(
        input_dimension=122,
        method=method,
        embedding_dimension=args.embedding_dimension,
        encoder_dropout=args.encoder_dropout,
        millet_dropout=args.millet_dropout,
        timemil_dropout=args.timemil_dropout,
        millet_positional_encoding=not args.no_millet_positional_encoding,
        timemil_mask_ratio=args.timemil_mask_ratio,
    ).to(device)


def select_epoch(
    args: argparse.Namespace,
    method: str,
    training: list[PatientBag],
    validation: list[PatientBag],
    device: torch.device,
    seed: int,
) -> int:
    training = device_bags(training, device)
    validation = device_bags(validation, device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(args, method, device)
    labels = np.asarray([bag.label for bag in training])
    positive_weight = torch.tensor(
        (labels == 0).sum() / max(1, (labels == 1).sum()),
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(seed)
    best_epoch, best_auc, stale = 1, -np.inf, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = generator.permutation(len(training))
        for start in range(0, len(order), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start : start + args.batch_size]:
                bag = training[int(index)]
                target = torch.tensor(float(bag.label), device=device)
                losses.append(
                    criterion(
                        model(
                            tensor_clips(bag, device),
                            warmup=(
                                method == "timemil"
                                and epoch <= args.timemil_warmup_epochs
                            ),
                            patch_mask=method == "timemil",
                        ),
                        target,
                    )
                )
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch % 2 == 0 or epoch == args.max_epochs:
            validation_auc = auc_or_half(
                [bag.label for bag in validation], predict(model, validation, device)
            )
            if validation_auc > best_auc + 1e-5:
                best_auc, best_epoch, stale = validation_auc, epoch, 0
            else:
                stale += 2
            if stale >= args.patience:
                break
    return best_epoch


def fit_outer_model(
    args: argparse.Namespace,
    method: str,
    outer_training: list[PatientBag],
    inner_ids: set[int],
    validation_ids: set[int],
    device: torch.device,
    seed: int,
) -> tuple[QEEGMILModel, int, np.ndarray, np.ndarray]:
    inner_raw = [bag for bag in outer_training if bag.patient_id in inner_ids]
    validation_raw = [
        bag for bag in outer_training if bag.patient_id in validation_ids
    ]
    inner_center, inner_scale = fit_scaler(inner_raw)
    inner = [transform_bag(bag, inner_center, inner_scale) for bag in inner_raw]
    validation = [
        transform_bag(bag, inner_center, inner_scale) for bag in validation_raw
    ]
    epoch = select_epoch(args, method, inner, validation, device, seed)

    center, scale = fit_scaler(outer_training)
    training = [transform_bag(bag, center, scale) for bag in outer_training]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(args, method, device)
    train_epochs(
        model,
        training,
        device,
        epoch,
        seed,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.timemil_warmup_epochs,
    )
    return model, epoch, center, scale


def validate_folds(folds: list[int]) -> list[int]:
    if not folds:
        raise ValueError("At least one fold is required")
    if len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must be unique values selected from 0,1,2,3,4")
    return folds


def run(args: argparse.Namespace, workspace: Path, device: torch.device) -> None:
    index = cache_index(workspace)
    clinical = pd.read_csv(
        workspace / "data/final_short_merged.csv",
        dtype={"short_recording_id": str},
    )
    clinical = clinical[clinical.pre_post_treatment_label.eq("PRE")].copy()
    routine = pd.read_csv(
        workspace / "data/final_test.csv", dtype={"short_recording_id": str}
    )
    prediction_rows: list[dict] = []
    training_rows: list[dict] = []
    shuffle_rows: list[dict] = []

    for task in args.tasks:
        development = load_bags(clinical, index, task)
        evaluation = load_bags(routine, index, task)
        manifest = pd.read_csv(
            workspace / f"kfold_split/data/{task}/fold_manifest.csv"
        )
        fold_of = manifest.set_index("patient_id").fold.astype(int).to_dict()
        if set(development) != set(fold_of) or set(evaluation) != set(fold_of):
            raise RuntimeError(f"Patient/fold mismatch for {task}")

        for method in args.methods:
            for fold in args.folds:
                test_ids = {pid for pid, value in fold_of.items() if value == fold}
                validation_ids = {
                    pid for pid, value in fold_of.items() if value == (fold + 1) % 5
                }
                outer_ids = set(fold_of) - test_ids
                inner_ids = outer_ids - validation_ids
                outer = [development[pid] for pid in sorted(outer_ids)]
                raw_test = [evaluation[pid] for pid in sorted(test_ids)]
                seed_predictions, seed_clip_predictions, seed_epochs = [], [], []
                seed_models: list[QEEGMILModel] = []
                seed_test_bags: list[dict[int, PatientBag]] = []

                for seed_index in range(args.seeds):
                    seed = (
                        SEED_BASE
                        + 120000
                        + 10000 * METHODS.index(method)
                        + 1000 * seed_index
                        + 100 * fold
                    )
                    model, epoch, center, scale = fit_outer_model(
                        args,
                        method,
                        outer,
                        inner_ids,
                        validation_ids,
                        device,
                        seed,
                    )
                    test = [transform_bag(bag, center, scale) for bag in raw_test]
                    seed_predictions.append(predict(model, test, device))
                    seed_clip_predictions.append(
                        {
                            bag.patient_id: predict(
                                model, split_clip_bags(bag), device
                            )
                            for bag in test
                        }
                    )
                    seed_epochs.append(epoch)
                    seed_models.append(model)
                    seed_test_bags.append({bag.patient_id: bag for bag in test})
                    training_rows.append(
                        {
                            "task": task,
                            "method": method,
                            "fold": fold,
                            "seed_index": seed_index,
                            "seed": seed,
                            "selected_epoch": epoch,
                            "parameters": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                        }
                    )

                ensemble = np.mean(seed_predictions, axis=0)
                for patient_index, (bag, probability) in enumerate(
                    zip(raw_test, ensemble)
                ):
                    clip_probabilities = np.mean(
                        [
                            values[bag.patient_id]
                            for values in seed_clip_predictions
                        ],
                        axis=0,
                    )
                    prediction_rows.append(
                        {
                            "task": task,
                            "method": method,
                            "fold": fold,
                            "patient_id": bag.patient_id,
                            "label": bag.label,
                            "patient_probability": float(probability),
                            "routine_clip_1_probability": float(
                                clip_probabilities[0]
                            ),
                            "routine_clip_2_probability": float(
                                clip_probabilities[1]
                            ),
                            "selected_epoch_mean": float(np.mean(seed_epochs)),
                            "seed_probability_sd": (
                                float(
                                    np.std(
                                        [
                                            values[patient_index]
                                            for values in seed_predictions
                                        ],
                                        ddof=1,
                                    )
                                )
                                if len(seed_predictions) > 1
                                else 0.0
                            ),
                            **{
                                f"seed_{seed_index}_probability": float(
                                    values[patient_index]
                                )
                                for seed_index, values in enumerate(seed_predictions)
                            },
                        }
                    )
                    if method == "timemil":
                        shuffled = shuffled_patient_probabilities(
                            seed_models,
                            seed_test_bags,
                            bag.patient_id,
                            device,
                            list(TASK_COLUMN).index(task),
                            fold,
                            args.order_shuffle_draws,
                        )
                        shuffle_rows.append(
                            {
                                "task": task,
                                "method": method,
                                "fold": fold,
                                "patient_id": bag.patient_id,
                                "label": bag.label,
                                "original_probability": float(probability),
                                "mean_shuffled_probability": float(shuffled.mean()),
                                "original_minus_mean_shuffled": float(
                                    probability - shuffled.mean()
                                ),
                                "shuffle_probability_sd": float(
                                    shuffled.std(ddof=1)
                                ),
                                "shuffle_draws": args.order_shuffle_draws,
                                **{
                                    f"shuffle_{draw}_probability": float(value)
                                    for draw, value in enumerate(shuffled)
                                },
                            }
                        )
                print(f"{task} {method} fold {fold} complete", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(
        args.output_dir / "patient_predictions.csv", index=False
    )
    pd.DataFrame(training_rows).to_csv(
        args.output_dir / "training_runs.csv", index=False
    )
    pd.DataFrame(shuffle_rows).to_csv(
        args.output_dir / "timemil_order_shuffle.csv", index=False
    )
    metadata = {
        "unit_of_supervision": "patient",
        "development_view": "four Clinical Clips per patient",
        "evaluation_view": "two separately sampled Routine Clips per patient",
        "aggregation": (
            "pool windows within each clip, average clip logits with equal weight, "
            "then apply sigmoid"
        ),
        "input": "122-dimensional qEEG vector per non-overlapping 30-second window",
        "encoder": (
            f"two-layer window MLP with {args.embedding_dimension}-dimensional output"
        ),
        "outer_folds": args.folds,
        "seeds_per_fold": args.seeds,
        "sources": SOURCE_URLS,
        "millet": {
            "scope": "faithful implementation of the four pooling equations",
            "attention": "Linear-tanh-Linear-sigmoid with dimension 8; not softmax",
            "position": not args.no_millet_positional_encoding,
            "order_control": (
                "not repeated: the four pooling operators are permutation invariant; "
                "the optional sinusoidal position input can make the complete head "
                "order sensitive when enabled"
            ),
            "adaptations": [
                "qEEG window MLP replaces the papers' raw-series feature extractors",
                "each clinical clip is pooled separately before clip-equal averaging",
                "binary BCE and the benchmark nested-fold training protocol are used",
            ],
        },
        "timemil": {
            "scope": "faithful pooling adaptation, not exact end-to-end reproduction",
            "wavelet_bases": 3,
            "wavelet": "learned Mexican-hat scale and shift, kernel length 19",
            "attention": (
                "two pre-normalized Nystrom MHSA layers, 8 heads, "
                f"{args.embedding_dimension // 2} landmarks, six inverse iterations"
            ),
            "warmup_epochs": args.timemil_warmup_epochs,
            "training_mask_ratio": args.timemil_mask_ratio,
            "held_out_order_shuffle_draws": args.order_shuffle_draws,
            "adaptations": [
                "precomputed qEEG windows replace the raw multivariate series and InceptionTime backbone",
                "each clinical clip receives its own class token and temporal coordinates",
                "clip logits are averaged equally at patient level",
                "AdamW without the released repository's external Lookahead wrapper is used",
                "the two WPE blocks use independent parameters; this follows the paper architecture rather than an apparent public-code parameter alias",
                "a signed numerical floor prevents undefined wavelet scales at exactly zero",
            ],
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size_patients": args.batch_size,
            "maximum_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
        },
        "governance": (
            "patient_predictions.csv is local-only because response labels are not "
            "part of the governed public EEG v1.0 linkage"
        ),
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    paper = _paths.runtime()
    workspace = _paths.workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASK_COLUMN), default=list(TASK_COLUMN)
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-dimension", type=int, default=128)
    parser.add_argument("--encoder-dropout", type=float, default=0.20)
    parser.add_argument("--millet-dropout", type=float, default=0.10)
    parser.add_argument("--timemil-dropout", type=float, default=0.20)
    parser.add_argument("--timemil-mask-ratio", type=float, default=0.50)
    parser.add_argument("--timemil-warmup-epochs", type=int, default=10)
    parser.add_argument("--order-shuffle-draws", type=int, default=10)
    parser.add_argument("--no-millet-positional-encoding", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paper / "analysis/generated/sparse_evidence/closest_mil_baselines",
    )
    args = parser.parse_args()

    args.folds = validate_folds(args.folds)
    if (
        args.seeds < 1
        or args.max_epochs < 1
        or args.batch_size < 1
        or args.order_shuffle_draws < 10
    ):
        raise ValueError(
            "--seeds, --max-epochs, and --batch-size must be positive; "
            "--order-shuffle-draws must be at least 10"
        )
    if not 0 <= args.timemil_mask_ratio <= 1:
        raise ValueError("--timemil-mask-ratio must lie in [0, 1]")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Closest-MIL baselines require an explicit CUDA device")
    torch.cuda.set_device(device)
    run(args, workspace, device)


if __name__ == "__main__":
    main()
