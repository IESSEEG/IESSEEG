#!/usr/bin/env python
"""Run the frozen-encoder, window-to-recording response screen.

One PRE or POST Clinical Clip is one prediction example. Native foundation-
model windows are instances, and one recording-level response loss is applied
after aggregation. Patients, not recordings, define all train, validation, and
test folds.

Row-level outputs contain patient IDs and restricted response labels. They
must remain under the ignored local output directory.
"""

from __future__ import annotations
import legacy_paths as _paths

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_closest_mil_baselines import MILLETPooling, TimeMILPooling  # noqa: E402


ENCODERS = (
    "biot", "labram", "cbramod", "eegpt",
    "luna", "reve", "codebrain", "csbrain",
)
TASK_COLUMNS = {
    "immediate": "immediate_responder",
    "sustained": "meaningful_responder",
}
METHODS = (
    "mean", "max", "top10", "deepsets", "attention",
    "temporal_attention", "multiscale", "millet_instance",
    "millet_attention", "millet_additive", "millet_conjunctive",
    "psmil", "timemil",
)
SCALES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
SEED = 20260909


@dataclass
class RecordingBag:
    recording_id: str
    patient_id: int
    label: int
    windows: np.ndarray | torch.Tensor


def derive_training_seed(
    model: str,
    task: str,
    method: str,
    fold: int,
    base_seed: int = SEED,
) -> int:
    """Return the canonical seed independent of CLI subset ordering."""
    return (
        base_seed
        + 100000 * ENCODERS.index(model)
        + 10000 * tuple(TASK_COLUMNS).index(task)
        + 100 * METHODS.index(method)
        + fold
    )


class RecordingHead(torch.nn.Module):
    """Map one sequence of frozen window embeddings to one response logit."""

    def __init__(self, input_dim: int, method: str, dropout: float = 0.20):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        hidden = 64
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
        )
        self.local_head = torch.nn.Linear(hidden, 1)
        self.temporal = torch.nn.Conv1d(hidden, hidden, 5, padding=2)
        self.attention_v = torch.nn.Linear(hidden, 32)
        self.attention_u = torch.nn.Linear(hidden, 32)
        self.attention_w = torch.nn.Linear(32, 1, bias=False)
        self.scale_logits = torch.nn.Parameter(torch.zeros(len(SCALES)))
        self.bias = torch.nn.Parameter(torch.zeros(()))

        self.deepset_phi = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        )
        self.deepset_rho = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

        if method.startswith("millet_"):
            self.millet = MILLETPooling(
                hidden,
                method.removeprefix("millet_"),
                attention_dimension=8,
                dropout=0.10,
                positional_encoding=True,
            )
        if method == "timemil":
            self.timemil = TimeMILPooling(hidden, dropout=dropout)

        # PSMIL probability-space attention adaptation. The prototype bank and
        # its momentum update follow the public PSMIL implementation; the
        # external frozen EEG representation replaces its supplied features.
        self.ps_classifier = torch.nn.Parameter(torch.empty(hidden, 2))
        torch.nn.init.kaiming_uniform_(self.ps_classifier)
        self.ps_attention = torch.nn.Linear(2, 1)
        self.register_buffer("ps_prototypes", torch.zeros(hidden, 2))
        self.register_buffer("ps_initialized", torch.tensor(False))

    def _attention(self, h: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.attention_v(h))
        gate = torch.sigmoid(self.attention_u(h))
        return torch.softmax(self.attention_w(value * gate).squeeze(-1), dim=0)

    @staticmethod
    def _mask_deciles(h: torch.Tensor, ratio: float = 0.50) -> torch.Tensor:
        if ratio <= 0 or len(h) < 10:
            return h
        result = h.clone()
        n_deciles = 10
        interval = len(h) // n_deciles
        selected = int(ratio * n_deciles)
        for index in torch.randperm(n_deciles, device=h.device)[:selected]:
            start = int(index) * interval
            result[start:start + interval] = torch.randn(
                (), device=h.device, dtype=h.dtype
            )
        return result

    def _psmil(self, h: torch.Tensor, label: int | None, update: bool) -> torch.Tensor:
        if not bool(self.ps_initialized):
            with torch.no_grad():
                initial = h.detach().mean(dim=0)
                self.ps_prototypes.copy_(initial[:, None].repeat(1, 2))
                self.ps_initialized.fill_(True)

        # Clone because the momentum bank is updated for the next recording
        # before this recording's loss is backpropagated.
        prototypes = self.ps_prototypes.detach().clone()
        prototype_probability = torch.softmax(h @ prototypes, dim=1)
        weight = torch.softmax(
            self.ps_attention(prototype_probability).squeeze(-1), dim=0
        )
        pooled = torch.sum(weight[:, None] * h, dim=0, keepdim=True)
        class_logits = pooled @ self.ps_classifier

        if update:
            if label is None:
                raise ValueError("PSMIL prototype update requires a training label")
            with torch.no_grad():
                selected = torch.argmax(prototype_probability[:, label])
                critical = h[selected].detach()
                updated = torch.nn.functional.normalize(
                    0.99 * self.ps_prototypes[:, label] + 0.01 * critical,
                    dim=0,
                )
                self.ps_prototypes[:, label].copy_(updated)
        return (class_logits[0, 1] - class_logits[0, 0]).reshape(())

    def forward(
        self,
        windows: torch.Tensor,
        *,
        epoch: int = 0,
        label: int | None = None,
        update_psmil: bool = False,
    ) -> torch.Tensor:
        h = self.adapter(windows)

        if self.method == "deepsets":
            return self.deepset_rho(self.deepset_phi(h).mean(dim=0)).reshape(())
        if self.method.startswith("millet_"):
            return self.millet(h.unsqueeze(0)).reshape(())
        if self.method == "timemil":
            if self.training:
                h = self._mask_deciles(h)
            return self.timemil(
                h.unsqueeze(0), warmup=self.training and epoch <= 10
            ).reshape(())
        if self.method == "psmil":
            return self._psmil(h, label, update_psmil)

        if self.method == "temporal_attention":
            residual = h
            h = torch.nn.functional.gelu(
                self.temporal(h.transpose(0, 1).unsqueeze(0))
                .squeeze(0).transpose(0, 1)
            ) + residual

        local = self.local_head(h).squeeze(-1)
        if self.method == "mean":
            logit = local.mean()
        elif self.method == "max":
            logit = local.max()
        elif self.method == "top10":
            k = max(1, math.ceil(0.10 * len(local)))
            logit = torch.topk(local, k).values.mean()
        elif self.method in {"attention", "temporal_attention"}:
            logit = torch.sum(self._attention(h) * local)
        elif self.method == "multiscale":
            order = torch.argsort(local, descending=True)
            pooled = []
            for fraction in SCALES:
                k = max(1, math.ceil(fraction * len(local)))
                pooled.append(local[order[:k]].mean())
            logit = torch.dot(
                torch.stack(pooled), torch.softmax(self.scale_logits, dim=0)
            )
        else:
            raise KeyError(self.method)
        return (logit + self.bias).reshape(())


def label_value(value: str) -> int:
    if value == "Responder":
        return 1
    if value == "Non-responder":
        return 0
    raise ValueError(value)


def load_recordings(
    metadata: pd.DataFrame,
    representation_dir: Path,
    task: str,
) -> list[RecordingBag]:
    column = TASK_COLUMNS[task]
    rows: list[RecordingBag] = []
    for row in metadata.itertuples(index=False):
        value = str(getattr(row, column))
        if value == "UNKNOWN":
            continue
        recording_id = str(row.short_recording_id).removesuffix(".0")
        path = representation_dir / f"{recording_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        windows = np.load(path).astype(np.float32)
        if windows.ndim != 2 or not len(windows) or not np.isfinite(windows).all():
            raise RuntimeError(f"Invalid representation {path}: {windows.shape}")
        rows.append(RecordingBag(
            recording_id=recording_id,
            patient_id=int(row.patient_id),
            label=label_value(value),
            windows=windows,
        ))
    if len(rows) != 200 or len({row.patient_id for row in rows}) != 50:
        raise RuntimeError(
            f"Expected 200 recordings from 50 response patients, got "
            f"{len(rows)} recordings from {len({row.patient_id for row in rows})} patients"
        )
    return rows


def fit_scaler(recordings: list[RecordingBag]) -> tuple[np.ndarray, np.ndarray]:
    # Equal weight per recording and equal weight per window within recording.
    means = np.stack([
        np.asarray(row.windows, dtype=np.float64).mean(axis=0)
        for row in recordings
    ])
    seconds = np.stack([
        np.square(np.asarray(row.windows, dtype=np.float64)).mean(axis=0)
        for row in recordings
    ])
    center = means.mean(axis=0)
    variance = seconds.mean(axis=0) - np.square(center)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform(
    recordings: list[RecordingBag], center: np.ndarray, scale: np.ndarray
) -> list[RecordingBag]:
    return [
        RecordingBag(
            row.recording_id,
            row.patient_id,
            row.label,
            np.clip((np.asarray(row.windows) - center) / scale, -10, 10)
            .astype(np.float32),
        )
        for row in recordings
    ]


def to_device(
    recordings: list[RecordingBag], device: torch.device
) -> list[RecordingBag]:
    return [
        RecordingBag(
            row.recording_id,
            row.patient_id,
            row.label,
            torch.as_tensor(row.windows, device=device),
        )
        for row in recordings
    ]


def metric(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def predict(
    model: RecordingHead,
    recordings: list[RecordingBag],
) -> np.ndarray:
    model.eval()
    result = []
    with torch.inference_mode():
        for row in recordings:
            logit = model(row.windows)
            result.append(torch.sigmoid(logit).item())
    return np.asarray(result)


def train_epoch(
    model: RecordingHead,
    recordings: list[RecordingBag],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    rng: np.random.Generator,
    epoch: int,
    batch_size: int,
) -> None:
    model.train()
    order = rng.permutation(len(recordings))
    for start in range(0, len(order), batch_size):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in order[start:start + batch_size]:
            row = recordings[int(index)]
            logit = model(
                row.windows,
                epoch=epoch,
                label=row.label,
                update_psmil=model.method == "psmil",
            )
            target = torch.tensor(float(row.label), device=logit.device)
            losses.append(criterion(logit, target))
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def make_training_objects(
    model: RecordingHead,
    recordings: list[RecordingBag],
    device: torch.device,
) -> tuple[torch.optim.Optimizer, torch.nn.Module]:
    labels_by_patient = {
        row.patient_id: row.label for row in recordings
    }
    labels = np.asarray(list(labels_by_patient.values()))
    pos_weight = torch.tensor(
        (labels == 0).sum() / max(1, (labels == 1).sum()),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return optimizer, criterion


def select_epoch(
    method: str,
    input_dim: int,
    training: list[RecordingBag],
    validation: list[RecordingBag],
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[int, float]:
    torch.manual_seed(seed)
    model = RecordingHead(input_dim, method).to(device)
    optimizer, criterion = make_training_objects(model, training, device)
    rng = np.random.default_rng(seed)
    best_epoch, best_auc, stale = 1, -np.inf, 0
    labels = np.asarray([row.label for row in validation])
    for epoch in range(1, max_epochs + 1):
        train_epoch(
            model, training, optimizer, criterion, rng, epoch, batch_size
        )
        if epoch % 2:
            continue
        scores = predict(model, validation)
        auc, _ = metric(labels, scores)
        if auc > best_auc + 1e-5:
            best_epoch, best_auc, stale = epoch, auc, 0
        else:
            stale += 2
        if stale >= patience:
            break
    return best_epoch, best_auc


def fit_fixed_epochs(
    method: str,
    input_dim: int,
    training: list[RecordingBag],
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
) -> RecordingHead:
    torch.manual_seed(seed)
    model = RecordingHead(input_dim, method).to(device)
    optimizer, criterion = make_training_objects(model, training, device)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        train_epoch(
            model, training, optimizer, criterion, rng, epoch, batch_size
        )
    return model


def main() -> None:
    paper = _paths.runtime()
    project = _paths.workspace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ENCODERS, required=True)
    parser.add_argument("--condition", choices=("PRE", "POST"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_COLUMNS), default=list(TASK_COLUMNS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--representation-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paper / "analysis/generated/clip_response_screen",
    )
    args = parser.parse_args()

    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be between 0 and 4")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Response training requires explicit CUDA")
    torch.cuda.set_device(device)

    if args.representation_root is None:
        root_name = (
            "long_frozen_representations"
            if args.condition == "PRE"
            else "post_frozen_representations"
        )
        representation_root = paper / "analysis/generated" / root_name
    else:
        representation_root = args.representation_root
    representation_dir = representation_root / args.model

    metadata = pd.read_csv(
        project / "data/final_short_merged.csv",
        dtype={"short_recording_id": str},
    )
    metadata = metadata[
        metadata.pre_post_treatment_label.eq(args.condition)
        & metadata.case_control_label.eq("CASE")
    ].copy()
    if len(metadata) != 200 or metadata.patient_id.nunique() != 50:
        raise RuntimeError(
            f"Expected 200 {args.condition} recordings from 50 patients"
        )

    manifest = pd.read_csv(representation_dir / "manifest.csv")
    if len(manifest) != 200 and not (
        args.condition == "PRE" and len(manifest) == 600
    ):
        raise RuntimeError(f"Unexpected representation manifest: {len(manifest)}")
    dimensions = manifest.embedding_dim.unique()
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent embedding dimensions: {dimensions}")
    input_dim = int(dimensions[0])

    output_dir = args.output_root / args.condition.lower() / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "recording_predictions.csv"
    run_path = output_dir / "training_runs.csv"
    existing_predictions = (
        pd.read_csv(prediction_path, dtype={"recording_id": str})
        if prediction_path.exists() else pd.DataFrame()
    )
    existing_runs = pd.read_csv(run_path) if run_path.exists() else pd.DataFrame()
    prediction_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for task in args.tasks:
        recordings = load_recordings(metadata, representation_dir, task)
        manifest_path = project / f"kfold_split/data/{TASK_COLUMNS[task]}/fold_manifest.csv"
        folds = pd.read_csv(manifest_path)
        fold_of = folds.set_index("patient_id").fold.astype(int).to_dict()
        if set(fold_of) != {row.patient_id for row in recordings}:
            raise RuntimeError(f"Patient/fold mismatch for {task}")

        for method in args.methods:
            for fold in args.folds:
                completed = False
                if not existing_runs.empty:
                    completed = bool((
                        existing_runs.task.eq(task)
                        & existing_runs.method.eq(method)
                        & existing_runs.fold.eq(fold)
                        & existing_runs.seed.eq(args.seed)
                    ).any())
                if completed:
                    print(
                        f"[{args.condition} {args.model}] {task} {method} fold {fold} already complete",
                        flush=True,
                    )
                    continue

                test_ids = {pid for pid, value in fold_of.items() if value == fold}
                validation_ids = {
                    pid for pid, value in fold_of.items() if value == (fold + 1) % 5
                }
                outer_ids = set(fold_of) - test_ids
                inner_ids = outer_ids - validation_ids
                inner_raw = [row for row in recordings if row.patient_id in inner_ids]
                validation_raw = [row for row in recordings if row.patient_id in validation_ids]
                outer_raw = [row for row in recordings if row.patient_id in outer_ids]
                test_raw = [row for row in recordings if row.patient_id in test_ids]

                inner_center, inner_scale = fit_scaler(inner_raw)
                inner = to_device(transform(inner_raw, inner_center, inner_scale), device)
                validation = to_device(
                    transform(validation_raw, inner_center, inner_scale), device
                )
                seed = derive_training_seed(
                    args.model, task, method, fold, args.seed
                )
                best_epoch, validation_auc = select_epoch(
                    method, input_dim, inner, validation, device, seed,
                    args.max_epochs, args.patience, args.batch_size,
                )
                del inner, validation

                outer_center, outer_scale = fit_scaler(outer_raw)
                outer = to_device(transform(outer_raw, outer_center, outer_scale), device)
                test = to_device(transform(test_raw, outer_center, outer_scale), device)
                final = fit_fixed_epochs(
                    method, input_dim, outer, device, seed,
                    best_epoch, args.batch_size,
                )
                scores = predict(final, test)
                labels = np.asarray([row.label for row in test])
                test_auc, test_auprc = metric(labels, scores)
                for row, score in zip(test, scores):
                    prediction_rows.append({
                        "condition": args.condition,
                        "model": args.model,
                        "task": task,
                        "method": method,
                        "fold": fold,
                        "recording_id": row.recording_id,
                        "patient_id": row.patient_id,
                        "label": row.label,
                        "probability": float(score),
                        "n_windows": len(row.windows),
                        "selected_epoch": best_epoch,
                        "seed": args.seed,
                        "training_seed": seed,
                    })
                run_rows.append({
                    "condition": args.condition,
                    "model": args.model,
                    "task": task,
                    "method": method,
                    "fold": fold,
                    "seed": args.seed,
                    "training_seed": seed,
                    "selected_epoch": best_epoch,
                    "validation_clip_auroc": validation_auc,
                    "test_clip_auroc": test_auc,
                    "test_clip_auprc": test_auprc,
                    "n_inner_patients": len(inner_ids),
                    "n_validation_patients": len(validation_ids),
                    "n_outer_patients": len(outer_ids),
                    "n_test_patients": len(test_ids),
                    "n_test_recordings": len(test),
                })
                del outer, test, final
                torch.cuda.empty_cache()

                existing_predictions = pd.concat(
                    [existing_predictions, pd.DataFrame(prediction_rows)],
                    ignore_index=True,
                )
                existing_runs = pd.concat(
                    [existing_runs, pd.DataFrame(run_rows)], ignore_index=True
                )
                existing_predictions.to_csv(prediction_path, index=False)
                existing_runs.to_csv(run_path, index=False)
                prediction_rows.clear()
                run_rows.clear()
                print(
                    f"[{args.condition} {args.model}] {task} {method} fold {fold} "
                    f"complete: AUROC={test_auc:.3f}, epoch={best_epoch}",
                    flush=True,
                )

    metadata_out = {
        "condition": args.condition,
        "model": args.model,
        "input": "native-window frozen foundation representations",
        "prediction_unit": "one Clinical Clip recording",
        "split_unit": "patient",
        "foundation_encoder_trainable": False,
        "methods": list(args.methods),
        "tasks": list(args.tasks),
        "folds": list(args.folds),
        "seed": args.seed,
        "selection": "patient-disjoint validation fold; clip AUROC; early stopping",
        "primary_metric": "clip AUROC",
        "psmil_scope": "probability-space attention with the public prototype-bank update; no probability-alignment loss",
        "row_level_output": "local only",
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata_out, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
