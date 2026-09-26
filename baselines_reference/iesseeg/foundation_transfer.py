"""Shared train and inference loops for patch-based EEG foundation models.

CodeBrain and CSBrain both consume 200 Hz EEG arranged as one-second
patches, but expose different backbone implementations.  Their adapters
provide the model; this module keeps the patient split, sampling, checkpoint
selection, and clip aggregation identical.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from .data.raw_dataset import InMemoryRandomDataset
from .data.splits import encode_labels
from .training import CosineWithWarmup, EarlyStopping, clip_level_scores, patient_level_split


def require_cuda(cuda_index):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for foundation-model training and inference")
    device = torch.device(f"cuda:{cuda_index}")
    torch.cuda.set_device(device)
    return device


def read_info(csv_path, label_key):
    df = pd.read_csv(csv_path)
    df["label"] = encode_labels(df, label_key)
    recording_col = next(c for c in df.columns if "recording_id" in c)
    return list(zip(df["patient_id"], df[recording_col].astype(str), df["label"]))


def make_loaders(args):
    info = read_info(args.train_csv, args.label_key)
    train_info, val_info = patient_level_split(info, args.val_size, args.seed)
    print(
        f"[DATA] {len({p for p, _, _ in train_info})} train / "
        f"{len({p for p, _, _ in val_info})} val patients; "
        f"{len(train_info)} / {len(val_info)} clips"
    )

    common = dict(
        data_dir=args.data_root,
        sample_rate=200,
        window_sec=args.epoch_length,
        n_channels=19,
        scale_factor=0.01,
        window_transform="patch",
        verbose=False,
    )
    train_ds = InMemoryRandomDataset(
        info_list=train_info,
        mode="train",
        step_sec=args.epoch_length,
        train_iterations=args.train_iterations,
        **common,
    )
    val_ds = InMemoryRandomDataset(
        info_list=val_info,
        mode="val",
        step_sec=args.val_step_sec,
        **common,
    )
    return {
        "train": DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=InMemoryRandomDataset.collate_fn,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=InMemoryRandomDataset.collate_fn,
        ),
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_seen = 0.0, 0
    probs, labels, recordings, patients = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["waveform"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * y.shape[0]
            n_seen += y.shape[0]
            probs.append(torch.softmax(logits.detach().float(), dim=-1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
            recordings.extend(batch["recording_id"])
            patients.extend(batch["patient_id"])

    return {
        "loss": total_loss / max(n_seen, 1),
        "prob": np.concatenate(probs) if probs else np.array([]),
        "label": np.concatenate(labels) if labels else np.array([]),
        "recording_id": recordings,
        "patient_id": patients,
    }


def train_model(args, build_model):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = require_cuda(args.cuda)
    print(f"[SETUP] device={device} task={args.label_key}")

    loaders = make_loaders(args)
    model = build_model(args.pretrained, dropout=args.dropout).to(device)
    print(f"[MODEL] {args.model_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(len(loaders["train"]), 1)
    scheduler = CosineWithWarmup(
        optimizer,
        total_steps=steps_per_epoch * args.epochs,
        warmup_steps=0,
        min_lr=args.min_lr,
    )
    stopper = EarlyStopping(patience=args.patience)
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)

    for epoch in range(args.epochs):
        train_result = run_epoch(model, loaders["train"], criterion, device, optimizer, scheduler)
        val_result = run_epoch(model, loaders["val"], criterion, device)
        clip = clip_level_scores(
            val_result["patient_id"],
            val_result["recording_id"],
            val_result["prob"],
            val_result["label"],
        )
        multiclass = clip["known_label"].nunique() > 1
        val_acc = (
            balanced_accuracy_score(clip["known_label"], clip["pred_label"])
            if multiclass
            else float("nan")
        )
        val_auc = (
            roc_auc_score(clip["known_label"], clip["pred_prob"])
            if multiclass
            else float("nan")
        )
        print(
            f"[EPOCH {epoch:3d}] train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} val_clip_acc={val_acc:.3f} "
            f"val_clip_auc={val_auc:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if stopper.update(val_result["loss"], epoch):
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": stopper.best,
                    "args": vars(args),
                },
                args.model_out,
            )
            if args.save_preds_dir:
                os.makedirs(args.save_preds_dir, exist_ok=True)
                clip.to_csv(
                    os.path.join(args.save_preds_dir, "best_val_preds.csv"), index=False
                )
        elif stopper.should_stop:
            print(f"[STOP] no validation-loss improvement for {args.patience} epochs")
            break

    print(
        f"[DONE] best epoch {stopper.best_epoch} val_loss={stopper.best:.4f} "
        f"-> {args.model_out}"
    )


def infer_model(args, build_model):
    device = require_cuda(args.cuda)
    print(f"[SETUP] device={device} task={args.label_key}")
    info = read_info(args.inference_csv, args.label_key)
    print(f"[DATA] {len(info)} clips from {len({p for p, _, _ in info})} patients")

    dataset = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=info,
        mode="test",
        sample_rate=200,
        window_sec=args.epoch_length,
        step_sec=args.epoch_length,
        n_channels=19,
        scale_factor=0.01,
        window_transform="patch",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=InMemoryRandomDataset.collate_fn,
    )

    checkpoint = torch.load(args.model_file, map_location="cpu", weights_only=False)
    model = build_model(args.pretrained, dropout=args.dropout)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["waveform"].to(device, non_blocking=True).float()
            probs = torch.softmax(model(x).float(), dim=-1)[:, 1].cpu().numpy()
            for i, prob in enumerate(probs):
                rows.append(
                    {
                        "patient_id": batch["patient_id"][i],
                        "recording_id": batch["recording_id"][i],
                        "start_ind": batch["start_ind"][i],
                        "end_ind": batch["end_ind"][i],
                        "pred_prob": float(prob),
                        "pred_label": int(prob >= 0.5),
                        "known_label": int(batch["label"][i]),
                    }
                )

    window_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    window_path = args.out_csv.replace(".csv", "_window.csv")
    window_df.to_csv(window_path, index=False)

    clip_df = window_df.groupby("recording_id").agg(
        patient_id=("patient_id", "first"),
        pred_prob=("pred_prob", "mean"),
        known_label=("known_label", "first"),
    ).reset_index()
    clip_df["pred_label"] = (clip_df["pred_prob"] >= 0.5).astype(int)
    clip_df = clip_df[
        ["patient_id", "recording_id", "pred_prob", "pred_label", "known_label"]
    ]
    clip_df.to_csv(args.out_csv, index=False)
    print(
        f"[DONE] {len(window_df)} windows / {len(clip_df)} clips -> {args.out_csv}"
    )
