#!/usr/bin/env python
"""Fine-tune REVE on one IESSEEG task and fold.

REVE is pre-trained at 200 Hz, IESSEEG's native rate, and takes electrode
positions explicitly, so the signal reaching the model is the released
data itself with only channel selection and per-window standardisation
applied.

REVE's weights are distributed under a Responsible Use Agreement whose
redistribution clause covers derivatives, so fine-tuned checkpoints stay
local and are not published.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from iesseeg.data.raw_dataset import InMemoryRandomDataset
from iesseeg.data.splits import encode_labels
from iesseeg.training import (
    CosineWithWarmup, EarlyStopping, clip_level_scores, layer_decay_param_groups,
    patient_level_split,
)
from reve_data import REVE_CHANNELS, REVE_SAMPLE_RATE, channel_positions, make_recording_transform
from reve_model import ReveClassifier


def make_loaders(args):
    df = pd.read_csv(args.train_csv)
    df["label"] = encode_labels(df, args.label_key)
    recording_col = [c for c in df.columns if "recording_id" in c][0]
    info = list(zip(df["patient_id"], df[recording_col].astype(str), df["label"]))

    train_info, val_info = patient_level_split(info, args.val_size, args.seed)
    print(f"[DATA] {len({p for p,_,_ in train_info})} train / "
          f"{len({p for p,_,_ in val_info})} val patients; "
          f"{len(train_info)} / {len(val_info)} clips")

    common = dict(
        data_dir=args.data_root, sample_rate=REVE_SAMPLE_RATE,
        window_sec=args.epoch_length, step_sec=args.epoch_length,
        n_channels=args.source_channels, scale_factor=1.0,
        window_transform="zscore",
        recording_transform=make_recording_transform(
            source_rate=args.source_sfreq, target_rate=REVE_SAMPLE_RATE
        ),
        verbose=False,
    )

    train_ds = InMemoryRandomDataset(
        info_list=train_info, mode="train",
        train_iterations=args.train_iterations, **common
    )
    val_common = dict(common, step_sec=args.val_step_sec or args.epoch_length)
    val_ds = InMemoryRandomDataset(info_list=val_info, mode="val", **val_common)

    return {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            collate_fn=InMemoryRandomDataset.collate_fn),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=InMemoryRandomDataset.collate_fn),
    }


def run_epoch(model, loader, positions, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)

    total_loss, n_seen = 0.0, 0
    probs, labels, recordings, patients = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["waveform"].to(device, non_blocking=True).float()
            y = batch["label"].to(device, non_blocking=True)
            pos = positions.unsqueeze(0).expand(x.shape[0], -1, -1)

            logits = model(x, pos)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * y.shape[0]
            n_seen += y.shape[0]
            probs.append(torch.softmax(logits.detach().float(), dim=-1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
            recordings += list(batch["recording_id"])
            patients += list(batch["patient_id"])

    return dict(
        loss=total_loss / max(n_seen, 1),
        prob=np.concatenate(probs) if probs else np.array([]),
        label=np.concatenate(labels) if labels else np.array([]),
        recording_id=recordings, patient_id=patients,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained_repo", default="brain-bzh/reve-base")
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--save_preds_dir", default=None)
    parser.add_argument("--label_key", required=True)
    parser.add_argument("--epoch_length", type=int, default=30, help="Window seconds.")
    parser.add_argument("--source_sfreq", type=int, default=200)
    parser.add_argument("--source_channels", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train_iterations", type=int, default=2500)
    parser.add_argument("--val_step_sec", type=int, default=None)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] device={device} task={args.label_key}")

    loaders = make_loaders(args)
    model = ReveClassifier(args.pretrained_repo, n_classes=2).to(device)
    print(f"[MODEL] REVE encoder loaded "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    positions = channel_positions(REVE_CHANNELS).to(device)

    criterion = nn.CrossEntropyLoss()
    n_layers = model.n_encoder_layers + 2
    optimizer = torch.optim.AdamW(layer_decay_param_groups(
        model, lambda n: model.layer_of(n, n_layers), n_layers,
        args.lr, args.weight_decay, args.layer_decay,
    ))

    steps_per_epoch = max(len(loaders["train"]), 1)
    scheduler = CosineWithWarmup(
        optimizer, steps_per_epoch * args.epochs,
        steps_per_epoch * args.warmup_epochs, args.min_lr,
    )
    stopper = EarlyStopping(patience=args.patience)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)

    for epoch in range(args.epochs):
        train_result = run_epoch(model, loaders["train"], positions, criterion,
                                 device, optimizer, scheduler)
        val_result = run_epoch(model, loaders["val"], positions, criterion, device)

        clip = clip_level_scores(val_result["patient_id"], val_result["recording_id"],
                                 val_result["prob"], val_result["label"])
        multiclass = clip["known_label"].nunique() > 1
        val_acc = balanced_accuracy_score(clip["known_label"], clip["pred_label"]) if multiclass else float("nan")
        val_auc = roc_auc_score(clip["known_label"], clip["pred_prob"]) if multiclass else float("nan")

        print(f"[EPOCH {epoch:3d}] train_loss={train_result['loss']:.4f} "
              f"val_loss={val_result['loss']:.4f} val_clip_acc={val_acc:.3f} "
              f"val_clip_auc={val_auc:.3f} lr={optimizer.param_groups[-1]['lr']:.2e}")

        if stopper.update(val_result["loss"], epoch):
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": stopper.best, "args": vars(args)}, args.model_out)
            if args.save_preds_dir:
                os.makedirs(args.save_preds_dir, exist_ok=True)
                clip.to_csv(os.path.join(args.save_preds_dir, "best_val_preds.csv"), index=False)
        elif stopper.should_stop:
            print(f"[STOP] no val improvement for {args.patience} epochs")
            break

    print(f"[DONE] best epoch {stopper.best_epoch} val_loss={stopper.best:.4f} "
          f"-> {args.model_out}")


if __name__ == "__main__":
    main()
