#!/usr/bin/env python
"""Fine-tune LUNA on one IESSEEG task and fold.

Follows the fine-tuning recipe from the upstream LUNA repository: AdamW
with layer-wise learning-rate decay, a cosine schedule with warmup,
channel-wise input normalisation, and early stopping on validation loss.

The pre-trained weights are the released LUNA checkpoint; only the
classification head is new. Note that LUNA's weights are CC BY-ND 4.0,
which permits fine-tuning for internal use but not redistribution of the
resulting weights, so fine-tuned checkpoints stay local.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
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
from luna_data import (
    LUNA_SAMPLE_RATE, TCP_MONTAGE, channel_locations, make_recording_transform,
)
from models.LUNA import LUNA

N_CHANNELS = len(TCP_MONTAGE)


def build_model(checkpoint, num_classes=2, device="cuda:0"):
    """Instantiate LUNA-Base and load the pre-trained backbone."""
    from safetensors.torch import load_file

    model = LUNA(patch_size=40, embed_dim=64, num_heads=2, depth=8,
                 num_queries=4, drop_path=0.1, num_classes=num_classes)

    state = load_file(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The released checkpoint carries the pre-training reconstruction
    # head; the classification head is new. Anything else missing would
    # mean the architecture and checkpoint disagree.
    backbone_missing = [k for k in missing if not k.startswith("classifier.")]
    if backbone_missing:
        raise RuntimeError(f"Checkpoint is missing backbone weights: {backbone_missing[:8]}")
    print(f"[MODEL] loaded {len(state) - len(unexpected)}/{len(state)} pretrained tensors; "
          f"{len(missing)} new head tensors")
    return model.to(device)


def param_groups_with_layer_decay(model, base_lr, weight_decay, decay=0.75):
    """Layer-wise learning-rate decay over LUNA's stem, blocks and head."""
    n_layers = len(model.blocks) + 2

    def layer_of(name):
        if name.startswith(("patch_embed", "freq_embed", "channel_location_embedder",
                            "cross_attn", "mask_token")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return n_layers - 1  # norm and classifier

    return layer_decay_param_groups(
        model, layer_of, n_layers, base_lr, weight_decay, decay
    )


def make_loaders(args, device):
    df = pd.read_csv(args.train_csv)
    df["label"] = encode_labels(df, args.label_key)

    recording_col = [c for c in df.columns if "recording_id" in c][0]
    info = list(zip(df["patient_id"], df[recording_col].astype(str), df["label"]))

    train_info, val_info = patient_level_split(info, args.val_size, args.seed)
    print(f"[DATA] {len({p for p,_,_ in train_info})} train / "
          f"{len({p for p,_,_ in val_info})} val patients; "
          f"{len(train_info)} / {len(val_info)} clips")

    common = dict(
        data_dir=args.data_root, sample_rate=LUNA_SAMPLE_RATE,
        window_sec=args.epoch_length, step_sec=args.epoch_length,
        n_channels=args.source_channels, scale_factor=1.0,
        window_transform="zscore",
        recording_transform=make_recording_transform(
            source_rate=args.source_sfreq, target_rate=LUNA_SAMPLE_RATE
        ),
        verbose=False,
    )

    train_ds = InMemoryRandomDataset(
        info_list=train_info, mode="train",
        train_iterations=args.train_iterations, **common
    )
    # Validation exists only to pick the epoch to keep, so it steps
    # through each clip coarsely instead of tiling it exhaustively. That
    # still leaves tens of windows per clip to average over while cutting
    # most of the per-epoch cost. Test-time inference is unaffected: it
    # always tiles with a full non-overlapping stride.
    val_common = dict(common, step_sec=args.val_step_sec or args.epoch_length)
    val_ds = InMemoryRandomDataset(info_list=val_info, mode="val", **val_common)

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            collate_fn=InMemoryRandomDataset.collate_fn),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=InMemoryRandomDataset.collate_fn),
    }
    return loaders, val_ds


def run_epoch(model, loader, locations, criterion, device, optimizer=None, scheduler=None):
    """One pass; trains when an optimizer is given, evaluates otherwise."""
    training = optimizer is not None
    model.train(training)

    total_loss, n_seen = 0.0, 0
    probs, labels, recordings, patients = [], [], [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["waveform"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            coords = locations.unsqueeze(0).expand(x.shape[0], -1, -1)

            # Upstream passes an all-false mask during fine-tuning, which
            # leaves the signal intact but jitters the electrode
            # coordinates slightly -- a mild augmentation. At evaluation
            # we pass None instead, which skips the jitter and makes
            # clip-level scores deterministic.
            mask = torch.zeros(x.shape, dtype=torch.bool, device=device) if training else None
            logits, _ = model(x, mask, coords)
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

    result = dict(
        loss=total_loss / max(n_seen, 1),
        prob=np.concatenate(probs) if probs else np.array([]),
        label=np.concatenate(labels) if labels else np.array([]),
        recording_id=recordings,
        patient_id=patients,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--pretrained", required=True, help="LUNA_base.safetensors")
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--save_preds_dir", default=None)
    parser.add_argument("--label_key", required=True)
    parser.add_argument("--epoch_length", type=int, default=30, help="Window seconds.")
    parser.add_argument("--source_sfreq", type=int, default=200)
    parser.add_argument("--source_channels", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=2.5e-6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train_iterations", type=int, default=5000)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--val_step_sec", type=int, default=None,
                        help="Stride between validation windows; defaults to the "
                             "window length (exhaustive tiling).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", default="0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"[SETUP] device={device} task={args.label_key}")

    loaders, _ = make_loaders(args, device)
    model = build_model(args.pretrained, num_classes=2, device=device)
    locations = torch.from_numpy(channel_locations()).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        param_groups_with_layer_decay(model, args.lr, args.weight_decay, args.layer_decay)
    )

    steps_per_epoch = max(len(loaders["train"]), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_at(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    state = {"step": 0}

    class CosineWithWarmup:
        def step(self):
            state["step"] += 1
            scale = lr_at(state["step"])
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = max(base * scale, args.min_lr)

    scheduler = CosineWithWarmup()

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    best_loss, best_epoch, epochs_without_improvement = float("inf"), -1, 0

    for epoch in range(args.epochs):
        train_result = run_epoch(model, loaders["train"], locations, criterion,
                                 device, optimizer, scheduler)
        val_result = run_epoch(model, loaders["val"], locations, criterion, device)

        clip = clip_level_scores(val_result["patient_id"], val_result["recording_id"],
                                 val_result["prob"], val_result["label"])
        val_acc = (balanced_accuracy_score(clip["known_label"], clip["pred_label"])
                   if clip["known_label"].nunique() > 1 else float("nan"))
        val_auc = (roc_auc_score(clip["known_label"], clip["pred_prob"])
                   if clip["known_label"].nunique() > 1 else float("nan"))

        print(f"[EPOCH {epoch:3d}] train_loss={train_result['loss']:.4f} "
              f"val_loss={val_result['loss']:.4f} val_clip_acc={val_acc:.3f} "
              f"val_clip_auc={val_auc:.3f} lr={optimizer.param_groups[-1]['lr']:.2e}")

        if val_result["loss"] < best_loss - 1e-5:
            best_loss, best_epoch, epochs_without_improvement = val_result["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": best_loss, "args": vars(args)}, args.model_out)
            if args.save_preds_dir:
                os.makedirs(args.save_preds_dir, exist_ok=True)
                clip.to_csv(os.path.join(args.save_preds_dir, "best_val_preds.csv"), index=False)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[STOP] no val improvement for {args.patience} epochs")
                break

    print(f"[DONE] best epoch {best_epoch} val_loss={best_loss:.4f} -> {args.model_out}")


if __name__ == "__main__":
    main()
