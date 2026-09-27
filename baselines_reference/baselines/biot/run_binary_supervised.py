import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE
from model_zoo import build_model
from inmem_raw_dataset import InMemoryRandomDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv
from data_utils.classification_metrics import binary_metrics_fn

def setup_seed(seed):
    """Simple function to fix the random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args

    def training_step(self, batch, batch_idx):
        X, y = batch['waveform'], batch['label'].float()
        prob = self.model(X)
        loss = BCE(prob, y)  # focal_loss(prob, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_probs  = []
        self._val_labels = []

    # ──────────────────────────────────────────────────────────────
    # 2) validation_step – store outputs in the buffers
    # ──────────────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        x, y = batch['waveform'], batch['label'].float()
        logits = self.model(x)
        probs  = torch.sigmoid(logits).detach().cpu().numpy()
        self._val_probs.append(probs)
        self._val_labels.append(y.detach().cpu().numpy())

    # ──────────────────────────────────────────────────────────────
    # 3) epoch–end hook (no arguments!)
    # ──────────────────────────────────────────────────────────────
    def on_validation_epoch_end(self):
        probs  = np.concatenate(self._val_probs,  axis=0)
        labels = np.concatenate(self._val_labels, axis=0)

        # print(f"Shape probs: {probs.shape}, labels: {labels.shape}")

        # edge-case: all 0 or all 1
        if labels.sum() * (len(labels) - labels.sum()) != 0:
            self.threshold = np.sort(probs)[-int(labels.sum())]
            metrics = binary_metrics_fn(
                labels,
                probs,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            metrics = dict(accuracy=0.0, balanced_accuracy=0.0,
                           pr_auc=0.0,  roc_auc=0.0)

        # Lightning 2.0 logging
        self.log("val_acc",   metrics["accuracy"],          prog_bar=True,  sync_dist=True)
        self.log("val_bacc",  metrics["balanced_accuracy"], prog_bar=False, sync_dist=True)
        self.log("val_pr_auc",metrics["pr_auc"],             prog_bar=False, sync_dist=True)
        self.log("val_auroc", metrics["roc_auc"],            prog_bar=True,  sync_dist=True)

        # (optional) save epoch-level csv
        # self._dump_preds_csv(probs, labels, self.current_epoch, split="val")

        print({k: round(v, 4) for k, v in metrics.items()})

    def on_test_epoch_start(self):
        self._test_probs  = []
        self._test_labels = []

    def test_step(self, batch, batch_idx):
        x, y = batch['waveform'], batch['label'].float()
        probs = torch.sigmoid(self.model(x)).detach().cpu().numpy()
        self._test_probs.append(probs)
        self._test_labels.append(y.detach().cpu().numpy())

    def on_test_epoch_end(self):
        probs  = np.concatenate(self._test_probs,  axis=0)
        labels = np.concatenate(self._test_labels, axis=0)
        metrics = binary_metrics_fn(
            labels, probs,
            metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
            threshold=self.threshold,
        )
        self.log_dict({f"test_{k}": v for k, v in metrics.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]

    def _dump_preds_csv(self, probs, labels, epoch_idx, split="val"):
        """
        Save a CSV of the probabilities and labels at the given epoch.
        We assume:
          probs  => np.array shape (N,)  or (N,1)
          labels => np.array shape (N,)
        """
        # If user didn't set `args.save_preds_dir`, skip
        if not self.args.save_preds_dir:
            return

        os.makedirs(self.args.save_preds_dir, exist_ok=True)
        df = pd.DataFrame({
            "prob": probs,
            "label": labels
        })
        out_path = os.path.join(
            self.args.save_preds_dir,
            f"{split}_epoch{epoch_idx:03d}.csv"
        )
        df.to_csv(out_path, index=False)
        print(f"[Save preds] wrote {len(df)} rows to {out_path}")

def supervised(args):
    setup_seed(args.seed)

    df = pd.read_csv(args.train_csv)
    rec_ids = df["short_recording_id"].values
    labels  = create_label_from_meta_csv(df, args.label_key)
    pt_ids  = df["patient_id"].values

    labels_names, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(labels_names, counts))}")

    # 3) train-test split
    train_idx, test_idx = train_test_split(
        np.arange(len(rec_ids)),
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed
    )
    print(f"Split => train={len(train_idx)}, test={len(test_idx)}")

    # We'll store train_list = list of (patient_id, rec_id, lbl)
    train_list = [(pt_ids[i], rec_ids[i], labels[i]) for i in train_idx]
    test_list  = [(pt_ids[i], rec_ids[i], labels[i]) for i in test_idx]

    # 4) Build Dataset => We'll pass mode='train' or 'test'
    train_ds = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=train_list,
        mode='train',
        sample_rate=args.sfreq,
        window_sec=args.sample_length,
        step_sec=args.sample_length,
        n_channels=18,
        scale_factor=1.0,
        train_iterations=10000,
        verbose=False
    )

    test_ds = InMemoryRandomDataset(
        data_dir=args.data_root,
        info_list=test_list,
        mode='test',
        sample_rate=args.sfreq,
        window_sec=args.sample_length,
        step_sec=args.sample_length,
        n_channels=18,
        scale_factor=1.0,
        train_iterations=5000,
        verbose=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=train_ds.collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=test_ds.collate_fn
    )

    backbone = build_model(args.model, args)

    lightning_model = LitModel_finetune(args, backbone)

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sfreq}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",     # The metric you log
        mode="max",              # we want the best (largest) val_auroc
        save_top_k=1,            # keep only the best
        filename=args.model_out,
        dirpath=args.model_dir,
        save_last=True           # optionally also keep last.ckpt
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[args.cuda],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=test_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=float, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sfreq", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    parser.add_argument("--model_dir", type=str, default="ckpts", help="Directory to save model checkpoint(s).")
    parser.add_argument("--model_out", type=str, default="best.ckpt", help="Checkpoint filename to save best model.")
    parser.add_argument("--save_preds_dir", type=str, default="", help="Folder to save CSV files of predictions.")
    parser.add_argument("--cuda", type=int, default=0, help="Which GPU device to use (0,1,...)")
    parser.add_argument(
        "--label_key", type=str, default="case_control_label", help="Key for the label in the CSV file"
    )

    args = parser.parse_args()
    print(args)

    supervised(args)
