import torch
import torch.nn as nn
import copy
import os
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from finetune_evaluator import Evaluator
import pandas as pd
from pathlib import Path

from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss
)

class Trainer:
    """
    A trainer that uses the same approach from your code:
     - depends on param: downstream_dataset => picks BCE vs CrossEntropy vs MSE
     - depends on param: epochs => training loop
     - logs best weights (on 'val' set) => final test
    """

    def __init__(self, params, data_loader, model):
        """
        data_loader is expected to be a dict:
          {
            'train': DataLoader,
            'val':   DataLoader,
            'test':  DataLoader
          }
        model => the CBraMod-based model with e.g. param.use_pretrained_weights
        """
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Build evaluator for val & test
        self.val_eval  = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        # Decide criterion based on #classes or dataset name
        # If user wants to do explicit: e.g. 2 => BCE, >2 => CE, or 'SEED-VIG' => MSE
        if self.params.num_of_classes > 2:
            # multi-class
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).to(self.device)
        elif self.params.num_of_classes == 2:
            # binary
            self.criterion = BCEWithLogitsLoss().to(self.device)
        else:
            # regression
            self.criterion = MSELoss().to(self.device)

        # Setup optimizer
        # (Below code is a direct adaptation from your snippet)
        # We'll separate backbone from others if "backbone" in name
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                # freeze if needed
                if self.params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        # Build optimizer
        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW(
                    [
                        {'params': backbone_params, 'lr': self.params.lr},
                        {'params': other_params,   'lr': self.params.lr * 5}
                    ],
                    weight_decay=self.params.weight_decay
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay
                )
        else: # SGD
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD(
                    [
                        {'params': backbone_params, 'lr': self.params.lr},
                        {'params': other_params,   'lr': self.params.lr * 5}
                    ],
                    momentum=0.9,
                    weight_decay=self.params.weight_decay
                )
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.params.lr,
                    momentum=0.9,
                    weight_decay=self.params.weight_decay
                )

        # We'll do a simple scheduler: CosineAnnealing over total steps
        self.data_length = len(self.data_loader['train'])  # #batches in training
        T_max_steps = self.params.epochs * self.data_length
        # avoid T_max=0 if data_length=0
        if T_max_steps < 1:
            T_max_steps = 1
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=T_max_steps,
            eta_min=1e-6
        )

        print(f"Trainer created with #train batches={self.data_length}, T_max={T_max_steps}.")
        print(self.model)

        self.best_model_states = None  # store best

    def train_for_multiclass(self):
        """
        In your original snippet:
         - track best kappa
         - at end => load best states => do test
        """
        f1_best = 0
        kappa_best = 0
        best_epoch = 0

        for epoch_i in range(self.params.epochs):
            start_t = timer()
            train_loss = self._train_one_epoch_multiclass()
            # Evaluate on val => get (acc, kappa, f1, cm)
            acc_val, kappa_val, f1_val, cm_val = self.val_eval.get_metrics_for_multiclass(self.model)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_i+1}/{self.params.epochs} => "
                  f"TrainLoss={train_loss:.4f}, Val_Acc={acc_val:.4f}, "
                  f"Val_Kappa={kappa_val:.4f}, Val_F1={f1_val:.4f}, LR={lr_now:.6f}, "
                  f"Time={(timer()-start_t)/60:.2f}min")

            if kappa_val > kappa_best:
                print(f"[+] kappa improved from {kappa_best:.4f} to {kappa_val:.4f}, saving best model.")
                kappa_best = kappa_val
                f1_best    = f1_val
                best_epoch = epoch_i+1
                self.best_model_states = copy.deepcopy(self.model.state_dict())

        # load best and run test
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states, map_location=self.device)

        acc_test, kappa_test, f1_test, cm_test = self.test_eval.get_metrics_for_multiclass(self.model)
        print(f"Final Test => Acc={acc_test:.4f}, Kappa={kappa_test:.4f}, F1={f1_test:.4f}\n{cm_test}")

        # Save final
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        ckpt_path = os.path.join(self.params.model_dir,
            f"epoch{best_epoch}_acc_{acc_test:.4f}_kappa_{kappa_test:.4f}_f1_{f1_test:.4f}.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Best model saved to {ckpt_path}.")

    def _train_one_epoch_multiclass(self):
        self.model.train()
        losses = []
        for batch in tqdm(self.data_loader['train'], desc="TrainBatch", leave=False):
            x = batch["waveform"].to(self.device)
            y = batch["label"].to(self.device)  # shape => (B,)

            self.optimizer.zero_grad()
            logits = self.model(x)  # shape => (B, #classes)
            loss   = self.criterion(logits, y)
            loss.backward()
            if self.params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
            self.optimizer.step()
            self.optimizer_scheduler.step()
            losses.append(loss.item())
        return np.mean(losses)

    def train_for_binaryclass(self):
        metric_best = float('inf')
        best_epoch = 0

        for epoch_i in range(self.params.epochs):
            start_t = timer()
            train_loss = self._train_one_epoch_binary()
            # Evaluate on val => (acc, pr_auc, roc_auc, cm)
            acc_val, pr_auc_val, roc_val, cm_val, meta_val, prob_val, pred_val = \
                    self.val_eval.get_metrics_for_binaryclass(self.model, return_meta=True)
            loss_val = self.val_eval.get_val_loss_binary(self.model, self.criterion)
            self._dump_epoch_preds("val",   epoch_i, meta_val, prob_val, pred_val)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_i+1}/{self.params.epochs} => "
                  f"TrainLoss={train_loss:.4f}, ValLoss={loss_val:.4f}, "
                  f"Val_Acc={acc_val:.4f}, "
                  f"Val_PR_AUC={pr_auc_val:.4f}, Val_ROC_AUC={roc_val:.4f}, "
                  f"LR={lr_now:.6f}, Time={(timer()-start_t)/60:.2f}min")

            if metric_best > loss_val:
                print(f"[+] Validation loss improved from {metric_best:.4f} to {loss_val:.4f}, saving best model.")
                metric_best = loss_val
                best_epoch = epoch_i+1
                self.best_model_states = copy.deepcopy(self.model.state_dict())

        # load best
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        # final test
        acc_t, pr_t, roc_t, cm_t, meta_t, prob_t, pred_t = \
                self.test_eval.get_metrics_for_binaryclass(self.model, return_meta=True)
        self._dump_epoch_preds("test", epoch_i, meta_t, prob_t, pred_t)
        print(f"Final Test => Loss={metric_best:.4f}, Acc={acc_t:.4f}, PR_AUC={pr_t:.4f}, ROC_AUC={roc_t:.4f}\n{cm_t}")

        # save final
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        torch.save(self.model.state_dict(), self.params.model_out)
        print(f"Best model saved to {self.params.model_out}.")

    def _train_one_epoch_binary(self):
        self.model.train()
        losses = []
        for batch_i, batch in tqdm(enumerate(self.data_loader['train']),
                          total=len(self.data_loader['train']),
                          desc="TrainBatch", leave=False):
            x = batch["waveform"].to(self.device)
            y = batch["label"].float().to(self.device)  # shape => (B,) for binary

            self.optimizer.zero_grad()
            logits = self.model(x).view(-1)  # shape => (B,)

            if self.params.debug:
                print(f"\n[Batch {batch_i}] x.shape={tuple(x.shape)}")
                x_mean = x.mean(dim=(1,2,3))
                print(f"   x_mean (first 5) => {x_mean[:5]} ... (min={x_mean.min():.3f}, max={x_mean.max():.3f})")
                unique_labels, counts = y.unique(return_counts=True)
                print(f"   y distribution => {unique_labels} : {counts}")
                print(f"   logits[:5]={logits[:5].data.cpu().numpy()} ... min={logits.min():.2f}, max={logits.max():.2f}")

            loss   = self.criterion(logits, y)
            loss.backward()

            if self.params.debug:
                total_gnorm = 0.0
                # Example: separate the "backbone" from "classifier" param grads
                back_gnorm = 0.0
                class_gnorm = 0.0
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_gnorm += param_norm ** 2
                        if 'backbone' in n:
                            back_gnorm += param_norm**2
                        else:
                            class_gnorm += param_norm**2

                total_gnorm = total_gnorm**0.5
                back_gnorm  = back_gnorm**0.5
                class_gnorm = class_gnorm**0.5
                print(f"   Grad-norm => total={total_gnorm:.4f}, backbone={back_gnorm:.4f}, classifier={class_gnorm:.4f}")

            if self.params.clip_value > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                if total_norm.isinf() or total_norm.isnan():
                    print(f"Gradient norm is {total_norm} before clipping!")
            self.optimizer.step()
            self.optimizer_scheduler.step()
            losses.append(loss.item())
        return np.mean(losses)

    def train_for_regression(self):
        best_r2 = -1e9
        best_epoch = 0

        for epoch_i in range(self.params.epochs):
            start_t = timer()
            train_loss = self._train_one_epoch_regression()
            # Evaluate => (corrcoef, r2, rmse)
            corr_val, r2_val, rmse_val = self.val_eval.get_metrics_for_regression(self.model)

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch_i+1}/{self.params.epochs} => "
                  f"TrainLoss={train_loss:.4f}, Val_R2={r2_val:.4f}, "
                  f"LR={lr_now:.6f}, Time={(timer()-start_t)/60:.2f}min")

            if r2_val > best_r2:
                print(f"[+] R2 improved from {best_r2:.4f} to {r2_val:.4f}. Saving best.")
                best_r2 = r2_val
                best_epoch = epoch_i+1
                self.best_model_states = copy.deepcopy(self.model.state_dict())

        # load best
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states, map_location=self.device)

        # final test => (corrcoef, r2, rmse)
        corr_test, r2_test, rmse_test = self.test_eval.get_metrics_for_regression(self.model)
        print(f"Final Test => Corr={corr_test:.4f}, R2={r2_test:.4f}, RMSE={rmse_test:.4f}")

        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        torch.save(self.model.state_dict(), self.params.model_out)
        print(f"Best model saved to {self.params.model_out}.")

    def _train_one_epoch_regression(self):
        self.model.train()
        losses = []
        for batch in tqdm(self.data_loader['train'], desc="TrainBatch", leave=False):
            x = batch["waveform"].to(self.device)
            y = batch["label"].float().to(self.device)  # shape => (B,)

            self.optimizer.zero_grad()
            logits = self.model(x).view(-1)  # shape => (B,)
            loss   = self.criterion(logits, y)
            loss.backward()
            if self.params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
            self.optimizer.step()
            self.optimizer_scheduler.step()
            losses.append(loss.item())
        return np.mean(losses)

    def _dump_epoch_preds(self, split_name, epoch, meta_list, probs, preds):
        """
        meta_list: list of dicts coming back from Evaluator (one per sample)
        probs/preds: 1-D numpy arrays
        """
        out_rows = []
        for meta, p, y_hat in zip(meta_list, probs, preds):
            row = {
                "patient_id":   meta["patient_id"],
                "recording_id": meta["recording_id"],
                "start_ind":    meta["start_ind"],
                "end_ind":      meta["end_ind"],
                "label":        meta["label"],
                "probability":  float(p),
                "prediction":   int(y_hat),
                "epoch":        epoch + 1,
                "split":        split_name
            }
            out_rows.append(row)

        out_df = pd.DataFrame(out_rows)

        save_dir = Path(self.params.save_preds_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"epoch{epoch+1:03d}_{split_name}.csv"
        out_df.to_csv(fname, index=False)
        print(f"[SAVE] wrote {len(out_df)} rows to {fname}")
