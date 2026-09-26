import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    r2_score,
    mean_squared_error
)
from tqdm import tqdm

class Evaluator:
    """
    A simplified evaluator class that uses the same style of metrics
    as the original code:
      - For multiclass => (acc, kappa, f1, cm)
      - For binary    => (acc, pr_auc, roc_auc, cm)
      - For regression => (corrcoef, r2, rmse)
    We assume the batch is a dict with keys:
      ["waveform", "label", "patient_id", "recording_id", "start_ind", "end_ind"]
    Where "waveform" => shape (B,C,seq_len,200).
    """

    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{params.cuda}" if torch.cuda.is_available() else "cpu")

    def get_metrics_for_multiclass(self, model):
        # We'll read the entire dataset, gather preds, compare to y
        model.eval()
        all_preds = []
        all_labels= []
        with torch.no_grad():
            for batch in self.data_loader:
                x = batch["waveform"].to(self.device)
                y = batch["label"].to(self.device)
                logits = model(x)  # shape => (B, #classes)
                pred_y = torch.argmax(logits, dim=-1)

                all_preds.append(pred_y.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # balanced acc
        acc   = balanced_accuracy_score(all_labels, all_preds)
        # Kappa
        kappa = cohen_kappa_score(all_labels, all_preds)
        # weighted F1
        f1    = f1_score(all_labels, all_preds, average='weighted')
        cm    = confusion_matrix(all_labels, all_preds)

        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model, return_meta=False):
        model.eval()
        probs, preds, labels, metas = [], [], [], []

        with torch.no_grad():
            for batch_i, batch in enumerate(self.data_loader):
                x   = batch["waveform"].to(self.device)
                y   = batch["label"].to(self.device)
                log = model(x).contiguous().view(-1)
                p   = torch.sigmoid(log).detach()

                if getattr(self.params, "debug", False):
                    print(f"[Eval|Binary] Batch {batch_i}, x.shape={x.shape}, x.mean={x.mean():.3f}")
                    print(f"   p[:5]= {p[:5].cpu().numpy()}, range=({p.min():.3f},{p.max():.3f})")

                probs.append(p.cpu().numpy())
                preds.append((p >= 0.5).long().cpu().numpy())
                labels.append(y.cpu().numpy())

                if return_meta:
                    B = len(y)
                    for i in range(B):
                        metas.append({
                            "patient_id":   batch["patient_id"][i],
                            "recording_id": batch["recording_id"][i],
                            "start_ind":    batch["start_ind"][i],
                            "end_ind":      batch["end_ind"][i],
                            "label":        int(y[i].item())
                        })

        probs  = np.concatenate(probs)
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)

        acc = balanced_accuracy_score(labels, preds)
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        cm      = confusion_matrix(labels, preds)

        if return_meta:
            return acc, pr_auc, roc_auc, cm, metas, probs, preds
        return acc, pr_auc, roc_auc, cm

    def get_val_loss_binary(self, model, criterion):
        """
        Compute the average validation loss for a binary classification task.
        model: a torch.nn.Module
        criterion: e.g. torch.nn.BCEWithLogitsLoss
        """
        model.eval()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in self.data_loader:
                x = batch["waveform"].to(self.device)
                y = batch["label"].float().to(self.device)  # shape => (B,)

                logits = model(x).view(-1)  # (B,)
                loss   = criterion(logits, y)

                batch_size = x.size(0)
                total_loss  += loss.item() * batch_size
                total_count += batch_size

        return total_loss / total_count if total_count > 0 else 0.0

    def get_metrics_for_regression(self, model):
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch in self.data_loader:
                x = batch["waveform"].to(self.device)
                y = batch["label"].to(self.device)
                logits = model(x).view(-1)  # shape => (B,)
                all_preds.append(logits.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # correlation
        corrcoef = np.corrcoef(all_labels, all_preds)[0,1]
        r2  = r2_score(all_labels, all_preds)
        rmse = mean_squared_error(all_labels, all_preds)**0.5

        return corrcoef, r2, rmse
