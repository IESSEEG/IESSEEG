"""CUDA ridge probes, nested regularization selection and patient statistics.

Scores are continuous ridge-classifier outputs, NOT calibrated probabilities.
The primary AUROC compares patient pairs scored by the same outer-fold model.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)


def require_cuda(device):
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Explicit working CUDA device required; no CPU fallback")
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    return dev


def standardize(train, test):
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0
    a = np.where(np.isfinite(train), train, median)
    b = np.where(np.isfinite(test), test, median)
    center, scale = a.mean(0), a.std(0)
    scale[scale < 1e-8] = 1
    return (a - center) / scale, (b - center) / scale, (median, center, scale)


def auc(y, score, folds=None):
    y, score = np.asarray(y), np.asarray(score)
    if folds is None:
        return float(roc_auc_score(y, score))
    concordant = pairs = 0
    for f in np.unique(folds):
        p = score[(folds == f) & (y == 1)]
        n = score[(folds == f) & (y == 0)]
        d = p[:, None] - n[None, :]
        concordant += (d > 0).sum() + 0.5 * (d == 0).sum()
        pairs += d.size
    if not pairs:
        raise ValueError("No evaluable patient pairs")
    return float(concordant / pairs)


def batch_auc(labels, scores, folds=None):
    # labels and scores: permutation/draw by patient
    difference = scores[:, :, None] - scores[:, None, :]
    pairs = (labels[:, :, None] == 1) & (labels[:, None, :] == 0)
    if folds is not None:
        pairs &= folds[None, :, None] == folds[None, None, :]
    return ((difference > 0) * pairs + 0.5 * (difference == 0) * pairs).sum(
        (1, 2)
    ) / pairs.sum((1, 2))


def probe_maps(x, folds, device):
    """Outcome-independent linear maps for nested train/validation/test fits."""
    maps = []
    for fold in range(5):
        test = np.flatnonzero(folds == fold)
        val = np.flatnonzero(folds == (fold + 1) % 5)
        inner = np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5))
        outer = np.flatnonzero(folds != fold)
        stage_maps = []
        for tr, te in ((inner, val), (outer, test)):
            a, b, scaler = standardize(x[tr], x[te])
            a = torch.as_tensor(a, dtype=torch.float64, device=device)
            b = torch.as_tensor(b, dtype=torch.float64, device=device)
            d = max(1, x.shape[1])
            k = a @ a.T / d
            cross = b @ a.T / d
            identity = torch.eye(len(tr), dtype=torch.float64, device=device)
            h = []
            for alpha in ALPHAS:
                # Average squared loss + alpha ||w||^2; unpenalized intercept.
                mat = torch.linalg.solve(k + len(tr) * alpha * identity, cross.T).T
                mat = mat - mat.mean(dim=1, keepdim=True) + 1 / len(tr)
                h.append(mat.cpu().numpy())
            stage_maps.append((tr, te, np.stack(h), scaler))
        maps.append(stage_maps)
    return maps


def predict_with_maps(maps, labels, folds):
    labels = np.atleast_2d(labels)
    output = np.empty(labels.shape, dtype=float)
    choices = np.empty((len(labels), 5), dtype=int)
    for f, (validation, testing) in enumerate(maps):
        tr, va, h, _ = validation
        # permutation x alpha x validation patient
        scores = np.einsum("avt,pt->pav", h, 2 * labels[:, tr] - 1)
        metrics = np.stack(
            [batch_auc(labels[:, va], scores[:, j]) for j in range(len(ALPHAS))], axis=1
        )
        # Fixed tie policy: choose larger regularization among equal AUROCs.
        selected = len(ALPHAS) - 1 - np.argmax(metrics[:, ::-1], axis=1)
        tr, te, h, _ = testing
        all_scores = np.einsum("avt,pt->pav", h, 2 * labels[:, tr] - 1)
        output[:, te] = all_scores[np.arange(len(labels)), selected]
        choices[:, f] = selected
    return output, choices


def bootstrap_indices(labels, folds, draws=2000, seed=20260920):
    rng = np.random.default_rng(seed)
    strata = [
        np.flatnonzero((labels == y) & (folds == f)) for f in range(5) for y in (0, 1)
    ]
    return np.concatenate(
        [rng.choice(s, (draws, len(s)), replace=True) for s in strata if len(s)], axis=1
    )


def summarize(labels, scores, folds, draws=2000):
    ix = bootstrap_indices(labels, folds, draws)
    # Stratified indices preserve fold membership at every position.
    values = batch_auc(labels[ix], scores[ix], folds[ix[0]])
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "n_patients": len(labels),
        "n_positive": int(labels.sum()),
        "patient_auroc": auc(labels, scores, folds),
        "ci_low": lo,
        "ci_high": hi,
        "pooled_oof_auroc": auc(labels, scores),
        "pooled_oof_auprc": float(average_precision_score(labels, scores)),
        "ci_unit": "patient; fold/class stratified; fitted predictions fixed",
    }


def permutation_labels(labels, folds, n, seed=20260920):
    rng = np.random.default_rng(seed)
    out = np.tile(labels, (n, 1))
    for row in out:
        for f in range(5):
            ix = np.flatnonzero(folds == f)
            row[ix] = rng.permutation(row[ix])
    return out


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result
