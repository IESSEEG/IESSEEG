"""Clip metrics with patient-clustered uncertainty for fixed OOF predictions."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auc_draws(frame, draws=2000, seed=20260923):
    """Within-fold pair-weighted AUROC; resample patients within fold/label.

    The same seed gives paired draws when frames have the same patients/labels.
    This quantifies evaluation-sample uncertainty, not refitting uncertainty.
    """
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    point, pairs = 0., 0
    for _, fold in frame.groupby("fold", sort=True):
        banks = {}
        for label in (0, 1):
            banks[label] = [p.probability.to_numpy(float) for _, p in
                fold[fold.label.eq(label)].groupby("patient_id", sort=True)]
        if not banks[0] or not banks[1]:
            continue
        counts = {len(v) for b in banks.values() for v in b}
        if len(counts) != 1:
            raise ValueError("Expected equal clip counts per patient within this estimand")
        neg, pos = np.stack(banks[0]), np.stack(banks[1])
        difference = pos[:, None, :, None] - neg[None, :, None, :]
        matrix = ((difference > 0) + .5 * (difference == 0)).sum((2, 3))
        pn, nn = len(pos), len(neg)
        wp = rng.multinomial(pn, np.full(pn, 1/pn), size=draws)
        wn = rng.multinomial(nn, np.full(nn, 1/nn), size=draws)
        totals += np.einsum("bi,ij,bj->b", wp, matrix, wn)
        point += matrix.sum()
        pairs += pos.size * neg.size
    if not pairs:
        raise ValueError("No positive-negative pairs")
    return float(point / pairs), totals / pairs


def summary(frame):
    point, draws = auc_draws(frame)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(n_patients=int(frame.patient_id.nunique()), n_clips=len(frame),
        fold_comparable_clip_auroc=point, clustered_ci_low=float(lo), clustered_ci_high=float(hi),
        pooled_oof_clip_auroc=float(roc_auc_score(frame.label, frame.probability)),
        pooled_oof_clip_auprc=float(average_precision_score(frame.label, frame.probability)))
