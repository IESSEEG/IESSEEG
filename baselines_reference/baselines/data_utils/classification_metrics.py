"""Scikit-learn metrics used for BIOT and LaBraM checkpoint selection."""
import numpy as np
from sklearn import metrics as skm


def binary_metrics_fn(y_true, y_prob, metrics=None, threshold=0.5):
    names = metrics if metrics is not None else ['pr_auc', 'roc_auc', 'f1']
    predicted = (np.asarray(y_prob) >= threshold).astype(int)
    functions = {
        'pr_auc': (skm.average_precision_score, y_prob),
        'roc_auc': (skm.roc_auc_score, y_prob),
        'accuracy': (skm.accuracy_score, predicted),
        'balanced_accuracy': (skm.balanced_accuracy_score, predicted),
        'f1': (skm.f1_score, predicted),
        'precision': (skm.precision_score, predicted),
        'recall': (skm.recall_score, predicted),
        'cohen_kappa': (skm.cohen_kappa_score, predicted),
    }
    result = {}
    for name in names:
        if name not in functions:
            raise ValueError(f'Unsupported binary metric: {name}')
        function, values = functions[name]
        result[name] = function(y_true, values)
    return result


def multiclass_metrics_fn(y_true, y_prob, metrics=None):
    names = metrics if metrics is not None else ['accuracy', 'f1_macro', 'f1_micro']
    predicted = np.argmax(y_prob, axis=-1)
    functions = {
        'accuracy': skm.accuracy_score,
        'balanced_accuracy': skm.balanced_accuracy_score,
        'cohen_kappa': skm.cohen_kappa_score,
    }
    result = {}
    for name in names:
        if name in functions:
            result[name] = functions[name](y_true, predicted)
        elif name in {'f1_macro', 'f1_micro', 'f1_weighted'}:
            result[name] = skm.f1_score(y_true, predicted, average=name.removeprefix('f1_'))
        else:
            raise ValueError(f'Unsupported multiclass metric: {name}')
    return result
