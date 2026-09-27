"""Checkpoint-selection metrics retain the benchmark's scoring conventions."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'baselines_reference/baselines/data_utils/classification_metrics.py'
SPEC = importlib.util.spec_from_file_location('training_metrics', PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def test_binary_threshold_ties_and_average_precision():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.5, 0.5, 0.9])
    result = metrics.binary_metrics_fn(y, probability,
        metrics=['accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc'])
    assert result == pytest.approx({'accuracy': .75, 'balanced_accuracy': .75,
                                   'roc_auc': .875, 'pr_auc': 5/6})
    assert metrics.binary_metrics_fn(y, probability[:, None], metrics=['accuracy'])['accuracy'] == .75


def test_multiclass_metrics_used_by_labram():
    y = np.array([0, 1, 2, 2])
    probabilities = np.eye(3)[[0, 1, 2, 0]]
    result = metrics.multiclass_metrics_fn(y, probabilities,
        metrics=['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted'])
    assert result == pytest.approx({'accuracy': .75, 'balanced_accuracy': 5/6,
                                   'cohen_kappa': 7/11, 'f1_weighted': .75})
