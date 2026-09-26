#!/usr/bin/env python
"""Fit diagnostic qEEG with patient-disjoint train/validation/test and no refit.

Run after mamba activate seeg:
python experiments/diagnosis_qeeg_no_refit.py
Uses cached published-feature adaptations and the released diagnosis folds.
C is selected by validation AUROC (original ascending grid, first maximum).
The selected fitted classifier also supplies validation threshold and test scores.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iesseeg_paper.input_studies import TRIPLET, DIAGNOSTIC
from iesseeg_paper.splits import load_fold_manifest
from iesseeg_paper.evaluation import standardize
CS = (.01, .1, 1., 10., 100.)


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/input_studies_20260923')
    p.add_argument('--local-root', type=Path, default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/diagnosis_qeeg_no_refit_20260925')
    p.add_argument('--output-root', type=Path, default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/diagnosis_qeeg_no_refit_20260925')
    p.add_argument('--checkpoint-root', type=Path, default=Path(__import__('os').environ.get('IESSEEG_CHECKPOINT_ROOT', 'work/checkpoints') + '/diagnosis_qeeg_no_refit_20260925'))
    a = p.parse_args()
    for directory in (a.local_root, a.output_root, a.checkpoint_root):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = pd.read_csv(a.input_root/'inputs.csv', dtype={'recording_id': str})
    dev = inputs[inputs.kind.eq('smith_development')].sort_values(['patient_id', 'recording_id']).reset_index(drop=True)
    test = inputs[inputs.kind.eq('smith_evaluation')].sort_values(['patient_id', 'recording_id']).reset_index(drop=True)
    fm = load_fold_manifest('diagnosis').set_index('patient_id').fold
    for frame in (dev, test):
        frame['fold'] = frame.patient_id.map(fm)
        assert frame.fold.notna().all() and frame.recording_id.is_unique
        frame['label'] = frame.case_control_label.eq('CASE').astype(int)
    assert len(dev) == 400 and len(test) == 200
    # Only recover the two test label definitions; old test predictions are unused.
    labels = pd.read_csv(Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/main_benchmark_probabilities_20260925/smith_triplet_diagnosis.csv', dtype={'recording_id': str}).set_index('recording_id')
    np.testing.assert_array_equal(test.patient_id, labels.loc[test.recording_id, 'patient_id'])
    np.testing.assert_array_equal(test.label, labels.loc[test.recording_id, 'patient_label'])
    test['patient_label'] = test.label
    test['expert_label'] = labels.loc[test.recording_id, 'expert_label'].to_numpy()

    def features(frame, keys):
        rows = []
        for r in frame.itertuples():
            with np.load(a.input_root/'features/smith'/f'{r.kind}_{r.recording_id}.npz') as z:
                rows.append([float(z[k]) for k in keys])
        return np.array(rows)

    summaries, configurations, trials, fold_metrics = [], [], [], []
    for name, keys in [('smith_triplet', TRIPLET), ('smith_expanded', DIAGNOSTIC)]:
        x, xt = features(dev, keys), features(test, keys)
        predictions, validation = [], []
        for f in range(5):
            tr = ~dev.fold.isin([f, (f+1)%5]); va = dev.fold.eq((f+1)%5); te = test.fold.eq(f)
            train_ids, val_ids, test_ids = [set(frame.patient_id) for frame in (dev[tr], dev[va], test[te])]
            assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
            assert tuple(map(len, (train_ids, val_ids, test_ids))) == (60, 20, 20)
            train_x, val_x, scaler = standardize(x[tr], x[va])
            med, center, scale = scaler
            test_x = (np.where(np.isfinite(xt[te]), xt[te], med)-center)/scale
            candidates, val_probabilities, aucs = [], [], []
            for c in CS:
                model = LogisticRegression(C=c, max_iter=3000, solver='lbfgs', random_state=20260924).fit(train_x, dev.loc[tr, 'label'])
                vp = model.predict_proba(val_x)[:, 1]
                score = float(roc_auc_score(dev.loc[va, 'label'], vp))
                candidates.append(model); val_probabilities.append(vp); aucs.append(score)
                trials.append(dict(model=name, fold=f, C=c, validation_auroc=score))
            best = int(np.argmax(aucs)); model = candidates[best]; vp = val_probabilities[best]
            fpr, tpr, thresholds = roc_curve(dev.loc[va, 'label'], vp)
            finite = np.flatnonzero(np.isfinite(thresholds))
            cut = float(thresholds[finite[np.argmax((tpr-fpr)[finite])]])
            assert np.isclose(balanced_accuracy_score(dev.loc[va, 'label'], vp >= cut), (1 + np.max(tpr-fpr))/2)
            tp = model.predict_proba(test_x)[:, 1]
            assert np.isfinite(tp).all() and ((tp >= 0) & (tp <= 1)).all()
            payload = dict(model=model, scaler=scaler, features=list(keys), C=CS[best], threshold=cut, fold=f,
                           train_patients=sorted(train_ids), validation_patients=sorted(val_ids), test_patients=sorted(test_ids), refit=False)
            path = a.checkpoint_root/f'{name}_fold{f}.joblib'
            joblib.dump(payload, path)
            saved = joblib.load(path)
            np.testing.assert_allclose(saved['model'].predict_proba(val_x)[:, 1], vp, atol=1e-12)
            np.testing.assert_allclose(saved['model'].predict_proba(test_x)[:, 1], tp, atol=1e-12)
            pred = test.loc[te, ['recording_id', 'patient_id', 'fold', 'patient_label', 'expert_label']].copy()
            pred['probability'] = tp; pred['threshold'] = cut
            pred['predicted_05'] = (tp >= .5).astype(int)
            pred['predicted_validation_threshold'] = (tp >= cut).astype(int)
            predictions.append(pred)
            val = dev.loc[va, ['recording_id', 'patient_id', 'label', 'fold']].copy()
            val['model_fold'] = f; val['probability'] = vp; validation.append(val)
            configurations.append(dict(model=name, fold=f, C=CS[best], threshold=cut, validation_auroc=aucs[best],
                validation_balanced_accuracy=balanced_accuracy_score(val.label, vp >= cut), train_patients=60, validation_patients=20, test_patients=20))
        pred = pd.concat(predictions, ignore_index=True)
        assert len(pred) == 200 and pred.recording_id.is_unique and pred.patient_id.nunique() == 100
        assert pred.groupby('patient_id').size().eq(2).all()
        pred.to_csv(a.local_root/f'{name}_test_predictions.csv', index=False)
        pd.concat(validation, ignore_index=True).to_csv(a.local_root/f'{name}_validation_predictions.csv', index=False)
        for target in ('patient_label', 'expert_label'):
            scores = []
            for f, part in pred.groupby('fold'):
                score = roc_auc_score(part[target], part.probability); scores.append(score)
                fold_metrics.append(dict(model=name, target=target, fold=f, auroc=score,
                    balanced_accuracy_05=balanced_accuracy_score(part[target], part.predicted_05),
                    balanced_accuracy_validation_threshold=balanced_accuracy_score(part[target], part.predicted_validation_threshold)))
            summaries.append(dict(model=name, target=target, mean_fold_auroc=np.mean(scores),
                balanced_accuracy_05=balanced_accuracy_score(pred[target], pred.predicted_05),
                balanced_accuracy_validation_threshold=balanced_accuracy_score(pred[target], pred.predicted_validation_threshold)))
    for filename, rows in [('results', summaries), ('configurations', configurations), ('hyperparameter_trials', trials), ('fold_metrics', fold_metrics)]:
        pd.DataFrame(rows).to_csv(a.output_root/f'{filename}.csv', index=False)
    (a.output_root/'protocol.json').write_text(json.dumps(dict(
        training='Three fixed diagnosis patient folds, 240 Clinical Clips from 60 patients',
        validation='Next fixed patient fold, 80 Clinical Clips from 20 patients; patient diagnosis labels only',
        testing='Held-out fixed patient fold, 40 simulated routine EEG segments from 20 patients',
        preprocessing='Median imputation and standardization fitted on training clips only',
        C_grid=CS, C_selection='Maximum validation AUROC; first maximum in ascending C grid',
        threshold_selection='First finite ROC threshold maximizing validation balanced accuracy; prediction uses >=',
        refit=False, test_targets=['patient diagnosis label', 'clinician diagnosis label'],
        metrics='Equal mean of five fold AUROCs; balanced accuracy on combined held-out predictions',
        main_manuscript_modified=False, checkpoints=str(a.checkpoint_root),
        validation_checks='All three patient sets disjoint; reloaded classifier reproduces validation and test probabilities; 200 unique test clips from 100 patients'), indent=2)+'\n')
    names = {'smith_triplet': 'qEEG triplet', 'smith_expanded': 'Expanded qEEG'}
    lines = ['# Diagnosis qEEG with no refit', '',
        'Run `mamba activate seeg`, then `python experiments/diagnosis_qeeg_no_refit.py` from code_to_publish.', '',
        'Each fold uses 60 training, 20 validation and 20 test patients. The selected logistic classifier is used unchanged for both validation-threshold selection and test prediction. Both test targets use the threshold chosen with validation patient diagnosis labels. Feature extraction and the fixed outer folds are unchanged. Values below are percentages.', '',
        '| Model | Test label | Mean fold AUROC | Balanced accuracy, 0.5 | Balanced accuracy, validation threshold |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {names[r['model']]} | {r['target']} | {100*r['mean_fold_auroc']:.1f} | {100*r['balanced_accuracy_05']:.1f} | {100*r['balanced_accuracy_validation_threshold']:.1f} |")
    lines += ['', 'These newly fitted models use three training folds rather than the four-fold refit in the current manuscript. Their probabilities and AUROC can therefore differ from the current main table. Neither threshold uses test labels. Main manuscript tables have not been changed.']
    (a.output_root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print('COMPLETE: 10 selected classifiers; all validation checks passed')

if __name__ == '__main__':
    run()
