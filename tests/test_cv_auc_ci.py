"""Independent formula checks for the published CV-AUC CI and BA delta method."""
import numpy as np
import pandas as pd
import pytest
from iesseeg_paper.cv_auc_ci import cv_auc_ci,balanced_accuracy_if_ci

def fixture():
    return pd.DataFrame(dict(patient_id=np.arange(8),fold=np.repeat([0,1],4),
        label=[0,0,1,1]*2,probability=[.1,.8,.4,.9]*2,predicted_label=[0,1,0,1]*2))

def test_hand_calculated_influence_functions():
    d=fixture();a=cv_auc_ci(d);b=balanced_accuracy_if_ci(d)
    assert a['cv_auroc']==.75
    assert a['cv_auroc_se']==pytest.approx(np.sqrt(.25/8))
    assert b['balanced_accuracy']==.5
    assert b['balanced_accuracy_se']==pytest.approx(np.sqrt(.25/8))

def test_duplicate_observations_do_not_inflate_sample_size():
    d=fixture();twice=pd.concat([d,d],ignore_index=True)
    for fn,key in [(cv_auc_ci,'cv_auroc_se'),(balanced_accuracy_if_ci,'balanced_accuracy_se')]:
        assert fn(twice)[key]==pytest.approx(fn(d)[key])

def test_mixed_labels_within_patient_and_literal_reference():
    d=fixture();d['patient_id']=[0,1,0,1,2,3,2,3]
    # Independent literal implementation of Sections 4.1 and 5 (no ties).
    fold_means=[];aucs=[];p=d.label.mean();taubar=len(d)/d.patient_id.nunique()
    for _,g in d.groupby('fold'):
        pos=g[g.label==1].probability.tolist();neg=g[g.label==0].probability.tolist()
        auc=sum(x>y for x in pos for y in neg)/(len(pos)*len(neg));aucs.append(auc)
        values={}
        for x in g.itertuples():
            concordance=sum(x.probability>v for v in neg)/len(neg) if x.label else sum(v>x.probability for v in pos)/len(pos)
            ic=(concordance-auc)/(p if x.label else 1-p)
            values[x.patient_id]=values.get(x.patient_id,0)+ic/taubar
        fold_means.append(np.mean([v*v for v in values.values()]))
    result=cv_auc_ci(d)
    assert result['cv_auroc']==np.mean(aucs)
    assert result['cv_auroc_se']==pytest.approx(np.sqrt(np.mean(fold_means)/4))

def test_half_credit_for_ties():
    d=fixture();d['probability']=.5
    a=cv_auc_ci(d)
    assert a['cv_auroc']==.5 and a['cv_auroc_se']==0
    assert a['cross_class_tied_pairs']==8

def test_patient_may_not_cross_folds():
    d=fixture();d.loc[4,'patient_id']=0
    with pytest.raises(ValueError,match='exactly one fold'):cv_auc_ci(d)
    with pytest.raises(ValueError,match='exactly one fold'):balanced_accuracy_if_ci(d)

def test_equal_fold_not_pair_weighted():
    d=fixture();d.loc[4:,'label']=[0,0,0,1]
    result=cv_auc_ci(d)
    assert result['cv_auroc']==pytest.approx(.875)
    assert result['pair_weighted_auroc']==pytest.approx(6/7)

def test_ba_keeps_saved_threshold_decisions():
    d=fixture();d['predicted_label']=0
    assert balanced_accuracy_if_ci(d)['balanced_accuracy']==.5
    assert balanced_accuracy_if_ci(d)['balanced_accuracy_se']==0
