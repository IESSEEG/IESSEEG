"""Check added metrics against sklearn and AP influences against perturbations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from iesseeg.metrics import ap_influence, extra_metrics


def test_average_precision_influence_with_and_without_ties():
    rng=np.random.default_rng(91)
    for tied in [False,True]:
        y=np.array([0,1]*10); scores=rng.random(len(y))
        if tied:scores=np.round(scores,1)
        ap,ic=ap_influence(y,scores)
        np.testing.assert_allclose(ic.mean(),0,atol=1e-12)
        np.testing.assert_allclose(ap,average_precision_score(y,scores))
        for i in range(len(y)):
            eps=1e-7; weights=np.ones(len(y))*(1-eps)/len(y);weights[i]+=eps
            derivative=(average_precision_score(y,scores,sample_weight=weights)-ap)/eps
            np.testing.assert_allclose(derivative,ic[i],atol=1e-5,rtol=1e-4)


def test_metrics_use_fixed_threshold_and_keep_patient_clusters():
    d=pd.DataFrame(dict(label=[0,1,0,1]*5,probability=[.2,.8,.7,.4]*5,
                        predicted_label=[1]*20,patient_id=np.repeat(np.arange(10),2),
                        fold=np.repeat(np.arange(5),4)))
    result=extra_metrics(d,'label')
    for name in ['accuracy','precision','f1','sensitivity','specificity']:
        np.testing.assert_allclose(result[name],.5)
    np.testing.assert_allclose(result['average_precision'],average_precision_score(d.label,d.probability))
