"""Final-paper metrics, using patient-cluster influence functions."""
from iesseeg_paper.cv_auc_ci import cv_auc_ci, balanced_accuracy_if_ci
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, average_precision_score


def interval(value, influence, patients):
    """Asymptotic patient-cluster interval conditional on fitted predictions."""
    n = len(np.unique(patients))
    ic = pd.Series(influence, index=patients).groupby(level=0).sum() / (len(patients)/n)
    se = np.sqrt(np.mean(ic.to_numpy()**2)/n)
    return value, max(0., value-1.95996398454*se), min(1., value+1.95996398454*se)


def ap_influence(y, score):
    """Plug-in influence of non-interpolated AP, retaining tied-score groups.

    AP = sum_t P(Y=1,S=t)*precision(t)/P(Y=1).
    Differentiate the three empirical measures in this expression. This is
    separate from LeDell's AUROC method; it is not attributed to that paper.
    """
    y = np.asarray(y, float); score = np.asarray(score, float)
    prevalence = y.mean()
    thresholds = np.unique(score)
    above = score[:,None] >= thresholds[None,:]
    survival = above.mean(axis=0)
    precision = (above*y[:,None]).mean(axis=0)/survival
    positive_mass = ((score[:,None]==thresholds[None,:])*y[:,None]).mean(axis=0)
    value = float(positive_mass@precision/prevalence)
    own_precision = precision[np.searchsorted(thresholds,score)]
    ic = y/prevalence*(own_precision-value)
    ic += (above*(y[:,None]-precision)*positive_mass/survival).sum(axis=1)/prevalence
    return value, ic


def extra_metrics(d, label):
    y=d[label].to_numpy(int); h=(d.probability.to_numpy()>=.5).astype(int)
    pid=d.patient_id.to_numpy(); tp=y*h; correct=(y==h).astype(float)
    out={}
    def add(name, value, ic):
        value,lo,hi=interval(value,ic,pid)
        out.update({name:value,name+'_ci_low':lo,name+'_ci_high':hi})
    add('accuracy',correct.mean(),correct-correct.mean())
    for name,a,b in [('sensitivity',tp,y),('specificity',(1-y)*(1-h),1-y),
                     ('precision',tp,h),('f1',2*tp,y+h)]:
        if b.mean()==0:
            out.update({name:np.nan,name+'_ci_low':np.nan,name+'_ci_high':np.nan})
        else:
            value=a.mean()/b.mean();add(name,value,(a-value*b)/b.mean())
    ap_values=[];ap_ic=np.empty(len(d))
    for fold in sorted(d.fold.unique()):
        mask=d.fold.to_numpy()==fold
        ap,ic=ap_influence(y[mask],d.probability.to_numpy()[mask])
        np.testing.assert_allclose(ap,average_precision_score(y[mask],d.probability.to_numpy()[mask]),atol=1e-12)
        ap_values.append(ap);ap_ic[mask]=ic
    # Equal observation counts in each fold make this the IF of the mean fold AP.
    assert d.groupby('fold').size().nunique()==1
    add('average_precision',np.mean(ap_values),ap_ic)
    np.testing.assert_allclose(out['accuracy'],accuracy_score(y,h))
    if h.sum(): np.testing.assert_allclose(out['precision'],precision_score(y,h))
    np.testing.assert_allclose(out['f1'],f1_score(y,h))
    return out


def summarize(frame, label="label"):
    d = frame.copy()
    if not d.probability.between(0, 1).all():
        raise ValueError("Expected probabilities in [0, 1]")
    d['predicted_label'] = (d.probability >= .5).astype(int)
    auc = cv_auc_ci(d, label=label)
    auc.pop('fold_results')
    return {**auc, **balanced_accuracy_if_ci(d, label=label), **extra_metrics(d, label)}


def auc_influence(frame):
    """Patient contributions for paired differences of mean fold AUROCs."""
    d=frame.reset_index(drop=True)
    y=d.label.to_numpy();s=d.probability.to_numpy();prev=y.mean()
    influence=np.empty(len(d))
    for fold in sorted(d.fold.unique()):
        ix=np.flatnonzero(d.fold.to_numpy()==fold);yy=y[ix];pp=s[ix]
        delta=pp[yy==1,None]-pp[yy==0][None,:]
        h=(delta>0)+.5*(delta==0);auc=h.mean()
        v=np.empty(len(ix));v[yy==1]=(h.mean(1)-auc)/prev;v[yy==0]=(h.mean(0)-auc)/(1-prev)
        influence[ix]=v
    return pd.Series(influence,index=d.patient_id).groupby(level=0).sum()/(len(d)/d.patient_id.nunique())


def paired_auc(first, second):
    keys=['patient_id','recording_id','crop_id','fold','label']
    a=first.sort_values(keys).reset_index(drop=True)
    b=second.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys],b[keys])
    estimate=cv_auc_ci(a)['cv_auroc']-cv_auc_ci(b)['cv_auroc']
    diff=auc_influence(a)-auc_influence(b)
    se=float(np.sqrt(np.mean(diff.to_numpy()**2)/len(diff)))
    return dict(delta_auroc=estimate,ci_low=estimate-1.95996398454*se,ci_high=estimate+1.95996398454*se)
