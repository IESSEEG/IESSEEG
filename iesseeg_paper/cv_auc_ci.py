"""Influence-function CV-AUROC intervals from LeDell et al. (2015).

DOI: 10.1214/15-EJS1035, Sections 4.1 and 5. The point estimate is the
unweighted mean of fold AUROCs, NOT the benchmark's legacy pair-weighted mean.
Global class proportions, fold-specific concordances and AUC centering, and
patient influence sums / global mean observations per patient follow cvAUC's
ci.cvAUC and ci.pooled.cvAUC algorithms. Variance uses mean squared influence,
not a sample variance with a degrees-of-freedom correction. Equal patient
counts across folds are required here, as in our fixed five-fold manifests.

We assign half credit to tied scores in BOTH AUC and influence contributions.
The authors' reference R implementation uses strict comparisons for influence
contributions (equivalent in the absence of cross-class ties). Half credit
makes the influence function consistent with the usual tie-aware AUC target.
No training, thresholds, or patient-to-recording averaging takes place here.
"""
from statistics import NormalDist
import numpy as np
import pandas as pd


def cv_auc_ci(frame, label="label", score="probability", patient="patient_id",
              fold="fold", confidence=0.95):
    cols=[label, score, patient, fold]
    if frame.empty or frame[cols].isna().any().any():
        raise ValueError("Complete nonempty predictions are required")
    d=frame[cols].copy()
    if not set(d[label].unique()).issubset({0,1}) or d[label].nunique()!=2:
        raise ValueError("Binary labels 0 and 1 are required")
    if not np.isfinite(d[score].to_numpy(float)).all():
        raise ValueError("Scores must be finite")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    if (d.groupby(patient)[fold].nunique()!=1).any():
        raise ValueError("Each patient must belong to exactly one fold")
    sizes=d.groupby(fold)[patient].nunique()
    if len(sizes)<2 or sizes.nunique()!=1:
        raise ValueError("At least two equally sized patient folds are required")
    n_patients=d[patient].nunique()
    mean_observations=len(d)/n_patients
    prevalence=float(d[label].mean())
    per_fold=[]
    for f,g in d.groupby(fold,sort=True):
        y=g[label].to_numpy(int);s=g[score].to_numpy(float)
        positive=s[y==1];negative=s[y==0]
        if not len(positive) or not len(negative):
            raise ValueError("Each fold must include both evaluated classes")
        delta=positive[:,None]-negative[None,:]
        concordance=(delta>0).astype(float)+.5*(delta==0)
        auc=float(concordance.mean())
        influence=np.empty(len(g))
        influence[y==1]=(concordance.mean(axis=1)-auc)/prevalence
        influence[y==0]=(concordance.mean(axis=0)-auc)/(1-prevalence)
        # Sum within patient, not within label: clinician labels can differ
        # between the two observations of one patient.
        patient_ic=pd.Series(influence,index=g[patient].to_numpy()).groupby(level=0).sum()/mean_observations
        per_fold.append(dict(fold=int(f),auroc=auc,n_positive=len(positive),
            n_negative=len(negative),n_patients=g[patient].nunique(),
            influence_second_moment=float(np.mean(patient_ic.to_numpy()**2)),
            tied_pairs=int((delta==0).sum())))
    cvauc=float(np.mean([v["auroc"] for v in per_fold]))
    variance=float(np.mean([v["influence_second_moment"] for v in per_fold]))/n_patients
    se=float(np.sqrt(variance));z=NormalDist().inv_cdf((1+confidence)/2)
    pairs=np.array([v["n_positive"]*v["n_negative"] for v in per_fold])
    weighted=float(np.average([v["auroc"] for v in per_fold],weights=pairs))
    return dict(cv_auroc=cvauc,cv_auroc_se=se,cv_auroc_ci_low=max(0.,cvauc-z*se),
        cv_auroc_ci_high=min(1.,cvauc+z*se),pair_weighted_auroc=weighted,
        confidence=confidence,n_predictions=len(d),n_patients=n_patients,
        n_folds=len(per_fold),cross_class_tied_pairs=sum(v["tied_pairs"] for v in per_fold),
        ci_method="LeDell2015_influence_function_patient_cluster",fold_results=per_fold)


def balanced_accuracy_if_ci(frame, label="label", predicted="predicted_label",
                            patient="patient_id", fold="fold", confidence=0.95):
    """Cluster delta-method CI for pooled held-out balanced accuracy.

    This is OUR standard ratio-estimator influence-function derivation, not a
    balanced-accuracy algorithm provided by LeDell et al. Keep held-out binary
    decisions (including validation-selected thresholds) fixed. With p=P(Y=1),
    sensitivity Se and specificity Sp, the row influence is
      .5*[Y/p*(I(pred=1)-Se) + (1-Y)/(1-p)*(I(pred=0)-Sp)].
    Sum row influences per patient and divide by global mean rows per patient;
    variance of the estimate is mean(patient influence squared)/N_patients.
    This is asymptotic and assumes independent patients and sufficient stability
    of cross-fitted predictions; it does not guarantee small-sample coverage.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie in (0, 1)")
    d=frame[[label,predicted,patient,fold]].copy()
    if d.empty or d.isna().any().any():
        raise ValueError("Complete nonempty predictions are required")
    if set(d[label].unique())!={0,1} or not set(d[predicted].unique()).issubset({0,1}):
        raise ValueError("Binary labels and decisions are required")
    if (d.groupby(patient)[fold].nunique()!=1).any():
        raise ValueError("Each patient must belong to exactly one fold")
    y=d[label].to_numpy(int);h=d[predicted].to_numpy(int);p=y.mean()
    sensitivity=float(h[y==1].mean());specificity=float((1-h[y==0]).mean())
    influence=.5*(y/p*(h-sensitivity)+(1-y)/(1-p)*(1-h-specificity))
    n=d[patient].nunique()
    patient_ic=pd.Series(influence,index=d[patient].to_numpy()).groupby(level=0).sum()/(len(d)/n)
    se=float(np.sqrt(np.mean(patient_ic.to_numpy()**2)/n))
    value=.5*(sensitivity+specificity)
    z=NormalDist().inv_cdf((1+confidence)/2)
    return dict(balanced_accuracy=value,balanced_accuracy_se=se,
        balanced_accuracy_ci_low=max(0.,value-z*se),
        balanced_accuracy_ci_high=min(1.,value+z*se),
        balanced_accuracy_ci_method="patient_cluster_delta_method",sensitivity=sensitivity,specificity=specificity)
