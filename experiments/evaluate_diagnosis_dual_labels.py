#!/usr/bin/env python
"""Evaluate saved diagnosis predictions against patient and expert clip labels.

No refitting. Keep both clips together in bootstrap samples, including patients
whose expert labels differ between clips. Stratification uses patient diagnosis.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score,average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]

def evaluate(frame,label,threshold=.5):
 rng=np.random.default_rng(20260923);draws=2000
 numer=np.zeros(draws);denom=np.zeros(draws);pointnum=0.;pointden=0.
 correct=np.zeros((draws,2));counts=np.zeros((draws,2))
 if frame.groupby("patient_id").fold.nunique().max()!=1 or frame.groupby("patient_id").patient_label.nunique().max()!=1:
  raise ValueError("Each diagnosis patient must have one fold and one patient diagnosis label")
 for _,fold in frame.groupby('fold',sort=True):
  groups=[v for _,v in fold.groupby('patient_id',sort=True)];n=len(groups)
  num=np.zeros((n,n));den=np.zeros((n,n))
  for i,a in enumerate(groups):
   pos=a.loc[a[label].eq(1),'probability'].to_numpy()
   for j,b in enumerate(groups):
    neg=b.loc[b[label].eq(0),'probability'].to_numpy();d=pos[:,None]-neg[None,:]
    num[i,j]=((d>0)+.5*(d==0)).sum();den[i,j]=len(pos)*len(neg)
  weights=np.zeros((draws,n))
  for target in [0,1]:
   ids=[i for i,g in enumerate(groups) if int(g.patient_label.iloc[0])==target]
   weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=draws)
  numer+=np.einsum('bi,ij,bj->b',weights,num,weights);denom+=np.einsum('bi,ij,bj->b',weights,den,weights)
  # Clinician labels can differ between clips; count actual target labels.
  for target in [0,1]:
   counts[:,target]+=weights@np.array([g[label].eq(target).sum() for g in groups])
   correct[:,target]+=weights@np.array([(g[label].eq(target)&((g.probability>=threshold)==target)).sum() for g in groups])
  pointnum+=num.sum();pointden+=den.sum()
 lo,hi=np.quantile(numer[denom>0]/denom[denom>0],[.025,.975])
 valid=(counts>0).all(axis=1)
 ba_lo,ba_hi=np.quantile((correct[valid]/counts[valid]).mean(axis=1),[.025,.975])
 return dict(auroc=pointnum/pointden,ci_low=lo,ci_high=hi,balanced_accuracy=balanced_accuracy_score(frame[label],frame.probability>=threshold),balanced_accuracy_ci_low=float(ba_lo),balanced_accuracy_ci_high=float(ba_hi),pooled_auroc=roc_auc_score(frame[label],frame.probability),pooled_average_precision=average_precision_score(frame[label],frame.probability),n_patients=frame.patient_id.nunique(),n_clips=len(frame))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,default=ROOT.parent/'data/final_test.csv');p.add_argument('--predictions',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/input_studies_20260923/smith_evaluation');a=p.parse_args()
 meta=pd.read_csv(a.labels,dtype={'short_recording_id':str}).rename(columns={'short_recording_id':'recording_id','patient_id':'metadata_patient_id'})
 assert len(meta)==200 and meta.recording_id.is_unique
 meta['patient_label']=meta.case_control_label.eq('CASE').astype(int);meta['expert_label']=meta.human_label.astype(int)
 rows=[]
 for path in sorted(a.predictions.glob('*_routine_predictions.csv')):
  f=pd.read_csv(path,dtype={'recording_id':str}).merge(meta[['recording_id','metadata_patient_id','patient_label','expert_label']],on='recording_id',validate='one_to_one')
  assert len(f)==200 and (f.patient_id==f.metadata_patient_id).all() and (f.label==f.patient_label).all()
  if 'decision_score' not in f:
   f=f.rename(columns={'probability':'decision_score'})
   f['probability']=expit(f.decision_score.to_numpy(float))
  else:
   np.testing.assert_allclose(f.probability,expit(f.decision_score),atol=1e-12,rtol=1e-12)
  for target in ['patient_label','expert_label']:rows.append(dict(model=path.name.removesuffix('_routine_predictions.csv'),target=target,**evaluate(f,target)))
 out=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/diagnosis_dual_labels_20260924';out.mkdir(parents=True,exist_ok=True)
 result=pd.DataFrame(rows);result.to_csv(out/'metrics.csv',index=False)
 text=['# Diagnosis: patient and expert clip labels','', 'Same saved five-fold predictions on the 200 expert-annotated 30-minute Routine Clips. Models were trained using patient diagnosis. AUROC compares predictions within each fold and weights folds by positive-negative clip pairs. Intervals resample patients within fold and patient-diagnosis strata; both clips always move together, even when expert labels differ. Logistic scores are converted to probabilities; balanced accuracy uses the fixed 0.5 probability threshold.','', '| Model | Target | AUROC [95% CI] | Balanced accuracy [95% CI] |','| --- | --- | --- | --- |']
 for r in rows:text.append(f"| {r['model']} | {r['target']} | {r['auroc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['balanced_accuracy']:.3f} [{r['balanced_accuracy_ci_low']:.3f}, {r['balanced_accuracy_ci_high']:.3f}] |")
 (out/'results.md').write_text('\n'.join(text)+'\n')
 (out/'protocol.json').write_text(json.dumps(dict(bootstrap_draws=2000,seed=20260923,threshold=.5,threshold_scale='probability',label_disagreements=int((meta.patient_label!=meta.expert_label).sum()),prediction_models=len(rows)//2),indent=2)+'\n')
 print(result.to_string(index=False))
if __name__=='__main__':main()
