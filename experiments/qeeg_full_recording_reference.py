#!/usr/bin/env python
"""Full original long-EEG qEEG reference, one feature vector per recording.

Uses the same features, fixed patient folds, classifier grids, and validation
rule as qeeg_window_scale.py. Reads each EDF from sample zero to its end.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,concurrent.futures,json,multiprocessing,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from iesseeg_paper.input_studies import read_interval,qeeg_features,RESPONSE
from iesseeg_paper.evaluation import require_cuda
from qeeg_window_scale import fit_classic,threshold,metrics,SEED

def initialize(device):require_cuda(device);torch.set_num_threads(1)
def extract(job):
 r,output,device=job;target=Path(output)/'features'/f"{r['pre_post_treatment_label']}_{r['recording_id']}.json"
 if target.exists():return str(target)
 start=time.time();signal=read_interval(Path(r['source']),0,None)
 actual=signal.shape[-1]/200
 if abs(actual-float(r['source_duration']))>1:raise ValueError('Full recording duration differs from source manifest')
 features=qeeg_features(signal,torch.device(device),diagnostic=False)
 features.update(recording_id=r['recording_id'],patient_id=int(r['patient_id']),visit=r['pre_post_treatment_label'],source=str(r['source']),start_seconds=0,elapsed_seconds=time.time()-start)
 target.write_text(json.dumps(features,indent=2)+'\n');return str(target)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--checkpoints',type=Path,required=True)
 p.add_argument('--coordinates',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/qeeg_window_scale_30crops_20260924/coordinates.csv');a=p.parse_args();require_cuda(a.device)
 out=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/qeeg_full_recording_20260924';(out/'features').mkdir(parents=True,exist_ok=True)
 result=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/qeeg_window_scale_30crops_20260924'
 frame=pd.read_csv(a.coordinates,dtype={'recording_id':str}).drop_duplicates(['pre_post_treatment_label','patient_id']).reset_index(drop=True)
 assert len(frame)==100
 frame.to_csv(out/'sources.csv',index=False)
 (out/'protocol.json').write_text(json.dumps(dict(input='entire original long EEG, sample zero to end',features=RESPONSE,task='sustained',validation='(test_fold+1)%5',refit=False,unit='recording',seed=SEED),indent=2)+'\n')
 jobs=[(r,str(out),a.device) for r in frame.to_dict('records') if not (out/'features'/f"{r['pre_post_treatment_label']}_{r['recording_id']}.json").exists()]
 print('FULL_RECORDINGS',len(jobs),'hours',frame.source_duration.sum()/3600,flush=True);start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(a.device,)) as pool:
  futures=[pool.submit(extract,j) for j in jobs]
  for i,f in enumerate(concurrent.futures.as_completed(futures),1):f.result();print('EXTRACT',i,len(jobs),'seconds',round(time.time()-start),flush=True)
 rows=[]
 for visit,part in frame.groupby('pre_post_treatment_label'):
  part=part.reset_index(drop=True);bank=[json.loads((out/'features'/f'{visit}_{rid}.json').read_text()) for rid in part.recording_id]
  x=np.array([[r[k] for k in RESPONSE] for r in bank]);y=part.label.to_numpy(int);folds=part.fold.to_numpy(int)
  for cls in ['ridge','logistic','xgboost']:
   preds=[];configs=[]
   for fold in range(5):
    tr=~np.isin(folds,[fold,(fold+1)%5]);va=folds==(fold+1)%5;te=folds==fold
    assert (tr.sum(),va.sum(),te.sum())==(30,10,10)
    s,v,cfg=fit_classic(x[tr],y[tr],x[va],y[va],x[te],cls,a.device,SEED+fold,a.checkpoints/f'{visit}_{cls}_fold{fold}.joblib')
    cut=.5 if cls=='logistic' else threshold(y[va],v)
    cfg.update(fold=fold,threshold=cut);configs.append(cfg)
    f=part.loc[te,['patient_id','recording_id','label','fold']].copy()
    if cls=='logistic':
     f['decision_score']=s
     s=expit(s)
    f['probability']=s;f['predicted_label']=(s>=cut).astype(int);preds.append(f)
   f=pd.concat(preds,ignore_index=True);stat,draws=metrics(f);stat.update(visit=visit,classifier=cls,unit='full_recording',duration_min_seconds=float(part.source_duration.min()),duration_max_seconds=float(part.source_duration.max()))
   rows.append(stat);f.to_csv(out/f'{visit}_{cls}_predictions.csv',index=False);np.save(out/f'{visit}_{cls}_bootstrap.npy',draws)
   (out/f'{visit}_{cls}_selection.json').write_text(json.dumps(configs,indent=2)+'\n')
   print('FIT',visit,cls,stat['auroc'],flush=True)
 pd.DataFrame(rows).to_csv(result/'full_recording_reference.csv',index=False)
 pd.DataFrame([{k:r.get(k) for k in ['recording_id','visit','duration_seconds','clean_seconds','clean_fraction','dfa_max_scale_seconds','pli_clean_epochs']} for r in [json.loads(f.read_text()) for f in (out/'features').glob('*.json')]]).to_csv(out/'feature_quality.csv',index=False)
 (out/'complete.json').write_text('{"complete":true}\n');print('COMPLETE full-recording reference',flush=True)
if __name__=='__main__':main()
