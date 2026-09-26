#!/usr/bin/env python
"""Frozen window heads and saved full-tuned windows, same probability-mean rule.
Row-level files stay local; checkpoints are external. No encoder retraining.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from iesseeg_paper.splits import load_fold_manifest,validate_patient_folds
from iesseeg.metrics import summarize
def evaluate(frame, label, threshold=.5):
 if threshold != .5: raise ValueError("The paper uses threshold 0.5")
 return summarize(frame,label)
MODELS=['biot','labram','cbramod','eegpt','luna','reve','codebrain','csbrain']
SEED=20260924

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--branches',nargs='+',choices=['frozen','finetuned'],default=['frozen','finetuned'])
 p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
 p.add_argument('--device',required=True)
 p.add_argument('--workspace',type=Path,default=ROOT.parent)
 p.add_argument('--features',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/features/long_frozen_representations')
 p.add_argument('--output',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/diagnosis_probability_mean_20260924')
 p.add_argument('--results',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/diagnosis_probability_mean_20260924')
 p.add_argument('--checkpoints',type=Path,default=Path(__import__('os').environ.get('IESSEEG_CHECKPOINT_ROOT', 'work/checkpoints') + '/diagnosis_probability_mean_20260924'))
 p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
 a=p.parse_args();assert a.device.startswith('cuda:') and torch.cuda.is_available()
 torch.cuda.set_device(a.device);torch.set_num_threads(2)
 for folder in [a.output,a.results,a.checkpoints]:folder.mkdir(parents=True,exist_ok=True)
 foldmap=load_fold_manifest('diagnosis').set_index('patient_id').fold
 meta=pd.read_csv(a.workspace/'data/final_test.csv',dtype={'short_recording_id':str}).rename(columns={'short_recording_id':'recording_id'})
 meta.recording_id=meta.recording_id.str.removesuffix('.0');meta['patient_label']=meta.case_control_label.eq('CASE').astype(int);meta['expert_label']=meta.human_label.astype(int);meta['fold']=meta.patient_id.map(foldmap)
 assert len(meta)==200 and meta.recording_id.is_unique
 dev=pd.read_csv(a.workspace/'data/final_short_merged.csv',dtype={'short_recording_id':str}).rename(columns={'short_recording_id':'recording_id'})
 dev=dev[dev.pre_post_treatment_label.eq('PRE')].copy();dev.recording_id=dev.recording_id.str.removesuffix('.0');dev['fold']=dev.patient_id.map(foldmap);dev['label']=dev.case_control_label.eq('CASE').astype(int)
 assert len(dev)==400 and dev.recording_id.is_unique
 for model in a.models:
  out=a.output/model;out.mkdir(exist_ok=True)
  if (out/('_'.join(a.branches)+'_complete.json')).exists():print('SKIP',model,flush=True);continue
  start=time.time();manifest=None
  if 'frozen' in a.branches:
   manifest=pd.read_csv(a.features/model/'manifest.csv',dtype={'recording_id':str}).set_index('recording_id')
  def load_bank(frame):
   bank=[];index=[]
   for i,r in enumerate(frame.itertuples()):
    x=np.load(a.features/model/f'{r.recording_id}.npy')
    assert x.ndim==2 and len(x)>0 and np.isfinite(x).all()
    assert int(manifest.loc[r.recording_id,'patient_id'])==r.patient_id
    bank.append(x);index.extend([i]*len(x))
   return np.concatenate(bank).astype('float32'),np.asarray(index)
  if 'frozen' in a.branches:
   xd,di=load_bank(dev);xt,ti=load_bank(meta)
  # Recompute every full-tuned clip score from its saved windows, not a clip-label prediction.
  full=[];source=[]
  if 'finetuned' in a.branches:
   for fold in range(5):
    base='kfold_split' if model in ['biot','labram','cbramod'] else 'baselines_release'
    path=a.workspace/base/'baselines'/model/'result/inference'/f'case_control_fold{fold}'/'inference_results_window.csv'
    f=pd.read_csv(path);rid=next(c for c in f if 'recording_id' in c);f=f.rename(columns={rid:'recording_id'})
    f.recording_id=f.recording_id.astype(str).str.removesuffix('.0')
    assert f.pred_prob.between(0,1).all() and f.pred_prob.notna().all()
    assert not f.duplicated(['recording_id','start_ind','end_ind']).any()
    if 'patient_id' in f:
     patient_lookup=meta.set_index('recording_id').patient_id
     assert (f.patient_id.astype(int)==f.recording_id.map(patient_lookup)).all()
    assert (f.groupby('recording_id').known_label.nunique()==1).all()
    g=f.groupby('recording_id').agg(probability=('pred_prob','mean'),n_windows=('pred_prob','size'),known_label=('known_label','first')).reset_index()
    g=g.merge(meta[['recording_id','patient_id','patient_label','expert_label','fold']],validate='one_to_one')
    assert len(g)==40 and g.fold.eq(fold).all() and (g.known_label.astype(int)==g.patient_label).all()
    if manifest is not None:
     assert (g.n_windows.to_numpy()==manifest.loc[g.recording_id,'n_windows'].to_numpy()).all(), 'Frozen/full window counts differ'
    original=pd.read_csv(path.with_name('inference_results.csv'))
    rid=next(c for c in original if 'recording_id' in c);original=original.rename(columns={rid:'recording_id'});original.recording_id=original.recording_id.astype(str).str.removesuffix('.0')
    merged=g.merge(original[['recording_id','pred_prob']],validate='one_to_one')
    np.testing.assert_allclose(merged.probability,merged.pred_prob,atol=1e-7,rtol=1e-7)
    full.append(g);source.append(str(path.relative_to(a.workspace)))
   full=pd.concat(full,ignore_index=True);validate_patient_folds(full,'diagnosis',require_complete=True);full.to_csv(out/'finetuned_predictions.csv',index=False)
  predictions=[];configs=[]
  if 'frozen' in a.branches:
   dfold=dev.fold.to_numpy()[di];ydev=dev.label.to_numpy()[di]
   for fold in range(5):
    saved=out/f'frozen_fold{fold}.csv';cfgpath=out/f'frozen_fold{fold}.json'
    if saved.exists() and cfgpath.exists():predictions.append(pd.read_csv(saved,dtype={'recording_id':str}));configs.append(json.loads(cfgpath.read_text()));continue
    torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold)
    tr=np.flatnonzero(~np.isin(dfold,[fold,(fold+1)%5]));va=np.flatnonzero(dfold==(fold+1)%5);te=np.flatnonzero(meta.fold.to_numpy()[ti]==fold)
    center=xd[tr].mean(0,dtype=np.float64);scale=xd[tr].std(0,dtype=np.float64);scale[scale<1e-6]=1
    x=torch.as_tensor(np.clip((xd-center)/scale,-10,10).astype('float32'),device=a.device)
    test=torch.as_tensor(np.clip((xt[te]-center)/scale,-10,10).astype('float32'),device=a.device)
    y=torch.as_tensor(ydev,dtype=torch.float32,device=a.device)
    net=torch.nn.Sequential(torch.nn.LayerNorm(x.shape[1]),torch.nn.Linear(x.shape[1],1)).to(a.device)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
    weight=torch.tensor(float((ydev[tr]==0).sum()/(ydev[tr]==1).sum()),device=a.device)
    def prob(data):
     net.eval()
     with torch.inference_mode():return torch.cat([net(chunk).flatten().sigmoid() for chunk in data.split(4096)]).cpu().numpy()
    best=-np.inf;bad=0
    for epoch in range(1,a.epochs+1):
     net.train()
     for ids in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/512)))):
      opt.zero_grad(set_to_none=True);z=net(x[ids]).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y[ids],pos_weight=weight)
      assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
     vp=prob(x[va]);v=pd.DataFrame({'clip':di[va],'prob':vp}).groupby('clip').prob.mean();auc=roc_auc_score(dev.label.to_numpy()[v.index],v)
     if auc>best+1e-5:best=float(auc);bad=0;bestep=epoch;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
     else:bad+=1
     if bad>=a.patience:break
    net.load_state_dict(state);probs=prob(test)
    cp=a.checkpoints/model/f'fold{fold}.pt';cp.parent.mkdir(parents=True,exist_ok=True)
    cfg=dict(fold=fold,best_epoch=bestep,stopped_epoch=epoch,validation_clip_auroc=best,seed=SEED+fold,train_patients=sorted(dev.loc[~dev.fold.isin([fold,(fold+1)%5]),'patient_id'].unique().tolist()),validation_patients=sorted(dev.loc[dev.fold.eq((fold+1)%5),'patient_id'].unique().tolist()),test_patients=sorted(meta.loc[meta.fold.eq(fold),'patient_id'].unique().tolist()))
    torch.save(dict(state_dict=state,center=center,scale=scale,protocol=cfg),cp)
    net.load_state_dict(torch.load(cp,map_location=a.device,weights_only=False)['state_dict']);np.testing.assert_allclose(probs,prob(test),atol=1e-7,rtol=1e-6)
    wp=pd.DataFrame({'recording_id':meta.recording_id.to_numpy()[ti[te]],'window_index':np.arange(len(te)),'probability':probs});wp.to_csv(out/f'frozen_fold{fold}_windows.csv',index=False)
    g=wp.groupby('recording_id').agg(probability=('probability','mean'),n_windows=('probability','size')).reset_index().merge(meta[['recording_id','patient_id','fold','patient_label','expert_label']],validate='one_to_one')
    assert len(g)==40;g.to_csv(saved,index=False);cfgpath.write_text(json.dumps(cfg,indent=2)+'\n');predictions.append(g);configs.append(cfg)
    del x,test,y,net,opt;torch.cuda.empty_cache();print('FOLD',model,fold,'best_epoch',bestep,'val',round(best,3),'elapsed',round(time.time()-start),flush=True)
   frozen=pd.concat(predictions,ignore_index=True);validate_patient_folds(frozen,'diagnosis',require_complete=True);frozen.to_csv(out/'frozen_predictions.csv',index=False)
  rows=[]
  for branch,frame in ([('frozen',frozen)] if 'frozen' in a.branches else [])+([('finetuned',full)] if 'finetuned' in a.branches else []):
   for target in ['patient_label','expert_label']:rows.append(dict(model=model,branch=branch,target=target,aggregation='mean_window_probability',**evaluate(frame,target,threshold=.5)))
  pd.DataFrame(rows).to_csv(a.results/f'{model}.csv',index=False)
  (out/'protocol.json').write_text(json.dumps(dict(model=model,frozen_head='train-window standardization and clipping [-10,10]; LayerNorm + Linear; inherited-window BCE',optimizer='AdamW',learning_rate=3e-4,weight_decay=1e-3,batch_size=512,max_epochs=a.epochs,patience=a.patience,selection='validation clip AUROC after probability averaging',refit=False,threshold=.5,full_tuned_sources=source,full_tuned_status='saved supervised windows; recomputed mean verified against original clip predictions; original backbone training recipes retained',windows='model-native preprocessing and encoding lengths retained; frozen/full counts compared when both branches are requested',fold_configs=configs),indent=2)+'\n')
  (out/('_'.join(a.branches)+'_complete.json')).write_text('{"complete": true}\n');print('COMPLETE',model,'seconds',round(time.time()-start),flush=True)
if __name__=='__main__':main()
