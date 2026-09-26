#!/usr/bin/env python
"""Fold-matched response aggregation: 30 random crops per long EEG.

Encoders are reused, never retrained. Features stay in RAM; persistent models
live in the explicit checkpoint directory. All test predictions remain local.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'legacy/analysis')]
from finetune_response_windows import WindowPredictor
from aggregate_finetuned_biot import prepare_windows,BatchedRecordingHead,LEARNED,SCORE
from iesseeg_paper.input_studies import read_interval
from iesseeg_paper.evaluation import require_cuda
from iesseeg_paper.splits import validate_patient_folds
from qeeg_window_scale import fit_classic,threshold,metrics
from run_duration_followup import GRIDS,aggregate
SEED=20260924

class Crops(Dataset):
    def __init__(self,frame,model,seconds):self.rows=frame.to_dict('records');self.model=model;self.seconds=seconds
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];used=1800//self.seconds*self.seconds
        raw=read_interval(Path(r['source']),float(r['start_seconds']),used)
        return prepare_windows(raw,self.model,self.seconds)

def worker_init(_):torch.set_num_threads(1)

def encode(a,frame,fold,settings):
    if a.branch=='finetuned':
        payload=torch.load(a.encoder_root/a.model/a.visit/f'fold{fold}'/'best.pt',map_location='cpu',weights_only=False)
        protocol=payload['protocol'];folds=frame.fold.to_numpy()
        for key,mask in [('train_patients',~np.isin(folds,[fold,(fold+1)%5])),('validation_patients',folds==(fold+1)%5),('test_patients',folds==fold)]:
            assert set(protocol[key])==set(frame.loc[mask,'patient_id']),key
        assert protocol['task']=='sustained' and protocol['condition']==a.visit and protocol['test_fold']==fold
    torch.manual_seed(SEED)
    model=WindowPredictor(a.model,a.workspace,a.device,settings)
    if a.branch=='finetuned':model.load_state_dict(payload['model'])
    model.eval()
    if a.branch=='finetuned':del payload
    loader=DataLoader(Crops(frame,a.model,settings['window_seconds']),batch_size=None,num_workers=a.workers,
        worker_init_fn=worker_init,multiprocessing_context='spawn' if a.workers else None,pin_memory=True)
    bank=[];logits=[];start=time.time()
    with torch.inference_mode():
        for i,windows in enumerate(loader):
            hs=[];zs=[]
            for w in windows.split(a.inference_batch_size):
                h=model.encode(w.to(a.device));hs.append(h.cpu().numpy());zs.append(model.head(h).flatten().cpu().numpy())
            bank.append(np.concatenate(hs));logits.append(np.concatenate(zs))
            if (i+1)%100==0:print('ENCODE',a.model,a.branch,a.visit,fold,i+1,len(frame),'seconds',round(time.time()-start),flush=True)
    del model;torch.cuda.empty_cache()
    x=np.stack(bank).astype('float32');z=np.stack(logits).astype('float32')
    if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Nonfinite model output')
    return x,z

def neural(x,y,tr,va,te,method,a,path,fold):
    torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold)
    center=x[tr].mean((0,1),dtype=np.float64);scale=x[tr].std((0,1),dtype=np.float64);scale[scale<1e-6]=1
    xx=torch.as_tensor(np.clip((x-center)/scale,-10,10).astype('float32'),device=a.device)
    yy=torch.as_tensor(y,dtype=torch.float32,device=a.device)
    window=method=='window_head'
    net=(torch.nn.Sequential(torch.nn.LayerNorm(x.shape[-1]),torch.nn.Linear(x.shape[-1],1)) if window else BatchedRecordingHead(x.shape[-1],method)).to(a.device)
    def forward(ids,train=False):
        if window:return net(xx[ids]).squeeze(-1)
        return net.forward_batch(xx[ids],yy[ids].long(),train)
    def predict(ids):
        net.eval()
        with torch.inference_mode():return np.concatenate([forward(v).cpu().numpy() for v in np.array_split(ids,math.ceil(len(ids)/64))])
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-3)
    weight=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()),device=a.device)
    best=-np.inf;bad=0;bestep=0;state=None
    for ep in range(1,a.epochs+1):
        net.train()
        for ids in np.array_split(rng.permutation(tr),math.ceil(len(tr)/64)):
            opt.zero_grad(set_to_none=True);z=forward(ids,True);target=yy[ids,None].expand_as(z) if window else yy[ids]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target,pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        v=predict(va);score=roc_auc_score(np.repeat(y[va],x.shape[1]) if window else y[va],v.ravel())
        if score>best+1e-5:best=float(score);bad=0;bestep=ep;state={k:v.cpu().clone() for k,v in net.state_dict().items()}
        else:bad+=1
        if bad>=a.patience:break
    net.load_state_dict(state);v=predict(va);s=predict(te)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,center=center,scale=scale,method=method,input_dim=x.shape[-1],best_epoch=bestep,validation_auroc=best,train_indices=tr,validation_indices=va,test_indices=te),path)
    net.load_state_dict(torch.load(path,map_location=a.device,weights_only=False)['state_dict'])
    np.testing.assert_allclose(s,predict(te),rtol=1e-5,atol=1e-6)
    allz=predict(np.arange(len(x))) if window else None
    del xx,net;torch.cuda.empty_cache()
    return s,v,dict(best_epoch=bestep,stopped_epoch=ep,validation_auroc=best),allz

def score_aggregate(z,method):
    p=1/(1+np.exp(-np.clip(z,-80,80)))
    if method=='mean_probability':return p.mean(1)
    if method=='mean_logit':return z.mean(1)
    if method=='max_probability':return p.max(1)
    if method=='median_probability':return np.median(p,axis=1)
    if method=='top10_probability':return np.sort(p,axis=1)[:,-max(1,math.ceil(.1*p.shape[1])):].mean(1)
    hist=np.stack([np.histogram(row,bins=np.linspace(0,1,11))[0]/len(row) for row in p]).astype('float32')
    if method=='raw':return p
    if method=='histogram':return hist
    if method=='hybrid':return np.c_[p,hist]
    raise ValueError(method)

def candidates():
    return [('score',m,'score') for m in SCORE]+[('arbitration',m,'logistic') for m in ['raw','histogram','hybrid']]+[('feature',m,c) for m in ['mean','mean_std'] for c in GRIDS]+[('learned',m,'learned') for m in LEARNED]

def evaluate(a,frame,x,z,fold,spec,duration,key=None):
    family,method,classifier=spec;count=duration//a.seconds
    if count<1:raise ValueError('Duration shorter than one encoder window')
    actual=count*a.seconds;x=x[:,:count];z=z[:,:count]
    folds=frame.fold.to_numpy();y=frame.label.to_numpy(int)
    tr=np.flatnonzero(~np.isin(folds,[fold,(fold+1)%5]));va=np.flatnonzero(folds==(fold+1)%5);te=np.flatnonzero(folds==fold)
    assert (len(tr),len(va),len(te))==(900,300,300)
    key=key or '__'.join(spec);dest=a.output/'fits'/f'T{duration}__{key}';dest.mkdir(parents=True,exist_ok=True)
    predfile=dest/f'fold{fold}.csv';cfgfile=dest/f'fold{fold}.json'
    if predfile.exists() and cfgfile.exists():return json.loads(cfgfile.read_text())
    ckpt=a.checkpoints/f'T{duration}__{key}'/f'fold{fold}'
    started=time.time()
    if family=='score':
        scores=score_aggregate(z,method);s,v=scores[te],scores[va];cfg=dict(validation_auroc=float(roc_auc_score(y[va],v)))
    elif family=='learned':s,v,cfg,_=neural(x,y,tr,va,te,method,a,ckpt.with_suffix('.pt'),fold)
    else:
        features=aggregate(x,method) if family=='feature' else score_aggregate(z,method)
        s,v,cfg=fit_classic(features[tr],y[tr],features[va],y[va],features[te],classifier,a.device,SEED+fold,ckpt.with_suffix('.joblib'))
    cut=threshold(y[va],v)
    out=frame.iloc[te][['patient_id','fold','label','crop_id']].copy();out['probability']=s;out['predicted_label']=(s>=cut).astype(int)
    out.to_csv(predfile,index=False)
    cfg.update(fold=fold,family=family,method=method,classifier=classifier,duration_requested=duration,duration_used=actual,threshold=cut,seconds=time.time()-started)
    cfgfile.write_text(json.dumps(cfg,indent=2)+'\n')
    print('FIT',a.model,a.branch,a.visit,fold,duration,key,'val',round(cfg['validation_auroc'],3),'seconds',round(cfg['seconds']),flush=True)
    return cfg

def summarize(a):
    rows=[]
    for dest in sorted((a.output/'fits').glob('*')):
        if len(list(dest.glob('fold*.csv')))!=5:continue
        cfg=[json.loads((dest/f'fold{f}.json').read_text()) for f in range(5)]
        pred=pd.concat([pd.read_csv(dest/f'fold{f}.csv') for f in range(5)],ignore_index=True)
        validate_patient_folds(pred,'sustained',True)
        stat,draws=metrics(pred)
        stat.update(model=a.model,branch=a.branch,visit=a.visit,procedure=dest.name,window_seconds=a.seconds,
                    duration_seconds=cfg[0]['duration_used'],mean_validation_auroc=float(np.mean([c['validation_auroc'] for c in cfg])))
        rows.append(stat);(dest/'summary.json').write_text(json.dumps(stat,indent=2)+'\n');np.save(dest/'bootstrap.npy',draws)
    a.results.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.results/f'{a.model}_{a.branch}_{a.visit}.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--branch',choices=['frozen','finetuned'],required=True);p.add_argument('--visit',choices=['PRE','POST'],required=True)
    p.add_argument('--workspace',type=Path,default=ROOT.parent);p.add_argument('--coordinates',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/qeeg_window_scale_30crops_20260924/coordinates.csv')
    p.add_argument('--storage',type=Path,default=Path(__import__('os').environ.get('IESSEEG_CHECKPOINT_ROOT', 'work/checkpoints') + ''))
    p.add_argument('--device',required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--inference-batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--folds',nargs='+',type=int,default=list(range(5)));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.device=require_cuda(a.device);torch.set_num_threads(1)
    a.encoder_root=a.storage/('sustained_window_finetuning_20260924' if a.model in ['biot','labram','cbramod'] else 'remaining_encoder_finetuning_20260924')/'runs'
    payload=torch.load(a.encoder_root/a.model/a.visit/'fold0/best.pt',map_location='cpu',weights_only=False)
    settings=payload['protocol']['config']['models'][a.model];a.seconds=settings['window_seconds'];del payload
    name=f'{a.model}_{a.branch}_{a.visit}'+('_smoke' if a.smoke else '')
    a.output=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/response_multicrop_20260924'/name;a.output.mkdir(parents=True,exist_ok=True)
    a.checkpoints=a.storage/'response_multicrop_20260924'/name
    a.results=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/response_multicrop_20260924'
    frame=pd.read_csv(a.coordinates,dtype={'recording_id':str});frame=frame[frame.pre_post_treatment_label.eq(a.visit)].reset_index(drop=True)
    assert len(frame)==1500 and frame.groupby('patient_id').size().eq(30).all()
    frame.to_csv(a.output/'coordinates.csv',index=False)
    protocol=dict(model=a.model,branch=a.branch,visit=a.visit,window_seconds=a.seconds,epochs=a.epochs,patience=a.patience,batch_size=64,seed=SEED,validation='(fold+1)%5',refit=False,crops_per_recording=30,duration_budgets=[30,60,120,300,600,1800],remainder='discard_incomplete_window',curve_selection='best learned and best non-score procedure per validation fold; mean probability reference',frozen_window_head='LayerNorm + linear, inherited-window BCE, frozen backbone',features='transient RAM; deterministically recomputed on resume')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frozen=None
    for fold in a.folds:
        done=a.output/f'fold{fold}_complete.json'
        if done.exists():continue
        if a.smoke:
            # Preserve all patients for split validation; extract one crop each.
            small=frame.groupby('patient_id',sort=False).head(1).reset_index(drop=True)
            x,z=encode(a,small,fold,settings)
            print('SMOKE',x.shape,z.shape,flush=True);return
        if a.branch=='frozen' and frozen is not None:x,z=frozen
        else:
            x,z=encode(a,frame,fold,settings)
            if a.branch=='frozen':frozen=(x,z.copy())
        f=frame.fold.to_numpy();y=frame.label.to_numpy(int);tr=np.flatnonzero(~np.isin(f,[fold,(fold+1)%5]));va=np.flatnonzero(f==(fold+1)%5);te=np.flatnonzero(f==fold)
        if a.branch=='frozen':
            _,_,_,z=neural(x,y,tr,va,te,'window_head',a,a.checkpoints/f'window_head_fold{fold}.pt',fold)
        records=[]
        for spec in candidates():records.append(evaluate(a,frame,x,z,fold,spec,1800))
        selected=[]
        for group in ['learned','other']:
            pool=[r for r in records if (r['family']=='learned' if group=='learned' else r['family'] in ['feature','arbitration'])]
            winner=max(pool,key=lambda r:round(r['validation_auroc'],12));selected.append((group,(winner['family'],winner['method'],winner['classifier'])))
        for duration in [30,60,120,300,600]:
            evaluate(a,frame,x,z,fold,('score','mean_probability','score'),duration)
            for group,spec in selected:evaluate(a,frame,x,z,fold,spec,duration,key='validation_selected_'+group)
        done.write_text(json.dumps(dict(fold=fold,selected=selected),indent=2)+'\n')
        summarize(a)
        if a.branch=='finetuned':del x,z
    summarize(a);(a.output/'complete.json').write_text('{"complete": true}\n')
    print('COMPLETE',name,flush=True)

if __name__=='__main__':main()
