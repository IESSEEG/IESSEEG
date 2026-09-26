#!/usr/bin/env python
"""Cross-window aggregation of fold-specific fine-tuned EEG representations.

Read best task-specific response checkpoints; fit heads on the same 30 training
patients and select with the same 10 validation patients. No outer-fold refit.
One prespecified continuous excerpt per original long recording, shared starts
and nested duration budgets. Never mix representations from different encoders
inside a fold. Individual outputs remain local; export only reviewed aggregates.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'legacy/analysis')]
from finetune_response_windows import WindowPredictor
from experiments import additional_response_encoders as extra
from iesseeg_paper.input_studies import read_interval,prepare_encoder_input
from iesseeg_paper.data import patient_info
from iesseeg_paper.splits import validate_patient_folds
from run_clip_response_screen import RecordingHead,METHODS,SCALES
from run_duration_followup import aggregate,make_classifier,predict_classifier,GRIDS
from clip_followup_metrics import summary
from iesseeg_paper.evaluation import standardize

SEED=20260924
LEARNED=[m for m in METHODS if m!='timemil']
SCORE=['mean_probability','mean_logit','max_probability','top10_probability','median_probability']


class BatchedRecordingHead(RecordingHead):
    """Same equations as RecordingHead, evaluated on equal-length bags in batches."""
    def forward_batch(self,x,labels=None,update=False):
        if self.method=='psmil':
            return torch.stack([self.forward(v,label=int(labels[i]) if labels is not None else None,
                update_psmil=update) for i,v in enumerate(x)])
        h=self.adapter(x)
        if self.method=='deepsets':return self.deepset_rho(self.deepset_phi(h).mean(1)).flatten()
        if self.method.startswith('millet_'):return self.millet(h).reshape(-1)
        if self.method=='temporal_attention':h=torch.nn.functional.gelu(self.temporal(h.transpose(1,2))).transpose(1,2)+h
        z=self.local_head(h).squeeze(-1)
        if self.method=='mean':out=z.mean(1)
        elif self.method=='max':out=z.max(1).values
        elif self.method=='top10':out=z.topk(max(1,math.ceil(.1*z.shape[1])),dim=1).values.mean(1)
        elif self.method in ['attention','temporal_attention']:
            w=torch.softmax(self.attention_w(torch.tanh(self.attention_v(h))*torch.sigmoid(self.attention_u(h))).squeeze(-1),dim=1)
            out=(w*z).sum(1)
        elif self.method=='multiscale':
            ordered=z.sort(dim=1,descending=True).values
            pooled=torch.stack([ordered[:,:max(1,math.ceil(s*z.shape[1]))].mean(1) for s in SCALES],1)
            out=pooled@torch.softmax(self.scale_logits,dim=0)
        else:raise ValueError(self.method)
        return out+self.bias


def prepare_windows(signal,model,seconds):
    """Match fine-tuning preprocessing, including per-window filtering for LaBraM and CBraMod."""
    samples=seconds*200
    assert signal.shape[-1]%samples==0
    if model=='biot':
        x=prepare_encoder_input(signal,model)
        windows=x.reshape(18,-1,samples).transpose(1,0,2).copy()
        windows=windows/(np.quantile(np.abs(windows),.95,axis=-1,keepdims=True)+1e-8)
    elif model in extra.NAMES:
        windows=np.stack([extra.prepare(signal[:,i:i+samples],model,seconds)
                          for i in range(0,signal.shape[-1],samples)])
    else:
        windows=np.stack([prepare_encoder_input(signal[:,i:i+samples],model).reshape(19,seconds,200)/100.
                          for i in range(0,signal.shape[-1],samples)])
    windows=windows.astype('float32')
    if not np.isfinite(windows).all():raise ValueError('Nonfinite encoder input')
    return windows


def extract(args,device):
    meta=pd.read_csv(args.manifest,dtype={'recording_id':str})
    meta=meta[(meta.kind=='long')&(meta.case_control_label=='CASE')].copy()
    pt=patient_info(dict(workspace=args.workspace),args.task)
    for visit in args.visits:
        frame=meta[meta.pre_post_treatment_label.eq(visit)].drop(columns=['label','fold'],errors='ignore')
        frame=frame.merge(pt,on='patient_id',validate='one_to_one').sort_values('patient_id').reset_index(drop=True)
        assert len(frame)==50 and frame.eligible.all()
        dest=args.output/'features'/visit;dest.mkdir(parents=True,exist_ok=True)
        frame[['patient_id','recording_id','source','source_duration','start_seconds','fold','label']].to_csv(dest/'coordinates.csv',index=False)
        pending=[f for f in range(5) if not (dest/f'fold{f}.npz').exists()]
        if not pending:continue
        raw=[]
        for i,r in frame.iterrows():
            signal=read_interval(Path(r.source),r.start_seconds,max(args.durations))
            windows=prepare_windows(signal,args.model,args.window_seconds)
            raw.append(windows)
            if (i+1)%10==0:print(f'INPUT {visit} {i+1}/50',flush=True)
        for fold in pending:
            ckpt=args.encoders/args.model/visit/f'fold{fold}'/'best.pt'
            payload=torch.load(ckpt,map_location='cpu',weights_only=False)
            protocol=payload['protocol']
            settings=protocol['config']['models'][args.model]
            assert settings['window_seconds']==args.window_seconds
            assert protocol['task']==args.task and protocol['test_fold']==fold and protocol['condition']==visit
            assert set(protocol['train_patients'])==set(frame[~frame.fold.isin([fold,(fold+1)%5])].patient_id)
            assert set(protocol['validation_patients'])==set(frame[frame.fold.eq((fold+1)%5)].patient_id)
            assert set(protocol['test_patients'])==set(frame[frame.fold.eq(fold)].patient_id)
            model=WindowPredictor(args.model,args.workspace,device,settings);model.load_state_dict(payload['model']);model.eval()
            embeddings=[];logits=[]
            with torch.inference_mode():
                for windows in raw:
                    hs=[];zs=[]
                    for start in range(0,len(windows),args.inference_batch_size):
                        x=torch.as_tensor(windows[start:start+args.inference_batch_size],device=device)
                        h=model.encode(x);z=model.head(h).flatten()
                        hs.append(h.cpu().numpy());zs.append(z.cpu().numpy())
                    embeddings.append(np.concatenate(hs));logits.append(np.concatenate(zs))
            path=dest/f'fold{fold}.npz'
            np.savez_compressed(path,embeddings=np.stack(embeddings),logits=np.stack(logits),
                patient_id=frame.patient_id.to_numpy(),labels=frame.label.to_numpy(),folds=frame.fold.to_numpy(),
                starts=frame.start_seconds.to_numpy(),encoder_best_epoch=payload['epoch'])
            print(f'FEATURES {visit} fold{fold} {np.stack(embeddings).shape}',flush=True)
            del model;torch.cuda.empty_cache()
        del raw


def scale_windows(x,train):
    mean=x[train].astype('float64').mean((0,1));std=x[train].astype('float64').std((0,1));std[std<1e-6]=1.
    return np.clip((x-mean)/std,-10,10).astype('float32'),(mean,std)


def fit_learned(x,y,train,val,test,method,device,seed,args,ckpt):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    scaled,scaler=scale_windows(x,train)
    xx=torch.as_tensor(scaled,device=device);yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    model=BatchedRecordingHead(x.shape[-1],method).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    weight=torch.tensor(float((y[train]==0).sum()/(y[train]==1).sum()),device=device)
    best=-float('inf');bad=0;best_epoch=0;state=None
    for epoch in range(1,args.epochs+1):
        model.train()
        for ids in np.array_split(rng.permutation(train),math.ceil(len(train)/8)):
            opt.zero_grad(set_to_none=True)
            z=model.forward_batch(xx[ids],yy[ids].long(),True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yy[ids],pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite learned aggregation loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.inference_mode():v=model.forward_batch(xx[val]).cpu().numpy()
        auc=float(roc_auc_score(y[val],v))
        if auc>best+1e-5:
            best=auc;bad=0;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        if bad>=args.patience:break
    model.load_state_dict(state);model.eval()
    with torch.inference_mode():score=model.forward_batch(xx[test]).cpu().numpy()
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,scaler=scaler,method=method,input_dim=x.shape[-1],best_epoch=best_epoch,
        validation_auroc=best,train_indices=train,val_indices=val,test_indices=test),ckpt)
    restored=BatchedRecordingHead(x.shape[-1],method).to(device)
    restored.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)['state_dict']);restored.eval()
    with torch.inference_mode():np.testing.assert_allclose(score,restored.forward_batch(xx[test]).cpu().numpy(),rtol=1e-5,atol=1e-6)
    return score,dict(best_epoch=best_epoch,stopped_epoch=epoch,validation_auroc=best)


def fit_classical(x,y,train,val,test,method,device,seed,ckpt):
    a,b,scaler=standardize(x[train],x[val])
    # standardize() defines its own serialization format; explicitly transform test below.
    best=-float('inf');winner=None;selected=None
    for hp in GRIDS[method]:
        model=make_classifier(method,hp,len(train),x.shape[-1],y[train],seed,device)
        model.fit(a,y[train]);value=round(float(roc_auc_score(y[val],predict_classifier(model,b))),12)
        if value>best:
            best=value;winner=model;selected=hp
    _,t,_=standardize(x[train],x[test])
    score=predict_classifier(winner,t)
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=winner,scaler=scaler,parameters=selected,train_indices=train,val_indices=val,test_indices=test),ckpt,compress=3)
    restored=joblib.load(ckpt)
    np.testing.assert_allclose(score,predict_classifier(restored['model'],t),rtol=1e-5,atol=1e-6)
    return score,dict(parameters=selected,validation_auroc=best)


def run(args,device):
    candidates=[('score',m,'score') for m in SCORE]+[('learned',m,'learned') for m in LEARNED]
    candidates += [('classical',m,a) for a in ['mean','mean_std'] for m in GRIDS]
    if args.families:candidates=[r for r in candidates if r[0] in args.families]
    for visit in args.visits:
        data=[dict(np.load(args.output/'features'/visit/f'fold{f}.npz')) for f in range(5)]
        for family,method,pooling in candidates:
            for duration in args.durations:
                key=f'{visit}__T{duration}__{family}__{pooling}__{method}'
                dest=args.output/'fits'/key;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'summary.json').exists():continue
                t=time.perf_counter();records=[];configs=[]
                for fold,z in enumerate(data):
                    x=z['embeddings'][:,:duration//args.window_seconds];logits=z['logits'][:,:duration//args.window_seconds]
                    y=z['labels'];folds=z['folds'];pids=z['patient_id']
                    train=np.flatnonzero((folds!=fold)&(folds!=(fold+1)%5));val=np.flatnonzero(folds==(fold+1)%5);test=np.flatnonzero(folds==fold)
                    assert (len(train),len(val),len(test))==(30,10,10)
                    if family=='score':
                        prob=1/(1+np.exp(-np.clip(logits,-80,80)))
                        if method=='mean_probability':score=prob[test].mean(1)
                        elif method=='mean_logit':score=logits[test].mean(1)
                        elif method=='max_probability':score=prob[test].max(1)
                        elif method=='median_probability':score=np.median(prob[test],axis=1)
                        else:score=np.sort(prob[test],axis=1)[:,-max(1,math.ceil(prob.shape[1]*.1)):].mean(1)
                        config={}
                    elif family=='learned':
                        score,config=fit_learned(x,y,train,val,test,method,device,SEED+fold+100*LEARNED.index(method),args,args.checkpoints/key/f'fold{fold}.pt')
                    else:
                        score,config=fit_classical(aggregate(x,pooling),y,train,val,test,method,device,SEED+fold,args.checkpoints/key/f'fold{fold}.joblib')
                    configs.append(dict(fold=fold,**config))
                    records.extend(dict(patient_id=int(pids[i]),fold=fold,label=int(y[i]),probability=float(s)) for i,s in zip(test,score))
                frame=pd.DataFrame(records);validate_patient_folds(frame,args.task,True)
                metrics=summary(frame)
                result=dict(model=args.model+'_finetuned',task=args.task,visit=visit,duration_seconds=duration,
                    window_seconds=args.window_seconds,family=family,method=method,pooling=pooling,seconds=time.perf_counter()-t,
                    patient_auroc=metrics['fold_comparable_clip_auroc'],ci_low=metrics['clustered_ci_low'],ci_high=metrics['clustered_ci_high'],
                    pooled_oof_auroc=metrics['pooled_oof_clip_auroc'],n_patients=metrics['n_patients'])
                frame.to_csv(dest/'predictions.csv',index=False)
                (dest/'selection.json').write_text(json.dumps(configs,indent=2)+'\n')
                (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
                print('RESULT '+json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--encoders',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--device',required=True)
    p.add_argument('--model',choices=['biot','labram','cbramod','reve','csbrain','codebrain'],default='biot')
    p.add_argument('--inference-batch-size',type=int,default=None)
    p.add_argument('--task',choices=['sustained','immediate'],default='sustained')
    p.add_argument('--stage',choices=['extract','fit','all'],default='all')
    p.add_argument('--visits',nargs='+',default=['PRE','POST'],choices=['PRE','POST'])
    p.add_argument('--durations',nargs='+',type=int,default=[30,60,120,300,600,1800])
    p.add_argument('--families',nargs='+',choices=['score','learned','classical'])
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
    args=p.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):raise RuntimeError('CUDA required')
    args.window_seconds={'biot':30,'labram':16,'cbramod':30,'reve':30,'csbrain':30,'codebrain':30}[args.model]
    args.inference_batch_size=args.inference_batch_size or (60 if args.model=='biot' else 16)
    if any(t%args.window_seconds or t<args.window_seconds for t in args.durations):raise ValueError('Durations must tile complete native windows')
    device=torch.device(args.device);torch.cuda.set_device(device);torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ['extract','all']:extract(args,device)
    if args.stage in ['fit','all']:run(args,device)

if __name__=='__main__':main()
