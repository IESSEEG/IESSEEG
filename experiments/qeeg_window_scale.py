#!/usr/bin/env python
"""qEEG window scale: single-window and fixed-30-minute aggregation performance.

30 random crops per recording, fixed patient folds, sustained response only.
No encoder training. Private coordinates/features/predictions remain local.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse, concurrent.futures, json, math, multiprocessing, sys, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
from iesseeg_paper.input_studies import read_interval,qeeg_features,RESPONSE,FS
from iesseeg_paper.data import patient_info
from iesseeg_paper.splits import validate_patient_folds
from iesseeg_paper.evaluation import require_cuda,standardize
from run_duration_followup import make_classifier,predict_classifier,GRIDS
from aggregate_finetuned_biot import BatchedRecordingHead
LENGTHS=(30,60,120,300,600,1800)
SEED=20260924


def initialize_worker(device):
    require_cuda(device);torch.set_num_threads(1)


def feature_path(root,r):
    return Path(root)/'features'/f"{r['pre_post_treatment_label']}_{r['recording_id']}_{r['crop_id']:02d}.npz"


def extract(job):
    r,root,device=job;p=feature_path(root,r)
    if p.exists():
        with np.load(p) as z:
            assert all(z[f'L{w}'].shape==(1800//w,3) for w in LENGTHS)
        return
    x=read_interval(Path(r['source']),float(r['start_seconds']),1800)
    bank={}
    for w in LENGTHS:
        values=[];quality=[]
        for s in range(0,1800,w):
            f=qeeg_features(x[:,s*FS:(s+w)*FS],torch.device(device),diagnostic=False)
            values.append([f.get(k,np.nan) for k in RESPONSE])
            quality.append([f['clean_seconds'],f.get('dfa_max_scale_seconds',np.nan),f.get('pli_clean_epochs',0)])
        bank[f'L{w}']=np.asarray(values);bank[f'Q{w}']=np.asarray(quality)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**bank)


def make_manifest(a):
    src=pd.read_csv(a.manifest,dtype={'recording_id':str})
    src=src[(src.kind=='long')&(src.case_control_label=='CASE')].copy()
    labels=patient_info(dict(workspace=a.workspace),'sustained')
    src=src.drop(columns=['label','fold'],errors='ignore').merge(labels,on='patient_id',validate='many_to_one')
    src=src.sort_values(['pre_post_treatment_label','patient_id']).reset_index(drop=True)
    assert len(src)==100 and src.groupby('pre_post_treatment_label').patient_id.nunique().eq(50).all()
    rng=np.random.default_rng(SEED);rows=[]
    for r in src.to_dict('records'):
        hi=int(np.floor(float(r['source_duration'])-1800))
        if hi<29:raise ValueError('Recording too short for 30 distinct integer-second crop starts')
        starts=rng.choice(hi+1,size=30,replace=False)
        for i,s in enumerate(starts):rows.append(dict(r,crop_id=i,start_seconds=int(s)))
    result=pd.DataFrame(rows)
    validate_patient_folds(result,'sustained',True)
    assert result.groupby(['pre_post_treatment_label','patient_id']).size().eq(30).all()
    path=a.output/'coordinates.csv'
    keep=['patient_id','recording_id','pre_post_treatment_label','crop_id','source','source_duration','start_seconds','label','fold']
    result=result[keep]
    if path.exists():pd.testing.assert_frame_equal(pd.read_csv(path,dtype={'recording_id':str}),result,check_dtype=False)
    else:result.to_csv(path,index=False)
    return result


def threshold(y,score):
    fpr,tpr,t=roc_curve(y,score);ok=np.isfinite(t)
    idx=np.flatnonzero(ok)[np.argmax((tpr-fpr)[ok])]
    return float(t[idx])


def score_model(model,x):
    return model.predict(x) if isinstance(model,Ridge) else predict_classifier(model,x)


def fit_classic(train_x,train_y,val_x,val_y,test_x,name,device,seed,path):
    tr,va,scaler=standardize(train_x,val_x)
    median,center,scale=scaler
    te=(np.where(np.isfinite(test_x),test_x,median)-center)/scale
    best=-np.inf;chosen=None;trials=[]
    for hp in GRIDS[name]:
        # Primal ridge is equivalent to the legacy dimension-normalized dual map,
        # avoiding a quadratic matrix for tens of thousands of small windows.
        model=Ridge(alpha=len(tr)*tr.shape[1]*hp['alpha'],fit_intercept=True,solver='cholesky') if name=='ridge' else make_classifier(name,hp,len(tr),tr.shape[1],train_y,seed,device)
        model.fit(tr,2*train_y-1 if name=='ridge' else train_y)
        v=score_model(model,va);auc=float(roc_auc_score(val_y,v));trials.append(dict(parameters=hp,validation_auroc=auc))
        if round(auc,12)>best:best=round(auc,12);chosen=(model,v,hp)
    model,v,hp=chosen;s=score_model(model,te)
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,scaler=scaler,parameters=hp),path,compress=3)
    saved=joblib.load(path);np.testing.assert_allclose(s,score_model(saved['model'],te),rtol=1e-6,atol=1e-7)
    return s,v,dict(parameters=hp,validation_auroc=best,trials=trials)


def fill(x,train):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x[train].reshape(-1,3),axis=0)
    med[~np.isfinite(med)]=0
    return np.where(np.isfinite(x),x,med).astype('float32'),med


def fit_neural(x,y,tr,va,te,method,a,fold,path):
    x,med=fill(x,tr);center=x[tr].astype('float64').mean((0,1));scale=x[tr].astype('float64').std((0,1));scale[scale<1e-6]=1
    scaled=np.clip((x-center)/scale,-10,10).astype('float32')
    xx=torch.as_tensor(scaled,device=a.device);yy=torch.as_tensor(y,dtype=torch.float32,device=a.device)
    torch.manual_seed(SEED+fold);rng=np.random.default_rng(SEED+fold)
    model=BatchedRecordingHead(3,method).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    weight=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()),device=a.device)
    best=-np.inf;bad=0;state=None
    for ep in range(1,a.epochs+1):
        model.train()
        for ids in np.array_split(rng.permutation(tr),math.ceil(len(tr)/64)):
            opt.zero_grad(set_to_none=True);z=model.forward_batch(xx[ids]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yy[ids],pos_weight=weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        model.eval()
        with torch.inference_mode():v=model.forward_batch(xx[va]).cpu().numpy()
        auc=float(roc_auc_score(y[va],v))
        if auc>best+1e-5:best=auc;bad=0;best_ep=ep;state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        else:bad+=1
        if bad>=a.patience:break
    model.load_state_dict(state);model.eval()
    with torch.inference_mode():v=model.forward_batch(xx[va]).cpu().numpy();s=model.forward_batch(xx[te]).cpu().numpy()
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=state,imputation=med,center=center,scale=scale,method=method,epoch=best_ep),path)
    restored=BatchedRecordingHead(3,method).to(a.device);restored.load_state_dict(torch.load(path,map_location=a.device,weights_only=False)['state_dict']);restored.eval()
    with torch.inference_mode():np.testing.assert_allclose(s,restored.forward_batch(xx[te]).cpu().numpy(),rtol=1e-5,atol=1e-6)
    return s,v,dict(best_epoch=best_ep,stopped_epoch=ep,validation_auroc=best)


def metrics(pred):
    """Patient bootstrap without materializing all window-pair comparisons."""
    rng=np.random.default_rng(20260923);draws=np.zeros(2000);num=den=0.
    correct=np.zeros((2000,2));counts=np.zeros((2000,2))
    if pred.groupby("patient_id").fold.nunique().max()!=1 or pred.groupby("patient_id").label.nunique().max()!=1:
        raise ValueError("Each response patient must have one fold and one outcome label")
    for _,f in pred.groupby('fold',sort=True):
        banks={label:[g.probability.to_numpy() for _,g in f[f.label==label].groupby('patient_id',sort=True)] for label in (0,1)}
        p,n=banks[1],banks[0];assert len({len(v) for v in p+n})==1
        matrix=np.zeros((len(p),len(n)))
        for j,neg in enumerate(n):
            ordered=np.sort(neg)
            for i,pos in enumerate(p):
                lower=np.searchsorted(ordered,pos,side='left');upper=np.searchsorted(ordered,pos,side='right')
                matrix[i,j]=(lower+.5*(upper-lower)).sum()/(len(pos)*len(neg))
        wp=rng.multinomial(len(p),np.full(len(p),1/len(p)),size=2000);wn=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=2000)
        draws+=np.einsum('bi,ij,bj->b',wp,matrix,wn);num+=matrix.sum();den+=matrix.size
        # Reuse the exact AUROC patient multiplicities; keep fitted thresholds fixed.
        for label,weights in [(1,wp),(0,wn)]:
            groups=[g for _,g in f[f.label==label].groupby("patient_id",sort=True)]
            correct[:,label]+=weights@np.array([(g.predicted_label==label).sum() for g in groups])
            counts[:,label]+=weights@np.array([len(g) for g in groups])
    draws/=den;lo,hi=np.quantile(draws,[.025,.975])
    ba_lo,ba_hi=np.quantile((correct/counts).mean(axis=1),[.025,.975])
    return dict(auroc=num/den,ci_low=float(lo),ci_high=float(hi),balanced_accuracy=float(balanced_accuracy_score(pred.label,pred.predicted_label)),balanced_accuracy_ci_low=float(ba_lo),balanced_accuracy_ci_high=float(ba_hi),n_patients=int(pred.patient_id.nunique()),n_predictions=len(pred)),draws


def save_fit(dest,pred,configs,info):
    validate_patient_folds(pred,'sustained',True)
    pred.to_csv(dest/'predictions.csv',index=False);(dest/'selection.json').write_text(json.dumps(configs,indent=2)+'\n')
    stat,draws=metrics(pred);stat=dict(info,**stat);np.save(dest/'bootstrap.npy',draws)
    (dest/'summary.json').write_text(json.dumps(stat,indent=2)+'\n')
    print('FIT',dest.name,f"AUROC={stat['auroc']:.3f} BA={stat['balanced_accuracy']:.3f}",flush=True)
    return stat


def fit(a,manifest):
    output=[];quality_rows=[]
    for visit,frame in manifest.groupby('pre_post_treatment_label',sort=True):
        frame=frame.reset_index(drop=True);y=frame.label.to_numpy(int);folds=frame.fold.to_numpy(int)
        for w in LENGTHS:
            bank=[];q=[]
            for r in frame.to_dict('records'):
                with np.load(feature_path(a.output,r)) as z:bank.append(z[f'L{w}']);q.append(z[f'Q{w}'])
            x=np.stack(bank);q=np.stack(q);n=x.shape[1]
            quality_rows.append(dict(visit=visit,window_seconds=w,n_windows=len(x)*n,incomplete_windows=int((~np.isfinite(x).all(-1)).sum()),median_clean_seconds=float(np.median(q[:,:,0])),median_dfa_max_scale=float(np.nanmedian(q[:,:,1]))))
            for classifier in ('ridge','logistic','xgboost'):
                key=f'{visit}__L{w}__single__{classifier}';dest=a.output/'fits'/key;dest.mkdir(parents=True,exist_ok=True)
                avgdest=a.output/'fits'/f'{visit}__L{w}__score_mean__{classifier}';avgdest.mkdir(parents=True,exist_ok=True)
                if (dest/'summary.json').exists() and (avgdest/'summary.json').exists():
                    output.extend([json.loads((d/'summary.json').read_text()) for d in (dest,avgdest)])
                else:
                    preds=[];averaged=[];configs=[];avgconfigs=[]
                    for f in range(5):
                        tr=np.flatnonzero((folds!=f)&(folds!=(f+1)%5));va=np.flatnonzero(folds==(f+1)%5);te=np.flatnonzero(folds==f)
                        assert (len(tr),len(va),len(te))==(900,300,300)
                        s,v,cfg=fit_classic(x[tr].reshape(-1,3),np.repeat(y[tr],n),x[va].reshape(-1,3),np.repeat(y[va],n),x[te].reshape(-1,3),classifier,a.device,SEED+f,a.checkpoints/key/f'fold{f}.joblib')
                        cut=threshold(np.repeat(y[va],n),v);configs.append(dict(cfg,fold=f,threshold=cut))
                        m=frame.iloc[te][['patient_id','fold','label','crop_id']].loc[frame.iloc[te].index.repeat(n)].reset_index(drop=True)
                        m['window_index']=np.tile(np.arange(n),len(te));m['probability']=s;m['predicted_label']=(s>=cut).astype(int);preds.append(m)
                        sv=s.reshape(-1,n).mean(1);vv=v.reshape(-1,n).mean(1);acut=threshold(y[va],vv)
                        m=frame.iloc[te][['patient_id','fold','label','crop_id']].copy();m['probability']=sv;m['predicted_label']=(sv>=acut).astype(int);averaged.append(m);avgconfigs.append(dict(fold=f,threshold=acut,source_model=key))
                    info=dict(visit=visit,window_seconds=w,classifier=classifier)
                    output.append(save_fit(dest,pd.concat(preds,ignore_index=True),configs,dict(info,method='single_window',unit='window')))
                    output.append(save_fit(avgdest,pd.concat(averaged,ignore_index=True),avgconfigs,dict(info,method='score_mean',unit='crop')))
            for method,classifier in [(p,c) for p in ('feature_mean','feature_mean_std') for c in ('ridge','logistic','xgboost')]+[('mean','learned'),('attention','learned')]:
                key=f'{visit}__L{w}__{method}__{classifier}';dest=a.output/'fits'/key;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'summary.json').exists():output.append(json.loads((dest/'summary.json').read_text()));continue
                preds=[];configs=[]
                for f in range(5):
                    tr=np.flatnonzero((folds!=f)&(folds!=(f+1)%5));va=np.flatnonzero(folds==(f+1)%5);te=np.flatnonzero(folds==f)
                    if classifier=='learned':s,v,cfg=fit_neural(x,y,tr,va,te,method,a,f,a.checkpoints/key/f'fold{f}.pt')
                    else:
                        filled,med=fill(x,tr);features=filled.mean(1)
                        if method=='feature_mean_std':features=np.c_[features,filled.std(1)]
                        s,v,cfg=fit_classic(features[tr],y[tr],features[va],y[va],features[te],classifier,a.device,SEED+f,a.checkpoints/key/f'fold{f}.joblib');cfg['window_imputation']=med.tolist()
                    cut=threshold(y[va],v);configs.append(dict(cfg,fold=f,threshold=cut))
                    m=frame.iloc[te][['patient_id','fold','label','crop_id']].copy();m['probability']=s;m['predicted_label']=(s>=cut).astype(int);preds.append(m)
                output.append(save_fit(dest,pd.concat(preds,ignore_index=True),configs,dict(visit=visit,window_seconds=w,classifier=classifier,method=method,unit='crop')))
            pd.DataFrame(output).to_csv(a.results/'all_results.csv',index=False)
            pd.DataFrame(quality_rows).to_csv(a.results/'feature_quality.csv',index=False)
    print('COMPLETE',len(output),'conditions',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/input_studies_20260923/inputs.csv')
    p.add_argument('--workspace',type=Path,default=ROOT.parent)
    p.add_argument('--output',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/qeeg_window_scale_30crops_20260924')
    p.add_argument('--results',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/qeeg_window_scale_30crops_20260924')
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--stage',choices=['all','extract','fit'],default='all');p.add_argument('--limit',type=int)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
    a=p.parse_args();require_cuda(a.device)
    a.output.mkdir(parents=True,exist_ok=True);a.results.mkdir(parents=True,exist_ok=True)
    manifest=make_manifest(a)
    protocol=dict(seed=SEED,task='sustained',crops_per_recording=30,total_seconds=1800,lengths=LENGTHS,validation='(test_fold+1)%5',refit=False,epochs=a.epochs,patience=a.patience,batch_size=64,features=RESPONSE)
    pp=a.output/'protocol.json';serialized=json.dumps(protocol,indent=2)+'\n'
    if pp.exists() and pp.read_text()!=serialized:raise ValueError('Existing protocol differs; use a new output directory')
    pp.write_text(serialized)
    if a.stage in ('all','extract'):
        jobs=[(r,str(a.output),a.device) for r in manifest.to_dict('records') if not feature_path(a.output,r).exists()]
        if a.limit:jobs=jobs[:a.limit]
        start=time.time();print('EXTRACT pending',len(jobs),flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(a.device,)) as pool:
            futures=[pool.submit(extract,j) for j in jobs]
            for i,f in enumerate(concurrent.futures.as_completed(futures),1):
                f.result()
                if i%10==0 or i==len(jobs):print(f'EXTRACT {i}/{len(jobs)} elapsed={time.time()-start:.0f}s',flush=True)
    if a.stage!='extract' and not a.limit:fit(a,manifest)


if __name__=='__main__':main()
