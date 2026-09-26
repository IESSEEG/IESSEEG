#!/usr/bin/env python
"""Matched duration comparisons using cached, frozen EEG window representations.

All selections use development validation patients. Every candidate's OOF results
are retained; no selection or smoothing uses the test-set duration curve.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,json,time,sys,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from iesseeg_paper.duration_models import MatchedRidge
from iesseeg_paper.splits import load_fold_manifest,validate_patient_folds
from iesseeg_paper.evaluation import require_cuda,standardize
from clip_followup_metrics import summary
SEED=20260924
TASK_FIELDS={'immediate':'immediate_responder','sustained':'meaningful_responder'}
PANELS={'diagnosis':('smith_pli_retained_mean','smith_beta_entropy_fd','beta_dfa_intercept_mean'),
        'response':('beta_dfa_intercept_mean','beta_entropy_350','raw_pli_percent_over_020')}
GRIDS={'ridge':[{'alpha':a} for a in [10.,1.,.1,.01,.001]],
       'logistic':[{'C':c} for c in [.01,.1,1.,10.]],
       'rbf_svm':[{'C':c} for c in [.1,1.,10.]],
       'extra_trees':[{'min_samples_leaf':v,'max_depth':d} for v,d in [(5,4),(2,None)]],
       'xgboost':[{'max_depth':d,'min_child_weight':m} for d,m in [(1,1),(2,3),(3,3)]]}


def conditions(frame):
    yield 'diagnosis','PRE',frame[frame.pre_post_treatment_label.eq('PRE')].sort_values('patient_id').reset_index(drop=True)
    for task in TASK_FIELDS:
        for visit in ['PRE','POST']:
            yield task,visit,frame[frame.case_control_label.eq('CASE')&frame.pre_post_treatment_label.eq(visit)].sort_values('patient_id').reset_index(drop=True)


def targets(meta,task):
    if task=='diagnosis':return meta.case_control_label.eq('CASE').to_numpy(int)
    return meta[TASK_FIELDS[task]].map({'Responder':1,'Non-responder':0}).to_numpy(int)


def load_windows(root,model,meta,T,task):
    values=[]
    for row in meta.itertuples():
        with np.load(root/'features'/model/f'long_{row.recording_id}.npz') as z:
            if model=='qeeg':
                values.append(np.array([z[f'T{T}_{k}'].item() for k in PANELS['diagnosis' if task=='diagnosis' else 'response']]))
            else:values.append(z[f'T{T}_L{10 if model=="labram" else 30}'].copy())
    return np.asarray(values)


def aggregate(x,method):
    if x.ndim==2:return x
    if method=='mean':return x.mean(1)
    if method=='max':return x.max(1)
    if method=='mean_std':return np.concatenate([x.mean(1),x.std(1)],axis=1)
    if method=='two_halves':
        cut=x.shape[1]//2
        return np.concatenate([x[:,:cut].mean(1),x[:,cut:].mean(1)],axis=1)
    raise ValueError(method)


def make_classifier(name,hp,n,d,y,seed,device):
    if name=='ridge':return MatchedRidge(alpha=hp['alpha'],device=device)
    if name=='logistic':return LogisticRegression(**hp,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=seed)
    if name=='rbf_svm':return SVC(**hp,kernel='rbf',gamma='scale',class_weight='balanced')
    if name=='extra_trees':return ExtraTreesClassifier(**hp,n_estimators=128,max_features=.7,class_weight='balanced',random_state=seed,n_jobs=2)
    if name=='xgboost':return xgb.XGBClassifier(**hp,n_estimators=150,learning_rate=.05,subsample=1.,colsample_bytree=1.,reg_lambda=1.,tree_method='hist',device=str(device),n_jobs=2,random_state=seed,eval_metric='logloss',scale_pos_weight=float((y==0).sum()/max(1,(y==1).sum())))
    raise ValueError(name)


def predict_classifier(model,x):
    if isinstance(model,xgb.XGBClassifier):return model.get_booster().predict(xgb.DMatrix(x))
    return model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)[:,1]


def classical(x,y,folds,name,device,checkpoint):
    scores=np.zeros(len(y));configs=[]
    for f in range(5):
        test=np.flatnonzero(folds==f);val=np.flatnonzero(folds==(f+1)%5)
        train=np.flatnonzero((folds!=f)&(folds!=(f+1)%5));outer=np.flatnonzero(folds!=f)
        a,b,_=standardize(x[train],x[val]);trials=[]
        for hp in GRIDS[name]:
            model=make_classifier(name,hp,len(train),x.shape[1],y[train],SEED+f,device)
            model.fit(a,y[train]);score=float(roc_auc_score(y[val],predict_classifier(model,b)))
            trials.append(dict(parameters=hp,validation_auroc=score))
        selected=int(np.argmax([round(t['validation_auroc'],12) for t in trials]));hp=trials[selected]['parameters']
        a,b,scaler=standardize(x[outer],x[test])
        model=make_classifier(name,hp,len(outer),x.shape[1],y[outer],SEED+f,device);model.fit(a,y[outer])
        if name=='xgboost':
            actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not actual.startswith('cuda'):raise RuntimeError(f'XGBoost failed to use CUDA: {actual}')
        scores[test]=predict_classifier(model,b)
        checkpoint.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(model=model,scaler=scaler,fold=f,parameters=hp,train_indices=outer,test_indices=test),checkpoint/f'fold{f}.joblib',compress=3)
        saved=joblib.load(checkpoint/f'fold{f}.joblib')
        np.testing.assert_allclose(scores[test],predict_classifier(saved['model'],b),rtol=1e-7,atol=1e-8)
        configs.append(dict(fold=f,selected=selected,trials=trials,validation_auroc=trials[selected]['validation_auroc'],parameters=hp))
    return scores,configs


class Head(torch.nn.Module):
    def __init__(self,d,method):
        super().__init__();self.method=method
        self.adapter=torch.nn.Sequential(torch.nn.Linear(d,128),torch.nn.LayerNorm(128),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(128,64),torch.nn.LayerNorm(64),torch.nn.GELU())
        self.local=torch.nn.Linear(64,1)
        self.v=torch.nn.Linear(64,32);self.u=torch.nn.Linear(64,32);self.w=torch.nn.Linear(32,1,bias=False)
    def forward(self,x):
        h=self.adapter(x);z=self.local(h).squeeze(-1)
        if self.method in ('window_bce','mean_bce'):out=z.mean(1)
        elif self.method=='max':out=z.max(1).values
        elif self.method=='top10':out=z.topk(max(1,math.ceil(z.shape[1]*.1)),dim=1).values.mean(1)
        elif self.method=='attention':
            weights=torch.softmax(self.w(torch.tanh(self.v(h))*torch.sigmoid(self.u(h))).squeeze(-1),dim=1)
            out=(z*weights).sum(1)
        else:raise ValueError(self.method)
        return z,out


def neural_scale(train,test):
    if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('Nonfinite neural input')
    center=train.astype('float64').mean((0,1));scale=train.astype('float64').std((0,1));scale[scale<1e-6]=1
    return np.clip((train-center)/scale,-10,10).astype('float32'),np.clip((test-center)/scale,-10,10).astype('float32'),(center,scale)


def neural_scores(model,x):
    model.eval()
    with torch.inference_mode():return np.concatenate([model(v)[1].cpu().numpy() for v in x.split(16)])


def fit_head(x,y,validation,method,device,seed,epochs,patience=16):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=Head(x.shape[-1],method).to(device);x=torch.as_tensor(x,device=device)
    yy=torch.as_tensor(y,dtype=torch.float32,device=device)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y==0).sum()/max(1,(y==1).sum())),device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    if validation is not None:vx=torch.as_tensor(validation[0],device=device);vy=validation[1]
    best=-np.inf;best_epoch=2;stale=0
    for ep in range(1,epochs+1):
        model.train()
        for batch in np.array_split(rng.permutation(len(y)),max(1,math.ceil(len(y)/16))):
            local,out=model(x[batch]);label=yy[batch]
            loss=criterion(local,label[:,None].expand_as(local)) if method=='window_bce' else criterion(out,label)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
        if validation is not None and ep%2==0:
            value=roc_auc_score(vy,neural_scores(model,vx))
            if value>best+1e-5:best=float(value);best_epoch=ep;stale=0
            else:stale+=2
            if stale>=patience:break
    return model,best_epoch,best


def neural(x,y,folds,method,device,checkpoint,max_epochs):
    scores=np.zeros(len(y));configs=[]
    for f in range(5):
        test=np.flatnonzero(folds==f);val=np.flatnonzero(folds==(f+1)%5);inner=np.flatnonzero((folds!=f)&(folds!=(f+1)%5));outer=np.flatnonzero(folds!=f)
        a,b,_=neural_scale(x[inner],x[val]);_,epoch,va=fit_head(a,y[inner],(b,y[val]),method,device,SEED+f,max_epochs)
        a,b,scaler=neural_scale(x[outer],x[test]);model,_,_=fit_head(a,y[outer],None,method,device,SEED+f,epoch)
        bx=torch.as_tensor(b,device=device);scores[test]=neural_scores(model,bx)
        checkpoint.mkdir(parents=True,exist_ok=True)
        path=checkpoint/f'fold{f}.pt';torch.save(dict(state_dict=model.state_dict(),scaler=scaler,method=method,input_dim=x.shape[-1],epoch=epoch,fold=f,train_indices=outer,test_indices=test),path)
        reloaded=Head(x.shape[-1],method).to(device);reloaded.load_state_dict(torch.load(path,map_location=device,weights_only=False)['state_dict'])
        np.testing.assert_allclose(scores[test],neural_scores(reloaded,bx),rtol=1e-6,atol=1e-7)
        configs.append(dict(fold=f,epoch=epoch,validation_auroc=va))
    return scores,configs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/input_studies_20260923')
    p.add_argument('--output-root',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/duration_followup_20260924')
    p.add_argument('--checkpoint-root',type=Path,required=True);p.add_argument('--device',required=True)
    p.add_argument('--stage',choices=['classical','neural'],required=True)
    p.add_argument('--models',nargs='+',default=['qeeg','biot','cbramod','labram'])
    p.add_argument('--classifiers',nargs='+',default=list(GRIDS));p.add_argument('--aggregations',nargs='+',default=['mean','mean_std','max','two_halves'])
    p.add_argument('--heads',nargs='+',default=['window_bce','mean_bce','attention','max','top10'])
    p.add_argument('--durations',nargs='+',type=int,default=[60,300,600,1800]);p.add_argument('--max-epochs',type=int,default=80)
    p.add_argument('--tasks',nargs='+',default=['diagnosis','immediate','sustained']);p.add_argument('--visits',nargs='+',default=['PRE','POST'])
    args=p.parse_args();device=require_cuda(args.device);torch.set_num_threads(2)
    frame=pd.read_csv(args.input_root/'inputs.csv',dtype={'recording_id':str});frame=frame[frame.kind.eq('long')&frame.eligible].copy()
    args.output_root.mkdir(parents=True,exist_ok=True);started=time.time();done=0
    # Mean pooling is completed first, then the other static aggregates.
    aggs=args.aggregations if args.stage=='classical' else ['learned']
    for agg in aggs:
      for model in args.models:
        if model=='qeeg' and (args.stage=='neural' or agg!='mean'):continue
        for task,visit,meta in conditions(frame):
          if task not in args.tasks or visit not in args.visits:continue
          y=targets(meta,task);folds=meta.patient_id.map(load_fold_manifest(task).set_index('patient_id').fold).to_numpy(int)
          assert meta.patient_id.is_unique and len(meta)==(100 if task=='diagnosis' else 50)
          for T in args.durations:
            windows=load_windows(args.input_root,model,meta,T,task)
            methods=args.classifiers if args.stage=='classical' else args.heads
            for method in methods:
              key=f'{args.stage}__{model}__{task}__{visit}__T{T}__{agg}__{method}'
              target=args.output_root/key;target.mkdir(exist_ok=True)
              if (target/'summary.json').exists():
                # Honor the prespecified tie rule despite floating-point AUROC roundoff.
                old_config=json.loads((target/'config.json').read_text());old=old_config['folds']
                if (method!='ridge' or old_config.get('ridge_solver')=='original_cuda_map') and (args.stage=='neural' or all(c['selected']==int(np.argmax([round(t['validation_auroc'],12) for t in c['trials']])) for c in old)):continue
              tick=time.time()
              if args.stage=='classical':scores,config=classical(aggregate(windows,agg),y,folds,method,device,args.checkpoint_root/key)
              else:scores,config=neural(windows,y,folds,method,device,args.checkpoint_root/key,args.max_epochs)
              pred=meta[['patient_id','recording_id']].copy();pred['label']=y;pred['fold']=folds;pred['probability']=scores
              validate_patient_folds(pred,task,require_complete=True);pred.to_csv(target/'predictions.csv',index=False)
              (target/'config.json').write_text(json.dumps(dict(ridge_solver='original_cuda_map' if method=='ridge' else None,folds=config,seed=SEED,stage=args.stage,method=method,max_epochs=args.max_epochs,versions=dict(torch=torch.__version__,xgboost=xgb.__version__),args={k:str(v) for k,v in vars(args).items()}),indent=2)+'\n')
              result=dict(stage=args.stage,model=model,task=task,visit=visit,T=T,aggregation=agg,classifier=method,mean_validation_auroc=float(np.mean([v['validation_auroc'] for v in config])),seconds=time.time()-tick,**summary(pred))
              (target/'summary.json').write_text(json.dumps(result,indent=2)+'\n');done+=1
              print(key,f"AUROC={result['fold_comparable_clip_auroc']:.3f}",f"{result['seconds']:.1f}s",flush=True)
    print(f'Completed {done} configurations in {time.time()-started:.1f}s',flush=True)
if __name__=='__main__':main()
