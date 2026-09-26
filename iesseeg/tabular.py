"""qEEG and context experiments reported in Tables 3–5."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from .dataset import Dataset
from .metrics import summarize
from iesseeg_paper.input_studies import TRIPLET, DIAGNOSTIC, RESPONSE
from iesseeg_paper.evaluation import standardize


def extract_qeeg(a):
    import torch
    from iesseeg_paper.input_studies import read_interval, qeeg_features
    from iesseeg_paper.evaluation import require_cuda
    device = require_cuda(a.device)
    torch.set_num_threads(1)
    ds = Dataset(a.data)
    r = ds.recordings
    if a.task == 'diagnosis':
        r = r[(r.subset.eq('clinician_selected') & r.visit.ne('POST')) | r.subset.eq('simulated_routine')]
    else:
        r = r[r.subset.eq('original_long') & r.patient_diagnosis_label.eq(1)]
    if a.limit: r = r.head(a.limit)
    out = a.work / 'qeeg' / a.task
    out.mkdir(parents=True, exist_ok=True)
    for i,row in enumerate(r.itertuples(),1):
        path = out / f'{row.recording_id}.json'
        if path.exists(): continue
        x = read_interval(ds.root / row.path)
        features = qeeg_features(x, device, diagnostic=a.task == 'diagnosis')
        path.write_text(json.dumps(features, indent=2)+'\n')
        print(f'qEEG {a.task}: {i}/{len(r)} {row.recording_id}', flush=True)


def _features(a, frame, fields, task):
    values = [json.loads((a.work / 'qeeg' / task / f'{rid}.json').read_text()) for rid in frame.recording_id]
    return np.array([[row[k] for k in fields] for row in values], float)


def _masks(frame, fold):
    f = frame.fold.to_numpy()
    masks = (~np.isin(f,[fold,(fold+1)%5]), f == (fold+1)%5, f == fold)
    groups = [set(frame.loc[m,'patient_id']) for m in masks]
    if any(groups[i] & groups[j] for i,j in [(0,1),(1,2),(0,2)]):
        raise ValueError('Patient overlap between partitions')
    return masks


def qeeg(a):
    ds = Dataset(a.data)
    r = ds.recordings.copy()
    r['fold'] = r.patient_id.map(ds.folds[a.task].set_index('patient_id').fold)
    r['label'] = r['patient_diagnosis_label' if a.task == 'diagnosis' else 'treatment_response_label']
    (a.work / 'predictions').mkdir(parents=True, exist_ok=True)
    (a.work / 'fits').mkdir(exist_ok=True)
    if a.task == 'diagnosis':
        dev = r[r.subset.eq('clinician_selected') & r.visit.ne('POST')].sort_values(['patient_id','source_id']).reset_index(drop=True)
        test = r[r.subset.eq('simulated_routine')].sort_values(['patient_id','source_id']).reset_index(drop=True)
        jobs = [('diagnosis_triplet',TRIPLET,dev,test),('diagnosis_expanded',DIAGNOSTIC,dev,test)]
    else:
        jobs = []
        for visit in ('PRE','POST'):
            part = r[r.subset.eq('original_long') & r.visit.eq(visit)].sort_values('patient_id').reset_index(drop=True)
            jobs.append((f'response_{visit}_qeeg',RESPONSE,part,part))
    metrics = []
    for name,fields,dev,test in jobs:
        x,xt = _features(a,dev,fields,a.task),_features(a,test,fields,a.task)
        preds=[]
        for fold in range(5):
            tr,va,_ = _masks(dev,fold)
            te = test.fold.eq(fold).to_numpy()
            tx,vx,scaler = standardize(x[tr],x[va])
            med,center,scale = scaler
            ex = (np.where(np.isfinite(xt[te]),xt[te],med)-center)/scale
            grid = [.01,.1,1,10,100] if a.task == 'diagnosis' else [.01,.1,1,10]
            best=None
            for c in grid:
                options = dict(solver='lbfgs',random_state=20260924) if a.task=='diagnosis' else dict(solver='liblinear',class_weight='balanced',random_state=20260924+fold)
                model=LogisticRegression(C=c,max_iter=3000,**options).fit(tx,dev.loc[tr,'label'].astype(int))
                score=float(roc_auc_score(dev.loc[va,'label'],model.predict_proba(vx)[:,1]))
                if a.task == 'response': score=round(score,12)
                if best is None or score>best[0]:best=score,c,model
            score,c,model=best
            p=test.loc[te,['recording_id','patient_id','fold','label','patient_diagnosis_label','clinician_diagnosis_label']].copy()
            p['probability']=model.predict_proba(ex)[:,1];preds.append(p)
            joblib.dump(dict(model=model,scaler=scaler,features=list(fields),C=c,validation_auroc=score,train_patients=sorted(set(dev.loc[tr,'patient_id'])),validation_patients=sorted(set(dev.loc[va,'patient_id'])),test_patients=sorted(set(test.loc[te,'patient_id']))),a.work/'fits'/f'{name}_fold{fold}.joblib')
        pred=pd.concat(preds,ignore_index=True)
        pred.to_csv(a.work/'predictions'/f'{name}.csv',index=False)
        targets=['patient_diagnosis_label','clinician_diagnosis_label'] if a.task=='diagnosis' else ['label']
        metrics.extend(dict(experiment=name,target=target,**summarize(pred,target)) for target in targets)
    out=a.work/'results';out.mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/f'{a.task}_qeeg.csv',index=False)
    print(pd.DataFrame(metrics)[['experiment','target','cv_auroc','balanced_accuracy']].to_string(index=False))


def context_extract(a):
    import torch
    from frozen_encoders import MODEL_IO
    from iesseeg_paper.input_studies import read_interval,prepare_encoder_input,encode
    from iesseeg_paper.evaluation import require_cuda
    device=require_cuda(a.device);torch.set_num_threads(1)
    model=MODEL_IO[a.model][0](a.work/'workspace',device)
    seconds={'biot':30,'cbramod':30,'labram':10}[a.model]
    for visit in a.visits:
        frame=pd.read_csv(a.work/'sampling'/f'context_{visit.lower()}.csv')
        if a.limit:frame=frame.head(a.limit)
        dest=a.work/'context_embeddings'/a.model/visit;dest.mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(frame.itertuples(),1):
            path=dest/f'{r.recording_id}_{r.crop_id:02d}.npy'
            if path.exists():continue
            raw=read_interval(Path(r.source),r.start_seconds,1800)
            prepared=prepare_encoder_input(raw,a.model)
            embedding=encode(model,a.model,prepared,seconds,device,batch_size=a.batch_size).mean(0)
            np.save(path,embedding)
            print('Context embeddings',a.model,visit,i,len(frame),flush=True)


def context_fit(a):
    ds=Dataset(a.data);lead=ds.participants.set_index('patient_id').lead_time_months
    out=a.work/'context';out.mkdir(exist_ok=True)
    results=[]
    for visit in a.visits:
        frame=pd.read_csv(a.work/'sampling'/f'context_{visit.lower()}.csv')
        if len(frame)!=500 or not frame.groupby('patient_id').size().eq(10).all():raise ValueError('Expected ten segments for each of 50 patients')
        q=_features(a,frame.rename(columns={'recording_id':'old_recording_id','source_recording_id':'recording_id'}),RESPONSE,'response')
        l=np.log1p(frame.patient_id.map(lead).to_numpy(float))[:,None]
        jobs=[('none','Q',{'Q':q}),('none','L',{'L':l}),('none','Q+L',{'Q':q,'L':l})]
        reduced={}
        for model in a.models:
            e=np.asarray([np.load(a.work/'context_embeddings'/model/visit/f'{r.recording_id}_{r.crop_id:02d}.npy') for r in frame.itertuples()],dtype=float)
            reduced[model]={}
            for fold in range(5):
                tr,_,_=_masks(frame,fold)
                scaler=StandardScaler().fit(e[tr]);scaled=scaler.transform(e)
                pls=PLSRegression(n_components=5,scale=False,max_iter=500,tol=1e-6).fit(scaled[tr],frame.loc[tr,'label'])
                reduced[model][fold]=pls.transform(scaled)
                joblib.dump(dict(scaler=scaler,pls=pls),out/f'{visit}_{model}_PLS5_fold{fold}.joblib')
            jobs.extend([(model,'E',{'E':e}),(model,'E+L',{'E':e,'L':l}),(model,'E+Q',{'E':e,'Q':q}),(model,'E+Q+L',{'E':e,'Q':q,'L':l})])
        for model,features,blocks in jobs:
            predictions=[]
            for fold in range(5):
                tr,va,te=_masks(frame,fold)
                x=np.concatenate([reduced[model][fold] if k=='E' else v for k,v in blocks.items()],axis=1)
                if not np.isfinite(x).all():raise ValueError('Nonfinite context features')
                scaler=StandardScaler().fit(x[tr]);x=scaler.transform(x)
                counts=frame.loc[tr].groupby('patient_id').size()
                weights=1/frame.loc[tr,'patient_id'].map(counts).to_numpy(float)
                best=None
                for c in [.001,.01,.1,1,10]:
                    clf=LogisticRegression(C=c,solver='lbfgs',max_iter=2000,tol=1e-9,random_state=20260925+fold).fit(x[tr],frame.loc[tr,'label'],sample_weight=weights)
                    score=float(roc_auc_score(frame.loc[va,'label'],clf.predict_proba(x[va])[:,1]))
                    if best is None or score>best[0]+1e-12:best=score,c,clf
                score,c,clf=best
                pred=frame.loc[te,['patient_id','recording_id','crop_id','fold','label']].copy();pred['probability']=clf.predict_proba(x[te])[:,1];predictions.append(pred)
                joblib.dump(dict(model=clf,scaler=scaler,blocks=list(blocks),C=c,validation_auroc=score),out/f'{visit}_{model}_{features}_fold{fold}.joblib')
            pred=pd.concat(predictions,ignore_index=True);pred.to_csv(out/f'{visit}_{model}_{features}_predictions.csv',index=False)
            results.append(dict(visit=visit,model=model,features=features,**summarize(pred)))
            print(visit,model,features,results[-1]['cv_auroc'],flush=True)
    pd.DataFrame(results).to_csv(out/'results.csv',index=False)
    context_contrasts(a)


def context_contrasts(a):
    from .metrics import paired_auc
    out=a.work/'context'
    rows=[]
    for visit in a.visits:
        for model in a.models:
            comparisons=[('E+Q','E',model),('E+Q','Q','none'),('E+L','E',model),('E+Q+L','E+Q',model),('E+Q+L','E+L',model),('E+Q+L','Q+L','none')]
            for first,second,reference_model in comparisons:
                aa=pd.read_csv(out/f'{visit}_{model}_{first}_predictions.csv')
                bb=pd.read_csv(out/f'{visit}_{reference_model}_{second}_predictions.csv')
                rows.append(dict(visit=visit,model=model,comparison=f'{first} vs. {second}',**paired_auc(aa,bb)))
    pd.DataFrame(rows).to_csv(out/'paired_differences.csv',index=False)
