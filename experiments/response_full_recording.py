#!/usr/bin/env python
"""Evaluate response on complete original recordings with fixed patient folds.

Reuse trained backbone checkpoints. Frozen window heads use the existing
30-crop inherited-window training recipe; no recording aggregator is trained.
Only window probabilities (not full-recording embeddings) are persisted.
"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments'), str(ROOT/'encoders')]
from finetune_response_windows import WindowPredictor
from experiments.response_windows import prepare_windows, encode, neural, worker_init
from iesseeg_paper.input_studies import read_interval
from iesseeg_paper.evaluation import require_cuda
from iesseeg_paper.splits import validate_patient_folds


def window_chunks(duration, seconds, batch_size):
    """Tile from zero; every complete window occurs once; drop only the tail."""
    n = int(math.floor(duration / seconds))
    return [(i, min(batch_size, n-i)) for i in range(0, n, batch_size)]


class RecordingChunks(Dataset):
    def __init__(self, row, model, seconds, batch_size):
        self.row, self.model, self.seconds = row, model, seconds
        self.chunks = window_chunks(float(row['source_duration']), seconds, batch_size)
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, index):
        first, count = self.chunks[index]
        raw = read_interval(Path(self.row['source']), first*self.seconds, count*self.seconds)
        return first, prepare_windows(raw, self.model, self.seconds)


def check_protocol(protocol, frame, fold, visit):
    folds = frame.fold.to_numpy()
    for key, mask in [('train_patients', ~np.isin(folds, [fold, (fold+1)%5])),
                      ('validation_patients', folds == (fold+1)%5), ('test_patients', folds == fold)]:
        if set(protocol[key]) != set(frame.loc[mask, 'patient_id']):
            raise ValueError(f'Checkpoint patient mismatch: {key}')
    if (protocol['task'], protocol['condition'], protocol['test_fold']) != ('sustained', visit, fold):
        raise ValueError('Checkpoint task/fold/visit mismatch')


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def load_frozen_heads(a, coords, settings):
    """Reuse identical heads from multicrop runs, or fit missing ones identically."""
    old = a.storage/'response_multicrop_20260924'/f'{a.model}_frozen_{a.visit}'
    paths = [old/f'window_head_fold{f}.pt' for f in range(5)]
    paths = [p if p.exists() else a.ckpts/f'window_head_fold{f}.pt' for f,p in enumerate(paths)]
    missing = [f for f,p in enumerate(paths) if not p.exists()]
    if missing:
        print('FROZEN_HEAD_FEATURES', a.model, a.visit, 'missing folds', missing, flush=True)
        x,_ = encode(a, coords, 0, settings)
        folds=coords.fold.to_numpy(); y=coords.label.to_numpy(int)
        for fold in missing:
            tr=np.flatnonzero(~np.isin(folds,[fold,(fold+1)%5]));va=np.flatnonzero(folds==(fold+1)%5);te=np.flatnonzero(folds==fold)
            neural(x,y,tr,va,te,'window_head',a,paths[fold],fold)
            print('FROZEN_HEAD_FIT',a.model,a.visit,fold,flush=True)
        del x
    heads={}
    provenance=[]
    folds=coords.fold.to_numpy()
    for fold,path in enumerate(paths):
        pay=torch.load(path,map_location='cpu',weights_only=False)
        for key,mask in [('train_indices',~np.isin(folds,[fold,(fold+1)%5])),('validation_indices',folds==(fold+1)%5),('test_indices',folds==fold)]:
            np.testing.assert_array_equal(pay[key],np.flatnonzero(mask))
        head=torch.nn.Sequential(torch.nn.LayerNorm(pay['input_dim']),torch.nn.Linear(pay['input_dim'],1)).to(a.device)
        head.load_state_dict(pay['state_dict']);head.eval()
        heads[fold]=(head, torch.as_tensor(pay['center'],device=a.device,dtype=torch.float64),torch.as_tensor(pay['scale'],device=a.device,dtype=torch.float64))
        provenance.append(dict(fold=fold,path=str(path),best_epoch=pay['best_epoch'],validation_window_auroc=pay['validation_auroc']))
    atomic_json(a.output/'frozen_heads.json',provenance)
    return heads


def predpath(a, branch, fold, row):
    return a.ckpts/'probabilities'/branch/f'fold{fold}'/f"patient{int(row['patient_id'])}.npz"


class MultiRecordingChunks(Dataset):
    def __init__(self, jobs, model, seconds, batch_size):
        self.jobs, self.model, self.seconds = jobs, model, seconds
        self.entries = [(j, first, n) for j, (row, _) in enumerate(jobs)
                        for first, n in window_chunks(float(row['source_duration']), seconds, batch_size)]
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, i):
        j, first, n = self.entries[i]
        row = self.jobs[j][0]
        raw = read_interval(Path(row['source']), first*self.seconds, n*self.seconds)
        return j, first, prepare_windows(raw, self.model, self.seconds)


def infer_many(a, model, rows, seconds, fixed_fold=None, frozen_heads=None):
    jobs=[]
    for row in rows:
        targets=[fixed_fold] if fixed_fold is not None else [int(row['fold']), (int(row['fold'])-1)%5]
        targets=[f for f in targets if not predpath(a,a.branch,f,row).exists()]
        if targets:jobs.append((row,targets))
    if not jobs:return
    dataset=MultiRecordingChunks(jobs,a.model,seconds,a.inference_batch_size)
    loader=DataLoader(dataset,batch_size=None,num_workers=a.workers,worker_init_fn=worker_init,
        pin_memory=True,**({'multiprocessing_context':'spawn','prefetch_factor':2} if a.workers else {}))
    active=None;expected=0;banks={};started=time.time()
    def finish(j):
        row,targets=jobs[j];n=int(float(row['source_duration'])//seconds)
        if expected!=n:raise ValueError('Incomplete long-recording coverage')
        for fold in targets:
            path=predpath(a,a.branch,fold,row);path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.partial.npz')
            np.savez_compressed(tmp,probability=np.concatenate(banks[fold]),patient_id=int(row['patient_id']),
                label=int(row['label']),patient_fold=int(row['fold']),model_fold=fold,window_seconds=seconds,
                source_duration=float(row['source_duration']),recording_id=str(row['recording_id']))
            tmp.replace(path)
        print('RECORDING',a.model,a.branch,a.visit,'patient',row['patient_id'],'folds',targets,'windows',n,'seconds',round(time.time()-started,1),flush=True)
    with torch.inference_mode():
        for j,first,x in loader:
            j,first=int(j),int(first)
            if j!=active:
                if active is not None:finish(active)
                active=j;expected=0;banks={f:[] for f in jobs[j][1]};started=time.time()
            if first!=expected:raise ValueError('Window coverage/order error')
            h=model.encode(x.to(a.device,non_blocking=True))
            for fold in banks:
                if frozen_heads is None:z=model.head(h).flatten()
                else:
                    head,center,scale=frozen_heads[fold]
                    z=head(((h.double()-center)/scale).clamp(-10,10).float()).flatten()
                probabilities=z.sigmoid().cpu().numpy()
                if not np.isfinite(probabilities).all():raise ValueError('Nonfinite probabilities')
                banks[fold].append(probabilities)
            expected+=len(x)
        if active is not None:finish(active)


def summarize(a,frame,seconds):
    records=[]; selections=[]
    for fold in range(5):
        part=frame[frame.fold.isin([fold,(fold+1)%5])]
        rows=[]
        for r in part.to_dict('records'):
            p=predpath(a,a.branch,fold,r)
            if not p.exists():return False
            with np.load(p) as z:
                if int(z['patient_id'])!=int(r['patient_id']) or int(z['model_fold'])!=fold:raise ValueError('Prediction identity mismatch')
                if len(z['probability'])!=int(float(r['source_duration'])//seconds):raise ValueError('Incomplete recording')
                rows.append(dict(patient_id=int(r['patient_id']),recording_id=str(r['recording_id']),fold=int(r['fold']),label=int(r['label']),probability=float(np.mean(z['probability'],dtype=np.float64)),n_windows=len(z['probability']),source_duration=float(r['source_duration']),used_seconds=len(z['probability'])*seconds))
        df=pd.DataFrame(rows);val=df[df.fold==(fold+1)%5];test=df[df.fold==fold].copy()
        cut=.5;test['predicted_label']=(test.probability>=cut).astype(int)
        records.append(test);selections.append(dict(fold=fold,threshold=cut,validation_patients=val.patient_id.tolist()))
    pred=pd.concat(records,ignore_index=True);validate_patient_folds(pred,'sustained',True)
    if len(pred)!=50 or pred.patient_id.duplicated().any():raise ValueError('Not one test prediction per patient')
    from iesseeg.metrics import summarize as paper_metrics
    stat=paper_metrics(pred);stat.update(model=a.model,branch=a.branch,visit=a.visit,unit='full_recording',window_seconds=seconds,aggregation='mean_probability',total_windows=int(pred.n_windows.sum()),min_used_seconds=int(pred.used_seconds.min()),max_used_seconds=int(pred.used_seconds.max()))
    pred.to_csv(a.output/f'{a.branch}_predictions.csv',index=False)
    atomic_json(a.output/f'{a.branch}_thresholds.json',selections)
    atomic_json(a.output/f'{a.branch}_summary.json',stat)
    a.results.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([stat]).to_csv(a.results/f'{a.model}_{a.branch}_{a.visit}.csv',index=False)
    atomic_json(a.output/f'{a.branch}_complete.json',stat)
    print('COMPLETE',json.dumps(stat),flush=True)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--visit',choices=['PRE','POST'],required=True)
    p.add_argument('--branches',nargs='+',choices=['frozen','finetuned'],default=['finetuned','frozen'])
    p.add_argument('--workspace',type=Path,default=ROOT.parent)
    p.add_argument('--coordinates',type=Path,default=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/qeeg_window_scale_30crops_20260924/coordinates.csv')
    p.add_argument('--storage',type=Path,default=Path(__import__('os').environ.get('IESSEEG_CHECKPOINT_ROOT', 'work/checkpoints') + ''))
    p.add_argument('--device',required=True);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--inference-batch-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--folds',nargs='+',type=int,choices=range(5),default=list(range(5)))
    a=p.parse_args()
    if a.folds!=list(range(5)) and a.branches!=['finetuned']:
        p.error('--folds subsets require --branches finetuned')
    a.device=require_cuda(a.device);torch.set_num_threads(1)
    coords=pd.read_csv(a.coordinates,dtype={'recording_id':str});coords=coords[coords.pre_post_treatment_label.eq(a.visit)].reset_index(drop=True)
    if len(coords)!=1500 or not coords.groupby('patient_id').size().eq(30).all():raise ValueError('Expected fixed 30-crop head-training manifest')
    validate_patient_folds(coords,'sustained',True)
    frame=coords.drop_duplicates('patient_id').sort_values('patient_id').reset_index(drop=True)
    frame['start_seconds']=0.
    a.encoder_root=a.storage/('sustained_window_finetuning_20260924' if a.model in ['biot','labram','cbramod'] else 'remaining_encoder_finetuning_20260924')/'runs'
    a.output=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'local/response_full_recording_20260925'/f'{a.model}_{a.visit}';a.output.mkdir(parents=True,exist_ok=True)
    a.ckpts=a.storage/'response_full_recording_20260925'/f'{a.model}_{a.visit}';a.ckpts.mkdir(parents=True,exist_ok=True)
    a.results=Path(__import__('os').environ.get('IESSEEG_WORK_ROOT', 'work')) / 'results/new/response_full_recording_20260925'
    recipe='sustained_window_finetuning.json' if a.model in ['biot','labram','cbramod'] else 'remaining_sustained_window_finetuning.json'
    settings=json.loads((ROOT/'configs'/recipe).read_text())['models'][a.model]
    if 'finetuned' in a.branches:
        for fold in range(5):
            pay=torch.load(a.encoder_root/a.model/a.visit/f'fold{fold}/best.pt',map_location='cpu',weights_only=False)
            check_protocol(pay['protocol'],frame,fold,a.visit)
            if settings != pay['protocol']['config']['models'][a.model]:
                raise ValueError('Checkpoint settings differ from configured recipe')
            del pay
    seconds=settings['window_seconds']
    atomic_json(a.output/'protocol.json',dict(model=a.model,visit=a.visit,task='sustained',window_seconds=seconds,input='entire original long recording, start zero',incomplete_tail='discard',aggregation='mean_probability',test_fold='f',validation_fold='(f+1)%5',frozen_head_training='existing multicrop inherited-window BCE recipe',seed=20260924,epochs=a.epochs,patience=a.patience,encoder_root=str(a.encoder_root)))
    frame.to_csv(a.output/'recordings.csv',index=False)
    if a.smoke:
        model=WindowPredictor(a.model,a.workspace,a.device,settings).eval()
        row=frame.iloc[0].to_dict();row['source_duration']=seconds*3+1
        ds=RecordingChunks(row,a.model,seconds,2)
        outputs=[]
        with torch.inference_mode():
            for first,x in ds:outputs.extend(model(torch.as_tensor(x,device=a.device)).cpu().tolist())
        assert len(outputs)==3 and np.isfinite(outputs).all()
        print('SMOKE_OK',a.model,seconds,len(outputs),flush=True);return
    for branch in a.branches:
        a.branch=branch
        if (a.output/f'{branch}_complete.json').exists():continue
        if branch=='finetuned':
            for fold in a.folds:
                part=frame[frame.fold.isin([fold,(fold+1)%5])]
                if all(predpath(a,branch,fold,r).exists() for r in part.to_dict('records')):continue
                payload=torch.load(a.encoder_root/a.model/a.visit/f'fold{fold}/best.pt',map_location='cpu',weights_only=False)
                model=WindowPredictor(a.model,a.workspace,a.device,settings);model.load_state_dict(payload['model']);model.eval();del payload
                infer_many(a,model,part.to_dict('records'),seconds,fixed_fold=fold)
                del model;torch.cuda.empty_cache()
                print('FOLD_DONE',a.model,branch,a.visit,fold,flush=True)
        else:
            heads=load_frozen_heads(a,coords,settings)
            torch.manual_seed(20260924)
            model=WindowPredictor(a.model,a.workspace,a.device,settings).eval()
            infer_many(a,model,frame.to_dict('records'),seconds,frozen_heads=heads)
            del model,heads;torch.cuda.empty_cache()
        if not summarize(a,frame,seconds):
            if a.folds!=list(range(5)):
                print('FOLD_SUBSET_COMPLETE',a.model,a.visit,a.folds,flush=True)
            else:raise RuntimeError('Run ended with missing recordings')

if __name__=='__main__':main()
