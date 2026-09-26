#!/usr/bin/env python
"""Extract traceable raw-EDF inputs for duration, state and encoding-length studies.

No raw EEG copies are written. Coordinates and compact derived features remain
under output-root. Re-running skips only completed per-example files.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'legacy/analysis')]
from iesseeg_paper.input_studies import (SEED,resolve_edf,read_interval,qeeg_features,
    prepare_encoder_input,encode)
from iesseeg_paper.evaluation import require_cuda


def build_manifest(workspace,out,long_root=None,short_root=None,routine_root=None):
    short=pd.read_csv(workspace/'data/final_short_merged.csv',dtype={'short_recording_id':str})
    test=pd.read_csv(workspace/'data/final_test.csv',dtype={'short_recording_id':str})
    rows=[]
    def add(row,source,kind,rid,maximum):
        path=resolve_edf(source,rid)
        with path.open('rb') as handle:
            header=handle.read(256)
        records=int(header[236:244]);record_seconds=float(header[244:252])
        if records<0:raise ValueError(f'Unknown EDF record count: {path}')
        duration=records*record_seconds
        row=dict(row);row.update(kind=kind,recording_id=rid,source=str(path),source_duration=duration)
        if kind=='long':
            row['sleep_awake_label']='UNKNOWN'
            row['short_recording_id']=''
        # One outcome-independent, reproducible random start. Nested durations
        # share this start, but filtering is repeated within each input budget.
        rng=np.random.default_rng(np.random.SeedSequence([SEED,int(row['patient_id']),
            sum((i+1)*ord(c) for i,c in enumerate(kind+rid))]))
        row['start_seconds']=float(rng.integers(0,int(duration-maximum)+1)) if duration>=maximum else np.nan
        row['eligible']=duration>=maximum
        rows.append(row)
    for _,r in short.groupby('long_recording_id',sort=True).first().reset_index().iterrows():
        add(r,long_root or workspace.parent/'long_eeg','long',str(r.long_recording_id),1800)
    for _,r in short[short.case_control_label.eq('CASE')].iterrows():
        add(r,short_root or workspace/'data/raw_data/edf','state',str(r.short_recording_id),480)
    for _,r in short[short.pre_post_treatment_label.eq('PRE')].iterrows():
        add(r,short_root or workspace/'data/raw_data/edf','smith_development',str(r.short_recording_id),0)
    for _,r in test.iterrows():
        add(r,routine_root or workspace/'data/raw_data/test_edf','smith_evaluation',str(r.short_recording_id),0)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'inputs.csv',index=False)
    print(frame.groupby('kind').eligible.agg(['count','sum']).to_string(),flush=True)
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--model',choices=['qeeg','smith','biot','cbramod','labram'],required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--build-manifest',action='store_true')
    p.add_argument('--long-edf-root',type=Path)
    p.add_argument('--short-edf-root',type=Path)
    p.add_argument('--routine-edf-root',type=Path)
    p.add_argument('--limit',type=int)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    device=require_cuda(a.device)
    torch.manual_seed(SEED);np.random.seed(SEED)
    if a.build_manifest:
        build_manifest(a.workspace,a.output_root,a.long_edf_root,a.short_edf_root,a.routine_edf_root)
    frame=pd.read_csv(a.output_root/'inputs.csv',dtype={'recording_id':str})
    kinds=['smith_development','smith_evaluation'] if a.model=='smith' else ['long','state']
    frame=frame[frame.kind.isin(kinds)&frame.eligible].reset_index(drop=True)
    if a.limit:frame=frame.iloc[:a.limit]
    if not 0<=a.shard_index<a.num_shards:raise ValueError('Invalid shard')
    total=len(frame)
    frame=frame.iloc[a.shard_index::a.num_shards]
    if a.model not in ('qeeg','smith'):
        from frozen_encoders import load_biot,load_cbramod,load_labram
        model={'biot':load_biot,'cbramod':load_cbramod,'labram':load_labram}[a.model](a.workspace,device)
    directory=a.output_root/'features'/a.model;directory.mkdir(parents=True,exist_ok=True)
    for i,r in frame.iterrows():
        target=directory/f'{r.kind}_{r.recording_id}.npz'
        diagnostic=(r.kind=='long' and r.pre_post_treatment_label=='PRE')
        if target.exists():
            with np.load(target) as previous:
                if a.model!='qeeg' or not diagnostic or 'T60_smith_beta_entropy_fd' in previous:continue
        tick=time.time();result={}
        if a.model=='smith':
            x=read_interval(Path(r.source))
            features=qeeg_features(x,device,diagnostic=True)
            for k,v in features.items():result[k]=np.array(v)
        else:
            durations=[60,300,480,600,1800] if r.kind=='long' else [120,240,480]
            for duration in durations:
                x=read_interval(Path(r.source),r.start_seconds,duration)
                if a.model=='qeeg':
                    features=qeeg_features(x,device,diagnostic=diagnostic)
                    for k,v in features.items():result[f'T{duration}_{k}']=np.array(v)
                else:
                    prepared=prepare_encoder_input(x,a.model)
                    lengths=([4,10,16] if a.model=='labram' else [4,10,30]) if r.kind=='long' and duration==480 else ([10] if a.model=='labram' else [30])
                    for length in lengths:
                        result[f'T{duration}_L{length}']=encode(model,a.model,prepared,length,device)
        # Atomic completion marker is the feature file itself.
        with target.with_suffix('.partial').open('wb') as handle:
            np.savez_compressed(handle,**result)
        target.with_suffix('.partial').replace(target)
        print(f'{a.model} {i+1}/{total} {r.kind} {r.recording_id} {time.time()-tick:.1f}s',flush=True)
    marker='complete.json' if a.num_shards==1 else f'complete_shard{a.shard_index}.json'
    (directory/marker).write_text(json.dumps(dict(model=a.model,expected_examples=len(frame),
        shard_index=a.shard_index,num_shards=a.num_shards,total_examples=total,
        seed=SEED,device=str(device),durations_long=[60,300,480,600,1800],
        durations_state=[120,240,480],encoding_lengths=[4,10,16] if a.model=='labram' else [4,10,30],
        sampling='one random start per EDF; no artifact-driven resampling',
        limit=a.limit),indent=2)+'\n')

if __name__=='__main__':main()
