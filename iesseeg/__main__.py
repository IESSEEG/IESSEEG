"""Command-line interface. Run `python -m iesseeg --help`."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .dataset import Dataset, prepare, link_file
ROOT = Path(__file__).resolve().parents[1]
MODELS = ['biot','labram','cbramod','eegpt','luna','reve','codebrain','csbrain']


def environment(a):
    work=a.work
    os.environ['IESSEEG_WORK_ROOT']=str(work)
    os.environ['IESSEEG_WORKSPACE']=str(work/'workspace')
    os.environ['IESSEEG_LEGACY_RUNTIME']=str(work/'runtime')
    os.environ['IESSEEG_CHECKPOINT_ROOT']=str(work/'checkpoints')
    os.environ['IESSEEG_DATA_ROOT']=str(work/'workspace/data')
    os.environ['IESSEEG_SPLIT_ROOT']=str(work/'splits')
    os.environ['IESSEEG_OUTPUT_ROOT']=str(work/'diagnosis_finetuned')
    os.environ['IESSEEG_TASKS']='case_control'
    os.environ['PYTHON_BIN']=sys.executable
    paths=[ROOT,ROOT/'experiments',ROOT/'legacy/analysis',ROOT/'baselines_reference']
    for path in reversed(paths):sys.path.insert(0,str(path))
    os.environ['PYTHONPATH']=os.pathsep.join(map(str,paths))+os.pathsep+os.environ.get('PYTHONPATH','')
    if a.pretrained:os.environ['IESSEEG_PRETRAINED_DIR']=str(a.pretrained.resolve())
    if a.device:os.environ['CUDA_DEVICE']=a.device.split(':')[-1]


def command(args, cwd=ROOT):
    import shlex
    print('$ '+shlex.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def preprocess(a):
    families={'biot':'biot','cbramod':'cbramod','labram':'labram','eegpt':'labram','reve':'labram','codebrain':'labram','csbrain':'labram','luna':'bipolar'}
    family=families[a.model]
    dirs={'biot':('scalp_eeg_data_200HZ_np_format_biot','biot_test'),'cbramod':('scalp_eeg_data_200HZ_np_format_cbramod','cbramod_test'),'labram':('scalp_eeg_data_200HZ_np_format_labram','labram_test'),'bipolar':('scalp_eeg_data_200HZ_np_format','baseline_test')}
    scripts=ROOT/'preprocessing_reference/eeg_preprocess'
    for subset,directory in zip(['selected','routine'],dirs[family]):
        raw=a.work/'workspace/data/raw_data'/('edf' if subset=='selected' else 'test_edf')
        dest=a.work/'workspace/data'/directory;dest.mkdir(parents=True,exist_ok=True)
        if family=='biot':
            args=[scripts/'biot_preprocessing.py','--num_biot_channels','18']
        elif family=='bipolar':
            args=[scripts/'preprocessing.py','--low_freq','.5','--high_freq','50','--reference_scheme','bipolar']
        else:
            args=[scripts/'cbramod_preprocessing.py','--low_freq','.1' if family=='labram' else '.3','--high_freq','75','--apply_notch','--notch_freq','60']
        if family=='cbramod' and subset=='routine':
            args += ['--trim_data','--trim_sec','60']
        command([sys.executable,*args,'--raw_data_path',raw,'--out_folder',dest,'--n_jobs',a.workers])
        rows=Dataset(a.data).ordered(subset)
        missing=[r for r in rows.source_id if not (dest/f'{r}.npz').is_file()]
        if missing:raise RuntimeError(f'Preprocessing failed for {len(missing)} files, including {missing[0]}')


def extract_diagnosis(a):
    command([sys.executable,ROOT/'legacy/analysis/extract_long_frozen_representations.py','--model',a.model,'--device',a.device,'--batch-size',a.batch_size,'--output-root',a.work/'local/features/long_frozen_representations',*(['--limit',a.limit] if a.limit else [])])


def diagnosis(a):
    if 'finetuned' in a.branches:
        folder=ROOT/'baselines_reference/baselines'/a.model
        for script in ['train_all.sh','inference_all.sh']:command(['bash',folder/script],cwd=folder)
        source=a.work/'diagnosis_finetuned'/a.model
        for base in ['kfold_split','baselines_release']:
            target=a.work/'workspace'/base/'baselines'/a.model
            link_file(source,target)
    command([sys.executable,ROOT/'experiments/diagnosis_probability_mean.py','--models',a.model,'--branches',*a.branches,'--device',a.device,'--workspace',a.work/'workspace'])


def response(a):
    first=a.model in ['biot','labram','cbramod']
    tag='sustained_window_finetuning_20260924' if first else 'remaining_encoder_finetuning_20260924'
    recipe=ROOT/'configs'/('sustained_window_finetuning.json' if first else 'remaining_sustained_window_finetuning.json')
    if 'finetuned' in a.branches:
        command([sys.executable,ROOT/'experiments/finetune_response_windows.py','--recipe',recipe,'--workspace',a.work/'workspace','--input-manifest',a.work/'inputs.csv','--output-root',a.work/'response_training','--checkpoint-root',a.work/'checkpoints'/tag/'runs','--model',a.model,'--device',a.device,'--conditions',*a.visits])
    for visit in a.visits:
        command([sys.executable,ROOT/'experiments/response_full_recording.py','--model',a.model,'--visit',visit,'--branches',*a.branches,'--workspace',a.work/'workspace','--coordinates',a.work/'sampling/response_head_training.csv','--storage',a.work/'checkpoints','--device',a.device,'--workers',a.workers,'--inference-batch-size',a.batch_size])


def export(a):
    import pandas as pd
    from .metrics import summarize
    rows=[]
    for model in MODELS:
        for branch in ['frozen','finetuned']:
            path=a.work/'local/diagnosis_probability_mean_20260924'/model/f'{branch}_predictions.csv'
            if path.exists():
                d=pd.read_csv(path)
                rows.extend(dict(task='diagnosis',model=model,branch=branch,target=t,**summarize(d,t)) for t in ['patient_label','expert_label'])
            for visit in ['PRE','POST']:
                path=a.work/'local/response_full_recording_20260925'/f'{model}_{visit}'/f'{branch}_predictions.csv'
                if path.exists():rows.append(dict(task='response_'+visit,model=model,branch=branch,target='label',**summarize(pd.read_csv(path))))
    out=a.work/'results';out.mkdir(exist_ok=True)
    for path in (a.work/'predictions').glob('*.csv'):
        d=pd.read_csv(path);targets=['patient_diagnosis_label','clinician_diagnosis_label'] if path.stem.startswith('diagnosis') else ['label']
        rows.extend(dict(task=path.stem,model='qeeg',branch='logistic',target=t,**summarize(d,t)) for t in targets)
    if not rows:raise FileNotFoundError('No held-out predictions found. Run experiments first.')
    frame=pd.DataFrame(rows);frame.to_csv(out/'classification_metrics.csv',index=False)
    frame[['task','model','branch','target','cv_auroc','balanced_accuracy','accuracy','sensitivity','specificity','precision','f1','average_precision']].to_csv(out/'classification_point_estimates.csv',index=False)
    print('Exported',len(frame),'evaluations to',out)


def main():
    parser=argparse.ArgumentParser(description='IESSEEG dataset and paper experiments')
    parser.add_argument('command',choices=['validate','prepare','preprocess','extract-diagnosis','diagnosis','response','extract-qeeg','qeeg','extract-context','context','metrics'])
    parser.add_argument('--data',type=Path,required=True,help='Downloaded Capur/IESSEEG dataset root')
    parser.add_argument('--work',type=Path,required=True,help='Writable directory for caches, checkpoints, and outputs')
    parser.add_argument('--pretrained',type=Path,help='Directory containing official pretrained checkpoints')
    parser.add_argument('--device',help='Explicit CUDA device, e.g. cuda:0; required for signal extraction and neural methods')
    parser.add_argument('--model',choices=MODELS,default='biot')
    parser.add_argument('--models',nargs='+',choices=['biot','labram','cbramod'],default=['biot','labram','cbramod'])
    parser.add_argument('--task',choices=['diagnosis','response'],default='diagnosis')
    parser.add_argument('--visits',nargs='+',choices=['PRE','POST'],default=['PRE','POST'])
    parser.add_argument('--branches',nargs='+',choices=['frozen','finetuned'],default=['frozen','finetuned'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--limit',type=int,help='Extraction smoke check only; incomplete caches cannot be used for paper results')
    parser.add_argument('--metadata-only',action='store_true')
    a=parser.parse_args();a.data=a.data.expanduser().resolve();a.work=a.work.expanduser().resolve();a.work.mkdir(parents=True,exist_ok=True)
    if a.command in ['preprocess','extract-diagnosis','diagnosis','response','extract-qeeg','extract-context'] and (not a.device or not a.device.startswith('cuda:')):
        parser.error('Select an explicit CUDA device with --device cuda:N')
    if a.command in ['preprocess','extract-diagnosis','diagnosis','response','extract-qeeg','extract-context']:
        import torch
        if not torch.cuda.is_available():parser.error('CUDA is unavailable; no CPU fallback is used')
        torch.cuda.set_device(a.device)
    environment(a)
    if a.command=='validate':print(json.dumps(Dataset(a.data).validate(not a.metadata_only),indent=2));return
    if a.command=='prepare':print('Prepared',prepare(a.data,a.work));return
    if not (a.work/'dataset.json').exists():parser.error('Run prepare with the same --data and --work first')
    if json.loads((a.work/'dataset.json').read_text())['dataset']!=str(a.data):parser.error('This work directory is already linked to a different dataset')
    if a.command in ['extract-qeeg','qeeg','extract-context','context']:
        from . import tabular
        {'extract-qeeg':tabular.extract_qeeg,'qeeg':tabular.qeeg,'extract-context':tabular.context_extract,'context':tabular.context_fit}[a.command](a)
    else:{'preprocess':preprocess,'extract-diagnosis':extract_diagnosis,'diagnosis':diagnosis,'response':response,'metrics':export}[a.command](a)

if __name__=='__main__':main()
