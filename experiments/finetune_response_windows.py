#!/usr/bin/env python
"""Fine-tune encoders on randomly sampled long-EEG windows with inherited labels.

One training run per outer fold: three patient folds train, one validates, one
is held out. The validation-selected checkpoint is not refitted. Source EEG and
upstream model files are read-only. This stage does not fit recording aggregators.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/iesseeg_numba')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'encoders')]
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, Sampler
from iesseeg_paper.input_studies import read_interval, prepare_encoder_input, SCALP
from iesseeg_paper.data import patient_info
from frozen_encoders import load_biot, load_cbramod, load_labram
from experiments import additional_response_encoders as extra


class WindowDataset(Dataset):
    """Read a requested coordinate, with no outcome-dependent rejection/resampling."""
    def __init__(self, frame, model, seconds):
        self.rows = frame.to_dict('records')
        self.model, self.seconds = model, seconds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, coordinate):
        index, start = coordinate
        r = self.rows[index]
        raw = read_interval(Path(r['source']), float(start), self.seconds)
        if self.model in extra.NAMES:
            x = extra.prepare(raw, self.model, self.seconds)
        else:
            x = prepare_encoder_input(raw, self.model)
        if self.model == 'biot':
            x = x / (np.quantile(np.abs(x), .95, axis=-1, keepdims=True) + 1e-8)
        elif self.model not in extra.NAMES:
            x = x.reshape(19, self.seconds, 200) / 100.
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite input; refusing to silently replace a window')
        return (torch.from_numpy(x), float(r['label']), int(r['patient_id']), float(start))


class Coordinates(Sampler):
    """Each epoch samples patients evenly, then starts uniformly at 200-Hz ticks."""
    def __init__(self, frame, seconds, count, seed, fixed_per_patient=False):
        self.frame, self.seconds, self.count = frame, seconds, count
        self.seed, self.epoch, self.fixed = seed, 0, fixed_per_patient

    def values(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        n = len(self.frame)
        if self.fixed:
            indices = np.repeat(np.arange(n), self.count)
        else:
            indices = np.tile(np.arange(n), math.ceil(self.count/n))[:self.count]
            # Randomize which patients receive the remaining one draw.
            indices = rng.permutation(n)[indices]
            rng.shuffle(indices)
        durations = self.frame.source_duration.to_numpy(float)
        return [(int(i), float(rng.integers(0, int((durations[i]-self.seconds)*200)+1))/200)
                for i in indices]

    def __iter__(self):
        return iter(self.values())

    def __len__(self):
        return self.count * len(self.frame) if self.fixed else self.count


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(frame, name, cfg, seed, train=False, workers=8):
    seconds = cfg['models'][name]['window_seconds']
    count = cfg['samples_per_epoch'] if train else cfg['validation_windows_per_patient']
    sampler = Coordinates(frame, seconds, count, seed, fixed_per_patient=not train)
    ds = WindowDataset(frame, name, seconds)
    loader = DataLoader(ds, batch_size=cfg['models'][name].get('batch_size',cfg['batch_size']), sampler=sampler,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0, worker_init_fn=worker_init,
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if workers else {}))
    return loader, sampler


class WindowPredictor(torch.nn.Module):
    def __init__(self, name, workspace, device, settings):
        super().__init__()
        self.name = name
        if name in extra.NAMES:
            self.encoder,positions,width=extra.load(name,workspace,device,settings)
            if positions is not None: self.register_buffer('positions',positions)
            self.head=torch.nn.Sequential(torch.nn.LayerNorm(width),
                torch.nn.Dropout(settings.get('head_dropout',0.)),torch.nn.Linear(width,1)).to(device)
            return
        self.encoder = {'biot':load_biot, 'cbramod':load_cbramod, 'labram':load_labram}[name](workspace, device)
        # Integer channel-index tensors are not trainable parameters.
        for p in self.encoder.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)
        if name == 'labram':
            import utils
            from timm.layers import DropPath
            self.input_chans = utils.get_input_chans(SCALP)
            for block, probability in zip(self.encoder.blocks,
                    np.linspace(0., settings['drop_path'], len(self.encoder.blocks))):
                block.drop_path = DropPath(float(probability)) if probability else torch.nn.Identity()
        width = 256 if name == 'biot' else 200
        # A compact task head; not CBraMod's original large flattened MLP.
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width,1)).to(device)

    def encode(self, x):
        if self.name in extra.NAMES: return extra.encode(self,x)
        if self.name == 'labram':
            return self.encoder.forward_features(x, input_chans=self.input_chans,
                                                 return_patch_tokens=True).mean(1)
        z = self.encoder(x)
        return z.mean((1,2)) if self.name == 'cbramod' else z

    def forward(self, x):
        return self.head(self.encode(x)).flatten()


def make_optimizer(model, settings):
    groups = {}
    depth = len(model.encoder.blocks) if model.name == 'labram' else 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer = depth + 1
        if model.name == 'labram' and name.startswith('encoder.'):
            local = name.removeprefix('encoder.')
            if local.startswith('blocks.'):
                layer = int(local.split('.')[1]) + 1
            elif local.startswith(('patch_embed.', 'pos_embed', 'time_embed', 'cls_token')):
                layer = 0
        scale = settings['layer_decay'] ** (depth+1-layer) if depth else 1.
        if model.name=='luna' and name.startswith('encoder.blocks.'):
            scale=settings['layer_decay']**(len(model.encoder.blocks)-int(name.split('.')[2]))
        if model.name=='eegpt' and name.startswith('encoder.target_encoder.'):
            local=name.removeprefix('encoder.target_encoder.')
            d=len(model.encoder.target_encoder.blocks)
            layer=int(local.split('.')[1])+1 if local.startswith('blocks.') else (d+1 if local.startswith('norm') else 0)
            scale=settings['layer_decay']**(d+1-layer)
        # LaBraM official code excludes normalization, biases and positional embeddings from decay.
        no_decay = model.name in ('labram','eegpt','reve') and (p.ndim == 1 or name.endswith('.bias') or
                    name.endswith(('pos_embed', 'time_embed', 'cls_token')))
        decay = 0. if no_decay else settings['weight_decay']
        key = (scale, decay)
        groups.setdefault(key, dict(params=[], lr_scale=scale,
                                    lr=settings['lr']*scale, weight_decay=decay))['params'].append(p)
    optimizer_class=getattr(torch.optim,settings['optimizer'],None)
    if settings['optimizer']=='StableAdamW':
        from third_party.reve_optimizer.stable_adamw import StableAdamW
        optimizer_class=StableAdamW
    optimizer = optimizer_class(list(groups.values()),
                    lr=settings['lr'], betas=tuple(settings['betas']), eps=settings['eps'])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert expected == actual, 'Optimizer omitted trainable tensors'
    return optimizer


def learning_rate(settings, step, steps_per_epoch, max_epochs):
    if settings['scheduler'] == 'constant':
        return settings['lr']
    warm = settings['warmup_epochs'] * steps_per_epoch
    total = max_epochs * steps_per_epoch
    if step < warm:
        initial = settings.get('warmup_lr', settings['min_lr'])
        return initial + (settings['lr']-initial) * (step+1)/warm
    if settings['scheduler']=='plateau': return settings['lr']
    fraction = min(1., (step-warm)/max(1, total-warm-1))
    return settings['min_lr'] + .5*(settings['lr']-settings['min_lr'])*(1+math.cos(math.pi*fraction))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores, patients, starts = [], [], [], []
    total_loss, n = 0., 0
    for x,y,pid,start in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, dtype=torch.float32)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite validation/test logits')
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum').item()
        n += len(y)
        labels.extend(y.cpu().tolist()); scores.extend(logits.sigmoid().cpu().tolist())
        patients.extend(pid.tolist()); starts.extend(start.tolist())
    frame = pd.DataFrame(dict(patient_id=patients, start_seconds=starts, label=labels, probability=scores))
    return dict(window_auroc=float(roc_auc_score(labels,scores)),loss=total_loss/n), frame


def atomic_save(payload, path):
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(path)


def train_fold(args, cfg, rows, fold, condition, device):
    settings = cfg['models'][args.model]
    seed = cfg['seed'] + fold*100 + (0 if condition=='PRE' else 10000)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    validation_fold = (fold+1)%5
    train = rows[~rows.fold.isin([fold,validation_fold])].reset_index(drop=True)
    val = rows[rows.fold.eq(validation_fold)].reset_index(drop=True)
    test = rows[rows.fold.eq(fold)].reset_index(drop=True)
    assert (len(train),len(val),len(test)) == (30,10,10)
    assert not set(train.patient_id)&set(val.patient_id)
    assert not set(train.patient_id)&set(test.patient_id)
    assert all(f.label.nunique()==2 for f in [train,val,test])
    tag = Path(args.model)/condition/f'fold{fold}'
    out, ckpts = args.output_root/tag, args.checkpoint_root/tag
    out.mkdir(parents=True, exist_ok=True); ckpts.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=cfg, model=args.model, condition=condition, task=cfg['task'],
                    test_fold=fold, validation_fold=validation_fold,
                    train_patients=train.patient_id.tolist(), validation_patients=val.patient_id.tolist(),
                    test_patients=test.patient_id.tolist(), seed=seed,
                    head='pooled encoder features, LayerNorm, Linear to one logit',
                    input_manifest=str(args.input_manifest), checkpoint_root=str(ckpts))
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing run uses a different protocol; choose a new output directory')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    if (out/'complete.json').exists():
        print(f'SKIP complete {tag}',flush=True); return
    train_loader, train_sampler = make_loader(train,args.model,cfg,seed,True,cfg['workers'])
    val_loader, val_sampler = make_loader(val,args.model,cfg,seed+1,False,cfg['workers'])
    model=WindowPredictor(args.model,args.workspace,device,settings)
    optimizer=make_optimizer(model,settings)
    plateau=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5)
             if settings['scheduler']=='plateau' else None)
    start_epoch=0; best=-float('inf'); bad=0; best_epoch=0
    last=ckpts/'last.pt'; best_path=ckpts/'best.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.to(device)
        if plateau is not None and saved.get('scheduler') is not None: plateau.load_state_dict(saved['scheduler'])
        start_epoch=saved['epoch'];best=saved['best'];bad=saved['bad'];best_epoch=saved['best_epoch']
        if 'numpy_rng' in saved: np.random.set_state(saved['numpy_rng'])
        if 'python_rng' in saved: random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],device)
    print(f'START {tag}: 30 train / 10 validation / 10 test patients; {cfg["samples_per_epoch"]} draws/epoch; resume epoch {start_epoch}',flush=True)
    for epoch in range(start_epoch,cfg['max_epochs']):
        if bad>=cfg['early_stopping']['patience']:
            break
        tick=time.perf_counter(); model.train(); train_sampler.epoch=epoch
        sum_loss=0.;seen=0
        for step,(x,y,_,_) in enumerate(train_loader):
            rate=learning_rate(settings,epoch*len(train_loader)+step,len(train_loader),cfg['max_epochs'])
            if plateau is None or epoch<settings['warmup_epochs']:
                for group in optimizer.param_groups: group['lr']=rate*group['lr_scale']
            else: rate=optimizer.param_groups[0]['lr']/optimizer.param_groups[0]['lr_scale']
            optimizer.zero_grad(set_to_none=True)
            x=x.to(device,non_blocking=True);y=y.to(device,dtype=torch.float32)
            if settings.get('mixup_alpha',0)>0:
                lam=float(np.random.beta(settings['mixup_alpha'],settings['mixup_alpha']))
                permutation=torch.randperm(len(x),device=device)
                x=lam*x+(1-lam)*x[permutation];y=lam*y+(1-lam)*y[permutation]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad'],error_if_nonfinite=True)
            optimizer.step(); sum_loss+=loss.item()*len(y);seen+=len(y)
        metrics,_=evaluate(model,val_loader,device)
        if plateau is not None and epoch+1>=settings['warmup_epochs']: plateau.step(metrics['window_auroc'])
        if metrics['window_auroc']>best+cfg['early_stopping']['min_delta']:
            best=metrics['window_auroc'];best_epoch=epoch+1;bad=0
            atomic_save(dict(model=model.state_dict(),epoch=best_epoch,validation=metrics,protocol=protocol),best_path)
        elif epoch+1>settings['warmup_epochs']:
            bad+=1
        record=dict(epoch=epoch+1,train_loss=sum_loss/seen,validation=metrics,
                    best_epoch=best_epoch,bad_epochs=bad,seconds=time.perf_counter()-tick,lr=rate)
        with (out/'epochs.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
        atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                         best=best,bad=bad,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(device),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                         scheduler=plateau.state_dict() if plateau is not None else None),last)
        print(json.dumps(dict(run=str(tag),**record)),flush=True)
    del train_loader,val_loader
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval()
    test_cfg=dict(cfg,validation_windows_per_patient=cfg['test_windows_per_patient'])
    test_loader,test_sampler=make_loader(test,args.model,test_cfg,seed+2,False,cfg['workers'])
    metrics,predictions=evaluate(model,test_loader,device)
    predictions['fold']=fold;predictions['condition']=condition
    predictions.to_csv(out/'window_predictions.csv',index=False)
    # The resulting scores are window-level diagnostics, not the later aggregation benchmark.
    done=dict(model=args.model,condition=condition,fold=fold,task=cfg['task'],best_epoch=best_epoch,
              best_validation_window_auroc=best,test_window_metrics=metrics,checkpoint=str(best_path))
    (out/'complete.json').write_text(json.dumps(done,indent=2)+'\n')
    print('COMPLETE '+json.dumps(done),flush=True)
    del model,optimizer,test_loader
    torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=ROOT/'configs/sustained_window_finetuning.json')
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--model',choices=['biot','cbramod','labram',*extra.NAMES],required=True)
    p.add_argument('--device',required=True)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--conditions',nargs='+',choices=['PRE','POST'])
    p.add_argument('--smoke',action='store_true',help='One epoch, 64 training windows and two validation/test windows per patient')
    args=p.parse_args();cfg=json.loads(args.recipe.read_text())
    if cfg['task'] not in ['sustained','immediate']:raise ValueError('Unknown response endpoint')
    if cfg['refit']:raise ValueError('No refit authorized')
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):raise RuntimeError('CUDA required')
    device=torch.device(args.device);torch.cuda.set_device(device);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    if args.smoke:
        cfg.update(max_epochs=1,samples_per_epoch=64,validation_windows_per_patient=2,test_windows_per_patient=2,workers=2)
    if any(f not in range(5) for f in args.folds):raise ValueError('Invalid fold')
    manifest=pd.read_csv(args.input_manifest,dtype={'recording_id':str})
    manifest=manifest[(manifest.kind=='long')&(manifest.case_control_label=='CASE')]
    patients=patient_info(dict(workspace=args.workspace),cfg['task'])
    for condition in args.conditions or cfg['conditions']:
        frame=manifest[manifest.pre_post_treatment_label.eq(condition)].drop(columns=['fold','label'],errors='ignore')
        frame=frame.merge(patients,on='patient_id',validate='one_to_one').sort_values('patient_id').reset_index(drop=True)
        if len(frame)!=50 or frame.patient_id.nunique()!=50:raise ValueError('Incomplete long EEG cohort')
        if not all(Path(x).exists() for x in frame.source):raise FileNotFoundError('Missing source EDF')
        for fold in args.folds: train_fold(args,cfg,frame,fold,condition,device)

if __name__=='__main__':main()
