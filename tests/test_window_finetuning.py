"""Sampling and optimizer checks for long-EEG inherited-window training."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

path=Path(__file__).resolve().parents[1]/'experiments/finetune_response_windows.py'
spec=importlib.util.spec_from_file_location('window_ft',path)
ft=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def test_sampling_balances_patients_and_respects_recording_end():
    frame=pd.DataFrame({'source_duration':[50.,500.,5000.]})
    sample=ft.Coordinates(frame,30,100,42)
    one=sample.values()
    counts=np.bincount([i for i,_ in one])
    assert counts.max()-counts.min()<=1
    assert all(0<=t<=frame.source_duration.iloc[i]-30 for i,t in one)
    assert one==sample.values()
    sample.epoch=1
    assert one!=sample.values()


def test_validation_uses_equal_fixed_numbers_per_patient():
    frame=pd.DataFrame({'source_duration':[50.,500.,5000.]})
    sample=ft.Coordinates(frame,16,64,43,True)
    assert len(sample)==192
    assert np.array_equal(np.bincount([i for i,_ in sample]),[64,64,64])
    assert list(sample)==list(sample)


def test_layer_decay_does_not_omit_encoder_or_head():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.name='labram';self.encoder=torch.nn.Module()
            self.encoder.patch_embed=torch.nn.Linear(2,2)
            self.encoder.blocks=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(12)])
            self.head=torch.nn.Linear(2,1)
    settings=dict(layer_decay=.65,lr=5e-4,weight_decay=.05,optimizer='AdamW',betas=[.9,.999],eps=1e-8)
    model=Model();opt=ft.make_optimizer(model,settings)
    groups={id(p):g for g in opt.param_groups for p in g['params']}
    assert np.isclose(groups[id(model.encoder.patch_embed.weight)]['lr'],5e-4*.65**13)
    assert groups[id(model.head.weight)]['lr']==5e-4
    assert groups[id(model.head.bias)]['weight_decay']==0


def test_cosine_warmup_and_end():
    s=dict(scheduler='cosine',warmup_epochs=5,warmup_lr=1e-6,min_lr=1e-6,lr=5e-4)
    assert ft.learning_rate(s,0,80,100)<ft.learning_rate(s,399,80,100)
    assert np.isclose(ft.learning_rate(s,399,80,100),5e-4)
    assert np.isclose(ft.learning_rate(s,7999,80,100),1e-6)
