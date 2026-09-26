"""Differentiable adapters for the remaining released EEG encoders.

The encoder is loaded from the same pretrained checkpoint as the frozen study.
Native fixed tensors retain their status; computational backbone parameters train.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from iesseeg_paper.input_studies import SCALP, prepare_encoder_input
from frozen_encoders import load_eegpt, load_luna, load_reve, load_codebrain, load_csbrain

NAMES = ('eegpt', 'luna', 'reve', 'codebrain', 'csbrain')
ROOT = Path(__file__).resolve().parents[1]


def prepare(raw, name, seconds):
    if name == 'luna':
        sys.path.insert(0, str(ROOT/'baselines_reference/baselines/luna'))
        from luna_data import TCP_MONTAGE
        names = [*SCALP, 'A1', 'A2']
        x = np.stack([raw[names.index(a)]-raw[names.index(b)]
                      for a,b in (p.split('-') for p in TCP_MONTAGE)])
        x = resample_poly(x, 32, 25, axis=-1)
        return (x-x.mean(-1,keepdims=True))/(x.std(-1,keepdims=True)+1e-8)
    x = prepare_encoder_input(raw, name)
    if name == 'eegpt':
        from braindecode.preprocessing import exponential_moving_standardize
        return exponential_moving_standardize(
            np.ascontiguousarray(resample_poly(x,5,4,axis=-1),dtype=np.float64),
            factor_new=.001,init_block_size=None,eps=1e-4)
    if name == 'reve':
        return (x-x.mean(-1,keepdims=True))/(x.std(-1,keepdims=True)+1e-8)
    return x.reshape(19,seconds,200)/100.


def load(name, workspace, device, settings):
    # Do not modify the original third-party checkouts.
    os.environ.setdefault('IESSEEG_CODEBRAIN_ROOT',str(workspace/'external_models/CodeBrain'))
    os.environ.setdefault('IESSEEG_CSBRAIN_ROOT',str(workspace/'external_models/CSBrain'))
    result = dict(eegpt=load_eegpt,luna=load_luna,reve=load_reve,
                  codebrain=load_codebrain,csbrain=load_csbrain)[name](workspace,device)
    encoder, positions = result if isinstance(result,tuple) else (result,None)
    for key,p in encoder.named_parameters():
        # Frozen loaders disable the entire network for EEGPT/REVE only.
        if name in ('eegpt','reve') and p.is_floating_point(): p.requires_grad_(True)
        if key.startswith(('classifier.','decoder_head.','final_layer.','lm_head_t.','lm_head_f.')):
            p.requires_grad_(False)
    if name == 'luna':
        from timm.layers import DropPath
        for block, probability in zip(encoder.blocks,np.linspace(0,settings['drop_path'],len(encoder.blocks))):
            for attr in ('drop_path1','drop_path2'):
                setattr(block,attr,DropPath(float(probability)) if probability else torch.nn.Identity())
    if name == 'codebrain':
        # The official S4 dropout uses Dropout/Dropout2d modules.
        for module in encoder.modules():
            if isinstance(module,(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d)):
                module.p=settings['dropout']
    return encoder,positions,dict(eegpt=512,luna=256,reve=512,codebrain=200,csbrain=200)[name]


def encode(model,x):
    if model.name == 'luna':
        from einops import rearrange
        pos=model.positions.unsqueeze(0).expand(len(x),-1,-1)
        tokens,_=model.encoder.prepare_tokens(x,pos,mask=None)
        latent,_=model.encoder.cross_attn(tokens)
        latent=rearrange(latent,'(b t) q d -> b t (q d)',b=len(x))
        for block in model.encoder.blocks: latent=block(latent)
        return model.encoder.norm(latent).mean(1)
    if model.name == 'reve':
        pos=model.positions.unsqueeze(0).expand(len(x),-1,-1)
        return model.encoder.attention_pooling(model.encoder(x,pos))
    z=model.encoder(x)
    if model.name=='codebrain' and z.ndim==3: z=z.unsqueeze(0)
    return z.mean((1,2))
