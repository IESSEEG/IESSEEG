"""IESSEEG adapter for the official CodeBrain EEGSSM checkpoint."""

import os
import sys

import torch
import torch.nn as nn


def _upstream_root():
    default = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "external_models", "CodeBrain")
    )
    root = os.environ.get("IESSEEG_CODEBRAIN_ROOT", default)
    if not os.path.isfile(os.path.join(root, "Models", "SSSM.py")):
        raise FileNotFoundError(
            "CodeBrain source not found. Clone https://github.com/jingyingma01/CodeBrain "
            "and set IESSEEG_CODEBRAIN_ROOT to that checkout."
        )
    return root


def _backbone_class():
    root = _upstream_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from Models.SSSM import SSSM

    return SSSM


class CodeBrainClassifier(nn.Module):
    """Official pretrained backbone with a length-independent binary head."""

    def __init__(self, pretrained, dropout=0.1):
        super().__init__()
        SSSM = _backbone_class()
        self.backbone = SSSM(
            in_channels=200,
            res_channels=200,
            skip_channels=200,
            out_channels=200,
            num_res_layers=8,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=4096,
            codebook_size_f=4096,
            if_codebook=False,
        )
        state = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.backbone.load_state_dict(state, strict=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(200, 2))

    def forward_features(self, x):
        features = self.backbone(x)
        if features.ndim == 3:
            features = features.unsqueeze(0)
        return features.mean(dim=(1, 2))

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(pretrained, dropout=0.1):
    return CodeBrainClassifier(pretrained, dropout=dropout)
