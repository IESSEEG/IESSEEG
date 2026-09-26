# model_zoo.py
import torch
from model import (
    SPaRCNet, ContraWR, CNNTransformer,
    FFCL, STTransformer, BIOTClassifier
)

def build_model(name, args):
    """Return an instantiated backbone + (optionally) load weights."""
    name = name.upper()

    if name == "SPARCNET":
        net = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sfreq * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4, growth_rate=16, bn_size=16,
            drop_rate=0.5, conv_bias=True, batch_norm=True,
        )

    elif name == "CONTRAWR":
        net = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif name == "CNNTRANSFORMER":
        net = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            dropout=0.2, nhead=4, emb_size=256,
        )

    elif name == "FFCL":
        net = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sfreq * args.sample_length),
            shrink_steps=20,
        )

    elif name == "STTRANSFORMER":
        net = STTransformer(
            emb_size=256, depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(args.sfreq * args.sample_length),
            n_channels=args.in_channels,
        )

    elif name == "BIOT":
        net = BIOTClassifier(
            n_classes=args.n_classes,
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        # optional paper checkpoint
        # check if pretrain_model_path is provided
        if getattr(args, "pretrain_model_path", None) is not None and args.sfreq == 200:
            net.biot.load_state_dict(torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False))
            print(f"[✓] loaded BIOT weights: {args.pretrain_model_path}")

    else:
        raise ValueError(f"Unknown model name: {name}")

    return net
