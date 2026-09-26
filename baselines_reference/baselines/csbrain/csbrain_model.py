"""IESSEEG adapter for the official CSBrain checkpoint."""

import os
import sys

import torch
import torch.nn as nn


CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
]
REGIONS = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
TOPOLOGY = {
    0: ["FP1", "F3", "F7", "FZ", "F4", "F8", "FP2"],
    4: ["C3", "CZ", "C4"],
    1: ["P3", "PZ", "P4"],
    2: ["T3", "T5", "T6", "T4"],
    3: ["O1", "O2"],
}


def topology_order(channels=CHANNELS, regions=REGIONS):
    groups = {}
    for index, (channel, region) in enumerate(zip(channels, regions)):
        groups.setdefault(region, []).append((index, channel))
    order = []
    for region in sorted(groups):
        order.extend(
            index
            for index, _ in sorted(
                groups[region], key=lambda item: TOPOLOGY[region].index(item[1])
            )
        )
    return order


def _upstream_root():
    default = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "external_models", "CSBrain")
    )
    root = os.environ.get("IESSEEG_CSBRAIN_ROOT", default)
    if not os.path.isfile(os.path.join(root, "models", "CSBrain.py")):
        raise FileNotFoundError(
            "CSBrain source not found. Clone https://github.com/yuchen2199/CSBrain "
            "and set IESSEEG_CSBRAIN_ROOT to that checkout."
        )
    return root


def _backbone_class():
    root = _upstream_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.CSBrain import CSBrain

    return CSBrain


class CSBrainClassifier(nn.Module):
    """Official pretrained backbone adapted to the 19-channel 10-20 montage."""

    def __init__(self, pretrained, dropout=0.1):
        super().__init__()
        CSBrain = _backbone_class()
        self.backbone = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
            brain_regions=REGIONS,
            sorted_indices=topology_order(),
        )
        state = torch.load(pretrained, map_location="cpu", weights_only=False)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.backbone.load_state_dict(state, strict=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(200, 2))

    def forward_features(self, x):
        return self.backbone(x).mean(dim=(1, 2))

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(pretrained, dropout=0.1):
    return CSBrainClassifier(pretrained, dropout=dropout)
