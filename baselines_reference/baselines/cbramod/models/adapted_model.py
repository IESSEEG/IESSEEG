import torch
import torch.nn as nn

from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        """
        param: Namespace containing:
          - param.use_pretrained_weights (bool)
          - param.foundation_dir (str) -> path to .pth weights
          - param.cuda (int) -> which GPU device
          - param.dropout (float)
          - param.num_of_classes (int) -> e.g. 2 for binary, >2 for multi-class
          - other training hyperparams
        """
        super(Model, self).__init__()

        # 1) Build CBraMod backbone
        #    We'll set seq_len=30 if your dataset shape is (B,19,30,200).
        #    in_dim=200, out_dim=200, d_model=200 are from your original config.
        #    n_layer=12, nhead=8, etc.
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,  # adjust if your x has a different "seq_len"
            n_layer=12,
            nhead=8
        )

        # 2) Load pretrained weights if needed
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {param.foundation_dir}")
        else:
            print("No pretrained weights loaded. Training from scratch.")
        # Remove the final projection head from backbone
        self.backbone.proj_out = nn.Sequential()

        # 3) Decide how many output classes
        #    - If you're doing binary classification with BCEWithLogits,
        #      you might use output_dim=1
        #    - If multi-class with CrossEntropy, output_dim=param.num_of_classes
        if param.num_of_classes == 2:
            output_dim = 1
        else:
            output_dim = param.num_of_classes

        # 4) Build a simple classifier on top of flattened backbone
        #    The dimension "19*5*200" from your old code might be 19*(seq_len)*(patch_size).
        #    If your shape is (B,19,30,200) => that is 19*30*200=114000
        #    adjust the MLP hidden dims as you like
        self.classifier = nn.Sequential(
            nn.Linear(19 * 30 * 200, 30 * 200),  # 19->some hidden
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(30 * 200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, output_dim)
        )

    def forward(self, x):
        """
        x shape => (batch_size, n_channels=19, seq_len=30, patch_size=200).

        1) backbone => (B, n_channels, seq_len, hidden_dim) or something,
           then we flatten to (B, n_channels*seq_len*hidden_dim).
        2) final classifier => either shape (B,1) for binary or (B,C) for multi-class.
        """
        bz, ch_num, seq_len, patch_size = x.shape
        # pass x through CBraMod backbone
        feats = self.backbone(x)
        # feats shape => (bz, 200, ch_num, seq_len) or (bz, ch_num, seq_len, 200), etc,
        #   depending on your CBraMod internal arrangement.
        #   If your code ends up with feats shape => (bz, ch_num, seq_len, 200),
        #   we flatten that to a single vector per sample:
        feats = feats.contiguous().view(bz, ch_num * seq_len * 200)

        # pass flattened feats into classifier
        out = self.classifier(feats)  # shape => (B, output_dim)

        # if binary => shape (B,1). If multi-class => shape (B, C).
        # your trainer can handle either shape:
        # BCEWithLogits => (B,1)
        # CrossEntropy   => (B,C)
        return out
