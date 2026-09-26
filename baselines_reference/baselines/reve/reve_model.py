"""REVE with a classification head.

The released REVE checkpoint is a feature extractor: it returns
per-channel, per-patch embeddings rather than class scores. This wraps it
with the model's own attention pooling over those embeddings followed by
a linear classifier, so the fine-tuned model is the pre-trained encoder
plus one new layer -- the same shape of adaptation the other foundation
models in this benchmark get.
"""

import torch
import torch.nn as nn

REVE_REPO = "brain-bzh/reve-base"


class ReveClassifier(nn.Module):
    """Pre-trained REVE encoder + attention pooling + linear head."""

    def __init__(self, repo_id=REVE_REPO, n_classes=2, dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(repo_id, revision="fa9a2163a4b7c0a42c8e28b56077ef9c368944dc", trust_remote_code=True)
        embed_dim = self.encoder.config.embed_dim

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, eeg, pos):
        # (B, C, T) -> (B, C, patches, E), then pooled to (B, E) by the
        # encoder's own attention pooling.
        features = self.encoder(eeg, pos)
        pooled = self.encoder.attention_pooling(features)
        return self.head(self.dropout(pooled))

    def layer_of(self, name, n_layers):
        """Depth index for layer-wise learning-rate decay.

        Patch embedding and positional machinery sit at the input, the
        transformer blocks step through the middle, and the new head
        trains at the full rate.
        """
        if name.startswith("head") or name.startswith("dropout"):
            return n_layers - 1
        if "cls_query_token" in name or "final_layer" in name:
            return n_layers - 2
        for part in name.split("."):
            if part.isdigit():
                return min(int(part) + 1, n_layers - 2)
        return 0

    @property
    def n_encoder_layers(self):
        return int(getattr(self.encoder.config, "depth", 22))
