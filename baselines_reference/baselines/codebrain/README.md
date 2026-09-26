# CodeBrain adapter

The adapter uses 30-second, 19-channel windows at 200 Hz. Inputs in microvolts
are divided by 100. Mean pooling over pretrained features is followed by a
binary linear head.

Provide the upstream source through `IESSEEG_CODEBRAIN_ROOT` and place
`CodeBrain.pth` in `IESSEEG_PRETRAINED_DIR`. See the
[model setup guide](../../../docs/MODELS.md#codebrain-and-csbrain) for downloads.
