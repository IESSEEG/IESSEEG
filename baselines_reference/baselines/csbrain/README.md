# CSBrain adapter

The adapter uses 30-second, 19-channel windows at 200 Hz, with an explicit
10–20 electrode ordering. Inputs in microvolts are divided by 100. Mean pooling
over pretrained features is followed by a binary linear head.

Provide the upstream source through `IESSEEG_CSBRAIN_ROOT` and place
`CSBrain.pth` in `IESSEEG_PRETRAINED_DIR`. Source and weights are obtained
separately; see the [model setup guide](../../../docs/MODELS.md#codebrain-and-csbrain).
