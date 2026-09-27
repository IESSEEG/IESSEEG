# CSBrain adapter

The adapter uses 30-second, 19-channel windows at 200 Hz, with an explicit
10–20 electrode ordering. Inputs in microvolts are divided by 100. Mean pooling
over pretrained features is followed by a binary linear head.

Run the [model preparation script](../../../docs/MODELS.md) to download the
upstream source and checkpoint and configure their paths.
