# CodeBrain baseline

This adapter fine-tunes the official ICLR 2026 CodeBrain EEGSSM checkpoint on
the fixed IESSEEG patient folds. It uses 30-second, 19-channel referential
windows at 200 Hz. Signals stored in microvolts are divided by 100, matching
the scale used by the official clinical-dataset loaders.

Clone the official implementation and expose it with
`IESSEEG_CODEBRAIN_ROOT` if it is not at `../external_models/CodeBrain` relative
to this repository. Download `CodeBrain.pth` from
<https://huggingface.co/YjMajy/CodeBrain> into `pretrained-models/`.
The official source repository is Apache-2.0 licensed; pretrained weights are
downloaded separately and are not committed here.

The released downstream heads flatten a dataset-specific number of channels
and seconds. IESSEEG instead applies mean pooling over the unchanged pretrained
features followed by a binary linear head, so the head does not encode a
dataset-specific input length.
