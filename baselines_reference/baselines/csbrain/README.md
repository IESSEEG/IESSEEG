# CSBrain baseline

This adapter fine-tunes the official NeurIPS 2025 CSBrain checkpoint on the
fixed IESSEEG patient folds. It uses 30-second, 19-channel referential windows
at 200 Hz and supplies an explicit topological ordering for the standard 10-20
electrodes. Signals stored in microvolts are divided by 100, equivalent to the
official loader's conversion of volt-scale inputs.

Clone the official implementation and expose it with `IESSEEG_CSBRAIN_ROOT` if
it is not at `../external_models/CSBrain` relative to this repository. Obtain
`CSBrain.pth` from the checkpoint link in
<https://github.com/yuchen2199/CSBrain> and place it in `pretrained-models/`.
The official repository does not currently include a license file. This
benchmark therefore imports a user-supplied checkout and does not redistribute
the upstream source or checkpoint.

The released downstream heads flatten a dataset-specific number of channels
and seconds. IESSEEG instead applies mean pooling over the unchanged pretrained
features followed by a binary linear head, so the head does not encode a
dataset-specific input length.
