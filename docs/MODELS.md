# Pretrained model setup

Weights are supplied separately under their upstream terms. This repository
preserves the original model implementations and loads the pretrained encoder
parameters before training the benchmark head or fine-tuning.

Set `IESSEEG_PRETRAINED_DIR` (or pass `--pretrained`) to a directory containing:

| Model | File | Provider |
| --- | --- | --- |
| BIOT | `EEG-six-datasets-18-channels.ckpt` | [BIOT](https://github.com/ycq091044/BIOT) |
| LaBraM | `labram-base.pth` | [LaBraM](https://github.com/935963004/LaBraM) |
| CBraMod | `pretrained_weights.pth` | [CBraMod](https://github.com/wjq-learning/CBraMod) |
| LUNA | `LUNA_base.safetensors` | [PulpBio/LUNA](https://huggingface.co/PulpBio/LUNA) |
| CodeBrain | `CodeBrain.pth` | [YjMajy/CodeBrain](https://huggingface.co/YjMajy/CodeBrain) |
| CSBrain | `CSBrain.pth` | [CSBrain checkpoint instructions](https://github.com/yuchen2199/CSBrain) |

## EEGPT

The experiment uses the Braindecode implementation. Keep EEGPT in a separate
environment because the other models were run with an older Braindecode:

```bash
python -m venv --system-site-packages /models/venv_eegpt
/models/venv_eegpt/bin/pip install --no-deps braindecode==1.7.0
/models/venv_eegpt/bin/pip install skorch torchinfo docstring-inheritance
hf download braindecode/eegpt-pretrained --local-dir /models/eegpt-pretrained
export IESSEEG_EEGPT_UPSTREAM=/models/eegpt-pretrained
export IESSEEG_PYTHON_EEGPT=/models/venv_eegpt/bin/python
```

Invoke `python -m iesseeg` with `/models/venv_eegpt/bin/python` for EEGPT feature
extraction and response experiments. The diagnosis shell runner also recognizes
`IESSEEG_PYTHON_EEGPT`.

## REVE

The frozen and response loaders require both the model and electrode-position
model in a Hugging Face cache. Download them before running:

```bash
hf download brain-bzh/reve-base --revision fa9a2163a4b7c0a42c8e28b56077ef9c368944dc --cache-dir /models/hf_cache/hub
hf download brain-bzh/reve-positions --revision befa5b57a455b77cf302daf610c2e9ed8140bace --cache-dir /models/hf_cache/hub
export IESSEEG_HF_CACHE=/models/hf_cache
export HF_HOME=/models/hf_cache
```

REVE loads the upstream model's custom Python implementation with
`trust_remote_code=True`. These dependencies are not copied into this repository.

## CodeBrain and CSBrain

Provide the original source checkouts as external dependencies:

```bash
git clone https://github.com/jingyingma01/CodeBrain.git /models/CodeBrain
git clone https://github.com/yuchen2199/CSBrain.git /models/CSBrain
export IESSEEG_CODEBRAIN_ROOT=/models/CodeBrain
export IESSEEG_CSBRAIN_ROOT=/models/CSBrain
```

Install the upstream requirements, including `opt_einsum` used by CodeBrain, in the CUDA environment. The adapters check the expected source
modules before loading. CSBrain source and weights are not redistributed here.

## Native model inputs

| Model | Diagnosis window | Response window | Signal input |
| --- | ---: | ---: | --- |
| BIOT | 30 s | 30 s | 18 bipolar channels, 200 Hz |
| LaBraM | 10 s | 16 s | 19 scalp channels, 200 Hz |
| CBraMod | 30 s | 30 s | 19 scalp channels, 200 Hz |
| EEGPT | 4 s | 4 s | 19 scalp channels, resampled to 250 Hz |
| LUNA | 30 s | 30 s | 22 bipolar channels, resampled to 256 Hz |
| REVE | 30 s | 30 s | 19 scalp channels, 200 Hz |
| CodeBrain | 30 s | 30 s | 19 scalp channels, 200 Hz |
| CSBrain | 30 s | 30 s | 19 scalp channels, 200 Hz |

Scaling and filtering are model-specific. The diagnosis pipeline preprocesses
complete segments before windowing. In the original CBraMod diagnosis pipeline,
simulated routine EEG inputs additionally discard the first and last 60 seconds
after filtering, leaving 56 complete 30-second windows per routine segment.
The release retains this preprocessing setting to reproduce the reported results;
CBraMod training segments and the other diagnosis input families are not trimmed. The response pipeline reads and prepares
native windows from long recordings. Section 6 computes a mean of frozen
embeddings within each 30-minute segment and uses 10-second LaBraM windows.

Do not interchange frozen and fine-tuned checkpoints or caches between tasks.
No fine-tuned weights are included in the release.
