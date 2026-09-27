# Pretrained models

The [setup script](../scripts/setup.sh) downloads the pretrained weights for all
eight models. To download weights separately after installation:

```bash
python scripts/prepare_models.py --output pretrained
source pretrained/env.sh
```

To prepare only selected models:

```bash
python scripts/prepare_models.py --models biot labram cbramod --output pretrained
source pretrained/env.sh
```

The script downloads checkpoints, obtains the CodeBrain and CSBrain source
checkouts, and writes the environment variables used by the benchmark to
`env.sh`. It uses fixed upstream revisions and reuses existing downloads.
If a download fails, rerun the same command. Model files remain outside Git.

| Model | Provider | Local checkpoint |
| --- | --- | --- |
| BIOT | [BIOT](https://github.com/ycq091044/BIOT) | `EEG-six-datasets-18-channels.ckpt` |
| LaBraM | [LaBraM](https://github.com/935963004/LaBraM) | `labram-base.pth` |
| CBraMod | [CBraMod](https://huggingface.co/weighting666/CBraMod) | `pretrained_weights.pth` |
| EEGPT | [Braindecode EEGPT](https://huggingface.co/braindecode/eegpt-pretrained) | `eegpt/model.safetensors` |
| LUNA | [PulpBio/LUNA](https://huggingface.co/PulpBio/LUNA) | `LUNA_base.safetensors` |
| REVE | [REVE](https://huggingface.co/brain-bzh/reve-base) and [positions](https://huggingface.co/brain-bzh/reve-positions) | `hf_cache/hub/` |
| CodeBrain | [CodeBrain](https://huggingface.co/YjMajy/CodeBrain) | `CodeBrain.pth` |
| CSBrain | [CSBrain](https://github.com/yuchen2199/CSBrain#finetuning-csbrain-on-downstream-datasets) | `CSBrain.pth` |

All paths in the table are relative to the selected output directory.
Checkpoint licenses are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).

EEGPT uses Braindecode 1.7.0 from the shared requirements. REVE loads Python
model code from the pinned upstream Hugging Face revisions.

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
complete segments before windowing. For CBraMod diagnosis,
simulated routine EEG inputs additionally discard the first and last 60 seconds
after filtering, leaving 56 complete 30-second windows per routine segment.
CBraMod training segments and the other diagnosis input families are not
trimmed. Response preprocessing is applied to individual windows from long
recordings. The recording-context analysis averages frozen embeddings within
each 30-minute segment and uses 10-second LaBraM windows.

The download script provides pretrained weights. Fine-tuned checkpoints are
created by the [diagnosis](experiments/diagnosis.md) and
[response](experiments/response.md) training commands.
