#!/usr/bin/env python
"""Check the shared model environment and an explicitly selected CUDA device."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error('Use Python 3.11 for the benchmark environment.')
    for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'torchvision',
                 'torchaudio', 'pytorch_lightning', 'timm', 'transformers', 'mne',
                 'pyedflib', 'braindecode', 'linear_attention_transformer',
                 'rotary_embedding_torch', 'safetensors', 'huggingface_hub', 'gdown'):
        importlib.import_module(name)
    from braindecode.models import EEGPT
    import torch
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('A working CUDA device is required, for example --device cuda:0.')
    torch.cuda.set_device(device)
    result = torch.ones(2, device=device).sum().item()
    assert result == 2
    print(f'Environment ready: Python 3.11, PyTorch {torch.__version__}, {device} ({torch.cuda.get_device_name(device)}).')


if __name__ == '__main__':
    main()
