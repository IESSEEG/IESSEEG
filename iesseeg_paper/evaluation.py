"""CUDA selection and training-set feature standardization."""
import numpy as np
import torch

def require_cuda(device):
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Explicit working CUDA device required; no CPU fallback")
    torch.cuda.set_device(dev)
    torch.set_num_threads(2)
    return dev


def standardize(train, test):
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0
    a = np.where(np.isfinite(train), train, median)
    b = np.where(np.isfinite(test), test, median)
    center, scale = a.mean(0), a.std(0)
    scale[scale < 1e-8] = 1
    return (a - center) / scale, (b - center) / scale, (median, center, scale)
