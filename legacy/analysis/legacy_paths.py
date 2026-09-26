"""Runtime roots for the historical experiments bundled with the paper code."""
import os
from pathlib import Path

def workspace():
    return Path(os.environ.get('IESSEEG_WORKSPACE','local/input_workspace')).expanduser().resolve()

def runtime():
    return Path(os.environ.get('IESSEEG_LEGACY_RUNTIME','local/legacy_runtime')).expanduser().resolve()

def reference():
    return Path(__file__).resolve().parents[2] / 'baselines_reference'

def baselines():
    return reference() / 'baselines'

def pretrained(fallback):
    root = os.environ.get('IESSEEG_PRETRAINED_DIR')
    return Path(root) / Path(fallback).name if root else Path(fallback)
