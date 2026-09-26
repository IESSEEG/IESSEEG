"""Tests for input-study invariants; no private EEG required by default."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
pytest.importorskip('mne')
pytest.importorskip('pyedflib')
from iesseeg_paper.input_studies import phase_pairs,dfa_both,dfa_intercept,FS
def test_pairwise_pli_matches_numpy_reference():
    rng=np.random.default_rng(15)
    phase=rng.uniform(-np.pi,np.pi,size=(3,19,1600)).astype(np.float32)
    i,j=np.triu_indices(19,1)
    expected=np.abs(np.sign(np.sin(phase[:,i]-phase[:,j])).mean(-1))
    actual=phase_pairs(torch.tensor(phase)).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_dfa_intercept_preserves_existing_response_definition():
    rng=np.random.default_rng(12)
    beta=rng.standard_normal((19,30*FS))
    new,exponent,high=dfa_both(beta)
    old=dfa_intercept(beta,FS).mean()
    np.testing.assert_allclose(new,old,atol=1e-12)
    assert np.isfinite(exponent) and high==3
