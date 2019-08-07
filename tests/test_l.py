from __future__ import print_function
import healpy as hp
from lenspyx import lensing
import numpy as np

def test_len():
    lmax = 200
    nside = 256
    cls_unl = np.ones(lmax + 1, dtype=float)
    tunl = hp.synalm(cls_unl, new=True)
    dlm = np.zeros_like(tunl)
    hp.almxfl(dlm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)), inplace=True)
    unlmap = hp.alm2map(tunl, nside)
    lenmap = lensing.alm2lenmap(tunl, dlm, nside, verbose=True, nband=8, facres=-1)
    assert np.max(np.abs(lenmap - unlmap)) / np.std(unlmap) < 1e-5

if __name__ == '__main__':
    test_len()