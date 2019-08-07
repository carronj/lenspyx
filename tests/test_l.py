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
    lenmap = lensing.alm2lenmap(tunl, [dlm, None], nside, verbose=True, nband=8, facres=-2)
    assert np.max(np.abs(lenmap - unlmap)) / np.std(unlmap) < 1e-5
    d1Re, d1Im = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, lmax)
    lenmap2 = lensing.alm2lenmap(tunl, [d1Re, d1Im], nside, verbose=True, nband=8, facres=-2)
    assert np.allclose(lenmap, lenmap2, rtol=1e-10)
    lenmap3 = lensing.alm2lenmap(tunl, [dlm, dlm.copy()], nside, verbose=True, nband=8, facres=-2)
    assert np.all(lenmap2 == lenmap3)

if __name__ == '__main__':
    test_len()