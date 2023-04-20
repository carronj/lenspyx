from __future__ import print_function
import healpy as hp
from lenspyx import lensing
import numpy as np

def test_t(experimental=False):
    lmax = 200
    nside = 256
    cls_unl = np.ones(lmax + 1, dtype=float)
    tunl = hp.synalm(cls_unl, new=True)
    dlm = np.zeros_like(tunl)
    hp.almxfl(dlm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)), inplace=True)
    T = hp.alm2map(tunl, nside)
    T1 = lensing.alm2lenmap(tunl, [dlm, None], nside, verbose=True, nband=8, facres=-2, experimental=experimental)
    assert np.max(np.abs(T - T1)) / np.std(T) < 1e-5
    if not experimental:
        d1Re, d1Im = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, lmax)
        T2 = lensing.alm2lenmap(tunl, [d1Re, d1Im], nside, verbose=True, nband=8, facres=-2)
        assert np.max(np.abs(T - T2)) / np.std(T) < 1e-5
        T3 = lensing.alm2lenmap(tunl, [dlm, dlm.copy()], nside, verbose=True, nband=8, facres=-2)
        assert np.all(T2 == T3)

def test_pol(experimental=False):
    lmax = 200
    nside = 256
    facres= -1
    cls_unl = np.ones(lmax + 1, dtype=float)
    tunl = hp.synalm(cls_unl, new=True)
    dlm = np.zeros_like(tunl)
    hp.almxfl(dlm, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)), inplace=True)
    Q, U = hp.alm2map_spin([tunl, np.zeros_like(tunl)], nside, 2, lmax)
    Q1, U1 = lensing.alm2lenmap_spin([tunl, None], [dlm, None], nside, 2, verbose=True, nband=8, facres=facres, experimental=experimental)

    assert np.max(np.abs(Q - Q1)) / np.std(Q) < 1e-5, np.max(np.abs(Q - Q1)) / np.std(Q)
    assert np.max(np.abs(U - U1)) / np.std(U) < 1e-5, np.max(np.abs(U - U1)) / np.std(U)

    if not experimental:
        #FIXME: this version does not have the Red Imd input caller at the moment
        d1Re, d1Im = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, lmax)
        Q2, U2 = lensing.alm2lenmap_spin([tunl ,None], [d1Re, d1Im], nside, 2, verbose=True, nband=8, facres=facres, experimental=experimental)
        assert np.allclose(Q2, Q1, rtol=1e-10)
        assert np.allclose(U2, U1, rtol=1e-10)

        Q3, U3 = lensing.alm2lenmap_spin([tunl,tunl * 0.], [d1Re, d1Im], nside, 2, verbose=True, nband=8, facres=facres, experimental=experimental)
        assert np.all(Q3 == Q2)
        assert np.all(U3 == U2)


if __name__ == '__main__':
    test_t(experimental=True)
    test_pol(experimental=True)
