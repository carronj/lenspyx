import numpy as np
from lenspyx.utils_hp import synalm
from lenspyx.utils import timer
from lenspyx.tests.helper import cls_unl
from ducc0.sht.experimental import synthesis_general

def syn_fibo(N:int, lmax:int, nthreads=4):
    """Number of points is P = 2N + 1"""
    npix = 2 * N + 1
    Psi = (1 + np.sqrt(5.))/2
    tim = timer('fibo', False)
    i = np.arange(-N, N+1, dtype=int)
    loc = np.empty((npix, 2), dtype=float)
    loc[:, 0] = np.arcsin(i / (N + 0.5)) + 0.5 * np.pi
    loc[:, 1] = ((2 * np.pi / Psi) * i)%(2 * np.pi)
    del i
    tim.add('%.5f Mpix'%(npix/1e6))
    alm = np.atleast_2d(synalm(cls_unl['tt'][:lmax + 1], lmax, lmax))
    tim.add('synalm lmax %s'%lmax)
    m = synthesis_general(alm=alm, spin=0, lmax=lmax, mmax=lmax, loc=loc, nthreads=nthreads)
    tim.add('spin 0 synthesis_general')
    print(tim)
    return m, alm


if __name__ == '__main__':
    m = syn_fibo(10, 10)