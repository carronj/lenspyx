from __future__ import  annotations
import pylab as pl
import numpy as np
import time
from ducc0.sht.experimental import alm2leg
from ducc0.misc import GL_thetas
def wigner_pos(cl, theta, s1: np.ndarray[int], s2:int, nthreads=0): # version returning s1|s2| and s1 -|s2| on a single call
    if np.isscalar(s1):
        mval = np.array([s1], dtype=int)
        cls = [cl]
    else:
        mval = s1.astype(int)
        cls = cl
    sgn_fac = -1 if s2 else 1
    lmax = len(cls[0]) - 1 # FIXME: now all same lmax's
    mstart = np.arange(len(mval), dtype=int) * (lmax + 1)
    glm_r = np.empty((1,len(mval) * (lmax + 1)), dtype=complex)
    prefac = np.sqrt(np.arange(1, 2 * lmax + 3, 2)) * (sgn_fac / np.sqrt(4 * np.pi))
    for is1, this_cl in enumerate(cls):
        glm_r[0, is1 * (lmax + 1):  (is1 + 1) * (lmax + 1)].real = this_cl * prefac
    mode = 'GRAD_ONLY' if s2 else 'STANDARD'
    t0 = time.time()
    leg = alm2leg(alm=glm_r, spin=abs(s2), lmax=lmax, mval=mval, mstart=mstart, theta=theta, mode=mode, nthreads=nthreads).squeeze()
    dt = time.time() - t0
    if s2:
        sp = leg[0].real + leg[1].imag
        sm = leg[0].real - leg[1].imag
        return sp, sm * (1 if s2 % 2 == 0 else -1), dt
    return leg.real, leg.real, dt


if __name__ == '__main__':
    lmax = 4000
    cl = np.ones(lmax + 1, dtype=float)
    s2 = 2
    npts = (3 * lmax) // 2 + 1

    theta = GL_thetas(npts)
    nruns = 10
    ns = np.arange(1, 16)
    for nthreads in np.arange(1, 9):
        times = ns * 0.
        for n in range(nruns):
            times += np.array([wigner_pos([cl] * n, theta, np.arange(1, n+1), s2, nthreads=nthreads)[-1] for n in ns ])
        times /= nruns
        pl.plot(times * 100, label=str(nthreads) + ' threads')
    pl.ylabel('exec time in ms')
    pl.xlabel('numbers of ms in alm2leg')
    pl.legend()
    pl.show()