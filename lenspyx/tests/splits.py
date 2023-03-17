"""Tests geometry splitings

"""
import time
import numpy as np
from lenspyx.remapping import utils_geom
from lenspyx.utils_hp import synalm
from multiprocessing import cpu_count

lmax = 200
spin = 0
nthreads = min(4, cpu_count())

gl_base = []
gl_base.append(utils_geom.Geom.get_healpix_geometry(2048))
gl_base.append(utils_geom.Geom.get_thingauss_geometry(3000, 2))
maxdiff = 0
for gl in gl_base:
    for nbands in [2, 3, 5, 10]:
        gls = gl.split(nbands, verbose=True)
        m = np.empty((1, gl.npix()), dtype=float)
        m2 = np.zeros((1, gl.npix()), dtype=float)

        tlm = np.atleast_2d(synalm(np.ones(lmax + 1) * 1, lmax, lmax))
        t0 = time.time()
        for g in gls:
            g.synthesis(tlm, spin, lmax, lmax, nthreads, map=m)
        t1 = time.time()
        print(t1 - t0)
        t2 = time.time()
        gl.synthesis(tlm, spin, lmax, lmax, nthreads, map=m2)
        t3 = time.time()
        print(t3 - t2)
        diff = np.max(np.abs(m2 - m) / np.std(m2))
        print('This should be zero', diff)
        maxdiff = max(maxdiff, diff)
assert maxdiff < 1e-13, maxdiff
