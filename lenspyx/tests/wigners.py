try:
    from plancklens.wigners import wigners as wigners_pl
except:
    assert 0, 'cant do this test witout plancklens'
import numpy as np
import time
from ducc0.misc import GL_thetas, GL_weights
from lenspyx.utils_hp import gauss_beam
from lenspyx.wigners import wigners

print('All these numbers must be small')
lmaxs = [500, 1000, 2000, 4000, 5000]
for lmax in lmaxs:
    npts = (3 * lmax) // 2 + 1
    ls = np.arange(lmax + 1)
    tht = GL_thetas(npts)[::-1]
    wg = GL_weights(npts, 1)[::-1] / (2 * np.pi)
    xg = np.cos(tht)
    cl_in = gauss_beam(1./180 / 60 * np.pi,lmax=lmax)
    maxdevs = []
    for s1 in range(-3, 4):
        for s2 in range(-3, 4):
            xi12 = wigners.wignerpos(cl_in, tht, s1, s2)
            clv2 = wigners_pl.wignercoeff(xi12 * wg, xg, s1, s2, lmax)
            t_ls = ls[max(abs(s1), abs(s2)):]
            maxdevs.append(np.max(np.abs(clv2[t_ls] / cl_in[t_ls] - 1.)))
    print('lmax %s %.3e'%(lmax, np.max(maxdevs)))
    assert np.max(maxdevs) < 1e-9, np.max(maxdevs)
    if lmax == 2000:
        for s1s2 in [(0, 0), (0, 2), (2, 0), (2, 2), (0, -2), (-2, 0), (-2, -2), (1, 3)]:
            s1, s2 = s1s2
            print('Timings s1 s2  %s %s'%(s1, s2))
            t0 = time.time()
            xi12 = wigners.wignerpos(cl_in, tht, s1, s2)
            dt1 = time.time() - t0
            t0 = time.time()
            xi12 = wigners_pl.wignerpos(cl_in, xg, s1, s2)
            dt2 = time.time() - t0
            print('speed-up %.3f'%(dt2 / dt1))