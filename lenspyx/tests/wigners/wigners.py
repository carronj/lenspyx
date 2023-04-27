"""Compares the Wigner functions to Plancklens's

"""
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
lmaxs = [500, 1000, 2000, 4000]
for lmax in lmaxs:
    npts = (3 * lmax) // 2 + 1
    ls = np.arange(lmax + 1)
    tht = GL_thetas(npts)[::-1]
    wg = GL_weights(npts, 1)[::-1] / (2 * np.pi)
    xg = np.cos(tht)
    cl_in = gauss_beam(1./180 / 60 * np.pi, lmax=lmax)
    maxdevs = []
    for s1 in range(-3, 4):
        for s2 in range(-3, 4):
            xi12 = wigners.wignerpos(cl_in, tht, s1, s2)
            xi12_2 = wigners.wigner4pos(cl_in, None, tht, s1, s2)[0 if s2 >= 0 else 1]
            clv2 = wigners_pl.wignercoeff(xi12 * wg, xg, s1, s2, lmax)
            clv3 = wigners.wignercoeff(xi12 * wg, tht, s1, s2, lmax)
            clv4 = wigners.wignercoeff(xi12_2 * wg, tht, s1, s2, lmax)
            t_ls = ls[max(abs(s1), abs(s2)):]
            maxdevs.append(np.max(np.abs(clv2[t_ls] / cl_in[t_ls] - 1.)))
            maxdevs.append(np.max(np.abs(clv4[t_ls] / cl_in[t_ls] - 1.)))
            maxdevs.append(np.max(np.abs(clv3[t_ls] / cl_in[t_ls] - 1.)))
            assert np.max(maxdevs) < 1e-9, (s1, s2)

    print('lmax %s %.3e'%(lmax, np.max(maxdevs)))
    assert np.max(maxdevs) < 1e-9, np.max(maxdevs)
    if lmax == 4000:
        print('speed-ups')
        for s1 in range(-2, 3):
            for s2 in range(-2, 3):
                t0 = time.time()
                xi12 = wigners.wignerpos(cl_in, tht, s1, s2)
                dt1 = time.time() - t0
                t0 = time.time()
                xi12 = wigners_pl.wignerpos(cl_in, xg, s1, s2)
                dt2 = time.time() - t0
                print('%2s %2s fwd %.3f'%(s1, s2, dt2 / dt1))
                t0 = time.time()
                cl = wigners.wignercoeff(xi12, tht, s1, s2, lmax)
                dt1 = time.time() - t0
                t0 = time.time()
                cl = wigners_pl.wignercoeff(xi12, xg, s1, s2, lmax)
                dt2 = time.time() - t0
                print('%2s %2s adj %.3f'%(s1, s2, dt2 / dt1))

