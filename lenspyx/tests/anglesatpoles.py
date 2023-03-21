"""Tests angles calculations at the poles compared to precictions


"""
from lenspyx.remapping import utils_geom
from lenspyx.tests import helper
import numpy as np
import pylab as pl

gl = utils_geom.Geom.get_gl_geometry(50000).restrict(0.0, 1e-3, True, update_ringstart=True)
ffi, _ = helper.syn_ffi_ducc_29()
ffi.geom = gl
ptg = ffi._get_ptg()
chi = ffi._get_gamma()
d1 = ffi._build_d1()
beta = np.arctan2(d1[1], d1[0])% (2 * np.pi)
asqd = d1[0] ** 2 + d1[1] ** 2
print(chi.shape, ptg.shape)
print(ffi.tim)
nph = int(gl.nph[0])
phi = 2 * np.pi / nph * np.arange(nph)
for ir in [0, gl.theta.size-1]:
    pl.figure()
    print(gl.theta[ir])
    sli = slice(ir * nph, ir * nph + nph)
    if gl.theta[ir] < np.pi * 0.05:
        pl.plot(phi, ptg[sli, 1], label='phip')
        pl.plot(phi, (beta[sli] + phi) %(2 * np.pi), label='b + phi')
        pl.plot(phi, chi[sli] % (2 * np.pi), label='chi')
        pl.plot(phi, (beta[sli])% (2 * np.pi) , label='beta')
        pl.legend()
        pl.show()
    if gl.theta[ir] > np.pi * 0.05:
        pl.plot(phi, ptg[sli, 1], label='phip')
        pl.plot(phi, (np.pi + phi - beta[sli]) %(2 * np.pi), label='phi + pi - beta')
        pl.plot(phi, chi[sli] % (2 * np.pi), label='chi')
        pl.plot(phi, (beta[sli]- 1*np.pi)% (2 * np.pi) , label='beta - pi')
        pl.legend()
        pl.show()
