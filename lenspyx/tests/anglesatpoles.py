"""Tests angles calculations at the poles compared to precictions


"""
from lenspyx.remapping import utils_geom
from lenspyx.tests import helper
import numpy as np
import pylab as pl

gl = utils_geom.Geom.get_cc_geometry(100000, 100).restrict(0.0, 1e-3, True, update_ringstart=True)
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
for ir in [0, 1, gl.theta.size-2, gl.theta.size-1]:
    pl.figure()
    sli = slice(ir * nph, ir * nph + nph)
    if gl.theta[ir] < (np.pi * 0.5):
        pl.plot(phi, ptg[sli, 1], label=r'$\phi^{\prime}$')
        pl.plot(phi, (beta[sli] + phi) %(2 * np.pi), label=r'$\beta + \phi$')
        pl.plot(phi, chi[sli] % (2 * np.pi), label=r'$\chi$')
        pl.plot(phi, (beta[sli])% (2 * np.pi) , label=r'$\beta$')
        pl.legend()
        pl.title(r'NP, $\theta = %.2f$ amin'%(gl.theta[ir] / np.pi * 180 * 60. ))
        print('tht dev NP in amin', np.max(np.abs(ptg[sli, 0] - np.sqrt(asqd[sli])))/np.pi * 180 * 60)
        pl.show()
    if gl.theta[ir] > (np.pi * 0.5):
        pl.plot(phi, ptg[sli, 1], label=r'$\phi^{\prime}$')
        pl.plot(phi, (np.pi + phi - beta[sli]) %(2 * np.pi), label=r'$\phi + \pi - \beta$')
        pl.plot(phi, chi[sli] % (2 * np.pi), label=r'$\chi$')
        pl.plot(phi, (beta[sli]- 1*np.pi)% (2 * np.pi) , label=r'$\beta - \pi$')
        print('tht dev SP in amin', np.max(np.abs( (np.pi-ptg[sli, 0]) - np.sqrt(asqd[sli])))/np.pi * 180 * 60)
        pl.legend()
        pl.title(r'SP, $\pi - \theta = %.2f$ amin'%( (np.pi-gl.theta[ir]) / np.pi * 180 * 60. ))
        pl.show()
