""" This plots the accuracy of the fwd remapping on a subset of rings as function of epsilon accuracy parameters


"""
import os

from lenspyx.tests.helper import  cls_unl
from lenspyx.tests.helper import syn_ffi_ducc_29 as syn_ffi_ducc
from lenspyx import cachers
from lenspyx.utils_hp import synalm
import numpy as np
import pylab as pl
from time import time
import matplotlib
from lenspyx.remapping.utils_geom import Geom
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)


res, nside, nthreads = 1.71, 2048, 8
lmax_len, mmax_len, dlmax = 4096, 4096, 1024
#res, nside, nthreads = 1.7, 2048, 8
#lmax_len, mmax_len, dlmax = 100, 100, 20
SAVE = False#(lmax_len == 4096) * (os.environ.get('ONED', 'SCRATCH') + '/ducclens/Tex/figs/epsilon.pdf')
OPTI = True

lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
thingauss = False # healpix rings
dlm_fac = 1.


ffi_ducc, ref_geom = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax, dlm_fac=dlm_fac,
                                  nthreads=nthreads)
ffi_ducc.single_prec = False

#ffi_ducc_opti, ref_geom = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax, target_res=res, nside=nside, dlm_fac=dlm_fac,
#                                  nthreads=nthreads, thingauss=thingauss, optiversion=True)
ffi_ducc.verbosity = 0
eblm = np.array([synalm(cls_unl['ee'][:lmax_unl+1], lmax_unl, mmax_unl),
                 synalm(cls_unl['bb'][:lmax_unl+1], lmax_unl, mmax_unl)])
Pexs = [] # exact pol.
pixels = []
phis = []
ofs_sorted = np.argsort(ffi_ducc.geom.ofs)
rings = [ffi_ducc.geom.weight.size//2, 0, 100, ffi_ducc.geom.weight.size-1]
for ir in rings:
    pixs = Geom.rings2pix(ffi_ducc.geom, [ofs_sorted[ir]])
    phi = Geom.rings2phi(ffi_ducc.geom, [ofs_sorted[ir]])

    if len(pixs) > 200:
        phi =  phi[:: len(pixs) // 100]
        pixs = pixs[:: len(pixs) // 100]
    Qex, Uex = ffi_ducc.gclm2lenpixs(eblm, mmax_unl, 2, pixs)
    Pexs.append(Qex + 1j * Uex)
    pixels.append(pixs)
    phis.append(phi)
angles = np.copy(ffi_ducc._build_angles())
Colors = ['C%s'%s for s in range(10)]
ls = ['-', '--', '-.', ':']
pl.figure(figsize=(20, 5))
norm = None
for ie, epsilon in enumerate([1e-3, 1e-6, 1e-8, 2e-13][::-1]):
    tffi_ducc, _ = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax, dlm_fac=dlm_fac,
                                      nthreads=nthreads, epsilon=epsilon)
    tffi_ducc = tffi_ducc.change_dlm([ffi_ducc.dlm, None], ffi_ducc.mmax_dlm, cacher=cachers.cacher_mem(safe=False))
    tffi_ducc.single_prec = (epsilon >= 1e-6)

    tffi_ducc.verbosity = 0
    tffi_ducc.cacher.cache('ptg', angles.astype(np.float64 if not tffi_ducc.single_prec else np.float32)) # avoiding angle calculation overhead
    t0 = time()
    this_eblm = eblm.astype(np.complex64 if tffi_ducc.single_prec else np.complex128)
    Q, U = tffi_ducc.gclm2lenmap(this_eblm, mmax_unl, 2, False)
    print(' %.3f exec time for eps' % (time() - t0), int(np.log10(epsilon)))

    if norm is None:
        norm = 1./np.sqrt(np.mean(Q ** 2 + U ** 2))
    for i, ir in enumerate(rings):
        # for rings with large numbers of theta we undersample
        tht = tffi_ducc.geom.theta[ofs_sorted[ir]]
        pl.semilogy(phis[i], norm * np.abs(Pexs[i] - (Q[pixels[i]] + 1j*U[pixels[i]])), c=Colors[i], ls=ls[ie], label=r'$\epsilon =10^{%i},  \theta = %.1f$ deg.'%(np.log10(epsilon), tht / np.pi * 180))

pl.xlabel(r'$\varphi$')
pl.ylabel(r'relative error in Pol.')
pl.xticks([0., np.pi * 0.5, np.pi , 3 * np.pi * 0.5, 2 * np.pi], [r'0', r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$' ])
pl.xlim(0. - 0.0125 * np.pi, 2 * np.pi + 0.0125 * np.pi)

if SAVE and os.path.exists(os.path.dirname(SAVE)):
    pl.savefig(SAVE, bbox_inches='tight')
pl.show()