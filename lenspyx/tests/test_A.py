from lenspyx.tests.helper import syn_ffi_ducc, cls_unl
from lenspyx.utils import timer
from lenspyx.utils_hp import Alm, synalm
import numpy as np
import pylab as pl
import healpy as hp

tim = timer(True)
ffi, _ = syn_ffi_ducc()
tim.add('ffi gen.')
Aunl = ffi.dlm2A()

lmax, mmax = 4096, 4096
tlm_unl = synalm(cls_unl['tt'][:lmax + 1], lmax, mmax)
tmap_len = ffi.gclm2lenmap(tlm_unl, mmax, 0, False)
tim.add('gen and lensing of Tlm')

points1 = tmap_len.copy()
for ofs, w, nph in zip(ffi.geom.ofs, ffi.geom.weight, ffi.geom.nph):
    points1[ofs:ofs + nph] *= w
points1 = points1 + 0j
points2 = tmap_len * Aunl
for ofs, w, nph in zip(ffi.geom.ofs, ffi.geom.weight, ffi.geom.nph):
    points2[ofs:ofs + nph] *= w
points2 = points2 + 0j
tim.add('weighting')

tlm_unl_1 = ffi.lenmap2gclm(points1, 0, lmax, mmax).astype(np.complex128)
tlm_unl_2 = ffi.lenmap2gclm(points2, 0, lmax, mmax).astype(np.complex128)
tim.add('delensing two maps')

pl.plot(hp.alm2cl(tlm_unl_1))
pl.plot(hp.alm2cl(tlm_unl - tlm_unl_1))
pl.plot(hp.alm2cl(tlm_unl - tlm_unl_2))
pl.loglog()
pl.show()

print(tim)