import numpy as np
import jcducc0
import time
import numexpr
from lenspyx.remapping.utils_geom import Geom
from multiprocessing import cpu_count
from lenspyx.tests.helper import syn_ffi_ducc_29, syn_ffi_ducc, cls_unl
from lenspyx.fortran.remapping import remapping

ffi, gl = syn_ffi_ducc_29(dlmax_gl=0,dlmax=0, epsilon=1e-7, nthreads=4)
gl = Geom.get_healpix_geometry(4096)
ffi.geom = gl
npix = gl.npix()

gclm = np.atleast_2d(ffi.dlm)
t0 = time.time()
valuesc = np.empty((npix,), dtype=complex)
valuesd = valuesc.view(float).reshape((npix, 2)).T  # real view onto complex array
spin = 3
gl.synthesis(gclm, spin, ffi.lmax_dlm, ffi.mmax_dlm, ffi.sht_tr, map=valuesd, mode='GRAD_ONLY' if spin > 0 else 'STANDARD')
t1 = time.time()
print(t1 - t0)
t0 = time.time()
valuesd2 = gl.synthesis(gclm, spin, ffi.lmax_dlm, ffi.mmax_dlm, ffi.sht_tr, mode='GRAD_ONLY' if spin > 0 else 'STANDARD')
t1 = time.time()
print(t1 - t0)
