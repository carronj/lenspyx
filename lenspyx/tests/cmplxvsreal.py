import numpy as np
import time
from multiprocessing import cpu_count
from lenspyx.tests.helper import syn_ffi_ducc_29

nthreads = min(cpu_count(), 4)
ffi, gl = syn_ffi_ducc_29(dlmax_gl=0,dlmax=0, epsilon=1e-7, nthreads=nthreads)
ffi.geom = gl
npix = gl.npix()

gclm = np.atleast_2d(ffi.dlm)
spin = 3
t0 = time.time()
valuesc = np.empty((npix,), dtype=complex)
valuesd = valuesc.view(float).reshape((npix, 2)).T  # real view onto complex array
gl.synthesis(gclm, spin, ffi.lmax_dlm, ffi.mmax_dlm, ffi.sht_tr, map=valuesd,
             mode='GRAD_ONLY' if spin > 0 else 'STANDARD')
t1 = time.time()
print(t1 - t0)
t0 = time.time()
valuesd2 = gl.synthesis(gclm, spin, ffi.lmax_dlm, ffi.mmax_dlm, ffi.sht_tr,
                        mode='GRAD_ONLY' if spin > 0 else 'STANDARD')
t1 = time.time()
print(t1 - t0)
