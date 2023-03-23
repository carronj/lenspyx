import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc, syn_ffi_ducc_29, syn_alms
import multiprocessing
from lenspyx.utils import timer

spin = 0
lmax_len, dlmax, dlmax_gl = 4096, 1024, 1024
nthreads = min(multiprocessing.cpu_count(), 4)

ffi, geom = syn_ffi_ducc(nthreads=nthreads, lmax_len=lmax_len, dlmax=dlmax, dlmax_gl=dlmax_gl, verbosity=1)
ffi_29, _ = syn_ffi_ducc_29(nthreads=nthreads,  lmax_len=lmax_len, dlmax=dlmax, dlmax_gl=dlmax_gl, verbosity=1)

alm = syn_alms(spin, ctyp=np.complex64 if ffi.single_prec else np.complex128)
ffi.tim = timer('', False)
ffi.gclm2lenmap(alm, None, spin, False)
print("u2nu: %s threads, %.3f Mpix / s"%(nthreads, geom.npix() * 1e-6 / ffi.tim.keys['u2nu']))

ffi_29.tim = timer('', False)
ffi_29.gclm2lenmap(alm, None, spin, False)
