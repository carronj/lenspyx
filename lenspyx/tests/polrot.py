import numpy as np
import jcducc0
import time
import numexpr
from lenspyx.remapping.utils_geom import Geom
from multiprocessing import cpu_count
from lenspyx.tests.helper import syn_ffi_ducc_29, syn_ffi_ducc, cls_unl
from lenspyx.fortran.remapping import remapping

ffi, gl = syn_ffi_ducc_29(dlmax_gl=1024,dlmax=1024, epsilon=1e-7, nthreads=8)
gl = Geom.get_healpix_geometry(2048)
ffi.geom = gl
gamma = ffi._get_gamma()
npix = gl.npix()
values = np.random.standard_normal(gl.npix()) + 1j * np.random.standard_normal(gl.npix())
values_in = values.copy()


spin = 2
for nt in range(1, cpu_count() + 1):
    values_f = values.copy()
    print('***** nthreads %s'%nt)
    t0 = time.time()
    remapping.apply_inplace(values_f, gamma, spin, nt)
    t1 = time.time()
    print('Fortran     : %.4f s'%(t1-t0))
    assert values_f[0] != values[0]
    values_in = values.copy()
    sj = 1j * spin
    t0 = time.time()
    values_in *= numexpr.evaluate('exp(sj * gamma)')
    t1 = time.time()
    print('Numexpr      : %.4f s'%(t1-t0))
    t0 = time.time()
    jcducc0.jc.apply_inplace(values=values_in.reshape((npix, 1)), gamma=gamma.reshape((npix, 1)), spin=spin, nthreads=nt)
    t1 = time.time()
    print('C++         : %.4f s'%(t1-t0))
    assert np.abs( (values_f[0] / values_in[0] - 1.) < 1e-15)
    values_in = values.copy()
    t0 = time.time()
    jcducc0.jc.apply_inplace_polar(values=values_in, gamma=gamma, spin=spin, nthreads=nt)
    t1 = time.time()
    assert np.abs( (values_f[0] / values_in[0] - 1.) < 1e-15)
    print('C++ polar   : %.4f s'%(t1-t0))
