import numpy as np
import time
from lenspyx.tests.helper import syn_ffi_ducc_29
from lenspyx.fortran.remapping import remapping
from multiprocessing import cpu_count
try:
    import jcducc0
    HAS_JCDUCC = True
except:
    HAS_JCDUCC = False

ffi, gl = syn_ffi_ducc_29(dlmax_gl=0, dlmax=0, epsilon=1e-7, nthreads=min(4, cpu_count()))
ffi.geom = gl
npix = gl.npix()
values = np.random.standard_normal(gl.npix()) + 1j * np.random.standard_normal(gl.npix())
values_real = np.array([values.real, values.imag])
values_in = values.copy()
spin = 2
gamma_rd = np.random.standard_normal(gl.npix())
gamma_true = ffi._get_gamma()
for gamma, label in zip([gamma_true, gamma_rd], ['CMB lensing', 'randoms']):
    print('-----------------------')
    gammaf = gamma.astype(np.float32)
    for nt in [2, 4]:
        values_f = values.copy()
        print('***** %s, nthreads %s'%(label, nt))
        t0 = time.time()
        remapping.apply_inplace(values_f, gamma, spin, nt)
        t1 = time.time()
        print('Fortran cd2cd    : %.4f s'%(t1-t0))
        assert np.max(np.abs(values_f[0:100] - values[0:100])) > 1e-5
        t0 = time.time()
        z = np.copy(values_f[0:100])
        remapping.apply_inplace(values_f, gamma, -spin, nt)
        t1 = time.time()
        print('Fortran cd2cd bwd : %.4f s'%(t1-t0))
        assert np.max(np.abs(values_f[0:100] - values[0:100])) < 1e-13
        if HAS_JCDUCC:
            values_f = values.copy()
            t0 = time.time()
            jcducc0.jc.apply_inplace_polar(values=values_f, gamma=gamma, spin=spin, nthreads=nt)
            t1 = time.time()
            assert np.max(np.abs(values_f[0:100] - z)) < 1e-13, np.max(np.abs(values_f[0:100] -z))
            print('C++ polar   : %.4f s' % (t1 - t0))
        values_f = values.astype(np.complex64)
        t0 = time.time()
        remapping.apply_inplacef(values_f, gammaf, spin, nt)
        t1 = time.time()
        print('Fortran c2c    : %.4f s'%(t1-t0))
        assert np.max(np.abs(values_f[0:100] - values[0:100])) > 1e-5
        t0 = time.time()
        z = np.copy(values_f[0:100])
        remapping.apply_inplacef(values_f, gammaf, -spin, nt)
        t1 = time.time()
        print('Fortran c2c bwd : %.4f s'%(t1-t0))
        assert np.max(np.abs(values_f[0:100] - values[0:100])) < 1e-5
