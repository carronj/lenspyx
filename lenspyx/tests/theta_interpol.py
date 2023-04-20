import numpy as np
from lenspyx.remapping import utils_geom
from lenspyx.tests import helper
from multiprocessing import cpu_count
from lenspyx.utils import timer
import ducc0

HAS_THETA_INTERPOL = 'theta_interpol' in ducc0.sht.experimental.synthesis.__doc__
assert HAS_THETA_INTERPOL, 'ducc0 version incompatible for this test'

lmax_unl = 5120
single_prec = False
spin = 2
ncomp = 2
nthreads = min(4, cpu_count())
sht_mode = 'STANDARD' if spin == 0 or ncomp == 2 else 'GRAD_ONLY'

geom = utils_geom.Geom.get_healpix_geometry(2048)
nrings = geom.theta.size
eblm = helper.syn_alms(spin, lmax_unl=lmax_unl, ctyp=np.complex64 if single_prec else np.complex128)[:ncomp]

tim = timer('', False)

tim.start('synthesis (%s %s %s)'%(lmax_unl, False, sht_mode))
m1 = geom.synthesis(eblm, spin, lmax_unl, lmax_unl, nthreads, theta_interpol=False, mode=sht_mode)
tim.close('synthesis (%s %s %s)'%(lmax_unl, False, sht_mode))
tim.start('synthesis (%s %s %s)'%(lmax_unl, True, sht_mode))
m2 = geom.synthesis(eblm, spin, lmax_unl, lmax_unl, nthreads, theta_interpol=True, mode=sht_mode)
tim.close('synthesis (%s %s %s)'%(lmax_unl, True, sht_mode))
print(tim)
