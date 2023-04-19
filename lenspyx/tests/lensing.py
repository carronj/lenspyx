from lenspyx.tests.helper import syn_alms, syn_dlm
from lenspyx import alm2lenmap, alm2lenmap_spin

spin = 0
lmax = 500
alm = syn_alms(0, lmax)
dlm = syn_dlm(lmax)

# default behavior
alm2lenmap(alm, [dlm, None])

