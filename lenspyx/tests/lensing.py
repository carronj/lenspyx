from lenspyx.tests.helper import syn_alms, syn_dlm
from lenspyx import alm2lenmap, alm2lenmap_spin

lmax = 500
alm = syn_alms(0, lmax)
eblm = syn_alms(2, lmax)
dlm = syn_dlm(lmax)

# default behavior
T = alm2lenmap(alm, [dlm, None])
Q, U = alm2lenmap_spin(eblm, [dlm, None], 2)

# thinned GL:
T = alm2lenmap(alm, [dlm, None], geometry=('thingauss', {'smax': 2, 'lmax':lmax}))
Q, U = alm2lenmap_spin(eblm, [dlm, None], 2, geometry=('thingauss', {'smax': 2, 'lmax':lmax}))




