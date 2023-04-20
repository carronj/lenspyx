import numpy as np
from lenspyx.tests.helper import syn_alms, syn_dlm
from lenspyx import alm2lenmap, alm2lenmap_spin
from lenspyx.remapping import utils_geom

lmax = 300
alm = syn_alms(0, lmax)  # tests T E B maps, here neglecting TE covariance and B is zero
eblm = syn_alms(2, lmax)
dlm = syn_dlm(lmax)

# T and Pol, on default geometry, with pure gradient lensing
T, Q, U = alm2lenmap([alm, eblm[0], eblm[1]], dlm)

# T and Pol, on default geometry, including curl deflection (here zero)
T2, Q2, U2 = alm2lenmap([alm, eblm[0], eblm[1]], [dlm, dlm * 0])
print('All numbers should be zero')
print(np.max(np.abs(Q2-Q)))
print(np.max(np.abs(U2-U)))
print(np.max(np.abs(T2-T)))

# T and Pol, same but no unlensed B (a little bit faster in principle)
T2, Q2, U2 = alm2lenmap([alm, eblm[0]], dlm)

print(np.max(np.abs(Q2-Q)))
print(np.max(np.abs(U2-U)))
print(np.max(np.abs(T2-T)))

# T-P independent constructions
T2 = alm2lenmap(alm, dlm)
Q2, U2 = alm2lenmap_spin(eblm, dlm, 2)

print(np.max(np.abs(Q2-Q)))
print(np.max(np.abs(U2-U)))
print(np.max(np.abs(T2-T)))

# Same with no unlensed B  (a little bit faster in principle)
Q2, U2 = alm2lenmap_spin(eblm[0], dlm, 2)
print(np.max(np.abs(Q2-Q)))
print(np.max(np.abs(U2-U)))

# Other geometries: thinned GL:
T = alm2lenmap(alm, dlm, geometry=('thingauss', {'smax': 2, 'lmax':lmax}))
Q, U = alm2lenmap_spin(eblm, dlm, 2, geometry=('thingauss', {'smax': 2, 'lmax':lmax}))

# Other geometries: fejer-1:
T = alm2lenmap(alm, dlm, geometry=('f1', {'ntheta': lmax + 2, 'nphi': 2 * lmax + 2}))
Q, U = alm2lenmap_spin(eblm, dlm, 2, geometry=('f1', {'ntheta': lmax + 2, 'nphi': 2 * lmax + 2}))

utils_geom.Geom.show_supported_geometries()
