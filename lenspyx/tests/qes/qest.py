"""This builds QEs from scratch in Planck-like settings, for an idealized full-sky configuration

    This goes through the following steps:

         * Generation of lensed maps with lenspyx.synfast
         * Addition of instrumental noise
         * Definition of inverse-variance filtering objects
         * Definition of QE constructor objects
         * Actual computation, for TT, Pol-only and GMV estimators and plots


    The entire things should take much less than a minute

"""
import numpy as np
from lenspyx import synfast, get_geom
from lenspyx.utils import get_ffp10_cls
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy, synalm
from lenspyx.qest.qest import Qlms, OpFilt

try:
    import pylab as pl
    PLOT = True

except ImportError:
    pl = None
    PLOT = False


def copy_cls(cls, include=()):
    """Returns cls for the desired fields only"""
    ret = {}
    for k in cls:
        if k[0] in include and k[1] in include:
            ret[k] = np.copy(cls[k])
    return ret


lmax_unl = 3000
lmax_filt, lmax_qlm = 2048, 400
geom_info = ('thingauss', {'lmax': 4000, 'smax': 2})

cls_unl, cls_len, cls_glen = get_ffp10_cls(lmax=lmax_unl)
geom = get_geom(geom_info)

beam = gauss_beam(5. / 180 / 60 * np.pi, lmax=lmax_filt)
inoise = {'tt': beam ** 2 / (35. / 180 / 60 * np.pi) ** 2,
          'ee': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2,
          'bb': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2}
transfs = {f: np.ones(lmax_filt + 1, dtype=float) for f in 'teb'}

# Generation of lensed CMBs
maps, (unl_alms, unl_lab) = synfast(cls_unl, lmax=lmax_unl, geometry=geom_info, verbose=True, alm=True)

# Lensed CMBs in harmonic space
tlm = geom.adjoint_synthesis(maps['T'], 0, lmax_filt, lmax_filt, 0).squeeze()
eblm = geom.adjoint_synthesis(maps['QU'], 2, lmax_filt, lmax_filt, 0)

# Applying transfer function
almxfl(tlm, transfs['t'], lmax_filt, True)
almxfl(eblm[0], transfs['e'], lmax_filt, True)
almxfl(eblm[1], transfs['b'], lmax_filt, True)

# Adding instrumental noise
tlm += synalm(1. / inoise['tt'], lmax_filt, lmax_filt)
eblm[0] += synalm(1. / inoise['ee'], lmax_filt, lmax_filt)
eblm[1] += synalm(1. / inoise['bb'], lmax_filt, lmax_filt)
alms = {'t': tlm, 'e': eblm[0], 'b': eblm[1]}

ls = np.arange(2, lmax_qlm + 1)
wls = ls ** 2 * (ls + 1) ** 2 * 1e7 / (2 * np.pi)
plm_in = alm_copy(unl_alms[unl_lab.index('p')], lmax_unl, lmax_qlm, lmax_qlm) # input lensing potental

for qe_key, qe_lab in zip(['ptt', 'p_p', 'p'], [r'$\hat \phi^{TT}$', r'$\hat \phi^{Pol}$', r'$\hat\phi^{GMV}$']):
    includes = ['t'] * (qe_key in ['ptt', 'p']) + ['e', 'b'] * (qe_key in ['p_p', 'p'])
    cls_filt = copy_cls(cls_len, include=includes)

    # inverse-variance filtering object
    filtr = OpFilt(cls_filt, transfs, inoise)

    # QE-calculator object:
    qlms_dd = Qlms(filtr, filtr, cls_len, lmax_qlm)

    # Calculation of (unormalized) lensing potentials (gradient and curl)
    plm, olm = qlms_dd.get_qlms(qe_key, alms, verbose=True)

    # Calculation of estimator normalization
    rp, ro = qlms_dd.get_response(qe_key, 'p', cls_len)

    # plots
    if PLOT:
        pl.plot(ls, wls / rp[ls] * alm2cl(plm, plm_in, lmax_qlm, lmax_qlm, lmax_qlm)[ls],
                label=r'$C_L^{\hat \phi \cdot \phi^{\rm in}}$')
        pl.plot(ls, wls / rp[ls] ** 2 * alm2cl(plm, plm, lmax_qlm, lmax_qlm, lmax_qlm)[ls],
                label=r'$C_L^{\hat \phi \hat \phi}$')
        pl.plot(ls, wls * cls_unl['pp'][ls], c='k', label=r'$C_L^{\phi\phi}$')
        pl.plot(ls, wls / rp[ls], ls='--', c='k', label = r'$R^{-1}_L$')
        pl.plot(ls, wls * (1. / rp[ls] + cls_unl['pp'][ls]), ls='-.', c='k', label = r'$C_L^{\phi\phi}+ R^{-1}_L$')
        pl.xlabel(r'$L$')
        pl.ylabel(r'$10^7\cdot L^2(L + 1)^2 C_L^{\phi\phi} / 2\pi$')
        pl.legend()
        pl.title(qe_lab)
        pl.show()
