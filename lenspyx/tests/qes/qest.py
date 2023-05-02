import numpy as np
import pylab as pl
from lenspyx import synfast, get_geom
from lenspyx.utils import get_ffp10_cls
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy, synalm
from lenspyx.qest.qest import Qlms, OpFilt


def copy_cls(cls, include=()):
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

# Lensed CMBs
maps, (unl_alms, unl_lab) = synfast(cls_unl, lmax=lmax_unl, geometry=geom_info, verbose=True, alm=True)

ls = np.arange(1, 2048)
for alm, f in zip(unl_alms, unl_lab):
    pl.loglog(ls, ls * (ls + 1) * alm2cl(alm, alm, lmax_unl, lmax_unl, lmax_unl)[ls], label=f)
    pl.plot(ls, ls * (ls + 1) * cls_unl[f + f][ls], c='k')
pl.legend()
pl.show()


# lensed CMBs in harmonic space
tlm = geom.adjoint_synthesis(maps['T'], 0, lmax_filt, lmax_filt, 0).squeeze()
eblm = geom.adjoint_synthesis(maps['QU'], 2, lmax_filt, lmax_filt, 0)

# Adding noise and transfer function
almxfl(tlm, transfs['t'], lmax_filt, True)
almxfl(eblm[0], transfs['e'], lmax_filt, True)
almxfl(eblm[1], transfs['b'], lmax_filt, True)

tlm += synalm(1. / inoise['tt'], lmax_filt, lmax_filt)
eblm[0] += synalm(1. / inoise['ee'], lmax_filt, lmax_filt)
eblm[1] += synalm(1. / inoise['bb'], lmax_filt, lmax_filt)

alms = {'t': tlm, 'e': eblm[0], 'b': eblm[1]}

ls = np.arange(1, lmax_qlm + 1)
wls = ls ** 2 * (ls + 1) ** 2 * 1e7 / (2 * np.pi)
plm_in = alm_copy(unl_alms[unl_lab.index('p')], lmax_unl, lmax_qlm, lmax_qlm)
for qe_key in ['ptt', 'p_p', 'p']:
    includes = ['t'] * (qe_key in ['ptt', 'p']) + ['e', 'b'] * (qe_key in ['p_p', 'p'])
    cls_filt = copy_cls(cls_len, include=includes)
    filtr = OpFilt(cls_filt, transfs, inoise)
    qlms_dd = Qlms(filtr, filtr, cls_len, lmax_qlm)
    plm, olm = qlms_dd.get_qlms(qe_key, alms, verbose=True)  # Unormalized potentials
    rp, ro = qlms_dd.get_response(qe_key, 'p', cls_len)
    pl.plot(ls, wls / rp[ls] * alm2cl(plm, plm_in, lmax_qlm, lmax_qlm, lmax_qlm)[ls], label='rec x input')
    pl.plot(ls, wls / rp[ls] ** 2 * alm2cl(plm, plm, lmax_qlm, lmax_qlm, lmax_qlm)[ls])
    pl.plot(ls, wls * alm2cl(plm_in, plm_in, lmax_qlm, lmax_qlm, lmax_qlm)[ls])
    pl.plot(ls, wls * cls_unl['pp'][ls], c='k', label='Cpp')
    pl.plot(ls, wls / rp[ls], ls='--', c='k', label = '1 / R')
    pl.plot(ls, wls * (1. / rp[ls] + cls_unl['pp'][ls]), ls='-.', c='k', label = 'Cpp + 1 / R')

    pl.legend()
    pl.title(qe_key)
    pl.show()

# Double-c
from plancklens import  qresp as qresp_pl
rp_pl = qresp_pl.get_response(qe_key, lmax_filt, 'p', cls_len, cls_len, filtr.get_fal(), lmax_qlm=lmax_qlm)[0]

