import numpy as np
import pylab as pl
from lenspyx import synfast, get_geom
from lenspyx.utils import get_ffp10_cls
from lenspyx.utils_hp import gauss_beam, almxfl, alm2cl, alm_copy
from lenspyx.remapping.utils_geom import Geom
from lenspyx.qest.qest import Qlms, OpFilt


def copy_cls(cls, include=()):
    ret = {}
    for k in cls:
        if k[0] in include and k[1] in include:
            ret[k] = np.copy(cls[k])
    return ret


lmax_unl = 3000
lmax_filt, lmax_qlm = 2048, 2048
geom_info = ('thingauss', {'lmax': 4000, 'smax': 2})

cls_unl, cls_len, cls_glen = get_ffp10_cls(lmax=lmax_unl)
maps, (unl_alms, unl_lab) = synfast(cls_unl, lmax=lmax_unl, geometry=geom_info, verbose=True, alm=True)

for alm, f in zip(unl_alms, unl_lab):
    pl.loglog(alm2cl(alm, alm, lmax_unl, lmax_unl, lmax_unl)[1:], label=f)
    pl.plot(cls_unl[f + f][1:], c='k')
pl.legend()
pl.show()

geom = get_geom(geom_info)
tlm = geom.adjoint_synthesis(maps['T'], 0, lmax_filt, lmax_filt, 0).squeeze()
eblm = geom.adjoint_synthesis(maps['QU'], 2, lmax_filt, lmax_filt, 0)
alms = {'t': tlm, 'e': eblm[0], 'b': eblm[1]}

beam = gauss_beam(5. / 180 / 60 * np.pi, lmax=lmax_filt)
inoise = {'tt': beam ** 2 / (35. / 180 / 60 * np.pi) ** 2,
          'ee': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2,
          'bb': beam ** 2 / (55. / 180 / 60 * np.pi) ** 2}
transfs = {f: np.ones(lmax_filt + 1, dtype=float) for f in 'teb'}

ls = np.arange(1, lmax_qlm + 1)
wls = ls ** 2 * (ls + 1) ** 2 * 1e7 / (2 * np.pi)
for qe_key in ['ptt']:
    cls_filt = copy_cls(cls_len, include=('t',))
    filtr = OpFilt(cls_filt, transfs, inoise)
    qlms = Qlms(filtr, filtr, cls_len, lmax_qlm)
    plm_in = alm_copy(unl_alms[unl_lab.index('p')], lmax_unl, lmax_qlm, lmax_qlm)

    plm, olm = qlms.get_qlms(qe_key, alms, verbose=True)  # Unormalized potentials
    rp, ro = qlms.get_response(qe_key, 'p', cls_len)
    pl.plot(ls, wls / rp[ls] * alm2cl(plm, plm_in, lmax_qlm, lmax_qlm, lmax_qlm)[ls])
    pl.plot(ls, wls / rp[ls] ** 2 * alm2cl(plm, plm, lmax_qlm, lmax_qlm, lmax_qlm)[ls])
    pl.plot(ls, wls * alm2cl(plm_in, plm_in, lmax_qlm, lmax_qlm, lmax_qlm)[ls])
    pl.plot(ls, wls * cls_unl['pp'][ls], c='k')

pl.show()



