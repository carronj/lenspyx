import pylab as pl
import time
import numpy as np

from lenspyx import utils
from lenspyx.qest import qest,  nhl, utils_qe as ut
from plancklens import nhl as nhl_pl


cl_unl, cl_len, cl_glen =  utils.get_ffp10_cls()
lmax_ivf, lmax_qlm,  nside = 3000, 3000, 2048

fal = {'tt': 1. / (cl_len['tt'][:lmax_ivf + 1] + (1. / 180 / 60 * np.pi) ** 2),
       'ee': 1. / (cl_len['ee'][:lmax_ivf + 1] + (2. / 180 / 60 * np.pi) ** 2),
       'bb': 1. / (cl_len['bb'][:lmax_ivf + 1] + (2. / 180 / 60 * np.pi) ** 2),
       }

fal['te'] = cl_len['te'][:lmax_ivf + 1] / fal['tt'] / fal['ee']
fal['te'][:100] *= 0.
fal['tt'][:100] *= 0.
fal['ee'][:100] *= 0.
fal['bb'][:100] *= 0.
source = 'p'

for k1, k2 in zip(['p_p', 'p', 'ptt', 'ptt'], ['p_p', 'p', 'ptt', 'p']):#[source + 'tt', source + '_p', source]:
    print(k1, k2)
    QE1 = qest._get_qes(k1, lmax_ivf, cl_glen)
    QE2 = QE1 if k2 == k1 else qest._get_qes(k2, lmax_ivf, cl_glen)
    QE1cpress = ut.qe_compress(QE1)
    QE2cpress = QE1cpress if k2 == k1 else ut.qe_compress(QE2)

    t0 = time.time()
    n1 = nhl_pl._get_nhl(QE1 , QE2, fal, lmax_out=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    n2 = nhl._get_nhl_pl(QE1 , QE2, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    n3= nhl._get_nhl(QE1cpress , QE2cpress, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    n4= nhl._get_nhl_pl2(QE1 , QE2, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    n4= nhl.get_nhl(k1, k2,  cl_glen, fal, lmax_ivf, lmax_ivf, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    ls = np.arange(2, lmax_qlm + 1)
    pl.loglog(ls, np.abs(n1[0][ls] / n2[0][ls] -1.), label=' x '.join([k1, k2]) + ' grad')
    pl.plot(ls, np.abs(n1[1][ls] / n2[1][ls] -1.),   label=' x '.join([k1, k2]) + ' curl')
    pl.plot(ls, np.abs(n3[0][ls] / n1[0][ls] -1.),   label=' x '.join([k1, k2]) + ' grad')
    pl.plot(ls, np.abs(n4[0][ls] / n1[0][ls] -1.),   label=' x '.join([k1, k2]) + ' grad')

pl.legend()
pl.show()