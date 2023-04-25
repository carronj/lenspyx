import pylab as pl
import time
import numpy as np

from lenspyx import utils
from lenspyx.qest import qest, qresp, utils_qe as ut
from plancklens import qresp as qresp_pl


cl_unl, cl_len, cl_glen =  utils.get_ffp10_cls()
lmax_ivf, lmax_qlm,  nside = 3000, 3000, 2048

#q1 = qest.eval_qe(k, lmax_ivf, cl_glen, get_alm, lmax_qlm=lmax_qlm)
fal = {'t':1./ (cl_len['tt'][:lmax_ivf + 1] + (1./ 180 / 60 * np.pi) ** 2),
       'e':1./ (cl_len['ee'][:lmax_ivf + 1] + (2./ 180 / 60 * np.pi) ** 2),
       'b':1./ (cl_len['bb'][:lmax_ivf + 1] + (2./ 180 / 60 * np.pi) ** 2),}

fal['t'][:100] *= 0.
fal['e'][:100] *= 0.
fal['b'][:100] *= 0.
source = 'p'
for k in ['p_p', 'p']:#[source + 'tt', source + '_p', source]:
    QE = qest._get_qes(k, lmax_ivf, cl_glen)
    QEcpress = ut.qe_compress(qest._get_qes(k, lmax_ivf, cl_glen))
    t0 = time.time()
    R1 = qresp_pl._get_response(QE , source, cl_len, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    R2 = qresp._get_response(QEcpress, source, cl_len, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    t0 = time.time()
    R3= qresp._get_response_pl(QE, source, cl_len, fal, lmax_qlm=lmax_qlm)
    print(time.time()-t0)
    ls = np.arange(2, lmax_qlm + 1)
    pl.loglog(ls, np.abs(R1[0][ls] / R2[0][ls] -1.), label=k + ' grad')
    pl.plot(ls, np.abs(R1[1][ls] / R2[1][ls] -1.), label=k + 'curl')
    pl.plot(ls, np.abs(R3[1][ls] / R2[1][ls] -1.), label=k + 'curl')

pl.legend()
pl.show()