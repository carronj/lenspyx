import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc_29, syn_alms

import pylab as pl
from duccjc.sht import synthesis_general_ringweight as syng_w, adjoint_synthesis_general_ringweight as adj_syng_w
from ducc0.sht import synthesis_general as syng, adjoint_synthesis_general as adj_syng
from lenspyx.utils_hp import alm_copy
from lenspyx.remapping import deflection_029, deflection_028 as deflection_28
from lenspyx.remapping.utils_geom import st2mmax
from lenspyx.utils_cap import fskycap, eps_opti, examine, args_default

if __name__ == '__main__':
    """This tests synthesis_general_cap

        This generates delfected positions according to LCDM,and compare output maps of full-sky vs cap routines

    """
    args = args_default()
    thta = 0 / 180 * np.pi
    thtb = 50 / 180 * np.pi # cap size
#    args.whiten = False


    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=args.nt,
                             verbosity=1, epsilon=10 ** (-args.epsilon))
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    eblm = syn_alms(args.spin, lmax_unl=lmax_unl, ctyp=np.complex64 if ffi.single_prec else np.complex128, white=args.whiten)
    eblm = np.atleast_2d(eblm)
    if args.gonly:
        eblm = eblm[:1]
    sht_mode = deflection_28.ducc_sht_mode(eblm, args.spin)
    ptg = ffi._get_ptg() # locations
    assert ptg.shape[-1] == 2, ptg.shape


    geom_trc = ffi.geom.restrict(thta, thtb, update_ringstart=False, northsouth_sym=False)
    slic = slice(np.min(geom_trc.ofs), np.max(geom_trc.ofs + geom_trc.nph))
    ptg_sliced = ptg[slic, :]
    geom_trc = ffi.geom.restrict(thta, thtb, update_ringstart=True, northsouth_sym=False)

    # Full-sky synthesis_general, at higher accuracy
    values = syng(lmax=lmax_unl, mmax=mmax_unl, alm=eblm, loc=ptg, spin=args.spin, epsilon=ffi.epsilon * 0.1,
                  nthreads=ffi.sht_tr, mode=sht_mode, verbose=ffi.verbosity * 1)

    if thta == 0:    # capped synthesis_general
        thtcap = np.max(ptg_sliced[:, 0]) * 1.0001
        eps_apo = eps_opti(lmax_unl, thtcap, dl=args.dl)
        new_mmax = min(int(st2mmax(args.spin, thtcap * (1. + eps_apo), lmax_unl)) + 1, mmax_unl) if args.adapt_mmax else mmax_unl
        print("Testing capped syng %2.f deg, fsky %.2f"%( (thtcap / np.pi * 180), fskycap(thtcap)))
        print('tht cap in deg %.0f, eps apo %.2f'%(thtcap/np.pi * 180, eps_apo))
        print('induced dlmax %s'%(int(lmax_unl *eps_apo)))
        print('reduction in mmax by a factor %.1f'%(mmax_unl/new_mmax))

        gclm = np.array([alm_copy(alm, mmax_unl, lmax_unl, new_mmax) for alm in eblm])
        values_w = syng_w(lmax=lmax_unl, mmax=new_mmax, alm=gclm, loc=ptg_sliced,
                ringweights=np.array([1.]), spin=args.spin, epsilon=ffi.epsilon,
                nthreads=ffi.sht_tr, mode=sht_mode, verbose=ffi.verbosity, thtcap=thtcap, eps_apo=eps_apo, apofct=args.apofct)
        print(np.max(np.abs(values_w - values[:, slic])))
        for i, (val, diff) in enumerate(zip(values[:, slic], values_w-values[:, slic])):
            examine(val, diff, geom_trc, ffi.epsilon, label='comp %s capped, weighted (dl=%s)'%(i, args.dl))
    pl.legend()
    pl.show()