import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc_29
import pylab as pl
from duccjc.sht import adjoint_synthesis_general_ringweight as adj_syng_w
from ducc0.sht import adjoint_synthesis_general as adj_syng
from lenspyx.utils_hp import alm_copy, alm2cl
from lenspyx.remapping.utils_geom import st2mmax
from lenspyx.utils_cap import fskycap, eps_opti, args_default

if __name__ == '__main__':
    """This tests adjoint_synthesis_general_cap
    
        This generates delfected positions according to LCDM,and compare output alm of full-sky vs cap routines
    
    """
    args = args_default()
    args.whiten = True

    thta = 0 / 180 * np.pi
    thtb = 50 / 180 * np.pi # cap size
    #dtheta = 7. / 180 * np.pi


    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=args.nt,
                             verbosity=1, epsilon=10 ** (-args.epsilon))
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    sht_mode = 'STANDARD' if (args.spin == 0 or not args.gonly) else 'GRAD ONLY'
    ncomp = 1 + (args.spin > 0) * (not args.gonly)
    ptg = ffi._get_ptg() # locations
    assert ptg.shape[-1] == 2, ptg.shape


    geom_trc = ffi.geom.restrict(thta, thtb, update_ringstart=False, northsouth_sym=False)
    slic = slice(np.min(geom_trc.ofs), np.max(geom_trc.ofs + geom_trc.nph))
    ptg_sliced = ptg[slic, :]
    geom_trc = ffi.geom.restrict(thta, thtb, update_ringstart=True, northsouth_sym=False)
    m = np.random.standard_normal((ncomp, ptg_sliced.shape[0]))
    # Full-sky adjoint synthesis_general, at higher accuracy
    eblm = adj_syng(map=m, lmax=lmax_unl, mmax=mmax_unl, loc=ptg_sliced, spin=args.spin, epsilon=ffi.epsilon * 0.1,
                  nthreads=ffi.sht_tr, mode=sht_mode, verbose=ffi.verbosity * 1)

    if thta == 0:    # capped synthesis_general
        thtcap = np.max(ptg_sliced[:, 0]) * 1.0001
        eps_apo = eps_opti(lmax_unl, thtcap, dl=args.dl)
        new_mmax = min(int(st2mmax(args.spin, thtcap * (1. + eps_apo), lmax_unl)) + 1, mmax_unl) if args.adapt_mmax else mmax_unl
        print("Testing capped syng %2.f deg, fsky %.2f"%( (thtcap / np.pi * 180), fskycap(thtcap)))
        print('tht cap in deg %.0f, eps apo %.2f'%(thtcap/np.pi * 180, eps_apo))
        print('induced dlmax %s'%(int(lmax_unl *eps_apo)))
        print('reduction in mmax by a factor %.1f'%(mmax_unl/new_mmax))

        eblm_2= adj_syng_w(map=m, lmax=lmax_unl, mmax=new_mmax, loc=ptg_sliced,
                spin=args.spin, epsilon=ffi.epsilon,
                nthreads=ffi.sht_tr, mode=sht_mode, verbose=ffi.verbosity, thtcap=thtcap, eps_apo=eps_apo, apofct=args.apofct)
        # Resize alms to full-sky adjoint synthesis general
        eblm_2 = np.array([alm_copy(alm, new_mmax, lmax_unl, mmax_unl) for alm in eblm_2])
        for i, (ref, diff) in enumerate(zip(eblm, eblm_2 - eblm)):
            cldiff = alm2cl(diff, diff, lmax_unl, mmax_unl, lmax_unl)
            clref = alm2cl(ref, ref, lmax_unl, mmax_unl, lmax_unl)
            pl.semilogy(np.sqrt(cldiff[2:]/clref[2:]), label='comp %s capped, weighted (dl=%s)'%(i, args.dl))
            pl.xlabel(r'$\ell$')
            pl.ylabel(r'$C_\ell$')
    pl.legend()
    pl.show()