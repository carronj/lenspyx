import os
import numpy as np
import lenspyx
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import Alm
from lenspyx import cachers
from lenspyx.utils_hp import synalm, almxfl
from lenspyx.remapping.deflection_028 import deflection as duccd28
from lenspyx.remapping.deflection_029 import deflection as duccd29
from lenspyx.remapping import utils_geom
path2cls = os.path.dirname(lenspyx.__file__)
cls_unl  = camb_clfile(path2cls + '/data/cls/FFP10_wdipole_lenspotentialCls.dat')
cls_len  = camb_clfile(path2cls + '/data/cls/FFP10_wdipole_lensedCls.dat')

def syn_alms(spin, lmax_unl=5120, ctyp=np.complex128):
    ncomp = 1 + (abs(spin) > 0)
    mmax_unl = lmax_unl
    rtyp = lenspyx.remapping.deflection_028.rtype[ctyp]
    eblm = np.empty( (ncomp, Alm.getsize(lmax_unl, mmax_unl)), dtype=ctyp)
    eblm[0] = synalm(cls_unl['ee' if abs(spin) > 0 else 'tt'][:lmax_unl + 1], lmax_unl, mmax_unl,  rlm_dtype=rtyp)
    if ncomp > 1:
       eblm[1] = synalm(cls_unl['bb'][:lmax_unl + 1], lmax_unl, mmax_unl, rlm_dtype=rtyp)
    return eblm

def syn_ffi_ducc(lmax_len = 4096, dlmax=1024, epsilon=1e-5, dlm_fac=1., nthreads=0, dlmax_gl=1024, verbosity=1, planned=False):
    """"Returns realistic LCDM deflection field scaled by dlm_fac

    """
    lmax_unl = lmax_len + dlmax
    lmax_dlm, mmax_dlm = lmax_unl, lmax_unl
    lmaxthingauss = lmax_unl + dlmax_gl
    plm = synalm(cls_unl['pp'][:lmax_dlm + 1], lmax_dlm, mmax_dlm)
    dlm = almxfl(plm, dlm_fac * np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2)), mmax_dlm, False)
    ref_geom = utils_geom.Geom.get_thingauss_geometry(lmaxthingauss, 2)
    ffi_ducc = duccd28(ref_geom, dlm, mmax_dlm, numthreads=nthreads, verbosity=verbosity, dclm=None, epsilon=epsilon,
                       cacher=cachers.cacher_mem(safe=False), planned=planned)
    return ffi_ducc, ref_geom

def syn_ffi_ducc_29(lmax_len = 4096, dlmax=1024, epsilon=1e-5, dlm_fac=1., nthreads=0, dlmax_gl=1024, verbosity=1):
    """"Returns realistic LCDM deflection field scaled by dlm_fac

    """
    lmax_unl = lmax_len + dlmax
    lmax_dlm, mmax_dlm = lmax_unl, lmax_unl
    lmaxthingauss = lmax_unl + dlmax_gl
    plm = synalm(cls_unl['pp'][:lmax_dlm + 1], lmax_dlm, mmax_dlm)
    dlm = almxfl(plm, dlm_fac * np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2)), mmax_dlm, False)
    ref_geom = utils_geom.Geom.get_thingauss_geometry(lmaxthingauss, 2)
    ffi_ducc = duccd29(ref_geom, dlm, mmax_dlm, numthreads=nthreads, verbosity=verbosity, dclm=None, epsilon=epsilon, cacher=cachers.cacher_mem(safe=False))
    return ffi_ducc, ref_geom

def syn_ffi_lenscarf(lmax_len = 4096, dlmax=1024, target_res=1.7, dlm_fac=1., nthreads=4, dlmax_gl=1024, verbose=False, nbands=1):
    """"Returns realistic LCDM deflection field scaled by dlm_fac

    """
    from lenscarf.remapping import deflection as scarfd
    lmax_unl = lmax_len + dlmax
    lmax_dlm, mmax_dlm = lmax_unl, lmax_unl
    lmaxthingauss = lmax_unl + dlmax_gl

    plm = synalm(cls_unl['pp'][:lmax_dlm + 1], lmax_dlm, mmax_dlm)
    dlm = almxfl(plm, dlm_fac * np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2)), mmax_dlm, False)
    ref_geom = utils_geom.Geom.get_thingauss_geometry(lmaxthingauss, 2)
    pbref_geom = utils_geom.pbdGeometry(ref_geom, utils_geom.pbounds(0., 2 * np.pi))
    fft_tr, sht_tr = nthreads, nthreads
    ffi = scarfd(pbref_geom, target_res, dlm, mmax_dlm, fft_tr, sht_tr, verbose=verbose, dclm=None, cacher=cachers.cacher_mem(safe=False))
    return ffi, ref_geom