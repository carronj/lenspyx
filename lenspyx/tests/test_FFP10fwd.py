"""This tests the exec. speed on FFP10-like accuracy maps


"""
from lenspyx.tests.helper import syn_ffi_ducc, cls_unl, cls_len
from lenscarf import cachers
import healpy as hp, numpy as np
import pylab as pl
from time import time

FORWARD = True #forward or adjoint operation
test_T = False
test_P = True  # this works nicely.
res, nside, nthreads = 0.75, 4096, 4
lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, 1024, 1024

#res, nside, nthreads = 3.,512, 3
#lmax_len, mmax_len, dlmax = 500, 500, 120


lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
thingauss = True # healpix rings if False
dlm_fac = 1.


ffi_ducc, ref_geom = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax,dlm_fac=dlm_fac,
                                  nthreads=nthreads)


def binit(cl, d=10):
    ret = cl.copy()
    for l in range(d, ret.size -d):
        ret[l] = np.mean(cl[l-d:l+d+1])
    return ret
if __name__ == '__main__':

    # 1st try: 39.341 exec time2nd try: 29.574 exec time
    # 1st try (JC) : 137.667 exec time

    if test_T:
        if FORWARD:
            tlm = hp.synalm(cls_unl['tt'][:lmax_unl+1])
            lmax_out, mmax_out = lmax_len, mmax_len
            lmax_in, mmax_in = lmax_unl, mmax_unl
        else:
            tlm = hp.synalm(cls_len['tt'][:lmax_len+1])
            lmax_out, mmax_out = lmax_unl, mmax_unl
            lmax_in, mmax_in = lmax_len, mmax_len

        t0 = time()
        len_tlm1 = ffi_ducc.lensgclm(tlm, mmax_in, 0, lmax_out, mmax_out, backwards= not FORWARD)
        print('1st try: %.3f exec time'%(time() - t0))
        t0 = time()
        len_tlm2 = ffi_ducc.lensgclm(tlm, mmax_in, 0, lmax_out, mmax_out, backwards= not FORWARD)
        print('2nd try: %.3f exec time'%(time() - t0))
        from plancklens import utils

        ls = np.arange(50, lmax_out+1)
        pl.plot(ls, binit(hp.alm2cl(len_tlm1) * utils.cli(cls_len['tt'][:lmax_out + 1]))[ls], label='ducc')

        # pl.plot(ls, binit(hp.alm2cl(len_tlmjc) *utils.cli(cls_len['tt'][:lmax_len+1]))[ls], label='jc')
        # pl.plot(ls, binit(hp.alm2cl(len_tlmjc - len_tlm1) *utils.cli(cls_len['tt'][:lmax_len+1]))[ls], label='diff')
        pl.legend()
        pl.axhline(1. - 0.01, c='k', ls='--')
        pl.axhline(1., c='k')
        pl.show()
    if test_P:
        if FORWARD:
            eblm = np.array([hp.synalm(cls_unl['ee'][:lmax_unl+1]),
                             hp.synalm(cls_unl['bb'][:lmax_unl+1])])
            lmax_out, mmax_out, lmax_in, mmax_in = lmax_len, mmax_len, lmax_unl, mmax_unl

        else:
            eblm = np.array([hp.synalm(cls_len['ee'][:lmax_len+1]),
                             hp.synalm(cls_len['bb'][:lmax_len+1])])
            lmax_out, mmax_out, lmax_in, mmax_in = lmax_unl, mmax_unl, lmax_len, mmax_len

        t0 = time()
        len_eb1 = ffi_ducc.lensgclm(eblm, mmax_in, 2, lmax_out, mmax_out, backwards= not FORWARD)
        print('1st try: %.3f exec time' % (time() - t0))
        t0 = time()
        len_eb2 = ffi_ducc.lensgclm(eblm,  mmax_in, 2, lmax_out, mmax_out, backwards= not FORWARD)
        print('2nd try: %.3f exec time' % (time() - t0))
        if FORWARD:
            from lenscarf.sims import sims_ffp10 # Make current lenspyx-like estimate splitting sky into bands
            lenspyx_like = sims_ffp10.cmb_len_ffp10(targetres=res, nbands=7, lmax_thingauss=lmax_unl + dlmax_gl, verbose=False)
            # HACK:
            lenspyx_like._get_dlm = lambda *args: (ffi_ducc.dlm, ffi_ducc.dlm * 0, lmax_unl, mmax_unl)
            t0 = time()
            len_ebjc = lenspyx_like._build_eb(0, unl_elm=eblm[0], unl_blm=eblm[1], lmax_len=lmax_out, mmax_len=mmax_out)
            print('1st try (JC) : %.3f exec time' % (time() - t0))
        from plancklens import utils

        ls = np.arange(100, lmax_len)
        pl.plot(ls, binit(hp.alm2cl(len_eb1[0]) *utils.cli(cls_len['ee'][:lmax_out+1]))[ls])
        pl.plot(ls, binit(hp.alm2cl(len_eb2[0]) *utils.cli(cls_len['ee'][:lmax_out+1]))[ls], label='ducc')
        if FORWARD:
            pl.plot(ls, binit(hp.alm2cl(len_ebjc[0]) *utils.cli(cls_len['ee'][:lmax_out+1]))[ls], label='jc lenspyx-like')
        pl.axhline(1. - 0.01, c='k', ls='--')
        pl.axhline(1., c='k')
        pl.legend()
        pl.show()
        pl.figure()

        pl.plot(ls, binit(hp.alm2cl(len_eb1[1]) *utils.cli(cls_len['bb'][:lmax_out+1]))[ls])
        pl.plot(ls, binit(hp.alm2cl(len_eb2[1]) *utils.cli(cls_len['bb'][:lmax_out+1]))[ls], label='ducc')
        if FORWARD:
            pl.plot(ls, binit(hp.alm2cl(len_ebjc[1]) *utils.cli(cls_len['bb'][:lmax_out+1]))[ls], label='jc lenspyx-like')

        pl.axhline(1. - 0.01, c='k', ls='--')
        pl.axhline(1., c='k')
        pl.legend()
        pl.show()