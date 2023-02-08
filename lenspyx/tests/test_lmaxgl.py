"""This tests the exec. speed on FFP10-like accuracy maps


"""
from lenspyx.tests.helper import syn_ffi_ducc, cls_unl, cls_len, duccd
from lenscarf import cachers
import healpy as hp, numpy as np
import pylab as pl
from time import time

def get_lensedcmb(ffi:duccd, dlmax_gl):
    lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, 1024, dlmax_gl
    lmax_unl = lmax_len + dlmax
    mmax_unl = lmax_unl
    eblm = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                     hp.synalm(cls_unl['bb'][:lmax_unl + 1])])
    return ffi.lensgclm(eblm, mmax_unl, 2, lmax_len, mmax_len, False)
def binit(cl, d=10):
    ret = cl.copy()
    for l in range(d, ret.size -d):
        ret[l] = np.mean(cl[l-d:l+d+1])
    return ret

def get_ffi(dlmax_gl, nthreads=4):
    lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, 1024, dlmax_gl
    ffi_ducc, ref_geom = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax,dlmax_gl=dlmax_gl,
                                      nthreads=nthreads)
    return ffi_ducc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test FFP10-like building')
    parser.add_argument('-nt', dest='nt', default=4, type=int, help='numbers of openMP threads')
    parser.add_argument('-dlmax_gl', dest='dlmax_gl', default=4, type=int, help='numbers of openMP threads')

    args = parser.parse_args()
    nthreads = args.nt
