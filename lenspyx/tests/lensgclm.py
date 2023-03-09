"""Times the deflection angles calculation etc"""
import os
import numpy as np
import healpy as hp
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_unl
import multiprocessing
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='spin', type=int, default=2, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=1024, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=1024, help='buffer to lensed alms')
    parser.add_argument('-n', dest='nt', type=int, default=4, help='number of threads')

    args = parser.parse_args()
    cpu_count = multiprocessing.cpu_count()

    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=cpu_count,
                             verbosity=0)
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    npix = geom.npix()
    nrings = ffi.geom.theta.size
    eblm = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                     hp.synalm(cls_unl['bb'][:lmax_unl + 1])])
    ffi._build_angles()
    for nthreads in range(1, args.nt + 1):
        ffi.sht_tr = nthreads
        os.environ['NUMEXPR_MAX_THREADS'] = str(nthreads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(nthreads)
        ffi.tim.keys = {}
        t0 = time.time()
        len_tlm1 = ffi.lensgclm(eblm, mmax_unl, args.spin, args.lmax_len, args.lmax_len, backwards=False)
        t1 = time.time()
        print(" %s threads, lmax %s, nrings %s npix %s:"%(ffi.sht_tr, ffi.lmax_dlm, nrings, npix))
        print('            calc: %.3f Mpix/s, total %.3f sec'%(npix / (t1 - t0) / 1e6, t1 - t0))
        #print(ffi.tim.keys)
