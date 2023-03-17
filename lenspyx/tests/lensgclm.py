"""Times the deflection angles calculation etc"""
import os
import numpy as np
import healpy as hp
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_unl
import multiprocessing
import argparse
import time
from lenspyx.utils import timer
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='spin', type=int, default=2, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=0, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=0, help='buffer to lensed alms')
    parser.add_argument('-n', dest='nt', type=int, default=4, help='number of threads')
    parser.add_argument('-bwd', dest='bwd', action='store_true', help='adjoint lensing')
    parser.add_argument('-gonly', dest='gonly', action='store_true', help='grad-only mode')

    args = parser.parse_args()
    cpu_count = multiprocessing.cpu_count()

    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=cpu_count,
                             verbosity=0)
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    npix = geom.npix()
    nrings = ffi.geom.theta.size
    eblm = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                     hp.synalm(cls_unl['bb'][:lmax_unl + 1])])
    eblm = eblm[:1 + (args.spin > 0) * (not args.gonly)]
    mode = 'STANDARD' if not args.gonly else 'GRAD_ONLY'
    ffi._build_angles()
    for nthreads in range(1, args.nt + 1):
        ffi.sht_tr = nthreads
        ffi.tim = timer('ffi timer', False)
        t0 = time.time()
        len_tlm1 = ffi.lensgclm(eblm, mmax_unl, args.spin, args.lmax_len, args.lmax_len, backwards=args.bwd, output_sht_mode=mode)
        t1 = time.time()
        print(" %s threads, spin %s lmax %s, nrings %s Mpix %s:"%(ffi.sht_tr, args.spin, ffi.lmax_dlm, nrings, npix/1e6))
        print('            calc: %.3f Mpix/s, total %.3f sec'%(npix / (t1 - t0) / 1e6, t1 - t0))
        print(ffi.tim)
