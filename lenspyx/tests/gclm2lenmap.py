"""Times the deflection angles calculation etc"""
import os
import numpy as np
import healpy as hp
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_unl
from lenspyx import lensing
import multiprocessing
import argparse
import time
from lenspyx.utils import timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='spin', type=int, default=2, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=1024, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=1024, help='buffer to lensed alms')
    parser.add_argument('-n', dest='nt', type=int, default=4, help='number of threads')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7, help='-log10 of nufft accuracy')

    args = parser.parse_args()
    cpu_count = multiprocessing.cpu_count()

    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=args.nt,
                             verbosity=0)
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    npix = geom.npix()
    nrings = ffi.geom.theta.size
    eblm = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                     hp.synalm(cls_unl['bb'][:lmax_unl + 1])])[0:1 + (args.spin != 0)]
    #t0 = time.time()
    #ffi._build_angles()
    #t1 = time.time()
    for nthreads in [args.nt]:
        ffi.sht_tr = nthreads
        os.environ['NUMEXPR_MAX_THREADS'] = str(nthreads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(nthreads)
        print("-----------------------")
        print('GL grid results: ')
        print(" %s threads, lmax %s, nrings %s, Mpix %s:"%(ffi.sht_tr, ffi.lmax_dlm, nrings, str(npix / 1e6)))
        ffi.tim = timer(False, 'deflection instance timer')
        t2 = time.time()
        len_tlm1 = ffi.gclm2lenmap(eblm, mmax_unl, args.spin, False)
        t3 = time.time()
        print(ffi.tim)
        print('            calc: %.3f Mpix/s, total %.3f sec'%(npix / (t3 - t2) / 1e6, t3 - t2))
        # Now healpix grid (nrings is 4 * nside or so)
        nside = args.lmax_len
        print("-----------------------")
        print('Healpix grid results: ')
        print(" %s threads, lmax %s, nrings %s, Mpix %s:"%(ffi.sht_tr, ffi.lmax_dlm, 4 * nside, str(12 * nside ** 2 / 1e6)))
        t4 = time.time()
        len_tlm2 = lensing.alm2lenmap_spin(eblm, [ffi.dlm, None], nside, args.spin, epsilon=10 ** (-args.epsilon) , verbose=True, experimental=True, nthreads=ffi.sht_tr)
        t5 = time.time()
        print('            calc: %.3f Mpix/s, total %.3f sec'%(12 * nside ** 2 / (t5 - t4) / 1e6, t5 - t4))
