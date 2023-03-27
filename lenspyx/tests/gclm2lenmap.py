"""Times the deflection angles calculation etc"""
import os
import numpy as np
import healpy as hp
from lenspyx.tests.helper import syn_ffi_ducc_29,  syn_alms, cls_unl
from lenspyx import cachers
import multiprocessing
import argparse
import time
from lenspyx.remapping import utils_geom
from lenspyx.utils import timer
import ducc0
import tracemalloc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='spin', type=int, default=2, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=1024, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=1024, help='buffer to lensed alms')
    parser.add_argument('-n', dest='nt', type=int, default=4, help='number of threads')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7, help='-log10 of nufft accuracy')
    parser.add_argument('-cis', dest='cis', action='store_true', help='test cis action')
    parser.add_argument('-gonly', dest='gonly', action='store_true', help='grad-only SHTs')
    parser.add_argument('-HL', dest='HL', type=int, default=0, help='also test Healpix pixelization with this nside')
    parser.add_argument('-alloc', dest='alloc',  type=int, default=0, help='tries pre-allocating ''alloc'' GB of memory')
    parser.add_argument('-tracemalloc', dest='tracemalloc',  action='store_true', help='trace memory usage')

    args = parser.parse_args()

    if args.alloc:
        if ducc0.misc.preallocate_memory(args.alloc):
            print('gclm2lenmap: allocated %s GB'%args.alloc)
        else:
            print('gclm2lenmap: allocation of %s GB failed'%args.alloc)

    cpu_count = multiprocessing.cpu_count()

    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=args.nt,
                             verbosity=1, epsilon=10 ** (-args.epsilon))
    lmax_unl, mmax_unl = args.lmax_len + args.dlmax, args.lmax_len + args.dlmax

    npix = geom.npix()
    nrings = ffi.geom.theta.size
    eblm = syn_alms(args.spin, lmax_unl=lmax_unl, ctyp=np.complex64 if ffi.single_prec else np.complex128)
    if args.tracemalloc:
        tracemalloc.start()
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
        if args.tracemalloc:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            print("     Tracemalloc first 4 lines:")
            for stat in top_stats[:4]:
                print(stat)
        if args.HL:
            # Now healpix grid (nrings is 4 * nside or so)
            nside = args.HL
            ffi.geom = utils_geom.Geom.get_healpix_geometry(nside)
            ffi.cacher = cachers.cacher_mem(safe=False)
            ffi._cis = args.cis
            ffi.tim = timer(False, 'deflection instance timer')
            print("-----------------------")
            print('Healpix grid results: ')
            print(" %s threads, lmax %s nside %s, nrings %s, Mpix %s:"%(ffi.sht_tr, ffi.lmax_dlm,nside, ffi.geom.theta.size, str(12 * nside ** 2 / 1e6)))
            t4 = time.time()
            len_tlm2 = ffi.gclm2lenmap(eblm, mmax_unl, args.spin, False)
            t5 = time.time()
            print(ffi.tim)
            print('            calc: %.3f Mpix/s, total %.3f sec'%(12 * nside ** 2 / (t5 - t4) / 1e6, t5 - t4))
            if args.tracemalloc:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("     Tracemalloc first 4 lines:")
                for stat in top_stats[:4]:
                    print(stat)