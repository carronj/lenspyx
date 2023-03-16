"""Times the deflection angles calculation etc"""
import numpy as np
import healpy as hp
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_unl
from lenspyx.remapping import utils_geom
import multiprocessing
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='s', type=int, default=2, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=1024, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=1024, help='buffer to lensed alms')
    parser.add_argument('-HL', dest='HL', action='store_true', help='use Healpix pixelization with nside=lmax_len')

    args = parser.parse_args()
    cpu_count =multiprocessing.cpu_count()

    ffi, ref_geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=cpu_count,
                                verbosity=0)
    if args.HL:
        ref_geom = utils_geom.Geom.get_healpix_geometry(args.lmax_len)
    gclm = np.empty((2, hp.Alm.getsize(ffi.lmax_dlm)), dtype=complex)
    gclm[0] = hp.synalm(cls_unl['ee'][:ffi.lmax_dlm + 1])
    gclm[1] = 0.
    elm, blm = gclm
    npix = ref_geom.npix()
    for nthreads in [min(4, cpu_count)]:
        print(" %s threads, lmax %s, Mpix %s:"%(nthreads, ffi.lmax_dlm, npix / 1e6))
        dt_g = 0.
        dt_s = 0.
        for trials in range(3):
            t0 = time.time()
            d1 = ref_geom.synthesis(gclm[0], args.s, ffi.lmax_dlm, ffi.mmax_dlm, nthreads, mode='GRAD_ONLY')
            t1 = time.time()
            dt_g += (t1 - t0)
            print('  spin %s  GRAD_ONLY synthesis: %.3f Mpix/s, total %.3f sec'%(args.s, npix / (t1 - t0) / 1e6, t1 - t0))
            t0 = time.time()
            d2 = ref_geom.synthesis(gclm, args.s, ffi.lmax_dlm, ffi.mmax_dlm, nthreads, mode='STANDARD')
            t1 = time.time()
            dt_s += (t1 - t0)
            print('  spin %s  STANDARD synthesis: %.3f Mpix/s, total %.3f sec'%(args.s, npix / (t1 - t0) / 1e6, t1 - t0))
            print('This should be zero:', np.max(np.abs(d1 - d2)))
        print('*** results for synthesis exec. time ratio : %.3f'%(dt_g / dt_s))
        dt_g = 0.
        dt_s = 0.
        for trials in range(3):
            d1c = d1.copy()
            t0 = time.time()
            g1 = ref_geom.adjoint_synthesis(d1c, args.s, ffi.lmax_dlm, ffi.mmax_dlm, nthreads, mode='GRAD_ONLY')
            t1 = time.time()
            dt_g += (t1 - t0)
            print('  spin %s GRAD_ONLY  ad synthesis: %.3f Mpix/s, total %.3f sec'%(args.s, npix / (t1 - t0) / 1e6, t1 - t0))
            d1c = d1.copy()
            t0 = time.time()
            g2, c2 = ref_geom.adjoint_synthesis(d1c, args.s, ffi.lmax_dlm, ffi.mmax_dlm, nthreads, mode='STANDARD')
            t1 = time.time()
            dt_s += (t1 - t0)
            print('  spin %s STANDARD  ad synthesis: %.3f Mpix/s, total %.3f sec'%(args.s, npix / (t1 - t0) / 1e6, t1 - t0))
            print('This should be zero:', np.max(np.abs(g1 - g2)))
        print('*** results for ad-synthesis exec. time ratio : %.3f'%(dt_g / dt_s))
