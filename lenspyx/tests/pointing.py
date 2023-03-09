"""Times the deflection angles calculation etc"""
import numpy as np

from lenspyx.tests.helper import syn_ffi_ducc_29
from lenspyx.utils_hp import almxfl
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
    from lenspyx.fortran import remapping as fremap

    cpu_count = multiprocessing.cpu_count()

    ffi, geom = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=cpu_count,
                             verbosity=0)
    gclm = np.zeros((2, ffi.dlm.size), dtype=complex)
    gclm[0] = ffi.dlm
    p2d = np.sqrt(np.arange(ffi.lmax_dlm + 1) * np.arange(1, ffi.lmax_dlm + 2, dtype=float))
    p2d[0] = 1.
    plm = almxfl(ffi.dlm, 1. / p2d, ffi.mmax_dlm, False)
    tht, phi0, nph, ofs = geom.theta, geom.phi0, geom.nph, geom.ofs
    npix = geom.npix()
    nrings = ffi.geom.theta.size
    for nthreads in [args.nt]:
        t0 = time.time()
        red, imd = ffi.geom.synthesis(gclm, 1, ffi.lmax_dlm, ffi.mmax_dlm, nthreads)
        t1 = time.time()
        thp_phip_mgamma = fremap.remapping.pointing(red, imd, tht, phi0, nph, ofs, nthreads)
        t2 = time.time()
        red2, imd2 = ffi.geom.synthesis_deriv1(np.atleast_2d(plm), ffi.lmax_dlm, ffi.mmax_dlm, nthreads)
        t3 = time.time()
        thp_phip_mgamma2 = fremap.remapping.pointingv2(red, imd, tht, phi0, nph, ofs, nthreads)
        t4 = time.time()
        print(" %s threads, lmax %s, nrings %s npix %s:"%(nthreads, ffi.lmax_dlm, nrings, npix))
        print('  1d        calc: %.3f Mpix/s, total %.3f sec'%(npix / (t1 - t0) / 1e6, t1 - t0))
        print('  pointingv1 calc: %.3f Mpix/s, total %.3f sec'%(npix / (t2 - t1) / 1e6, t2 - t1))
        print('  pointingv2 calc: %.3f Mpix/s, total %.3f sec'%(npix / (t4 - t3) / 1e6, t4 - t3))
        print('     fraction on pointingv1 and total time: %.3f,  %.3f s.'%( (t2 - t1) / (t2 - t0), (t2 - t0)))
        print('  1d deriv  calc: %.3f Mpix/s, total %.3f sec'%(npix / (t3 - t2) / 1e6, t3 - t2))
        print('     fraction on pointingv1 and total time: %.3f,  %.3f s.'%( (t2 - t1) / (t3 - t1), (t3-t1)))
        print(" This should be zero ", np.max(np.abs(red2 - red)/np.std(red)), np.max(np.abs(imd2 - imd)/np.std(imd)))
        print(" This should be zero ", np.max(np.abs(thp_phip_mgamma2 - thp_phip_mgamma)))