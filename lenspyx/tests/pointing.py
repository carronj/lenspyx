"""Times the deflection angles calculation etc"""
import numpy as np

from lenspyx.tests.helper import syn_ffi_ducc_29
from lenspyx.utils import timer
from lenspyx.remapping import utils_geom
from lenspyx.utils_hp import almxfl
import multiprocessing
import argparse
import time
import ducc0

try:
    from ducc0 import jc
    HAS_JCducc = True
except:
    HAS_JCducc = False

def cppangles(d1, geom, nthreads):
    # build ptg
    geom.sort(geom.ofs)
    gl = geom
    tim = timer('C++ angles', False)
    ptg_0 = np.empty((geom.npix(), 2), dtype=float)    # tht, phi
    for tht, phi0, nph, ofs in zip(gl.theta, gl.phi0, gl.nph, gl.ofs):
        ptg_0[ofs: ofs + nph, 0] = tht
        ptg_0[ofs: ofs + nph, 1] = phi0 + np.arange(nph) * ((2 * np.pi) / nph)
    tim.add('GL grid angles filling')
    ptg = ducc0.misc.get_deflected_angles(ptg_0, d1.T, nthreads=nthreads)
    tim.add('deflected angles')
    print(tim)
    return ptg

def cppangles_inplace(d1, geom, nthreads, v=0):
    print("cppangles_inplace v " + str(v))
    # build ptg
    geom.sort(geom.ofs)
    gl = geom
    tim = timer('C++ angles', False)
    ptg_0 = np.empty((geom.npix(), 2 + (v != 0)), dtype=float)  # tht, phi
    for tht, phi0, nph, ofs in zip(gl.theta, gl.phi0, gl.nph, gl.ofs):
        ptg_0[ofs: ofs + nph, 0] = tht
        ptg_0[ofs: ofs + nph, 1] = phi0 + np.arange(nph) * ((2 * np.pi) / nph)
    tim.add('GL grid angles filling')
    if v == 0:
        ptg = ducc0.jc.get_deflected_angles_inplace(ptg_0, d1.T, nthreads=nthreads)
    else:
        ptg = ducc0.jc.get_deflected_angles_wg(ptg_0, d1.T, nthreads=nthreads)
    tim.add('deflected angles')
    print(tim)
    return ptg

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

    ffi, _ = syn_ffi_ducc_29(lmax_len=args.lmax_len, dlmax=args.dlmax, dlmax_gl=args.dlmax_gl, nthreads=cpu_count,
                             verbosity=0)

    geom = utils_geom.Geom.get_healpix_geometry(2048)
    ffi.geom = geom
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
        d1 = ffi.geom.synthesis(gclm, 1, ffi.lmax_dlm, ffi.mmax_dlm, nthreads)
        t1 = time.time()
        thp_phip_mgamma = fremap.remapping.pointing(d1[0], d1[1], tht, phi0, nph, ofs, nthreads)
        t2 = time.time()
        red2, imd2 = ffi.geom.synthesis_deriv1(np.atleast_2d(plm), ffi.lmax_dlm, ffi.mmax_dlm, nthreads)
        t3 = time.time()
        thp_phip_mgamma2 = fremap.remapping.pointingv2(d1[0], d1[1], tht, phi0, nph, ofs, nthreads)
        t4 = time.time()
        print(" %s threads, lmax %s, nrings %s Mpix %s:"%(nthreads, ffi.lmax_dlm, nrings, npix / 1e6))
        print('  1d        calc: %.3f Mpix/s, total %.3f sec'%(npix / (t1 - t0) / 1e6, t1 - t0))
        print('  pointingv1 calc: %.3f Mpix/s, total %.3f sec'%(npix / (t2 - t1) / 1e6, t2 - t1))
        print('  pointingv2 calc: %.3f Mpix/s, total %.3f sec'%(npix / (t4 - t3) / 1e6, t4 - t3))
        print('     fraction on pointingv1 and total time: %.3f,  %.3f s.'%( (t2 - t1) / (t2 - t0), (t2 - t0)))
        print('  1d deriv  calc: %.3f Mpix/s, total %.3f sec'%(npix / (t3 - t2) / 1e6, t3 - t2))
        print('     fraction on pointingv1 and total time: %.3f,  %.3f s.'%( (t2 - t1) / (t3 - t1), (t3-t1)))
        red, imd = d1
        print(" This should be zero ", np.max(np.abs(red2 - red)/np.std(red)), np.max(np.abs(imd2 - imd)/np.std(imd)))
        print(" This should be zero ", np.max(np.abs(thp_phip_mgamma2 - thp_phip_mgamma)))
        t5 = time.time()
        thtp_phip_cpp = cppangles(d1, geom, nthreads)
        t6 = time.time()
        print('  C++ pointing calc: %.3f Mpix/s, total %.3f sec'%(npix / (t6 - t5) / 1e6, t6 - t5))
        print(" This should be zero (tht) ", np.max(np.abs(thtp_phip_cpp[:, 0] - (thp_phip_mgamma[0, :]))))
        print(" This should be zero (phi) ", np.max(np.abs( (thtp_phip_cpp[:, 1] - thp_phip_mgamma[1, :]%(2 * np.pi)))))
        if HAS_JCducc:
            for v in [0, 2]:
                t6 = time.time()
                thtp_phip_cpp = cppangles_inplace(d1, geom, nthreads, v=v)
                t7 = time.time()
                print('  C++ inplace pointing calc: %.3f Mpix/s, total %.3f sec'%(npix / (t7 - t6) / 1e6, t7 - t6))
                print(" This should be zero (tht) ", np.max(np.abs(thtp_phip_cpp[:, 0] - (thp_phip_mgamma[0, :]))))
                print(" This should be zero (phi)", np.max(np.abs( (thtp_phip_cpp[:, 1]%(2 * np.pi) - thp_phip_mgamma[1, :]%(2 * np.pi)))))
                if v > 0:
                    print(" This should be zero (mga)", np.max(np.abs( (thtp_phip_cpp[:, 2] - thp_phip_mgamma[2, :]))))