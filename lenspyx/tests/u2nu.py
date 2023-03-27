import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc, syn_ffi_ducc_29, syn_alms
import multiprocessing
from lenspyx.utils import timer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    parser.add_argument('-s', dest='spin', type=int, default=0, help='spin to test')
    parser.add_argument('-lmaxlen', dest='lmax_len', type=int, default=4096, help='lmax of lensed CMBs')
    parser.add_argument('-dlmaxgl', dest='dlmax_gl', type=int, default=1024, help='buffer to GL grid')
    parser.add_argument('-dlmax', dest='dlmax', type=int, default=1024, help='buffer to lensed alms')
    parser.add_argument('-eps', dest='epsilon', type=float, default=7, help='-log10 of nufft accuracy')
    parser.add_argument('-nmin', dest='ntmin', type=int, default=4, help='min number of threads')
    parser.add_argument('-nmax', dest='ntmax', type=int, default=4, help='max number of threads')

    args = parser.parse_args()

    spin = args.spin
    lmax_len, dlmax, dlmax_gl = args.lmax_len, args.dlmax, args.dlmax_gl
    lmax_unl = lmax_len + dlmax

    ffi, geom = syn_ffi_ducc(nthreads=args.ntmin, lmax_len=lmax_len, dlmax=dlmax, dlmax_gl=dlmax_gl, verbosity=1, epsilon=10**(-args.epsilon))

    for bwd in [False, True]:
        if bwd:
            lmax_in, mmax_in, lmax_out, mmax_out, = lmax_len, lmax_len, lmax_unl, lmax_unl
        else:
            lmax_in, mmax_in, lmax_out, mmax_out, = lmax_unl, lmax_unl, lmax_len, lmax_len
        alm = syn_alms(spin, lmax_unl=lmax_in, ctyp=np.complex64 if ffi.single_prec else np.complex128)
        for n in range(args.ntmin, args.ntmax + 1):
            ffi.sht_tr = n
            k = 'nu2u' if bwd else 'u2nu'
            ffi.tim = timer('', False)
            ffi.lensgclm(alm, mmax_in, spin, lmax_out, mmax_out, backwards=bwd)
            dt = ffi.tim.keys[k]
            print(k + ": %s threads, %.3f Mpix / s"%(ffi.sht_tr, geom.npix() * 1e-6 / dt))

