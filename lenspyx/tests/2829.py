"""This tests the exec. speed on FFP10-like accuracy maps

    This compares the 'synthesis_general' ducc0 version against the more python-based first versions

    '_general' appears slightly faster

"""
from lenspyx.tests.helper import syn_ffi_ducc, syn_ffi_ducc_29, cls_unl
import healpy as hp, numpy as np

def binit(cl, d=10):
    ret = cl.copy()
    for l in range(d, ret.size -d):
        ret[l] = np.mean(cl[l-d:l+d+1])
    return ret


def get_ffi(dlmax_gl, USE29, nthreads=4, dlmax=1024, epsilon=1e-5):
    lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, dlmax, dlmax_gl
    func = syn_ffi_ducc_29 if USE29 else syn_ffi_ducc
    ffi_ducc, ref_geom = func(lmax_len=lmax_len, dlmax=dlmax,dlmax_gl=dlmax_gl,
                                      nthreads=nthreads, verbosity=0, epsilon=epsilon)
    return ffi_ducc

if __name__ == '__main__':
    import argparse, os, time, json
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')

    args = parser.parse_args()
    spin, epsilon = 0, 1e-7
    single_prec = epsilon >= 1e-6
    lmax_len, mmax_len, dlmax = 4096, 4096, 1024
    lmax_unl = lmax_len + dlmax
    mmax_unl = lmax_unl
    dlmax_gl = 1024
    ebunl = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                      hp.synalm(cls_unl['bb'][:lmax_unl + 1])])
    ebunl = ebunl[0:1 + (spin > 0)]
    import multiprocessing
    cpu_count = min(multiprocessing.cpu_count(), 36)
    for tentative in [1, 2, 3]:
        ffi = get_ffi(dlmax_gl, False, nthreads=4)
        ptg = ffi._get_ptg()
        for nt in [4]:
            os.environ['OMP_NUM_THREADS'] = str(nt)
            print('doing %s_%s'%(nt, tentative))
            ffi = get_ffi(dlmax_gl, False, nthreads=nt, epsilon=epsilon)
            ffi.verbosity = 0
            ffi.cacher.cache('ptg', ptg.copy())
            t0 = time.time()
            S1 = ffi.gclm2lenmap(ebunl, mmax_unl, spin, False, polrot=False)
            ffi.tim.keys['lensgclm (total, lmax_unl %s )'%lmax_unl] = time.time() - t0
            print('28 fwd: %.3f'%(time.time() - t0))

            ffi._totalconvolves0 = True
            t0 = time.time()
            S1 = ffi.gclm2lenmap(ebunl, mmax_unl, spin, False, polrot=False)
            ffi.tim.keys['lensgclm (total, lmax_unl %s )' % lmax_unl] = time.time() - t0
            print('28 fwd (total convolve): %.3f' % (time.time() - t0))

            #print(ffi.tim)
            ffi29 = get_ffi(dlmax_gl, True, nthreads=nt, epsilon=epsilon)
            ffi29.verbosity = 0
            ffi29.cacher.cache('ptg', ptg.copy())
            t0 = time.time()
            S2 = ffi29.gclm2lenmap(ebunl, mmax_unl, spin, False,  polrot=False)
            ffi.tim.keys['lensgclm (total, lmax_unl %s )'%lmax_unl] = time.time() - t0
            print('29 fwd: %.3f'%(time.time() - t0))
            #print(ffi.tim)
            print(np.max(np.abs(S1 - S2)))
            #--------
            Sc = S2[0] +  (1j * S1[1] if spin > 0 else 0.)
            t0 = time.time()
            S4 = ffi.lenmap2gclm(Sc, spin, lmax_len, mmax_len)
            print('28 bwd: %.3f'%(time.time() - t0))

            ffi._totalconvolves0 = True
            t0 = time.time()
            S4 = ffi.lenmap2gclm(Sc, spin, lmax_len, mmax_len)
            print('28 bwd: %.3f'%(time.time() - t0))

            ffi29.verbosity = 1
            t0 = time.time()
            S3 = ffi29.lenmap2gclm(Sc, spin, lmax_len, mmax_len)
            print('29 bwd: %.3f'%(time.time() - t0))
            print(ffi29.tim)


