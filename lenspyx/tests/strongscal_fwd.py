"""This tests the exec. speed on FFP10-like accuracy maps


"""
from lenspyx.tests.helper import syn_ffi_ducc, syn_ffi_ducc_29, cls_unl
import healpy as hp, numpy as np

USE29 =False
spin = 2
def binit(cl, d=10):
    ret = cl.copy()
    for l in range(d, ret.size -d):
        ret[l] = np.mean(cl[l-d:l+d+1])
    return ret


def get_ffi(dlmax_gl, nthreads=4, dlmax=1024):
    lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, dlmax, dlmax_gl
    func = syn_ffi_ducc_29 if USE29 else syn_ffi_ducc
    ffi_ducc, ref_geom = func(lmax_len=lmax_len, dlmax=dlmax,dlmax_gl=dlmax_gl,
                                      nthreads=nthreads)
    return ffi_ducc

if __name__ == '__main__':
    import argparse, os, time, json
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    if os.path.exists('SCRATCH'):
        DIR = os.environ['SCRATCH'] + '/lenspyx/'
    else:
        #local ?
        DIR = os.environ['ONED'] + '/ducclens/Tex/figs/MacOSlocal'
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    args = parser.parse_args()
    lmax_len, mmax_len, dlmax = 4096, 4096, 1024
    lmax_unl = lmax_len + dlmax
    mmax_unl = lmax_unl
    dlmax_gl = 1024
    ebunl = np.array([hp.synalm(cls_unl['ee'][:lmax_unl + 1]),
                      hp.synalm(cls_unl['bb'][:lmax_unl + 1])]).astype(np.complex64)
    ebunl = np.atleast_2d(ebunl[:1 + (spin > 0)])
    import multiprocessing
    cpu_count = min(multiprocessing.cpu_count(), 36)
    for tentative in [1, 2, 3, 4]:
        for nt in [4]:
            os.environ['OMP_NUM_THREADS'] = str(nt)
            print('doing %s_%s'%(nt, tentative))
            json_file = DIR + '/sscal_fwd_%s%s_%s_sgl.json'%('v29_'*USE29, nt, tentative)
            ffi = get_ffi(dlmax_gl, nt)
            ffi.verbosity = 0
            t0 = time.time()
            ffi.lensgclm(ebunl, mmax_unl, spin, lmax_len, mmax_len, False)
            ffi.tim.keys['lensgclm (total, lmax_unl %s )'%lmax_unl] = time.time() - t0
            ffi.tim.dumpjson(json_file)
            print(json.load(open(json_file, 'r')))

