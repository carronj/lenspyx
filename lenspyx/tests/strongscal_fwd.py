"""This tests the exec. speed on FFP10-like accuracy maps


"""
from lenspyx.tests.helper import syn_ffi_ducc, syn_ffi_ducc_29, cls_unl, syn_alms
import healpy as hp, numpy as np

USE29 = True
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
                                      nthreads=nthreads, epsilon=1e-5)
    return ffi_ducc

if __name__ == '__main__':
    import argparse, os, time, json
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    if os.environ.get('SCRATCH', None) is not None:
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
    ebunl = syn_alms(spin, lmax_unl=lmax_unl, ctyp=np.complex64)[0:1] # grad-only
    import multiprocessing
    cpu_count = min(multiprocessing.cpu_count(), 36)
    for tentative in [1, 2]:
        for nt in range(1, cpu_count + 1):
            os.environ['OMP_NUM_THREADS'] = str(nt)
            print('doing %s_%s'%(nt, tentative))
            json_file = DIR + '/sscal_fwd_%s%s_%s_sgl.json'%('v29_'*USE29, nt, tentative)
            ffi = get_ffi(dlmax_gl, nt)
            ffi.verbosity = 1
            t0 = time.time()
            ffi.lensgclm(ebunl, mmax_unl, spin, lmax_len, mmax_len)
            ffi.tim.keys['lensgclm (total, lmax_unl %s )'%lmax_unl] = time.time() - t0
            ffi.tim.dumpjson(json_file)
