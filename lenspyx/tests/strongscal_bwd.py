"""This tests the exec. speed on FFP10-like accuracy maps


"""
from lenspyx.tests.helper import syn_ffi_ducc, cls_unl, cls_len, duccd
import healpy as hp, numpy as np

def binit(cl, d=10):
    ret = cl.copy()
    for l in range(d, ret.size -d):
        ret[l] = np.mean(cl[l-d:l+d+1])
    return ret


def get_ffi(dlmax_gl, nthreads=4, dlmax=1024):
    lmax_len, mmax_len, dlmax, dlmax_gl = 4096, 4096, dlmax, dlmax_gl
    ffi_ducc, ref_geom = syn_ffi_ducc(lmax_len=lmax_len, dlmax=dlmax,dlmax_gl=dlmax_gl,
                                      nthreads=nthreads)
    return ffi_ducc

if __name__ == '__main__':
    import argparse, os, time, json
    parser = argparse.ArgumentParser(description='test FFP10-like fwd building')
    DIR = os.environ['SCRATCH'] + '/lenspyx/'
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    args = parser.parse_args()
    lmax_len, mmax_len, dlmax = 4096, 4096, 1024
    lmax_unl = lmax_len + dlmax
    mmax_unl = lmax_unl
    dlmax_gl = 1024
    eblen = np.array([hp.synalm(cls_len['ee'][:lmax_len + 1]),
                      hp.synalm(cls_len['bb'][:lmax_len + 1])])
    for tentative in [1, 2]:
        for nt in range(1, 37):
            os.environ['OMP_NUM_THREADS'] = str(nt)
            print('doing %s_%s'%(nt, tentative))
            json_file = os.environ['SCRATCH'] + '/lenspyx/sscal_bwd_%s_%s.json'%(nt, tentative)
            ffi = get_ffi(dlmax_gl, nt)
            ffi.verbosity = 0
            t0 = time.time()
            ffi.lensgclm(eblen, mmax_len, 2, lmax_unl, mmax_unl, True)
            ffi.tim.keys['lensgclm (total, lmax in-out %s %s )'%(lmax_len, lmax_unl)] = time.time() - t0
            ffi.tim.dumpjson(json_file)
            print(json.load(open(json_file, 'r')))

