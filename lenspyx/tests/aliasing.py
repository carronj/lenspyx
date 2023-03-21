"""Tests grid-dependent aliasing"""
import ducc0.sht.experimental
import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_len
from lenspyx.utils_hp import synalm, Alm, alm2cl, alm_copy
from lenspyx.utils import timer
from lenspyx.remapping import utils_geom
import multiprocessing
import argparse
import pylab as pl

from lenspyx.utils import timer
if __name__ == '__main__':
    """This generate a Gaussian map with power extending to lmax_true, and tries reconstruct up tp lmax_wted
        using a grid with lmax_wted + dlmax_gl grid
        
        
    """
    parser = argparse.ArgumentParser(description='test aliasing')
    parser.add_argument('-s', dest='spin', type=int, default=0, help='spin to test')
    parser.add_argument('-lmax_true', dest='lmax_true', type=int, default=6900, help='lmax of lensed CMBs')
    args = parser.parse_args()

    ncomp, lmax_true, mmax_true = 1 + (args.spin > 0), args.lmax_true, args.lmax_true
    dlmax_gl = 1024
    lmax_wted, mmax_wted = 4096, 4096
    ntheta, nphi = (lmax_wted + dlmax_gl) + 2, 2 * (lmax_wted + dlmax_gl) + 2 # for CC and other grids
    nthreads = min(4, multiprocessing.cpu_count())
    alm = np.empty((ncomp, Alm.getsize(lmax_true, mmax_true)), complex)
    if args.spin > 0:
        alm[0] = synalm(cls_len['ee'][:lmax_true + 1], lmax_true, mmax_true)
        alm[1] = synalm(cls_len['bb'][:lmax_true + 1], lmax_true, mmax_true)
    else:
        alm[0] = synalm(cls_len['tt'][:lmax_true + 1], lmax_true, mmax_true)
    alm_cut = np.empty((ncomp, Alm.getsize(lmax_wted, mmax_wted)), complex)
    for i in range(ncomp):
        alm_cut[i] = alm_copy(alm[i], mmax_true, lmax_wted, mmax_wted)
    cls_in = [alm2cl(a, a, lmax_true, mmax_true, lmax_wted) for a in alm]
    gls = [utils_geom.Geom.get_thingauss_geometry(lmax_wted + dlmax_gl, args.spin),
           utils_geom.Geom.get_cc_geometry(lmax_wted + dlmax_gl + 2, 2 * (lmax_wted + dlmax_gl) + 2),
           utils_geom.Geom.get_cc_geometry(lmax_wted + dlmax_gl + 2, 2 * (lmax_wted + dlmax_gl) + 2)]
    gls += [utils_geom.Geom.get_healpix_geometry(int(np.sqrt(gls[0].npix()/12)))]
    gls_lab = ['thGL', 'CC lmax + 2', 'CC wMRtrick', 'HP same #pix as GL']
    fig, axes = pl.subplots(2, 1, figsize=(10, 5))
    for gl, gl_lab in zip(gls, gls_lab):
        tim = timer('this geom', False)
        m = gl.synthesis(alm, args.spin, lmax_true, mmax_true, nthreads=nthreads)
        tim.add(gl_lab + ' synthesis')
        if 'MRtrick' in gl_lab:
            alm_out = ducc0.sht.experimental.analysis_2d(map=np.atleast_2d(m).reshape((1, ntheta, nphi)),
                                                spin=args.spin, geometry='CC', lmax=lmax_wted, nthreads=nthreads)
        else:
            alm_out = gl.adjoint_synthesis(m, args.spin, lmax_wted, mmax_wted, nthreads=nthreads)
        tim.add(gl_lab + ' adjoint_synthesis')
        print(tim)
        cls_out = [alm2cl(a, a, lmax_wted, mmax_wted, lmax_wted) for a in alm_out]
        pl.sca(axes[0])
        pl.semilogy()
        ls = np.arange(max(1, args.spin), lmax_wted + 1)
        for i, lab in zip(range(ncomp), [gl_lab + ' G (%.2f Mpix)' % (gl.npix()/1e6), gl_lab + 'C']):
            pl.plot(ls, np.abs(cls_out[i][ls] / cls_in[i][ls] - 1), label=lab)
        pl.sca(axes[1])
        pl.semilogy()
        for i, lab in zip(range(ncomp), [gl_lab + ' G (%.2f Mpix)' % (gl.npix()/1e6), gl_lab + 'C']):
            diff = alm_out[i] - alm_cut[i]
            pl.plot(ls, np.sqrt(alm2cl(diff, diff, lmax_wted, mmax_wted, lmax_wted)[ls] / cls_in[i][ls]), label=lab)
            del diff
    pl.sca(axes[1])
    pl.title('residuals')
    pl.legend()
    pl.show()
