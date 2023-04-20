"""Tests grid-dependent aliasing on lensed map


"""
import ducc0.sht.experimental
import numpy as np
from lenspyx.tests.helper import syn_ffi_ducc_29, cls_len, cls_unl
from lenspyx.utils_hp import synalm, Alm, alm2cl, alm_copy, almxfl
from lenspyx.remapping import utils_geom
import multiprocessing
import argparse
import pylab as pl

from lenspyx.utils import timer
from lenspyx.remapping.deflection_029 import deflection as duccd29

def build_ref_alm(lmax_sky, lmax_wted, spin):
    # build reference set of alms integrating exactly up to order 3
    mmax_sky = lmax_sky
    gl = utils_geom.Geom.get_thingauss_geometry((3 * 5120 + lmax_wted) // 2 + 1, args.spin)
    ncomp = 1 + (spin > 0)
    alm_unl = np.empty((ncomp, Alm.getsize(lmax_sky, mmax_sky)), complex)
    if args.spin > 0:
        alm_unl[0] = synalm(cls_len['ee'][:lmax_sky + 1], lmax_sky, mmax_sky)
        alm_unl[1] = synalm(cls_len['bb'][:lmax_sky + 1], lmax_sky, mmax_sky)
    else:
        alm_unl[0] = synalm(cls_len['tt'][:lmax_sky + 1], lmax_sky, mmax_sky)
    dlm = synalm(cls_unl['pp'][:lmax_sky + 1], lmax_sky, mmax_sky)
    almxfl(dlm, np.sqrt(np.arange(lmax_sky + 1) * np.arange(1, lmax_sky + 2)), mmax_sky, True)
    nthreads = min(4, multiprocessing.cpu_count())
    ffi_ducc = duccd29(gl, dlm, mmax_sky, numthreads=nthreads, verbosity=True, dclm=None, epsilon=1e-11)
    alm_len = ffi_ducc.lensgclm(alm_unl, mmax_sky, spin, lmax_wted, mmax_wted)
    return alm_unl, alm_len, dlm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test aliasing')
    parser.add_argument('-s', dest='spin', type=int, default=2, help='spin to test')
    args = parser.parse_args()

    dlmax_gl = 1024
    lmax_wted = mmax_wted = 4096
    lmax_sky = mmax_sky = 5120
    ntheta, nphi = (lmax_sky + dlmax_gl) + 2, 2 * (lmax_sky + dlmax_gl) + 2 # for CC and other grids

    nthreads = min(4, multiprocessing.cpu_count())
    tim = timer('ref_alm', False)
    alm_unl, alm_ref, dlm = build_ref_alm(lmax_sky, lmax_wted, args.spin)
    tim.add('ref alm')
    print(tim)

    ncomp, alm_size = alm_ref.shape

    cls_in = [alm2cl(a, a, lmax_wted, mmax_wted, lmax_wted) for a in alm_ref]
    gls = [utils_geom.Geom.get_thingauss_geometry(lmax_sky + dlmax_gl, args.spin),
           utils_geom.Geom.get_cc_geometry(lmax_sky + dlmax_gl + 2, 2 * (lmax_sky + dlmax_gl) + 2).thinout(args.spin, True),
           utils_geom.Geom.get_f1_geometry(lmax_sky + dlmax_gl + 2, 2 * (lmax_sky + dlmax_gl) + 2).thinout(args.spin, True),
           utils_geom.Geom.get_f1_geometry(lmax_sky + dlmax_gl + 2, 2 * (lmax_sky + dlmax_gl) + 2)]
    gls += [utils_geom.Geom.get_healpix_geometry(int(np.sqrt(gls[0].npix()/12)))]
    gls_lab = ['thGL', 'thCC', 'thF1', 'F1 2d', 'HP same #pix as GL'] # 'CC lmax + 2', 'CC wMRtrick' # Cant uses on the poles for now
    fig, axes = pl.subplots(3, 1, figsize=(3 * 7, 5))
    ratios = []
    clres_ref = []
    for igeom, (gl, gl_lab) in enumerate(zip(gls, gls_lab)):
        tim = timer('this geom', False)
        ffi_ducc = duccd29(gl, dlm, mmax_sky, numthreads=nthreads, verbosity=True, dclm=None, epsilon=1e-11)
        if '2d' in gl_lab:
            m = ffi_ducc.gclm2lenmap(alm_unl, mmax_sky, args.spin, False)
            alm_out = ducc0.sht.experimental.analysis_2d(map=np.atleast_2d(m).reshape((ncomp, ntheta, nphi)),
                                                spin=args.spin, geometry=gl_lab[:2], lmax=lmax_wted, nthreads=nthreads)
        else:
            alm_out = ffi_ducc.lensgclm(alm_unl, mmax_sky, args.spin, lmax_wted, mmax_wted)
        tim.add(gl_lab + ' lensgclm')
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
            diff = alm_out[i] - alm_ref[i]
            pl.plot(ls, np.sqrt(alm2cl(diff, diff, lmax_wted, mmax_wted, lmax_wted)[ls] / cls_in[i][ls]), label=lab)
            del diff
        pl.sca(axes[2])
        for i, lab in zip(range(ncomp), [gl_lab + ' G (%.2f Mpix)' % (gl.npix() / 1e6), gl_lab + ' C']):
            diff = alm_out[i] - alm_ref[i]
            if igeom == 0:
                clres_ref.append(alm2cl(diff, diff, lmax_wted, mmax_wted, lmax_wted))
            else:
                this_ratio = np.sqrt(alm2cl(diff, diff, lmax_wted, mmax_wted, lmax_wted)[ls] / clres_ref[i][ls])
                pl.plot(ls, this_ratio, label=lab)
                ratios.append(this_ratio)
            del diff
    pl.sca(axes[1])
    pl.title('residuals')
    pl.legend()
    pl.sca(axes[2])
    pl.title('residuals / residuals ' + gls_lab[0])
    pl.ylim(0.75, 1.25)
    pl.axhline(1., c='k')
    pl.legend(ncol=len(gls_lab)- 1)
    pl.show()
