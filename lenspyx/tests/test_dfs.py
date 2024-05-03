import pylab as pl
import numpy as np
import time

import ducc0
from lenspyx.wigners import wigners
from lenspyx.remapping.bumpf import bump as bumpf
from lenspyx.remapping.deflection_028_bded import BdedGeom
from lenspyx.tests import helper
from lenspyx.utils import get_ffp10_cls
from lenspyx.utils_hp import synalm
from lenspyx.utils import timer

clu, cll, _ = get_ffp10_cls() # fiducial CMB spectra

# Test on patch extending ~ 40 degrees in latitude, apodized on 2 x 5 degrees
thtmin = 0.#129.475 / 180 * np.pi
thtmax = 40 / 180 * np.pi #159.399 / 180 * np.pi
dtht = 5. / 180 * np.pi

# parameters
lmax = 4000 # 4000
dlmax_gl = 100 # lmax of synthesis general will be lmax + dlmax_gl
eps = 1e-7
verbose = False
polrot = False # Do not care about polarization rotation here
spin = 2
RESET = True
PLOT_BUMP_FUNC = False
# setting up smooth bump function:
# This one has harmonic modes going like e^{-\sqrt{L}}
bump = lambda tht: bumpf.bump(thtmin, thtmax, dtht, tht)

# plot of bump legendre coeff:
band_limit_increase = 500
if PLOT_BUMP_FUNC:
    tht, wg = wigners.get_thgwg(20000)
    bump_L = wigners.wignercoeff(bump(tht) * wg, tht, 0, 0, lmax=10000)
    pl.loglog(np.arange(1,10002), np.abs(bump_L))
    pl.show()
    print('At this band-limit, the harmonic mode of the bump function has approximately '
      'decreased by a factor of %.2e' % (np.mean(np.abs(bump_L[band_limit_increase-100:band_limit_increase+100])) / bump_L[0]))
if True:
    lmax_tlm = lmax + dlmax_gl
    tlm = synalm(clu['ee'][:lmax_tlm + 1], lmax_tlm, lmax_tlm) # input array
    # deflection instances.
    # This one remaps using synthesis general:
    ffi_ducc, geom = helper.syn_ffi_ducc_29(lmax_len=lmax, dlmax_gl=dlmax_gl, epsilon=eps, verbosity=verbose)
    # This one according to oldest versions with exposed DFS maps:
    ffi_jc, _ = helper.syn_ffi_ducc(lmax_len=lmax, dlmax_gl=dlmax_gl, epsilon=eps, verbosity=verbose)

    # Add lat and long bounds for a cutout of the pointings
    # Patch crudely 120 degrees longitudinal extent, plus buffer, on original location, but here pole so 360
    dphi = 2 * np.pi * (geom.theta >= max(0., thtmin - dtht)) * (geom.theta <= (thtmax + dtht))
    bounded_geom = BdedGeom(geom, dphi)
    thts_trunc = bounded_geom.theta[np.where(bounded_geom.nph_bded > 0)]
    # Select the pointings to be on this region only
    ptg = bounded_geom.collectmap((ffi_jc._get_ptg()[:, :2]).T).T
    ffi_ducc._get_ptg = lambda : ptg

    print("%s pointing points, %.1f percent of the sky"%(ptg.shape[0], 100 * ptg.shape[0] / geom.npix() ))
    #ptg_full = ffi_jc._get_ptg()[:, :2]
    #ti = time.time()
    #tlm_len_ducc = np.atleast_2d(ffi_ducc.gclm2lenmap(tlm, None, spin, False, polrot=polrot, ptg=ptg_full))
    #baseline_ducc_time = (time.time() - ti)
    #print('%.2f sec for ducc-baseline with no tricks and full-sky pointing' % baseline_ducc_time)

    ti = time.time()
    tlm_len_ducc = np.atleast_2d(ffi_ducc.gclm2lenmap(tlm, None, spin, False, polrot=polrot, ptg=ptg))
    baseline_ducc_time = (time.time() - ti)
    print('%.2f sec for ducc-baseline with no tricks' % baseline_ducc_time)



    ti = time.time()
    tlm_len = np.atleast_2d(ffi_jc.gclm2lenmap(tlm, None, spin, False, polrot=polrot, ptg=ptg))
    baseline_jc_time = (time.time() - ti)
    print('%.2f sec for jc-baseline with no tricks' % baseline_jc_time)
    std = np.std(tlm_len_ducc)
    print('max-reldev between the two baselines ',np.abs(np.max(tlm_len_ducc[0] - tlm_len[0]))/std)


tlm_ref = tlm_len_ducc

def get_reldev(m1, m2):
    # collect deviations ring per ring
    meanreldev = np.zeros(bounded_geom.nrings_bded())
    maxreldev = np.zeros(bounded_geom.nrings_bded())
    for ir in range(bounded_geom.nrings_bded()):
        ofs = bounded_geom.ofs_bded[ir]
        nph = bounded_geom.nph_bded[ir]
        if nph > 0:
            dev = m1[0, ofs:ofs + nph] - m2[0, ofs:ofs + nph]
            meanreldev[ir] = np.sqrt(np.mean(dev ** 2)) / std
            maxreldev[ir] = np.sqrt(np.max(dev ** 2)) / std
    return meanreldev, maxreldev

# First, see what happens if we just throw aways rings above thtmax + dtht without any apodization:
ntheta = ducc0.fft.good_size(ffi_ducc.lmax_dlm + 2 + band_limit_increase)
tht = np.linspace(0., np.pi, ntheta)
ti = time.time()
tlm_len2 = np.atleast_2d(ffi_jc.gclm2lenmap(tlm, None, spin, False, _dfs_ringweights=bump(tht) > 0.,
                                            ntheta=ntheta, _dfs_scale=1, _forcefancydfs=True, polrot=polrot,
                                            ptg=ptg))
tf = time.time()
meanreldev, maxreldev = get_reldev(tlm_ref, tlm_len2)
pl.semilogy(thts_trunc / np.pi * 180, maxreldev,
            label=r'no apodization at all')

for band_limit_increase in [0, 100, 500]:
    for scale in  [3]:
        # magnifying the theta-range to reduce the number of theta-points
        #maybe can just do this changing the theta periodicity

        # forcing number of theta points in DFS map
        ntheta = ducc0.fft.good_size(ffi_ducc.lmax_dlm + 2 + band_limit_increase)
        # fraction of interval with non-zero points:
        ffi_jc.tim = timer(False)
        tht = np.linspace(0., np.pi, ntheta)
        ffi_jc.verbosity = verbose
        ti = time.time()
        tlm_len2 = np.atleast_2d(ffi_jc.gclm2lenmap(tlm, None, spin, False, _dfs_ringweights=bump(tht),
                        ntheta=ntheta, _dfs_scale=scale, _forcefancydfs=True, polrot=polrot, ptg=ptg))
        tf = time.time()
        print(ffi_jc.tim)
        # collect deviations ring per ring
        meanreldev, maxreldev = get_reldev(tlm_ref, tlm_len2)
        print("dN_theta %s, scale %s, %.2f sec" % (band_limit_increase, scale, tf - ti))
        print('Gain compared to jc baseline %.2f' % (baseline_jc_time / (tf-ti)))
        print('Gain compared to ducc baseline %.2f' % (baseline_ducc_time / (tf-ti)))
        #ln = pl.semilogy( meanreldev, label=r'(mean error)%.2f sec, dntheta %s' % (tf - ti, band_limit_increase))
        pl.semilogy(thts_trunc / np.pi * 180, maxreldev, label=r' %.2f sec, $\Delta N_\theta$ %s. $\theta^{\rm DFS} / \theta = %s$' % (tf - ti, band_limit_increase, scale))

pl.title('lmax %s ducc-base %.2f sec. jc-base %.2f sec, target eps %.1e' % (
lmax + dlmax_gl, baseline_ducc_time, baseline_jc_time, eps))
pl.xlabel(r'$\theta$ [deg]')
pl.ylabel(r'dev to baseline')
pl.axvline(thtmax / np.pi * 180, c='k', ls='--', label='start of apo')
pl.legend()
#pl.savefig('../test_dfs.pdf', bbox_inches='tight')
pl.show()