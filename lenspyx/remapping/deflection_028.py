from __future__ import annotations

import os
import numpy as np
import ducc0

from lenspyx.remapping.utils_angles import d2ang
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx import cachers
from lenspyx.utils import timer, blm_gauss
from lenspyx.remapping.utils_geom import Geom, pbdGeometry, pbounds
from multiprocessing import cpu_count
try:
    from lenspyx.fortran.remapping import remapping as fremap
    HAS_FORTRAN = True
except:
    HAS_FORTRAN = False

HAS_DUCCPOINTING = 'get_deflected_angles' in ducc0.misc.__dict__
HAS_DUCCROTATE = 'lensing_rotate' in ducc0.misc.__dict__
HAS_DUCCGRADONLY = 'mode:' in ducc0.sht.experimental.synthesis.__doc__

if HAS_DUCCPOINTING:
    from ducc0.misc import get_deflected_angles
if HAS_DUCCROTATE:
    from ducc0.misc import lensing_rotate

if not HAS_DUCCGRADONLY or not HAS_DUCCROTATE:
    print("You might need to update ducc0 to latest version")


ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longdouble): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longdouble: np.longcomplex}
rtype = {np.dtype(np.complex64): np.float32,
         np.dtype(np.complex128): np.float64,
         np.dtype(np.longcomplex): np.longdouble,
         np.complex64: np.float32,
         np.complex128: np.float64,
         np.longcomplex: np.longdouble}

def ducc_sht_mode(gclm, spin):

    gclm_ = np.atleast_2d(gclm)
    return 'GRAD_ONLY' if ((gclm_[0].size == gclm_.size) * (abs(spin) > 0)) else 'STANDARD'

class deflection:
    def __init__(self, lens_geom:Geom, dglm, mmax_dlm:int or None, numthreads:int=0,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None,
                 epsilon=1e-5, verbosity=0, single_prec=True, planned=False):
        """Deflection field object than can be used to lens several maps with forward or backward deflection

            Args:
                lens_geom: scarf.Geometry object holding info on the deflection operation pixelization
                dglm: deflection-field alm array, gradient mode (:math:`\sqrt{L(L+1)}\phi_{LM}` e.g.)
                numthreads: number of threads for the SHTs scarf-ducc based calculations (uses all available by default)
                cacher: cachers.cacher instance allowing if desired caching of several pieces of info;
                        Useless if only one maps is intended to be deflected, but useful if more.
                dclm: deflection-field alm array, curl mode (if relevant)
                mmax_dlm: maximal m of the dlm / dclm arrays, if different from lmax
                epsilon: desired accuracy on remapping


        """
        lmax = Alm.getlmax(dglm.size, mmax_dlm)
        if mmax_dlm is None:
            mmax_dlm = lmax
        if cacher is None:
            cacher = cachers.cacher_mem(safe=False)
        if numthreads <= 0:
            numthreads = cpu_count()

        # std deviation of deflection:
        s2_d = np.sum(alm2cl(dglm, dglm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        if dclm is not None:
            s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
            s2_d /= np.sqrt(2.)
        sig_d = np.sqrt(s2_d / lens_geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
        self.sig_d = sig_d
        self.dlm = dglm
        self.dclm = dclm

        self.lmax_dlm = lmax
        self.mmax_dlm = mmax_dlm


        self.cacher = cacher
        self.geom = lens_geom
        self.pbgeom = pbdGeometry(lens_geom, pbounds(0., 2 * np.pi))

        if verbosity:
            print("deflection: I set numthreads to " + str(numthreads))
        self.sht_tr = numthreads
        self.verbosity = verbosity
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result


        self.single_prec = single_prec * (epsilon > 1e-6) # Uses single precision arithmetic in some places
        self.single_prec_ptg = False
        self.tim = timer(False, 'deflection instance timer')

        self.planned = planned
        self.plans = {}

        if verbosity:
            print(" DUCC %s threads deflection instantiated"%self.sht_tr + self.single_prec * '(single prec)', self.epsilon)
        self._totalconvolves0 = False
        self.ofactor = 1.5  # upsampling grid factor (only used if _totalconvolves is set)

        self._cis = False

    def _get_ptg(self):
        # TODO improve this and fwd angles, e.g. this is computed twice for gamma if no cacher
        self._build_angles() if not self._cis else self._build_angleseig()
        return self.cacher.load('ptg')

    def _get_gamma(self):
        self._build_angles() if not self._cis else self._build_angleseig()
        return self.cacher.load('gamma')

    def _build_d1(self):
        if self.dclm is None:
            # undo p2d to use
            self.tim.reset()
            d1 = self.geom.synthesis(self.dlm, 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, mode='GRAD_ONLY')
            self.tim.add('build angles <- synthesis (GRAD_ONLY)')
        else:
            # FIXME: want to do that only once
            self.tim.reset()
            dgclm = np.empty((2, self.dlm.size), dtype=self.dlm.dtype)
            dgclm[0] = self.dlm
            dgclm[1] = self.dclm
            d1 = self.geom.synthesis(dgclm, 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr)
            self.tim.add('build angles <- synthesis (STANDARD)')
        return d1

    def _build_angles(self, fortran=True, calc_rotation=True):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]) :
            self.tim.start('build_angles')
            d1 = self._build_d1()
            # Probably want to keep red, imd double precision for the calc?
            if HAS_DUCCPOINTING:
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                      calc_rotation=calc_rotation, nthreads=self.sht_tr)
                self.tim.add('build angles <- th-phi%s (ducc)'%('-gm'*calc_rotation))
                if calc_rotation:
                    self.cacher.cache(fns[0], tht_phip_gamma[:, 0:2])
                    self.cacher.cache(fns[1], tht_phip_gamma[:, 2] if not self.single_prec else tht_phip_gamma[:, 2].astype(np.float32))
                else:
                    self.cacher.cache(fns[0], tht_phip_gamma)
                self.tim.close('build_angles')
                return
            npix = Geom.npix(self.geom)
            if fortran and HAS_FORTRAN:
                red, imd = d1
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                if self.single_prec_ptg:
                    thp_phip_gamma = fremap.fpointing(red, imd, tht, phi0, nph, ofs, self.sht_tr)
                else:
                    thp_phip_gamma = fremap.pointing(red, imd, tht, phi0, nph, ofs, self.sht_tr)
                self.tim.add('build angles <- th-phi-gm (ftn)')
                # I think this just trivially turns the F-array into a C-contiguous array:
                self.cacher.cache(fns[0], thp_phip_gamma.transpose()[:, 0:2])
                if calc_rotation:
                    self.cacher.cache(fns[1], thp_phip_gamma.transpose()[:, 2] if not self.single_prec else thp_phip_gamma.transpose()[:, 2].astype(np.float32))
                self.tim.close('build_angles')
                if self.verbosity:
                    print(self.tim)
                return
            elif fortran and not HAS_FORTRAN:
                print('Cant use fortran pointing building since import failed. Falling back on python impl.')
            thp_phip_gamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
            assert np.all(self.geom.theta > 0.) and np.all(self.geom.theta < np.pi), 'fix this (cotangent below)'
            red, imd = d1
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.rings2pix(self.geom, [ir])
                if pixs.size > 0:
                    t_red = red[pixs]
                    i_imd = imd[pixs]
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.theta[ir] * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(t_red, i_imd, thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip_gamma[0, sli] = thtp_
                    thp_phip_gamma[1, sli] = phip_
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_gamma[2, sli] = np.arctan2(i_imd, t_red ) - np.arctan2(i_imd, d * np.sin(d) * cot + t_red * np.cos(d))
                    startpix += len(pixs)
            self.tim.add('thts, phis and gammas  (python)')
            self.cacher.cache(fns[0], thp_phip_gamma.T[:, 0:2])
            if calc_rotation:
                self.cacher.cache(fns[1], thp_phip_gamma.T[:, 2] if not self.single_prec else thp_phip_gamma.T[:, 2].astype(np.float32) )
            self.tim.close('build_angles')
            assert startpix == npix, (startpix, npix)
            if self.verbosity:
                print(self.tim)
            return

    def _build_angleseig(self):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fn_ptg, fn_cischi = 'ptg', 'cischi'
        if not self.cacher.is_cached(fn_ptg) or not self.cacher.is_cached(fn_cischi):
            self.tim.start('build_angles')
            self.tim.reset()
            red, imd = self._build_d1()
            # Probably want to keep red, imd double precision for the calc?
            if HAS_FORTRAN:
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                thp_phip_cischi = fremap.pointingeig(red, imd, tht, phi0, nph, ofs, self.sht_tr)
                self.tim.add('build angles <- th-phi-cischi (ftn)')
                # I think this just trivially turns the F-array into a C-contiguous array:
                self.cacher.cache(fn_ptg, thp_phip_cischi.transpose()[:, 0:2])
                self.cacher.cache(fn_cischi,thp_phip_cischi[2] + 1j * thp_phip_cischi[3])
                self.tim.close('build_angles')
                if self.verbosity:
                    print(self.tim)
            else:
                assert 0

    def make_plan(self, lmax, spin):
        """Builds nuFFT plan for slightly faster transforms

            Useful if many remapping operations will be done from the same deflection field

        """
        if lmax not in self.plans:
            print("(NB: plan independent of spin)")
            self.tim.start('planning %s'%lmax)
            ntheta = ducc0.fft.good_size(lmax + 2)
            nphihalf = ducc0.fft.good_size(lmax + 1)
            nphi = 2 * nphihalf
            ptg = self._get_ptg()
            plan = ducc0.nufft.plan(nu2u=False, coord=ptg, grid_shape=(2 * ntheta - 2, nphi), epsilon=self.epsilon,
                                        nthreads=self.sht_tr, periodicity=2 * np.pi, fft_order=True)
            self.plans[lmax] = plan
            self.tim.close('planning %s'%lmax)
        return self.plans[lmax]

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.geom, dlm[0], mmax_dlm, numthreads=self.sht_tr, cacher=cacher, dclm=dlm[1],
                          verbosity=self.verbosity, epsilon=self.epsilon, single_prec=self.single_prec)

    def change_geom(self, lens_geom:Geom, cacher:cachers.cacher or None=None):
        """Returns a deflection instance with a different position-space geometry

                Args:
                    lens_geom: new geometry
                    cacher: cacher instance if desired


        """
        print("**** change_geom, DO YOU REALLY WANT THIS??")
        return deflection(lens_geom, self.dlm, self.mmax_dlm, self.sht_tr, cacher, self.dclm,
                          verbosity=self.verbosity, epsilon=self.epsilon, planned=self.planned)

    def gclm2lenpixs(self, gclm:np.ndarray, mmax:int or None, spin:int, pixs:np.ndarray[int], polrot=True):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation

            Note:
                The number of pixels must be small here, otherwise way too slow

            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so may take some time.

        """
        assert spin >= 0, spin
        gclm = np.atleast_2d(gclm)
        sth_mode = ducc_sht_mode(gclm, spin)
        ptg = self._get_ptg()
        thts, phis = ptg[pixs, 0], ptg[pixs, 1]
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        geom = Geom(thts.copy(), phis.copy(), nph, ofs, wt)
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.synthesis(gclm, spin, lmax, mmax, self.sht_tr, mode=sth_mode)[:, 0::2]
        # could do: complex view trick etc
        if spin and polrot:
            gamma = self._get_gamma()[pixs]
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()

    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
        """Produces deflected spin-weighted map from alm array and instance pointing information

            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation


        """
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        self.tim.start('gclm2lenmap')
        self.tim.reset()
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
            self.tim.add('type conversion')
        if spin == 0 and self._totalconvolves0: # this will probably just disappear
            # The code below would work just as well for spin-0 but seems slightly slower
            # For the moment this seems faster
            blm_T = blm_gauss(0, lmax_unl, 0)
            self.tim.add('blm_gauss')
            if ptg is None:
                ptg = self._build_angles()
            self.tim.add('ptg')
            assert mmax == lmax_unl
            # FIXME: this might only accept doubple prec input
            inter_I = ducc0.totalconvolve.Interpolator(gclm, blm_T, separate=False, lmax=lmax_unl,
                                                       kmax=0,
                                                       epsilon=self.epsilon, ofactor=self.ofactor,
                                                       nthreads=self.sht_tr)
            self.tim.add('interp. setup')
            ret = inter_I.interpol(ptg).squeeze()
            self.tim.add('interpolation')
            self.tim.close('gclm2lenmap')
            return ret
        # transform slm to Clenshaw-Curtis map
        ntheta = ducc0.fft.good_size(lmax_unl + 2)
        nphihalf = ducc0.fft.good_size(lmax_unl + 1)
        nphi = 2 * nphihalf
        # Is this any different to scarf wraps ?
        # NB: type of map, map_df, and FFTs will follow that of input gclm
        mode = ducc_sht_mode(gclm, spin)
        map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi,
                                spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=mode)
        self.tim.add('experimental.synthesis_2d (%s)'%mode)
        # extend map to double Fourier sphere map
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype if spin == 0 else ctype[map.dtype])
        if spin == 0:
            map_dfs[:ntheta, :] = map[0]
        else:
            map_dfs[:ntheta, :].real = map[0]
            map_dfs[:ntheta, :].imag = map[1]
        del map

        map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
        map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
        if (spin % 2) != 0:
            map_dfs[ntheta:, :] *= -1
        self.tim.add('map_dfs build')

        # go to Fourier space
        if spin == 0:
            tmp = np.empty(map_dfs.shape, dtype=ctype[map_dfs.dtype])
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp)
            del tmp
        else:
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=map_dfs)
        self.tim.add('map_dfs 2DFFT')

        if self.planned: # planned nufft
            assert ptg is None
            plan = self.make_plan(lmax_unl, spin)
            values = plan.u2nu(grid=map_dfs, forward=False, verbosity=self.verbosity)
            self.tim.add('planned u2nu')
        else:
            # perform NUFFT
            if ptg is None:
                ptg = self._get_ptg()
            self.tim.add('get ptg')
            # perform NUFFT
            values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg, forward=False,
                                      epsilon=self.epsilon, nthreads=self.sht_tr,
                                      verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
            self.tim.add('u2nu')

        if polrot * spin:
            if self._cis:
                cis = self._get_cischi()
                for i in range(polrot * abs(spin)):
                    values *= cis
                self.tim.add('polrot (cis)')
            else:
                if HAS_DUCCROTATE:
                    lensing_rotate(values, self._get_gamma(), spin, self.sht_tr)
                    self.tim.add('polrot (ducc)')
                else:
                    func = fremap.apply_inplace if values.dtype == np.complex128 else fremap.apply_inplacef
                    func(values, self._get_gamma(), spin, self.sht_tr)
                    self.tim.add('polrot (fortran)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        # Return real array of shape (2, npix) for spin > 0
        return values.real if spin == 0 else values.view(rtype[values.dtype]).reshape((values.size, 2)).T

    def lenmap2gclm(self, points:np.ndarray[complex or float], spin:int, lmax:int, mmax:int, gclm_out=None,
                    sht_mode='STANDARD'):
        """
            Note:
                points mst be already quadrature-weigthed

            Note:
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.

        """
        self.tim.start('lenmap2gclm')
        self.tim.reset()
        if spin == 0 and not np.iscomplexobj(points):
            points = points.astype(ctype[points.dtype]).squeeze()
        if spin > 0 and not np.iscomplexobj(points):
            points = (points[0] + 1j * points[1]).squeeze()
        ptg = self._get_ptg()
        self.tim.add('_get_ptg')

        ntheta = ducc0.fft.good_size(lmax + 2)
        nphihalf = ducc0.fft.good_size(lmax + 1)
        nphi = 2 * nphihalf
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=points.dtype)
        if self.planned:
            plan = self.make_plan(lmax, spin)
            map_dfs = plan.nu2u(points=points, out=map_dfs, forward=True, verbosity=self.verbosity)
            self.tim.add('planned nu2u')

        else:
            # perform NUFFT
            map_dfs = ducc0.nufft.nu2u(points=points, coord=ptg, out=map_dfs, forward=True,
                                       epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                       periodicity=2 * np.pi, fft_order=True)
            self.tim.add('nu2u')
        # go to position space
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=self.sht_tr, out=map_dfs)
        self.tim.add('c2c FFT')

        # go from double Fourier sphere to Clenshaw-Curtis grid
        if (spin % 2) != 0:
            map_dfs[1:ntheta - 1, :nphihalf] -= map_dfs[-1:ntheta - 1:-1, nphihalf:]
            map_dfs[1:ntheta - 1, nphihalf:] -= map_dfs[-1:ntheta - 1:-1, :nphihalf]
        else:
            map_dfs[1:ntheta - 1, :nphihalf] += map_dfs[-1:ntheta - 1:-1, nphihalf:]
            map_dfs[1:ntheta - 1, nphihalf:] += map_dfs[-1:ntheta - 1:-1, :nphihalf]
        map_dfs = map_dfs[:ntheta, :]
        map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=rtype[points.dtype])
        map[0] = map_dfs.real
        if spin > 0:
            map[1] = map_dfs.imag
        del map_dfs
        self.tim.add('Double Fourier')

        # adjoint SHT synthesis
        slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin,
                            lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=sht_mode, alm=gclm_out)
        self.tim.add('adjoint_synthesis_2d (%s)'%sht_mode)
        self.tim.close('lenmap2gclm')
        return slm.squeeze()

    def lensgclm(self, gclm:np.ndarray, mmax:int or None, spin:int, lmax_out:int, mmax_out:int or None,
                 gclm_out:np.ndarray=None, backwards=False, nomagn=False, polrot=True, out_sht_mode='STANDARD'):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

            Args:
                gclm: input gradient and possibly curl mode ((1 or 2, nalm)-shaped complex numpy.ndarray)
                mmax: set this for non-standard mmax != lmax in input array
                spin: spin-weight of the fields (larger or equal 0)
                lmax_out: desired output array lmax
                mmax_out: desired output array mmax (defaults to lmax_out if None)
                gclm_out(optional): output array (can be same as gclm provided it is large enough)
                backwards: forward or adjoint (not the same as inverse) lensing operation
                polrot(optional): includes small rotation of spin-weighted fields (defaults to True)
                out_sht_mode(optional): e.g. 'GRAD_ONLY' if only the output gradient mode is desired


            Note:
                 nomagn=True is a backward comptability thing to ask for inverse lensing


        """
        stri = 'lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards)
        self.tim.start(stri)
        self.tim.reset()
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if nomagn:
            assert backwards
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.geom.fsky() - 1.) < 1e-6:
            # no actual deflection and single-precision full-sky
            ncomp_out = 1 + (spin != 0) * (out_sht_mode == 'STANDARD')
            if gclm_out is None:
                gclm_out = np.empty((ncomp_out, Alm.getsize(lmax_out, mmax_out)),  dtype=gclm.dtype)
            assert gclm_out.ndim == 2 and gclm_out.shape[0] == ncomp_out, (gclm_out.shape, ncomp_out)
            gclm_2d = np.atleast_2d(gclm)
            gclm_out[0] = alm_copy(gclm_2d[0], mmax, lmax_out, mmax_out)
            if ncomp_out > 1:
                gclm_out[1] = 0. if input_sht_mode == 'GRAD_ONLY' else alm_copy(gclm_2d[1], mmax, lmax_out, mmax_out)
            self.tim.close(stri)
            return gclm_out.squeeze()
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards, polrot=polrot)
            self.tim.reset()
            if gclm_out is not None:
                assert gclm_out.dtype == ctype[m.dtype], 'type precision must match'
            gclm_out = self.geom.adjoint_synthesis(m, spin, lmax_out, mmax_out, self.sht_tr, alm=gclm_out,
                                                   mode=out_sht_mode)
            self.tim.add('adjoint_synthesis')
            self.tim.close('lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards))
            return gclm_out.squeeze()
        else:
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
                self.tim.add('type conversion')
            if spin == 0 and self._totalconvolves0:
                assert out_sht_mode == 'STANDARD', 'cant handle this here'
                # The code below works for any spin but this seems a little bit faster for non-zero spin
                # So keeping this for the moment
                lmax_unl = Alm.getlmax(gclm[0].size if abs(spin) > 0 else gclm.size, mmax)
                inter = ducc0.totalconvolve.Interpolator(lmax_out, spin, 1, epsilon=self.epsilon,
                                                         ofactor=self.ofactor, nthreads=self.sht_tr)
                I = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr, mode=input_sht_mode)
                for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                    I[ofs:ofs + nph] *= w
                self.tim.add('points')
                xptg = self._get_ptg()
                self.tim.add('_get_ptg')
                inter.deinterpol(xptg, np.atleast_2d(I))
                self.tim.add('deinterpol')
                blm = blm_gauss(0, lmax_out, spin)
                ret = inter.getSlm(blm).squeeze()
                self.tim.add('getSlm')
                self.tim.close('lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards))
                return ret
            if spin == 0:
                lmax_unl = Alm.getlmax(gclm.size, mmax)
                points = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr, mode=input_sht_mode)
                self.tim.add('points synthesis (%s)'%input_sht_mode)
                if nomagn:
                    points *= self.dlm2A()
                    self.tim.add('nomagn')
            else:
                assert gclm.ndim == 2, gclm.ndim
                lmax_unl = Alm.getlmax(gclm[0].size, mmax)
                if mmax is None:
                    mmax = lmax_unl
                pointsc = np.empty((self.geom.npix(),), dtype=np.complex64 if self.single_prec else np.complex128)
                points = pointsc.view(rtype[pointsc.dtype]).reshape((pointsc.size, 2)).T  # real view onto complex array
                self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr, map=points, mode=input_sht_mode)
                self.tim.add('points synthesis (%s)'%input_sht_mode)
                if nomagn:
                    points *= self.dlm2A()
                    self.tim.add('nomagn')
                if spin and polrot:
                    if HAS_DUCCROTATE:
                        lensing_rotate(pointsc, self._get_gamma(), -spin, self.sht_tr)
                        self.tim.add('polrot (ducc)')
                    elif HAS_FORTRAN:
                        func = fremap.apply_inplace if pointsc.dtype == np.complex128 else fremap.apply_inplacef
                        func(pointsc, self._get_gamma(), -spin, self.sht_tr)
                        self.tim.add('polrot (fortran)')
                    else:
                        pointsc *= np.exp((-1j * spin) * self._get_gamma())
                        self.tim.add('polrot (python)')

            assert points.ndim == 2 and not np.iscomplexobj(points)
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[:, ofs:ofs + nph] *= w
            self.tim.add('weighting')
            slm = self.lenmap2gclm(points, spin, lmax_out, mmax_out, sht_mode=out_sht_mode, gclm_out=gclm_out)
            self.tim.close(stri)
            if self.verbosity:
                print(self.tim)
            return slm

    def dlm2A(self):
        """Returns determinant of magnification matrix corresponding to input deflection field

            Returns:
                determinant of magnification matrix. Array of size input pixelization geometry

        """
        #FIXME splits in band with new offsets
        self.tim.start('dlm2A')
        geom, lmax, mmax, tr = self.geom, self.lmax_dlm, self.mmax_dlm, self.sht_tr
        dgclm = np.empty((2, self.dlm.size), dtype=self.dlm.dtype)
        dgclm[0] = self.dlm
        dgclm[1] = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
        d2k = -0.5 * get_spin_lower(1, self.lmax_dlm)  # For k = 12 \eth^{-1} d, g = 1/2\eth 1d
        d2g = -0.5 * get_spin_raise(1, self.lmax_dlm) #TODO: check the sign of this one
        glms = np.empty((2, self.dlm.size), dtype=self.dlm.dtype) # Shear
        glms[0] = almxfl(dgclm[0], d2g, self.mmax_dlm, False)
        glms[1] = almxfl(dgclm[1], d2g, self.mmax_dlm, False)
        klm = almxfl(dgclm[0], d2k, mmax, False)
        k = geom.synthesis(klm, 0, lmax, mmax, tr)
        g1, g2 = geom.synthesis(glms, 2, lmax, mmax, tr)
        d1, d2 = geom.synthesis(dgclm, 1, lmax, mmax, tr)
        if np.any(dgclm[1]):
            wlm = almxfl(dgclm[1], d2k, mmax, False)
            w = geom.synthesis(wlm, 0, lmax, mmax, tr)
        else:
            wlm, w = 0., 0.
        del dgclm, glms, klm, wlm
        d = np.sqrt(d1 * d1 + d2 * d2)
        max_d = np.max(d)
        if max_d > 0:
            f0 = np.sin(d) / d
            di = d
        else:
            from scipy.special import spherical_jn as jn
            f0 = jn(0, d)
            di = np.where(d > 0, d, 1.) # Something I can take the inverse of
        f1 = np.cos(d) - f0
        try: #FIXME
            import numexpr
            HAS_NUMEXPR = True
        except:
            HAS_NUMEXPR = False
        if HAS_NUMEXPR:
            A = numexpr.evaluate('f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)')
            A+= numexpr.evaluate('f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))')
        else:
            A  = f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)
            A += f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))
            #                 -      (   cos 2b * g1 + sin 2b * g2 )
        self.tim.close('dlm2A')
        return A.squeeze()

    def get_eigamma(self):
        red, imd = self._build_d1()
        eig = np.empty(self.geom.npix(), dtype=complex)
        self.geom.sort(self.geom.ofs)
        for ir, (nph, phi0, tht, of) in enumerate(zip(self.geom.nph, self.geom.phi0, self.geom.theta, self.geom.ofs)):
            sint = np.sin(tht)
            cost = np.cos(tht)
            sli = slice(of, of + nph)
            d = np.sqrt(red[sli] ** 2 + imd[sli] ** 2)
            sind = np.sin(d)
            cosd = np.cos(d)
            # FIXME
            eig[sli] = sint + (sint * red[sli] / d * (cosd - 1.) + cost * sind) * (red[sli] + 1j * imd[sli]) / d

        eig /= np.abs(eig)
        return eig

def get_spin_raise(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret
