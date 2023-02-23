#FIXME: always work ith alms arrays of shape [ncomp, nlm] ?
#FIXME: exact A
#TODO: spin-0 gradient conventions ?
#TODO: double-check delensing/adjoint for spin!= 0
from __future__ import annotations

import os

import numpy as np
from lenscarf.remapping import d2ang
from lenspyx.utils_hp import Alm, alm2cl,almxfl, alm_copy
from lenscarf import cachers
from lenspyx.utils import timer, blm_gauss
import healpy as hp
import ducc0
from lenspyx.remapping.utils_geom import Geom, pbdGeometry, pbounds

try:
    from lenspyx.fortran import remapping as fremap
    HAS_FORTRAN = True
except:
    print('deflection.py: could not load fortran module')
    HAS_FORTRAN = False

try:
    import numexpr
    HAS_NUMEXPR = True
except:
    HAS_NUMEXPR = False
    print("deflection.py::could not load numexpr, falling back on python impl.")


ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longfloat): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}
rtype = {np.dtype(np.complex64): np.float32,
         np.dtype(np.complex128): np.float64,
         np.dtype(np.longcomplex): np.longfloat,
         np.complex64: np.float32,
         np.complex128: np.float64,
         np.longcomplex: np.longfloat}

class deflection:
    def __init__(self, lens_geom:Geom, dglm, mmax_dlm:int or None, numthreads:int=0,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, epsilon=1e-5, verbosity=0):
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
            cacher = cachers.cacher_none()


        # std deviation of deflection:
        s2_d = np.sum(alm2cl(dglm, dglm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        if dclm is not None:
            s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        sig_d = np.sqrt(s2_d) / lens_geom.fsky()
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

        self.sht_tr = numthreads

        self.verbosity = verbosity
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result
        self.ofactor = 1.5  # upsampling grid factor

        print(" DUCC totalconvolve deflection instantiated", self.epsilon)

        self.single_prec = True # Uses single precision arithmetic in some places
        self.single_prec_ptg = False
        self.tim = timer(False, 'deflection instance timer')

        if HAS_NUMEXPR:
            os.environ['NUMEXPR_MAX_THREADS'] = str(numthreads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(numthreads)

        self._totalconvolves0 = False

    def _get_ptg(self):
        # TODO improve this and fwd angles
        return self._build_angles()[:, 0:2]  # -gamma in third argument

    def _get_mgamma(self):
        return self._build_angles()[:, 2]

    def _build_angles(self, fortran=True):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fn = 'ptg'
        if not self.cacher.is_cached(fn):
            self.tim.start('_build_angles')
            self.tim.reset()
            assert np.all(self.geom.theta > 0.) and np.all(self.geom.theta < np.pi), 'fix this (cotangent below)'
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            #red, imd = self.geom.synthesis_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
            red, imd = self.geom.synthesis([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr)
            # Probably want to keep red, imd double precision for the calc?
            npix = Geom.npix(self.geom)
            self.tim.add('d1 alm2map_spin')
            if fortran and HAS_FORTRAN and (np.abs(self.geom.fsky() - 1.) < 1e-5):
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                if self.single_prec_ptg:
                    thp_phip_mgamma = fremap.remapping.fpointing(red, imd, tht, phi0, nph, ofs)
                else:
                    thp_phip_mgamma = fremap.remapping.pointing(red, imd, tht, phi0, nph, ofs)
                self.tim.add('thts, phis and gammas  (fortran)')
                # I think this just trivially turns the F-array into a C-contiguous array:
                self.cacher.cache(fn, thp_phip_mgamma.transpose())
                self.tim.close('_build_angles')
                if self.verbosity:
                    print(self.tim)
                return thp_phip_mgamma.transpose()
            elif fortran and not HAS_FORTRAN:
                print('Cant use fortran pointing building since import failed. Falling back on python impl.')
            thp_phip_mgamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
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
                    thp_phip_mgamma[0, sli] = thtp_
                    thp_phip_mgamma[1, sli] = phip_
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_mgamma[2, sli] = -np.arctan2(i_imd, t_red ) + np.arctan2(i_imd, d * np.sin(d) * cot + t_red  * np.cos(d))
                    startpix += len(pixs)
            self.tim.add('thts, phis and gammas  (python)')
            thp_phip_mgamma = thp_phip_mgamma.transpose()
            self.cacher.cache(fn, thp_phip_mgamma)
            self.tim.close('_build_angles')
            assert startpix == npix, (startpix, npix)
            if self.verbosity:
                print(self.tim)
            return thp_phip_mgamma
        return self.cacher.load(fn)

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.geom, dlm[0], mmax_dlm, self.sht_tr, cacher, dlm[1],
                          verbosity=self.verbosity, epsilon=self.epsilon)

    def change_geom(self, lens_geom:Geom, cacher:cachers.cacher or None=None):
        """Returns a deflection instance with a different position-space geometry

                Args:
                    lens_geom: new geometry
                    cacher: cacher instance if desired


        """
        print("**** change_geom, DO YOU REALLY WANT THIS??")
        return deflection(lens_geom, self.dlm, self.mmax_dlm, self.sht_tr, cacher, self.dclm,
                          verbosity=self.verbosity, epsilon=self.epsilon)

    def gclm2lenpixs(self, gclm:np.ndarray, mmax:int or None, spin:int, pixs:np.ndarray[int], polrot=True):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation

            Note:
                The number of pixels must be small here, otherwise way too slow

            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so make take some time.

        """
        assert spin >= 0, spin
        gclm = np.atleast_2d(gclm)
        ptg = self._get_ptg()
        thts, phis, gamma = ptg[pixs, 0], ptg[pixs, 1], self._get_mgamma()[pixs] * (-1.)
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        geom = Geom(thts.copy(), phis.copy(), nph, ofs, wt)
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.synthesis(gclm, spin, lmax, mmax, self.sht_tr)[:, 0::2]
        if spin * polrot:
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()

    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
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
                ptg = self._get_ptg()
            self.tim.add('ptg')
            # FIXME: this might only accept doubple prec input
            inter_I = ducc0.totalconvolve.Interpolator(gclm, blm_T, separate=False, lmax=lmax_unl,
                                                       kmax=0,
                                                       epsilon=self.epsilon, ofactor=self.ofactor,
                                                       nthreads=self.sht_tr, mmax=mmax)
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
        map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi,
                                spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr)
        self.tim.add('experimental.synthesis_2d')

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

        # perform NUFFT
        if ptg is None:
            ptg = self._get_ptg()
        self.tim.add('get ptg')
        # perform NUFFT
        values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg, forward=False,
                                  epsilon=self.epsilon, nthreads=self.sht_tr,
                                  verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
        self.tim.add('u2nu')

        if spin * polrot: #TODO: at some point get rid of these exp(arctan...)
                          # maybe simplest to cache cis g and multpily in place a couple of times
            if HAS_NUMEXPR:
                mg = self._get_mgamma()
                js = - 1j * spin
                values *= numexpr.evaluate("exp(js * mg)")
                self.tim.add('polrot (numexpr)')
            else:
                values *= np.exp((-1j * spin) * self._get_mgamma())  # polrot. last entry is -gamma
                self.tim.add('polrot (python)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        return values.real if spin == 0 else (values.real, values.imag)

    def lenmap2gclm(self, points:np.ndarray[complex], spin:int, lmax:int, mmax:int):
        """
            Note:
                points mst be already quadrature-weigthed, and be complex

            Note:
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.

        """
        self.tim.start('lenmap2gclm')
        self.tim.reset()
        assert points.ndim == 1 and np.iscomplexobj(points), (points.ndim, points.dtype)

        ptg = self._get_ptg()
        self.tim.add('_get_ptg')

        ntheta = ducc0.fft.good_size(lmax + 2)
        nphihalf = ducc0.fft.good_size(lmax + 1)
        nphi = 2 * nphihalf
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=points.dtype)

        # perform NUFFT
        map_dfs = ducc0.nufft.nu2u(points=points, coord=ptg, out=map_dfs, forward=True,
                                   epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                   periodicity=2 * np.pi, fft_order=True)
        self.tim.add('map_dfs')
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
                                                          lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr)
        self.tim.add('map2alm_spin')
        self.tim.close('lenmap2gclm')
        if self.verbosity:
            print(self.tim)
        return slm.squeeze()

    def lensgclm(self, gclm:np.ndarray, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, nomagn=False, polrot=True):
        """Adjoint remapping operation from lensed alm space to unlensed alm space


            #FIXME: nomagn=True is a backward comptability thing to ask for inverse lensing
            #        but in this implementation it actually puts a magn...

        """
        stri = 'lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards)
        self.tim.start(stri)
        self.tim.reset()
        if nomagn:
            assert backwards
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.geom.fsky() - 1.) < 1e-6: # no actual deflection and single-precision full-sky
            if spin == 0:
                ret = alm_copy(gclm, mmax, lmax_out, mmax_out)
                self.tim.close(stri)
                return ret
            glmret = alm_copy(gclm[0], mmax, lmax_out, mmax_out)
            ret = np.array([glmret, alm_copy(gclm[1], mmax, lmax_out, mmax_out) if gclm[1] is not None else np.zeros_like(glmret)])
            self.tim.close(stri)
            return ret
        if not backwards:
            # FIXME: should return here 2d array?
            m = self.gclm2lenmap(gclm, mmax, spin, backwards)
            self.tim.reset()
            if spin == 0:
                ret = self.geom.adjoint_synthesis(m, spin, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm')
                self.tim.close('lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards))
                return ret.squeeze()
            else:
                assert polrot
                ret = self.geom.adjoint_synthesis(m, spin, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm_spin')
                self.tim.close('lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards))
                return ret
        else:
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
                self.tim.add('type conversion')
            if spin == 0 and self._totalconvolves0:
                # The code below works for any spin but this seems a little bit faster for non-zero spin
                # So keeping this for the moment
                lmax_unl = hp.Alm.getlmax(gclm[0].size if abs(spin) > 0 else gclm.size, mmax)
                inter = ducc0.totalconvolve.Interpolator(lmax_out, spin, 1, epsilon=self.epsilon,
                                                         ofactor=self.ofactor, nthreads=self.sht_tr)
                I = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr)
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
                # make complex if necessary
                lmax_unl = hp.Alm.getlmax(gclm.size, mmax)
                points = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr).squeeze()
                self.tim.add('points')
                if nomagn:
                    points *= self.dlm2A()
                    self.tim.add('nomagn')
            else:
                assert gclm.ndim == 2, gclm.ndim
                lmax_unl = hp.Alm.getlmax(gclm[0].size, mmax)
                if mmax is None:
                    mmax = lmax_unl
                points = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr)
                self.tim.add('points')
                if nomagn:
                    points *= self.dlm2A()
                    self.tim.add('nomagn')
                if polrot * spin:#TODO: at some point get rid of these exp(atan2)...
                          # maybe simplest to save cis gamma and twice multiply in place...
                    if HAS_NUMEXPR:
                        re, im = points
                        mg = self._get_mgamma()
                        js = + 1j * spin
                        points = numexpr.evaluate("(re + 1j * im) * exp(js * mg)")
                        self.tim.add('polrot (numexpr)')
                    else:
                        points = (points[0] + 1j * points[1]) * np.exp((1j * spin) * self._get_mgamma())
                        self.tim.add('polrot (python)')
                else:
                    points = points[0] + 1j * points[1]

            assert points.ndim == 1
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[ofs:ofs + nph] *= w
            self.tim.add('weighting')
            # make complex if necessary
            points2 = points.astype(ctype[points.dtype]) if spin == 0 else points
            slm = self.lenmap2gclm(points2, spin, lmax_out, mmax_out)
            self.tim.close(stri)
            if self.verbosity:
                print(self.tim)
            return slm

    def dlm2A(self):
        """Returns determinant of magnification matrix corresponding to input deflection field

            Returns:
                determinant of magnification matrix. Array of size input pixelization geometry

        #FIXME: get exact calc and test it on rot
        """
        self.tim.start('dlm2A')
        geom, lmax, mmax, tr = self.geom, self.lmax_dlm, self.mmax_dlm, self.sht_tr
        dgclm = np.empty((2, self.dlm.size), dtype=self.dlm.dtype)
        dgclm[0] = self.dlm
        dgclm[1] = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
        d2k = -0.5 * get_spin_lower(1, self.lmax_dlm)  # For k = 12 \eth^{-1} d, g = 1/2\eth 1d
        d2g = -0.5 * get_spin_raise(1, self.lmax_dlm)
        glms = np.empty((2, self.dlm.size), dtype=self.dlm.dtype) # Shear
        glms[0] = almxfl(dgclm[0], d2g, self.mmax_dlm, False)
        glms[1] = almxfl(dgclm[1], d2g, self.mmax_dlm, False)
        klm = almxfl(dgclm[0], d2k, mmax, False)
        k = geom.synthesis(klm, 0, lmax, mmax, tr)
        g1, g2 = geom.synthesis(glms, 2, lmax, mmax, tr)
        d1, d2 = geom.synthesis(dgclm, 1, lmax, mmax, tr)
        if np.any(dgclm[1]):
            w = geom.synthesis(almxfl(dgclm[1], d2k, mmax, False), 0, lmax, mmax, tr)
        else:
            w = 0.
        del dgclm, glms, klm
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
        if HAS_NUMEXPR:
            A = numexpr.evaluate('f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)')
            A+= numexpr.evaluate('f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))')
        else:
            A  = f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)
            A += f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))
            #                 -      (   cos 2b * g1 + sin 2b * g2 )
        self.tim.close('dlm2A')
        return A

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