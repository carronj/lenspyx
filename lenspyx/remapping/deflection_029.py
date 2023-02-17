from __future__ import annotations

import os

import numpy as np
from lenscarf.utils_hp import Alm, alm2cl, alm_copy
from lenscarf import cachers
from lenspyx.utils import timer,blm_gauss
from lenspyx.remapping.utils_geom import Geom
import ducc0
from ducc0.sht.experimental import synthesis, adjoint_synthesis, synthesis_general, adjoint_synthesis_general

try:
    from lenscarf.fortran import remapping as fremap
    HAS_FORTRAN = True
except:
    HAS_FORTRAN = False

try:
    import numexpr
    HAS_NUMEXPR = True
except:
    HAS_NUMEXPR = False
    print("deflection.py::could not load numexpr, falling back on python impl.")

# some helper functions

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
    def __init__(self, lens_geom:Geom, dglm:np.ndarray, mmax_dlm:int or None, numthreads:int=0,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, epsilon=1e-5, ofactor=1.5,verbosity=0):
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
                ofactor: upsampling parameter



        """
        lmax = Alm.getlmax(dglm.size, mmax_dlm)
        if mmax_dlm is None: mmax_dlm = lmax
        if cacher is None: cacher = cachers.cacher_none()


        # std deviation of deflection:
        s2_d = np.sum(alm2cl(dglm, dglm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        if dclm is not None:
            s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        sig_d = np.sqrt(s2_d)
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%(sig_d/np.pi * 180 * 60))
        self.sig_d = sig_d
        self.dlm = dglm
        self.dclm = dclm

        self.lmax_dlm = lmax
        self.mmax_dlm = mmax_dlm

        self.cacher = cacher
        self.geom = lens_geom

        # SHT's pand NUFFT parameters
        self.sht_tr = numthreads
        self.verbosity = verbosity
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result
        self.ofactor = ofactor  # upsampling grid factor

        print(" DUCC 029 deflection instantiated", self.epsilon, self.ofactor)

        self.single_prec = True # Uses single precision arithmetic in some places
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
            #red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
            red, imd = self.geom.alm2map(np.array([self.dlm, dclm]), 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr)
            # Probably want to keep red, imd double precision for the calc?
            self.tim.add('d1 alm2map_spin')
            if fortran and HAS_FORTRAN: # covering full phi range
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                if self.single_prec:
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
                assert 0 ,'Cant use fortran pointing building since import failed. Falling back on python impl.'
            else:
                assert 0, 'No python implementation available, get your fortran package to work'
        return self.cacher.load(fn)

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.geom, dlm[0], mmax_dlm, self.sht_tr, cacher, dlm[1],
                          verbosity=self.verbosity, epsilon=self.epsilon, ofactor=self.ofactor)

    def change_geom(self, lens_geom:Geom, cacher:cachers.cacher or None=None):
        """Returns a deflection instance with a different position-space geometry

                Args:
                    lens_geom: new pixelization geometry
                    cacher: cacher instance if desired


        """
        print("**** change_geom, DO YOU REALLY WANT THIS??")
        return deflection(lens_geom, self.dlm, self.mmax_dlm, self.sht_tr, cacher, self.dclm,
                          verbosity=self.verbosity, epsilon=self.epsilon, ofactor=self.ofactor)

    def gclm2lenpixs(self, gclm:np.ndarray, mmax:int or None, spin:int, pixs:np.ndarray[int]):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation

            Note:
                The number of pixels must be small here, otherwise way too slow

            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so make take some time.

        """
        assert spin >= 0, spin
        ptg = self._get_ptg()
        thts, phis, gamma = ptg[pixs, 0], ptg[pixs, 1], self._get_mgamma()[pixs] * (-1.)
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        geom = Geom(thts, phis.copy(), nph, ofs, wt) #copy necessary as this goes to C
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.alm2map(gclm, spin, lmax, mmax, self.sht_tr)[:, 0::2]
        if spin > 0 :
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()

    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        self.tim.start('gclm2lenmap')
        self.tim.reset()
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if ptg is None:
            ptg = self._get_ptg()
        assert ptg.shape[-1] == 2, ptg.shape
        print(ptg.shape, ptg.dtype, gclm.dtype, gclm.shape, lmax_unl, mmax)
        if self.single_prec: #FIXMEL synthesis general only accepts float
            ptg = ptg.astype(np.float64)

        if spin == 0:
            values = ducc0.sht.experimental.synthesis_general(lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg, spin=spin,
                                                          epsilon=self.epsilon, nthreads=self.sht_tr)
        else:
            npix = self.geom.npix()
            # This is a trick with two views of the same array to get complex values as output to multiply by the phase
            valuesc = np.empty((npix,), dtype=np.complex64 if self.single_prec else np.complex128)
            values = valuesc.view(np.float32 if self.single_prec else np.float64).reshape((npix, 2)).T
            ducc0.sht.experimental.synthesis_general(map=values, lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg,
                                                              spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr)
            if polrot:
                if HAS_NUMEXPR:
                    mg = self._get_mgamma()
                    js = - 1j * spin
                    valuesc *= numexpr.evaluate("exp(js * mg)")
                    self.tim.add('polrot (numexpr)')
                else:
                    valuesc *= np.exp((-1j * spin) * self._get_mgamma())  # polrot. last entry is -gamma
                    self.tim.add('polrot (python)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        return values

    def lensgclm(self, gclm:np.ndarray, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, polrot=True):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

        """
        self.tim.start('lengclm')
        self.tim.reset()
        gclm = np.atleast_2d(gclm)
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.geom.fsky() - 1.) < 1e-6: # no actual deflection and single-precision full-sky
            if spin == 0:
                return alm_copy(gclm[0], mmax, lmax_out, mmax_out)
            glmret = alm_copy(gclm[0], mmax, lmax_out, mmax_out)
            ret = np.array([glmret, alm_copy(gclm[1], mmax, lmax_out, mmax_out) if gclm[1] is not None else np.zeros_like(glmret)])
            self.tim.close('lengclm')
            return ret
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, False)
            self.tim.reset()
            if spin == 0:
                ret = self.geom.map2alm(m, spin, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm')
                self.tim.close('lengclm')
                return ret
            else:
                assert polrot
                ret = self.geom.map2alm(m, spin, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm_spin')
                self.tim.close('lengclm')
                if self.verbosity:
                    print(self.tim)
                return ret
        else:
            lmax_unl = Alm.getlmax(gclm[0].size, mmax)
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
            # minimum dimensions for a Clenshaw-Curtis grid at this band limit
            self.tim.start('points')
            points = self.geom.alm2map(gclm, spin, lmax_unl, mmax, self.sht_tr)
            self.tim.add('points, alm2map')
            if polrot * spin:#TODO: at some point get rid of these exp(atan2)...
                      # maybe simplest to save cis gamma and twice multiply in place...
                if HAS_NUMEXPR:
                    re, im = points
                    x = self._get_mgamma()
                    js = + 1j * spin
                    points = numexpr.evaluate("(re + 1j * im) * exp(js * x)")
                    points = np.array([points.real, points.imag])
                    self.tim.add('points, polrot (numexpr)')
                else:
                    points = points[0] + 1j * points[1]  # only needed if polrot
                    points *= np.exp( (1j * spin) * self._get_mgamma())  # ptg[:, 2] is -gamma
                    points = np.array([points.real, points.imag])
                    self.tim.add('points, polrot (python)')

            points2 = np.atleast_2d(points)
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[:, int(ofs):int(ofs + nph)] *= w
            self.tim.add('points, weighting)')

            # here we must turn back to 2-comp array
            self.tim.close('points')
            ptg = self._get_ptg()
            # FIXME: here ptg must be double prec
            self.tim.add('_get_ptg')
            slm = ducc0.sht.experimental.adjoint_synthesis_general(lmax=lmax_unl, mmax=mmax, map=points2, loc=ptg, spin=spin,
                                                              epsilon=self.epsilon, nthreads=self.sht_tr)
            self.tim.add('synthesis_general')
            self.tim.close('lengclm')

            return slm.squeeze()
