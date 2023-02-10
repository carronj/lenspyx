from __future__ import annotations

import os

import numpy as np
from lenscarf.remapping import d2ang
from lenscarf.utils_scarf import Geom, pbdGeometry, Geometry
from lenscarf.utils_hp import Alm, alm2cl, alm_copy
from lenscarf import cachers
from lenspyx.utils import timer,blm_gauss
import healpy as hp
import ducc0

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

# some helper functions

ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longfloat): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}


class deflection:
    def __init__(self, scarf_pbgeometry:pbdGeometry, dglm, mmax_dlm:int or None, numthreads:int=0,
                 cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, epsilon=1e-5, ofactor=1.5,verbosity=0):
        """Deflection field object than can be used to lens several maps with forward or backward deflection

            Args:
                scarf_pbgeometry: scarf.Geometry object holding info on the deflection operation pixelization
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
        s2_d = np.sum(alm2cl(dglm, dglm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1) ) / (4 * np.pi)
        if dclm is not None:
            s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1) ) / (4 * np.pi)
        sig_d = np.sqrt(s2_d)
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%(sig_d/np.pi * 180 * 60))
        self.sig_d = sig_d
        self.dlm = dglm
        self.dclm = dclm

        self.lmax_dlm = lmax
        self.mmax_dlm = mmax_dlm

        self.cacher = cacher
        self.pbgeom = scarf_pbgeometry
        self.geom = scarf_pbgeometry.geom
        self.fsky = Geom.fsky(scarf_pbgeometry.geom)

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = Geom.tbounds(scarf_pbgeometry.geom)
        self._pbds = scarf_pbgeometry.pbound  # (patch ctr, patch extent)
        self.sht_tr = numthreads

        self.verbosity = verbosity
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result
        self.ofactor = ofactor  # upsampling grid factor

        print(" DUCC totalconvolve deflection instantiated", self.epsilon, self.ofactor)

        self.single_prec = True # Uses single precision arithmetic in some places
        self.tim = timer(False, 'deflection instance timer')

        if HAS_NUMEXPR:
            os.environ['NUMEXPR_MAX_THREADS'] = str(numthreads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(numthreads)

    def _get_ptg(self):
        # TODO improve this and fwd angles
        return self._build_angles()  # -gamma in third argument

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
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            self.tim.add('d1 alm2map_spin')
            if fortran and HAS_FORTRAN and self.pbgeom.pbound.get_range() >= (2. * np.pi): # covering full phi range
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
                print('Cant use fortran pointing building since import failed. Falling back on python impl.')
            thp_phip_mgamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.pbounds2pix(self.geom, ir, self._pbds)
                if pixs.size > 0:
                    t_red = red[pixs]
                    i_imd = imd[pixs]
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.get_theta(ir) * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(t_red, i_imd, thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip_mgamma[0, sli] = thtp_
                    thp_phip_mgamma[1, sli] = phip_
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_mgamma[2, sli]  = -np.arctan2(i_imd, t_red ) + np.arctan2(i_imd, d * np.sin(d) * cot + t_red  * np.cos(d))
                    startpix += len(pixs)
            self.tim.add('thts, phis and gammas  (python)')
            thp_phip_mgamma = thp_phip_mgamma.transpose()
            self.cacher.cache(fn, thp_phip_mgamma)
            self.tim.close('_build_angle')
            assert startpix == npix, (startpix, npix)
            if self.verbosity:
                print(self.tim)
            return thp_phip_mgamma
        return self.cacher.load(fn)

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.pbgeom, dlm[0], mmax_dlm, self.sht_tr, cacher, dlm[1],
                          verbosity=self.verbosity, epsilon=self.epsilon, ofactor=self.ofactor)

    def change_geom(self, pbgeom:pbdGeometry, cacher:cachers.cacher or None=None):
        """Returns a deflection instance with a different position-space geometry

                Args:
                    pbgeom: new pbounded-scarf geometry
                    cacher: cacher instance if desired


        """
        print("**** change_geom, DO YOU REALLY WANT THIS??")
        return deflection(pbgeom, self.dlm, self.mmax_dlm, self.sht_tr, cacher, self.dclm,
                          verbosity=self.verbosity, epsilon=self.epsilon, ofactor=self.ofactor)

    def gclm2lenpixs(self, gclm:np.ndarray or list, mmax:int or None, spin:int, pixs:np.ndarray[int]):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation

            Note:
                The number of pixels must be small here, otherwise way too slow

            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so make take some time.

        """
        ptg = self._get_ptg()
        thts, phis, gamma = ptg[pixs, 0], ptg[pixs, 1], ptg[pixs, 2] * (-1.)
        nph = 2 * np.ones(thts.size, dtype=int) # I believe at least 2 points per ring if using scarf
        ofs = 2 * np.arange(thts.size, dtype=int)
        wt = np.ones(thts.size)
        geom = Geometry(thts.size, nph, ofs, 1, phis.copy(), thts.copy(), wt) #copy necessary as this goes to C
        #thts.size, nph, ofs, 1, phi0, thts, wt
        if abs(spin) > 0:
            lmax = Alm.getlmax(gclm[0].size, mmax)
            if mmax is None: mmax = lmax
            QU = geom.alm2map_spin(gclm, spin, lmax, mmax, self.sht_tr, [-1., 1.])[:, 0::2]
            QU = np.exp(1j * spin * gamma) * (QU[0] + 1j * QU[1])
            return QU.real, QU.imag
        lmax = Alm.getlmax(gclm.size, mmax)
        if mmax is None: mmax = lmax
        T = geom.alm2map(gclm, lmax, mmax, self.sht_tr, [-1., 1.])[0::2]
        return T

    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        self.tim.start('gclm2lenmap')
        self.tim.reset()
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
        if spin == 0: # The code below would work just as well for spin-0 but seems slightly slower
                     # For the moment this seems faster

            lmax_unl = Alm.getlmax(gclm.size, mmax)
            blm_T = blm_gauss(0, lmax_unl, 0)
            self.tim.add('blm_gauss')
            if ptg is None:
                ptg = self._get_ptg()
            self.tim.add('ptg')

            inter_I = ducc0.totalconvolve.Interpolator(np.atleast_2d(gclm), blm_T, separate=False, lmax=lmax_unl,
                                                       kmax=0,
                                                       epsilon=self.epsilon, ofactor=self.ofactor,
                                                       nthreads=self.sht_tr)
            self.tim.add('interp. setup')
            ret = inter_I.interpol(ptg).squeeze()
            self.tim.add('interpolation')
            return ret
        lmax_unl = Alm.getlmax(gclm.size if spin == 0 else gclm[0].size, mmax)
        if mmax is None: mmax = lmax_unl
        # transform slm to Clenshaw-Curtis map
        ntheta = ducc0.fft.good_size(lmax_unl + 2)
        nphihalf = ducc0.fft.good_size(lmax_unl + 1)
        nphi = 2 * nphihalf
        # Is this any different to scarf wraps ?
        # NB: type of map, map_df, and FFTs will follow that of input gclm
        map = ducc0.sht.experimental.synthesis_2d(alm=np.atleast_2d(gclm), ntheta=ntheta, nphi=nphi,
                                spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr)
        self.tim.add('experimental.synthesis_2d')

        # convert components to real or complex map
        map = map[0] if spin == 0 else map[0] + 1j * map[1]

        # extend map to double Fourier sphere map
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype)
        #    map_dfs = ducc0.misc.make_noncritical_from_shape((2*ntheta-2, nphi), dtype=np.dtype(map.dtype))
        map_dfs[:ntheta, :] = map
        map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
        map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
        if (spin % 2) != 0:
            map_dfs[ntheta:, :] *= -1
        self.tim.add('map_dfs build')

        # go to Fourier space
        if spin == 0:
            tmp = np.empty(map_dfs.shape, dtype=ctype[map.dtype])
            #        tmp = ducc0.misc.make_noncritical_from_shape(map_dfs.shape, dtype=np.dtype(ctype[map.dtype]))
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp)
        else:
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=map_dfs)
        self.tim.add('map_dfs 2DFFT')

        # perform NUFFT
        if ptg is None:
            ptg = self._get_ptg()
        self.tim.add('get ptg')
        values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg[:, 0:2], forward=False,
                                  epsilon=self.epsilon, nthreads=self.sht_tr,
                                  verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
        self.tim.add('u2nu')

        if polrot and spin != 0: #TODO: at some point get rid of these exp(arctan...)
            if HAS_NUMEXPR:
                x = ptg[:, 2]
                js = - 1j * spin
                values *= numexpr.evaluate("exp(js * x)")
                self.tim.add('polrot (numexpr)')
            else:
                values *= np.exp((-1j * spin) * ptg[:, 2])  # polrot. last entry is -gamma
                self.tim.add('polrot (python)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        return values.real if spin == 0 else (values.real, values.imag)

    def lensgclm(self, gclm:np.ndarray or list, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, polrot=True):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

        """
        self.tim.start('lengclm')
        self.tim.reset()
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.fsky - 1.) < 1e-6: # no actual deflection and single-precision full-sky
            if spin == 0: return alm_copy(gclm, mmax, lmax_out, mmax_out)
            glmret = alm_copy(gclm[0], mmax, lmax_out, mmax_out)
            ret = np.array([glmret, alm_copy(gclm[1], mmax, lmax_out, mmax_out) if gclm[1] is not None else np.zeros_like(glmret)])
            self.tim.close('lengclm')
            return ret
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards)
            self.tim.reset()
            if spin == 0:
                #TODO: this does not respect the input dtype ?
                ret = self.geom.map2alm(m, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm')
                self.tim.close('lengclm')
                return ret
            else:
                assert polrot
                #TODO: this does not respect the input dtype ?
                ret = self.geom.map2alm_spin(m, spin, lmax_out, mmax_out, self.sht_tr)
                self.tim.add('map2alm_spin')
                self.tim.close('lengclm')
                if self.verbosity:
                    print(self.tim)
                return ret
        else:
            if spin == 0:
                # The code below works for any spin but this seems a little bit faster for non-zero spin
                # So keeping this for the moment
                lmax_unl = hp.Alm.getlmax(gclm[0].size if abs(spin) > 0 else gclm.size, mmax)
                inter = ducc0.totalconvolve.Interpolator(lmax_out, spin, 1, epsilon=self.epsilon,
                                                         ofactor=self.ofactor, nthreads=self.sht_tr)
                if self.single_prec and gclm.dtype != np.complex64:  # will return same prec. output
                    I = self.geom.alm2map(gclm.astype(np.complex64), lmax_unl, mmax, self.sht_tr)
                else:
                    I = self.geom.alm2map(gclm, lmax_unl, mmax, self.sht_tr)
                for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                    I[int(ofs):int(ofs + nph)] *= w
                self.tim.add('points')
                xptg = self._get_ptg()
                self.tim.add('_get_ptg')
                inter.deinterpol(xptg, np.atleast_2d(I))
                self.tim.add('deinterpol')
                blm = blm_gauss(0, lmax_out, spin)
                ret = inter.getSlm(blm).squeeze()
                self.tim.add('getSlm')
                self.tim.close('lensgclm')
                return ret
            # minimum dimensions for a Clenshaw-Curtis grid at this band limit
            ntheta = ducc0.fft.good_size(lmax_out + 2)
            nphihalf = ducc0.fft.good_size(lmax_out + 1)
            nphi = 2 * nphihalf
            ptg = self._get_ptg()
            self.tim.add('_get_ptg')
            if spin == 0:
                # make complex if necessary
                lmax_unl = hp.Alm.getlmax(gclm.size, mmax)
                if self.single_prec and gclm.dtype != np.complex64:  # will return same prec. output
                    points = self.geom.alm2map(gclm.astype(np.complex64), lmax_unl, mmax, self.sht_tr, [-1., 1.]) + 0j
                else:
                    points = self.geom.alm2map(gclm, lmax_unl, mmax, self.sht_tr, [-1., 1.]) + 0j
                self.tim.add('points')
            else:
                lmax_unl = hp.Alm.getlmax(gclm[0].size, mmax)
                #TODO: experimential_sythesis return already complex
                if self.single_prec and gclm[0].dtype != np.complex64:
                    points = self.geom.alm2map_spin(gclm.astype(np.complex64), spin, lmax_unl, mmax, self.sht_tr, [-1., 1.])
                else:
                    points = self.geom.alm2map_spin(gclm, spin, lmax_unl, mmax, self.sht_tr, [-1., 1.])
                points = points[0] + 1j * points[1]
                self.tim.add('points')
                if polrot:#TODO: at some point get rid of these exp(atan2)...
                    if HAS_NUMEXPR:
                        x = ptg[:, 2]
                        js = + 1j * spin
                        points *= numexpr.evaluate("exp(js * x)")
                        self.tim.add('polrot (numexpr)')
                    else:
                        points *= np.exp( (1j * spin) * ptg[:, 2])  # ptg[:, 2] is -gamma
                        self.tim.add('polrot (python)')
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[int(ofs):int(ofs + nph)] *= w
            self.tim.add('weighting')

            map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128)

            # perform NUFFT
            map_dfs = ducc0.nufft.nu2u(points=points, coord=ptg[:, 0:2], out=map_dfs, forward=True,
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
            map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=np.float64)
            map[0] = map_dfs.real
            if spin > 0:
                map[1] = map_dfs.imag
            self.tim.add('Double Fourier')

            # adjoint SHT synthesis
            slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin,
                        lmax=lmax_out, mmax=mmax_out, geometry="CC", nthreads=self.sht_tr)
            self.tim.add('map2alm_spin')

            self.tim.close('lengclm')

            return slm.squeeze()
