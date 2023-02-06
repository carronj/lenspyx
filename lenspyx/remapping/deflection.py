import numpy as np
from lenscarf.remapping import d2ang
from lenscarf.utils_scarf import Geom, pbdGeometry
from lenscarf.utils_hp import Alm, alm2cl
from lenscarf import cachers
import healpy as hp
import ducc0



def blm_gauss(fwhm, lmax, spin:int):
    """Computes spherical harmonic coefficients of a circular Gaussian beam
    pointing towards the North Pole

    See an example of usage
    `in the documentation <https://healpy.readthedocs.io/en/latest/blm_gauss_plot.html>`_

    Parameters
    ----------
    fwhm : float, scalar
        desired FWHM of the beam, in radians
    lmax : int, scalar
        maximum l multipole moment to compute
    spin : bool, scalar
        if True, E and B coefficients will also be computed

    Returns
    -------
    blm : array with dtype numpy.complex128
          lmax will be as specified
          mmax is equal to spin
    """
    fwhm = float(fwhm)
    lmax = int(lmax)
    mmax = spin
    ncomp = 2 if spin > 0 else 1
    nval = hp.Alm.getsize(lmax, mmax)

    if mmax > lmax:
        raise ValueError("lmax value too small")

    blm = np.zeros((ncomp, nval), dtype=np.complex128)
    sigmasq = fwhm * fwhm / (8 * np.log(2.0))

    if spin == 0:
        for l in range(0, lmax + 1):
            blm[0, hp.Alm.getidx(lmax, l, spin)] = np.sqrt((2 * l + 1) / (4.0 * np.pi)) * np.exp(-0.5 * sigmasq * l * l)

    if spin > 0:
        for l in range(spin, lmax + 1):
            blm[0, hp.Alm.getidx(lmax, l, spin)] = np.sqrt((2 * l + 1) / (32 * np.pi)) * np.exp(-0.5 * sigmasq * l * l)
        blm[1] = 1j * blm[0]

    return blm

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

        # FIXME: can get d1 tbounds from geometry + buffers.
        self._tbds = Geom.tbounds(scarf_pbgeometry.geom)
        self._pbds = scarf_pbgeometry.pbound  # (patch ctr, patch extent)
        self.sht_tr = numthreads

        self.verbosity = verbosity
        self.epsilon = epsilon # accuracy of the totalconvolve interpolation result
        self.ofactor = ofactor  # upsampling grid factor

        print(" DUCC totalconvolve deflection instantiated", self.epsilon, self.ofactor)

    def _get_ptg(self):
        # TODO improve this and fwd angles
        return self._build_angles() # -gamma in third argument

    def _build_angles(self):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fn = 'ptg'
        if not self.cacher.is_cached(fn):
            dclm = np.zeros_like(self.dlm) if self.dclm is None else self.dclm
            red, imd = self.geom.alm2map_spin([self.dlm, dclm], 1, self.lmax_dlm, self.mmax_dlm, self.sht_tr, [-1., 1.])
            npix = Geom.pbounds2npix(self.geom, self._pbds)
            thp_phip_mgamma = np.empty( (3, npix), dtype=float) # (-1) gamma in last arguement
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
                    assert 0 < self.geom.theta[ir] < np.pi, 'Fix this'
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_mgamma[2, sli]  = -np.arctan2(i_imd, t_red ) + np.arctan2(i_imd, d * np.sin(d) * cot + t_red  * np.cos(d))
                    startpix += len(pixs)
            thp_phip_mgamma = thp_phip_mgamma.transpose()
            self.cacher.cache(fn, thp_phip_mgamma)
            assert startpix == npix, (startpix, npix)
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
        return deflection(pbgeom, self.dlm, self.mmax_dlm, self.sht_tr, cacher, self.dclm,
                          verbosity=self.verbosity, epsilon=self.epsilon, ofactor=self.ofactor)

    def gclm2lenmap(self, gclm:np.ndarray or list, mmax:int or None, spin, backwards:bool, nomagn=False, polrot=True, ptg=None):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        if spin == 0: # The code below would work just as well for spin-0 but seems slightly slower
                     # For the moment this seems faster
            lmax_unl = Alm.getlmax(gclm.size, mmax)
            blm_T = blm_gauss(0, lmax_unl, 0)
            if ptg is None:
                ptg = self._get_ptg()
            inter_I = ducc0.totalconvolve.Interpolator(np.atleast_2d(gclm), blm_T, separate=False, lmax=lmax_unl,
                                                       kmax=0,
                                                       epsilon=self.epsilon, ofactor=self.ofactor,
                                                       nthreads=self.sht_tr)
            return inter_I.interpol(ptg).squeeze()
        lmax_unl = Alm.getlmax(gclm.size if spin == 0 else gclm[0].size, mmax)
        if mmax is None: mmax = lmax_unl
        # transform slm to Clenshaw-Curtis map
        ntheta = lmax_unl + 2
        nphi = 2 * lmax_unl + 2
        # Is this any different to scarf wraps ?
        map = ducc0.sht.experimental.synthesis_2d(alm=np.atleast_2d(gclm), ntheta=ntheta, nphi=nphi,
                                spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr)

        # convert components to real or complex map
        map = map[0] if spin == 0 else map[0] + 1j * map[1]

        # extend map to double Fourier sphere map
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype)
        map_dfs[:ntheta, :] = map
        spinfac = 1 if (spin % 2) == 0 else -1
        map_dfs[ntheta:, :lmax_unl + 1] = spinfac * map[-2:0:-1, lmax_unl + 1:]
        map_dfs[ntheta:, lmax_unl + 1:] = spinfac * map[-2:0:-1, :lmax_unl + 1]

        # go to Fourier space
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2)

        # perform NUFFT
        if ptg is None:
            ptg = self._get_ptg()
        values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg[:, 0:2], forward=False,
                                  epsilon=self.epsilon, nthreads=self.sht_tr,
                                  verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
        if polrot and spin != 0:
            values *= np.exp( (-1j * spin) * ptg[:, 2]) # polrot. last entry is -gamma
        return values.real if spin == 0 else (values.real, values.imag)

    def lensgclm(self, gclm:np.ndarray or list, mmax:int or None, spin, lmax_out, mmax_out:int or None, backwards=False, nomagn=False, polrot=True):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

        """
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards, nomagn=nomagn)
            if spin == 0:
                return self.geom.map2alm(m, lmax_out, mmax_out, self.sht_tr)
            else:
                assert polrot
                return self.geom.map2alm_spin(m, spin, lmax_out, mmax_out, self.sht_tr)
        else:
            if spin == 0:
                # The code below works for any spin but this seems a little bit faster for non-zero spin
                # So keeping this for the moment
                lmax_unl = hp.Alm.getlmax(gclm[0].size if abs(spin) > 0 else gclm.size, mmax)
                inter = ducc0.totalconvolve.Interpolator(lmax_out, spin, 1, epsilon=self.epsilon,
                                                         ofactor=self.ofactor, nthreads=self.sht_tr)
                I = self.geom.alm2map(gclm, lmax_unl, mmax, self.sht_tr)
                for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                    I[int(ofs):int(ofs + nph)] *= w
                xptg = self._get_ptg()
                inter.deinterpol(xptg, np.atleast_2d(I))
                blm = blm_gauss(0, lmax_out, spin)
                return inter.getSlm(blm).squeeze()
            # minimum dimensions for a Clenshaw-Curtis grid at this band limit
            ntheta = lmax_out + 2
            nphi = 2 * lmax_out + 2
            ptg = self._get_ptg()
            if spin == 0:
                # make complex if necessary
                lmax_unl = hp.Alm.getlmax(gclm.size, mmax)
                points = self.geom.alm2map(gclm, lmax_unl, mmax, self.sht_tr, [-1., 1.]) + 0j
            else:
                lmax_unl = hp.Alm.getlmax(gclm[0].size, mmax)
                #NB: experimential_sythesis return already complex
                points = self.geom.alm2map_spin(gclm, spin, lmax_unl, mmax, self.sht_tr, [-1., 1.])
                points = points[0]+ 1j * points[1]
                if polrot:
                    points *= np.exp( (1j * spin) * ptg[:, 2])  # ptg[:, 2] is -gamma
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[int(ofs):int(ofs + nph)] *= w

            map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128)

            # perform NUFFT
            _ = ducc0.nufft.nu2u(points=points, coord=ptg[:, 0:2], out=map_dfs, forward=True,
                                    epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                    periodicity=2 * np.pi, fft_order=True)
            # go to position space
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2)

            # go from double Fourier sphere to Clenshaw-Curtis grid
            spinfac = 1 if (spin % 2) == 0 else -1
            map_dfs[1:ntheta - 1, :lmax_out + 1] += spinfac * map_dfs[-1:ntheta - 1:-1, lmax_out + 1:]
            map_dfs[1:ntheta - 1, lmax_out + 1:] += spinfac * map_dfs[-1:ntheta - 1:-1, :lmax_out + 1]
            map_dfs = map_dfs[:ntheta, :]
            map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=np.float64)
            map[0] = map_dfs.real
            if spin > 0:
                map[1] = map_dfs.imag
            # adjoint SHT synthesis
            slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin,
                        lmax=lmax_out, mmax=mmax_out, geometry="CC", nthreads=self.sht_tr)
            return slm.squeeze()