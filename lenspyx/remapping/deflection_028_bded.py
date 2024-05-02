from __future__ import annotations

import numpy as np

from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping import  deflection_028
from ducc0.misc import get_deflected_angles



class deflection(deflection_028.deflection):
    def __init__(self, *args, dphi_bd:np.ndarray[float]=None, **kwargs):
        """Deflection field object than can be used to lens several maps with forward or backward (adjoint) deflection

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
        super().__init__(*args, **kwargs)

        # Geometry object with additional truncation information
        self.geom_bded = BdedGeom(self.geom, dphi_bd)
        #TODO: the original instance already has something called like this
    def _build_angles(self, calc_rotation=True, **kwargs):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

            The only difference is that this occurs on a fraction of the sky only

        """
        assert deflection_028.HAS_DUCCPOINTING
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]) :
            self.tim.start('build_angles')
            d1 = self.geom_bded.collectmap(self._build_d1())
            assert d1.shape == (2, self.geom_bded.npix_bded())
            # Probably want to keep red, imd double precision for the calc?
            dphi = (2 * np.pi) / self.geom.nph
            tht, phi0, nph, ofs = self.geom_bded.theta, self.geom_bded.phi0, self.geom_bded.nph_bded, self.geom_bded.ofs_bded

            tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                      calc_rotation=calc_rotation, nthreads=self.sht_tr, dphi=dphi)
            self.tim.add('build angles <- th-phi%s (ducc)'%('-gm'*calc_rotation))
            if calc_rotation:
                self.cacher.cache(fns[0], tht_phip_gamma[:, 0:2])
                self.cacher.cache(fns[1], tht_phip_gamma[:, 2] if not self.single_prec else tht_phip_gamma[:, 2].astype(np.float32))
            else:
                self.cacher.cache(fns[0], tht_phip_gamma)
            self.tim.close('build_angles')
            return



class BdedGeom(Geom):
    def __init__(self, geom:Geom,  dphi:np.ndarray[float] or float):
        """Iso-latitude pixelisation of the sphere, with additional latitude and longitude truncation info.

                This may be used to work on a patch of the sky

                Args:
                    geom: base geometry object


        #FIXME: how to handle the desired phi0s in the best way ?
        # TODO: want here a phimax instead
        # Essentially we want here a phi_max argument
        #FIXME: how to handle zero rings in the best ways?
        """
        if np.isscalar(dphi):
            dphi = np.full(geom.theta.size, dphi)
        assert (dphi.size == geom.theta.size), ('inconsistent nphi_bd and geom.theta', dphi.size, geom.theta.size)
        assert np.all(np.sort(geom.ofs) == geom.ofs)
        super().__init__(geom.theta, geom.phi0, geom.nph, geom.ofs, geom.weight)
        nphi_bd = self._dphi2nmax(dphi)
        assert np.all(geom.nph >= nphi_bd), 'inconsistent nphi_bd and geom.nph'
        self.nph_bded = nphi_bd.astype(np.uint64)
        self.ofs_bded = np.insert(np.cumsum(nphi_bd[:-1]), 0, 0).astype(np.uint64)



        # index of starting pixel in truncated map (approximate)

    def npix_bded(self):
        """Number of pixels in truncated map


        """
        return int(np.sum(self.nph_bded))

    def nrings_bded(self):
        return np.sum(self.nph_bded > 0)

    def _dphi2nmax(self, dphi:np.ndarray[float]):
        """Finds for each ring the number of pixels in the truncated map that include phimax

        """
        assert np.min(dphi) >= 0 and np.max(dphi) <= 2. * np.pi
        imax = self.nph * ( dphi / (2. * np.pi))
        nmax = (np.int_(np.ceil(imax)) + 1) * (dphi > 0)
        return np.minimum(nmax, self.nph).astype(np.uint64)

    def collectmap(self, m: np.ndarray):
        """Collects interesting longitudes of the map pixelization into a smaller array


        """
        m = np.atleast_2d(m)
        assert [tm.size == self.npix() for tm in m]
        ncomp = m.shape[0]
        ret = np.empty((ncomp, self.npix_bded()), dtype=m.dtype)
        for ofs_bd, ofs, nphbd in zip(self.ofs_bded, self.ofs, self.nph_bded):
            ret[:, ofs_bd:ofs_bd+nphbd] = m[:, ofs:ofs+nphbd]
        return ret
