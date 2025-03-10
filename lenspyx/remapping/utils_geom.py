from __future__ import annotations
import numpy as np
from lenspyx import utils_hp
import ducc0
from ducc0.misc import GL_thetas, GL_weights
from ducc0.fft import good_size
from ducc0.sht.experimental import synthesis, adjoint_synthesis, synthesis_deriv1

def st2mmax(spin, tht, lmax):
    r"""Converts spin, tht and lmax to a maximum effective m, according to libsharp paper polar optimization formula Eqs. 7-8

        For a given mmax, one needs then in principle 2 * mmax + 1 longitude points for exact FFT's


    """
    T = max(0.01 * lmax, 100)
    b = - 2 * spin * np.cos(tht)
    c = -(T + lmax * np.sin(tht)) ** 2 + spin ** 2
    mmax = 0.5 * (- b + np.sqrt(b * b - 4 * c))
    return mmax


class Geom:
    def __init__(self, theta:np.ndarray[float], phi0:np.ndarray[float], nphi:np.ndarray[np.uint64], ringstart:np.ndarray[np.uint64], w:np.ndarray[float]):
        """Iso-latitude pixelisation of the sphere

                Args:
                    theta: rings co-latitudes in radians in [0, pi]
                    phi0: longitude offset of first point in each ring in radians
                    nphi: number of pixels in each ring
                    ringstart: index of first pixel of each ring in real space map
                    w: quadrature weight for each ring (used for SHT of 'analysis'-type )


        """
        for arr in [phi0, nphi, ringstart, w]:
            assert arr.size == theta.size

        argsort = np.argsort(ringstart) # We sort here the rings by order in the maps
        self.theta = theta[argsort].astype(np.float64)
        self.weight = w[argsort].astype(np.float64)
        self.phi0 = phi0[argsort].astype(np.float64)
        self.nph = nphi[argsort].astype(np.uint64)
        self.ofs = ringstart[argsort].astype(np.uint64)

    def npix(self):
        """Number of pixels

        """
        return int(np.sum(self.nph))

    def fsky(self):
        """Fractional area of the sky covered by the pixelization

        """
        return np.sum(self.weight * self.nph) / (4 * np.pi)

    def sort(self, arr:np.ndarray, inplace:bool):
        """Rearrange the arrays inplace, sorting them argsorting the input array


        """
        assert arr.size == self.theta.size, (arr.size, self.theta.size)
        asort = np.argsort(arr) # We sort here the rings by order in the maps
        if inplace:
            for ar in [self.theta, self.weight, self.phi0, self.nph, self.ofs]:
                ar[:] = ar[asort]
            return self
        return Geom(self.theta[asort], self.phi0[asort], self.nph[asort], self.ofs[asort], self.weight[asort])

    def restrict(self, tht_min:float, tht_max:float, northsouth_sym:bool, update_ringstart=False):
        """Returns a geometry with restricted co-latitude range

            Args:
                tht_min: min colatitude in radians
                tht_max: max colatitude in radians
                northsouth_sym: includes the equator-symmetrized region if set
                update_ringstart: The ringstart indices of the output object still refer to the original geometry map if not set

            Return:
                Geometry object with corresponding colatitude range

        """
        n_cond = (self.theta <= tht_max) & (self.theta >= tht_min)
        if northsouth_sym:
            n_cond = n_cond | (self.theta <= (np.pi - tht_min)) & (self.theta >= (np.pi - tht_max))
        band = np.where(n_cond)
        ofs = np.insert(np.cumsum(self.nph[band][:-1]), 0, 0) if update_ringstart else self.ofs[band]
        return Geom(self.theta[band], self.phi0[band], self.nph[band], ofs, self.weight[band])

    def thinout(self, spin, good_size_real=True):
        """Reduces the number of long points at high latitudes keeping the resolution similar to that of the equator

        """
        ntht, tht = self.theta.size, self.theta
        st = np.sin(tht)
        lmax = ntht - 1
        mmax = np.minimum(np.maximum(st2mmax(spin, tht, lmax), st2mmax(-spin, tht, lmax)), np.ones(ntht) * lmax)
        nph = np.array([good_size(int(np.ceil(2 * m + 1)), good_size_real) for m in mmax])
        ir_eq = np.argmax(st) # We force the longitude resolution not to degrade compared to the equator
        dph_eq = 1. / nph[ir_eq]
        nphi_eq = np.array([good_size(int(np.ceil(stx / dph_eq)), good_size_real) for stx in st])
        nph = np.where((st / nph) > dph_eq, nphi_eq, nph)
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        return Geom(tht, self.phi0, nph, ofs, self.weight / nph * self.nph)

    def split(self, nbands, verbose=False):
        """Split the pixelization into chunks

            Args:
                nbands(int): the colatitude range is uniformly split into 'nbands'

            Returns:
                list of Geom instances

            Notes:
                Respects north-south symmetry when present in the instance for faster SHTs

                The ringstarts of the geoms refers to the original full map

                There can be some small overlap between bands if the instance co-latitudes exactly match the separation points

        """
        thts = np.linspace(0., np.pi * 0.5, nbands + 1)
        th_ls = thts[:-1]
        th_us = thts[1:]
        th_ls[0] = 0.
        th_us[-1] = np.pi * 0.5
        geoms = []
        for th_l, th_u in zip(th_ls, th_us):
            geoms.append(self.restrict(th_l, th_u, True))
        npix = self.npix()
        npix_tot = np.sum([geo.npix() for geo in geoms])
        assert npix_tot >= npix, (npix, npix_tot, 'aaargh')
        if npix_tot > npix:
            if verbose:
                print('(split with overlap, %s additional pixels out of %s)'%(npix_tot-npix, npix))
        return geoms

    def synthesis(self, gclm: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, map:np.ndarray=None, **kwargs):
        """Wrapper to ducc forward SHT

            Return a map or a pair of map for spin non-zero, with the same type as gclm


        """
        gclm = np.atleast_2d(gclm)
        return synthesis(alm=gclm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs, map=map, **kwargs)

    def synthesis_deriv1(self, alm: np.ndarray, lmax:int, mmax:int, nthreads:int, **kwargs):
        """Wrapper to ducc synthesis_deriv1

        """
        alm = np.atleast_2d(alm)
        return synthesis_deriv1(alm=alm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs, **kwargs)

    def adjoint_synthesis(self, m: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, alm=None, apply_weights=True, **kwargs):
        """Wrapper to ducc backward SHT

            Return an array with leading dimension 1 for spin-0 or 2 for spin non-zero

            Note:
                This modifies the input map

        """
        m = np.atleast_2d(m)
        if apply_weights:
            for of, w, npi in zip(self.ofs, self.weight, self.nph):
                m[:, of:of + npi] *= w
        if alm is not None:
            assert alm.shape[-1] == utils_hp.Alm.getsize(lmax, mmax)
        return adjoint_synthesis(map=m, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                                 nthreads=nthreads, ringstart=self.ofs, alm=alm,  **kwargs)

    def alm2map_spin(self, gclm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, spin, lmax, mmax, nthreads, **kwargs)

    def map2alm_spin(self, m:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), spin, lmax, mmax, nthreads, **kwargs)

    def alm2map(self, gclm:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, 0, lmax, mmax, nthreads, **kwargs).squeeze()

    def map2alm(self, m:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), 0, lmax, mmax, nthreads, **kwargs).squeeze()

    @staticmethod
    def rings2pix(geom:Geom, rings:np.ndarray[int]):
        return np.concatenate([geom.ofs[ir] + np.arange(geom.nph[ir], dtype=np.uint64) for ir in rings])

    @staticmethod
    def phis(geom:Geom, ir):
        assert ir < geom.theta.size
        nph = geom.nph[ir]
        return (geom.phi0[ir] + np.arange(nph) * (2 * np.pi / nph)) % (2. * np.pi)

    @staticmethod
    def rings2phi(geom:Geom, rings:np.ndarray[int]):
        return np.concatenate([Geom.phis(geom, ir) for ir in rings])

    @staticmethod
    def get_supported_geometries():
        geoms = ''
        vs = vars(Geom)
        for k in list(vs.keys()):
            s = k.split('_')
            if len(s) == 3 and s[0] == 'get' and s[2] == 'geometry':
                geoms += ' ' + s[1]
        return geoms

    @staticmethod
    def show_supported_geometries():
        geoms = []
        vs = vars(Geom)
        for k in list(vs.keys()):
            s = k.split('_')
            if len(s) == 3 and s[0] == 'get' and s[2] == 'geometry':
                geoms.append(s)
        if len(geoms) > 0:
            print('supported geometries: ')
            for geo in geoms:
                print(geo[1] + ':')
                print(getattr(Geom, '_'.join(geo)).__doc__)
        else:
            print('no supported geometry found')


    @staticmethod
    def get_thingauss_geometry(lmax:int, smax:int, good_size_real=True):
        """Longitude-thinned Gauss-Legendre pixelization

            Args:
                lmax: number of latitude points is lmax + 1 (exact quadrature rules for this band-limit)
                smax: maximum spin-weight to be used on this grid (impact slightly the choice of longitude points)
                good_size_real(optional): decides on a FFT-friendly number of phi point for real if set or complex FFTs'
                                          (very slightly more points if set but largely inconsequential)


        """
        nlatf = lmax + 1  # full meridian GL points
        tht = GL_thetas(nlatf)
        wt = GL_weights(nlatf, 1)
        nlat = tht.size
        phi0 = np.zeros(nlat, dtype=float)
        mmax = np.minimum(np.maximum(st2mmax(smax, tht, lmax), st2mmax(-smax, tht, lmax)), np.ones(nlat) * lmax)
        nph = np.array([good_size(int(np.ceil(2 * m + 1)), good_size_real) for m in mmax])
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        return Geom(tht, phi0, nph, ofs, wt / nph)

    @staticmethod
    def get_healpix_geometry(nside:int):
        """Healpix pixelization

            Args:
                nside: healpix nside resolution parameter (npix is 12 * nside ** 2)


        """
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        geom = base.sht_info()
        area = (4 * np.pi) / (12 * nside ** 2)
        return Geom(w=np.full((geom['theta'].size, ), area), **geom)

    @staticmethod
    def get_cc_geometry(ntheta:int, nphi:int):
        """Clenshaw-Curtis pixelization

            Uniformly-spaced in latitude, one point on each pole

            Args:
                ntheta: number of latitude points
                nphi: number of longitude points


        """
        tht = np.linspace(0, np.pi, ntheta, dtype=float)
        phi0 = np.zeros(ntheta, dtype=float)
        nph = np.full((ntheta,), nphi, dtype=np.uint64)
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        w = ducc0.sht.experimental.get_gridweights('CC', ntheta)
        return Geom(tht, phi0, nph, ofs, w / nphi)

    @staticmethod
    def get_f1_geometry(ntheta:int, nphi:int):
        """Fejer-1 pixelization

            Uniformly-spaced in latitude, first and last point pi / 2N away from the poles

            Args:
                ntheta: number of latitude points
                nphi: number of longitude points


        """
        tht = np.linspace(0.5 * np.pi / ntheta, (ntheta - 0.5) * np.pi / ntheta, ntheta)
        phi0 = np.zeros(ntheta, dtype=float)
        nph = np.full((ntheta,), nphi, dtype=np.uint64)
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        w = ducc0.sht.experimental.get_gridweights('F1', ntheta)
        return Geom(tht, phi0, nph, ofs, w / nphi)


    @staticmethod
    def get_gl_geometry(lmax:int, good_size_real=True, nphi:int or None =None):
        """Gauss-Legendre pixelization

            Args:
                lmax: number of latitude points is lmax + 1 (exact quadrature rules for this band-limit)
                good_size_real(optional): decides on a FFT-friendly number of phi point for real if set or complex FFTs'
                                          (very slightly more points if set but largely inconsequential)

        """
        nlatf = lmax + 1  # full meridian GL points
        if nphi is None:
            nphi = good_size(2 * lmax + 1, good_size_real)
        tht = GL_thetas(nlatf)
        wt = GL_weights(nlatf, 1)
        phi0 = np.zeros(nlatf, dtype=float)
        nph = np.full((nlatf,), nphi, dtype=np.uint64)
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        return Geom(tht, phi0, nph, ofs, wt / nph)

    @staticmethod
    def get_tgl_geometry(lmax:int, smax:int, good_size_real=True):
        """Longitude-thinned Gauss-Legendre pixelization

            Args:
                lmax: number of latitude points is lmax + 1 (exact quadrature rules for this band-limit)
                smax: maximum spin-weight to be used on this grid (impact slightly the choice of longitude points)
                good_size_real(optional): decides on a FFT-friendly number of phi point for real if set or complex FFTs'
                                          (very slightly more points if set but largely inconsequential)

        """
        return Geom.get_thingauss_geometry(lmax, smax, good_size_real=good_size_real)


class pbounds:
    """Class to regroup simple functions handling sky maps longitude truncation

            Args:
                pctr: center of interval in radians
                prange: full extent of interval in radians

            Note:
                2pi periodicity

    """
    def __init__(self, pctr:float, prange:float):
        assert prange >= 0., prange
        self.pctr = pctr % (2. * np.pi)
        self.hext = min(prange * 0.5, np.pi) # half-extent

    def __repr__(self):
        return "ctr:%.2f range:%.2f"%(self.pctr, self.hext * 2)

    def __eq__(self, other:pbounds):
        return self.pctr == other.pctr and self.hext == other.hext

    def get_range(self):
        return 2 * self.hext

    def get_ctr(self):
        return self.pctr

    def contains(self, phs:np.ndarray):
        dph = (phs - self.pctr) % (2 * np.pi)  # points inside are either close to zero or 2pi
        return (dph <= self.hext) |((2 * np.pi - dph) <= self.hext)

class pbdGeometry:
    def __init__(self, geom: Geom, pbound: pbounds):
        """Gometry with additional info on longitudinal cuts


        """
        self.geom = geom
        self.pbound = pbound