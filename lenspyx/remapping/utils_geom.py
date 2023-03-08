from __future__ import annotations
import numpy as np
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
                    theta: rings co-latitudes in radians
                    phi0: longitude offset of first point in each ring
                    nphi: number of points in each ring
                    ofs: offsets of each ring in real space map
                    w: quadrature weight for each ring


        """
        for arr in [phi0, nphi, ringstart, w]:
            assert arr.size == theta.size

        argsort = np.argsort(nphi)[::-1] # We sort the rings by decreasing nphi
        self.theta = theta[argsort].astype(np.float64)
        self.weight = w[argsort].astype(np.float64)
        self.phi0 = phi0[argsort].astype(np.float64)
        self.nph = nphi[argsort].astype(np.uint64)
        self.ofs = ringstart[argsort].astype(np.uint64)

        self.sorted = True

    def npix(self):
        """Number of pixels

        """
        return int(np.sum(self.nph))

    def fsky(self):
        """Fractional area of the sky covered by the pixelization

        """
        return np.sum(self.weight * self.nph) / (4 * np.pi)

    def synthesis(self, gclm: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int):
        """Wrapper to ducc forward SHT

            Return a map or a pair of map for spin non-zero, with the same type as gclm

        """
        gclm = np.atleast_2d(gclm)
        return synthesis(alm=gclm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs)

    def synthesis_deriv1(self, alm: np.ndarray, lmax:int, mmax:int, nthreads:int):
        """Wrapper to ducc synthesis_deriv1

        """
        return synthesis_deriv1(alm=alm, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, phi0=self.phi0,
                         nthreads=nthreads, ringstart=self.ofs)

    def adjoint_synthesis(self, m: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int):
        """Wrapper to ducc backward SHT

            Return an array with leading dimension 1 for spin-0 or 2 for spin non-zero

            Note:
                This modifies the input map

        """
        m = np.atleast_2d(m)
        for of, w, npi in zip(self.ofs, self.weight, self.nph):
            m[:, of:of + npi] *= w
        return adjoint_synthesis(map=m, theta=self.theta, lmax=lmax, mmax=mmax, nphi=self.nph, spin=spin, phi0=self.phi0,
                                 nthreads=nthreads, ringstart=self.ofs)

    def alm2map_spin(self, gclm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.)):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, spin, lmax, mmax, nthreads)

    def map2alm_spin(self, m:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.)):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m, spin, lmax, mmax, nthreads)

    def alm2map(self, gclm:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.)):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, 0, lmax, mmax, nthreads).squeeze()

    def map2alm(self, m:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.)):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m, 0, lmax, mmax, nthreads).squeeze()

    @staticmethod
    def rings2pix(geom:Geom, rings:np.ndarray[int]):
        return np.concatenate([geom.ofs[ir] + np.arange(geom.nph[ir], dtype=int) for ir in rings])

    @staticmethod
    def phis(geom:Geom, ir):
        assert ir < geom.theta.size
        nph = geom.nph[ir]
        return (geom.phi0[ir] + np.arange(nph) * (2 * np.pi / nph)) % (2. * np.pi)

    @staticmethod
    def rings2phi(geom:Geom, rings:np.ndarray[int]):
        return np.concatenate([Geom.phis(geom, ir) for ir in rings])

    @staticmethod
    def get_thingauss_geometry(lmax:int, smax:int, zbounds:tuple[float, float]=(-1., 1.)):
        """Build a 'thinned' Gauss-Legendre geometry

            This uses polar optimization to reduce the number of points away from the equator

            Args:
                lmax: band-limit (or desired band-limit) on the equator
                smax: maximal intended spin-value (this changes the m-truncation by an amount ~smax)
                zbounds: pixels outside of provided cos-colatitude bounds will be discarded

            Note:
                'thinning' saves memory but hardly any computational time for the same latitude range


        """
        nlatf = lmax + 1  # full meridian GL points
        tht = GL_thetas(nlatf)
        tb = np.sort(np.arccos(zbounds))
        p = np.where((tb[0] <= tht) & (tht <= tb[1]))

        tht = tht[p]
        wt = GL_weights(nlatf, 1)[p]
        nlat = tht.size
        phi0 = np.zeros(nlat, dtype=float)
        mmax = np.minimum(np.maximum(st2mmax(smax, tht, lmax), st2mmax(-smax, tht, lmax)), np.ones(nlat) * lmax)
        nph = np.array([good_size(int(np.ceil(2 * m + 1))) for m in mmax])
        ofs = np.insert(np.cumsum(nph[:-1]), 0, 0)
        return Geom(tht, phi0, nph, ofs, wt / nph)

    @staticmethod
    def get_healpix_geometry(nside:int):
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        geom = base.sht_info()
        area = (4 * np.pi) / (12 * nside ** 2)
        return Geom(w=np.full((geom['theta'].size, ), area), **geom)

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