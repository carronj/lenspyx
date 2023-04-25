"""This module contains DUCC-based fonctions related to Wigner-small d tranforms

    This uses `alm2leg' and `leg2alm' from ducc0. They were not optimized for this, and the code could be improved further,
    but the compiler optimization is done so well that they still out-perform plancklens code it seems by large amounts.


"""
from __future__ import annotations
import numpy as np
from ducc0.sht.experimental import alm2leg, leg2alm
from ducc0.misc import GL_thetas, GL_weights

GL_cache = {}
verbose = False

def wignerpos(cl: np.darray[float], theta: np.darray[float], s1: int, s2: int):
    r"""Produces Wigner small-d transform defined by

        :math:`\sum_\ell \frac{2\ell + 1}{4\pi} C_\ell d^\ell_{s_1 s_2}(\theta)`

        Args:
            cl: spectrum of Wigner small-d transform
            theta: co-latitude in radians (in [0, pi])
            s1: first spin
            s2: second spin

        Returns:
            real array of same size than theta


    """
    if s1 == 0 and s2 != 0:
        # always want a spin 0 on the SHT side
        t_cl = cl if (s1 + s2) % 2 == 0 else -cl
        return wignerpos(t_cl, theta, s2, s1)
    if s1 < 0:
        t_cl = cl if (s1 + s2) % 2 == 0 else -cl
        return wignerpos(t_cl, theta, -s1, -s2)
    if s1 >= 0:
        lmax = len(cl) - 1
        s = abs(s2)
        mstart = np.array([0], dtype=int)
        mval = np.array([abs(s1)], dtype=int)
        glm_r = (cl * np.sqrt(np.arange(1, 2 * lmax + 3, 2)) / np.sqrt(4 * np.pi)).astype(complex)
        mode = 'GRAD_ONLY' if s else 'STANDARD'
        leg = alm2leg(alm=np.atleast_2d(glm_r), spin=s, lmax=lmax, mval=mval, mstart=mstart, theta=theta, mode=mode).squeeze()
        if s2 == 0:
            return leg.real
        if s2 > 0:
            return -(leg[0].real + leg[1].imag)
        return (-1 if s % 2 == 0 else 1) * (leg[0].real - leg[1].imag)

    assert 0, (s1, s2)


def wignercoeff(xi: np.ndarray[float], theta: np.ndarray[float], s1: int, s2: int, lmax: int):
    r"""Computes spectrum of Wigner small-d correlation function (adjoint to `wignerpos')

            :math:`2\pi \sum_\theta \xi(\theta) d^\ell_{s_1 s_2}(\theta)`

        Args:
            xi: Wigner function (real array on point per co-latitude)
            theta: co-latitude in radians (in [0, pi])
            s1: first spin
            s2: second spin
            lmax: calculates spectrum up to lmax (inclusive)


    """
    if s1 < 0:
        t_xi = xi if (s1 + s2) % 2 == 0 else -xi
        return wignercoeff(t_xi, theta, -s1, -s2, lmax)
    if s1 == 0 and s2 != 0:
        # always want a spin 0 on the SHT side
        t_xi = xi if (s1 + s2) % 2 == 0 else -xi
        return wignercoeff(t_xi, theta, s2, s1, lmax)
    mval = np.array([abs(s1)], dtype=int)
    mstart = np.array([0], dtype=int)
    fac = (2 * np.pi * np.sqrt(4 * np.pi)) / np.sqrt(np.arange(1, 2 * lmax + 3, 2))
    xis = xi.astype(complex)
    if s2 == 0:
        cl = leg2alm(leg=np.reshape(xis, (1, xi.size, 1)), spin=0, mval=mval, mstart=mstart, theta=theta,
                       lmax=lmax,  mode='STANDARD').squeeze().real
        sgn = 1 if s1 > 0 else (1 if abs(s1) % 2 == 0 else -1)
        return sgn * cl * fac
    else:
        xis = np.stack([xis, (1j * np.sign(s2)) * xis]).reshape((2, xi.size, 1))
        cl = leg2alm(leg=xis, spin=abs(s2), mval=mval, mstart=mstart, theta=theta, lmax=lmax,
                     mode='GRAD_ONLY').squeeze().real
        sgn = -1 if s2 > 0 else (-1 if abs(s2) % 2 == 0 else 1)
        return sgn * cl * fac


def get_thgwg(npts: int):
    """Gauss-Legendre integration points and weights from ducc0. Very fast.

        Args:
            number of points of quadrature rule

        Returns:
            tht: co-latitude points (array of size npts)
            wg: quadrature weights (array of size npts)


    """
    tht = GL_thetas(npts)
    wg = GL_weights(npts, 1) / (2 * np.pi)
    return tht, wg

def wignerc(cl1: np.ndarray[float or complex], cl2:np.ndarray[float or complex], sp1: int, s1: int, sp2: int, s2: int, lmax_out: int or None=None):
    """Spectrum of $ (\\xi_{sp1,s1} * \\xi_{sp2,s2})(\\cos \\theta)$ from their harmonic series.

        Uses Gauss-Legendre quadrature to solve this exactly.

    """
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmax_out = lmax1 + lmax2 if lmax_out is None else lmax_out
    lmaxtot = lmax1 + lmax2 + lmax_out
    spo = sp1 + sp2
    so = s1 + s2
    if np.any(cl1) and np.any(cl2):
        npts = (lmaxtot + 2 - lmaxtot % 2) // 2
        if not 'tht wg %s' % npts in GL_cache.keys():
            GL_cache['tht wg %s' % npts] = get_thgwg(npts)
        tht, wg = GL_cache['tht wg %s' % npts]
        if np.iscomplexobj(cl1):
            xi1 = wignerpos(np.real(cl1), tht, sp1, s1) + 1j * wignerpos(np.imag(cl1), tht, sp1, s1)
        else:
            xi1 = wignerpos(cl1, tht, sp1, s1)
        if np.iscomplexobj(cl2):
            xi2 = wignerpos(np.real(cl2), tht, sp2, s2) + 1j * wignerpos(np.imag(cl2), tht, sp2, s2)
        else:
            xi2 = wignerpos(cl2, tht, sp2, s2)
        xi1xi2w = xi1 * xi2 * wg
        if np.iscomplexobj(xi1xi2w):
            ret = wignercoeff(np.real(xi1xi2w), tht, spo, so, lmax_out)
            ret = ret + 1j * wignercoeff(np.imag(xi1xi2w), tht, spo, so, lmax_out)
            return ret
        else:
            return wignercoeff(xi1xi2w, tht, spo, so, lmax_out)

    else:
        return np.zeros(lmax_out + 1, dtype=float)