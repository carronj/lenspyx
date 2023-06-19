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

        Note:
            You can use *wigner4pos* instead if you also need the result for -s2 (e.g. for :math:`\xi_{\pm}`)

    """
    return wigner4pos(cl, None, theta, s1, s2)[0 if s2 >= 0 else 1]


def wigner4pos(gl: np.ndarray[float], cl: np.ndarray[float] or None, theta: np.ndarray[float], s1: int, s2: int):
    r"""Compute 4 Wigner correlation functions in one go

        Args:
            gl: first spectrum
            cl: second spectrum (can be set to None if irrelevant,  and is ignored if s2 is zero)
            theta: co-latitude in radians
            s1: int
            s2: int

        Returns:

            In the most general case, an array of shape (ncomp, ntheta) with:

            :math:`\sum_l g_l \frac{2l + 1}{4\pi} d^l_{s1, |s2|}(\theta)`
            :math:`\sum_l g_l \frac{2l + 1}{4\pi} d^l_{s1,-|s2|}(\theta)`
            :math:`\sum_l c_l \frac{2l + 1}{4\pi} d^l_{s1, |s2|}(\theta)`
            :math:`\sum_l c_l \frac{2l + 1}{4\pi} d^l_{s1,-|s2|}(\theta)`

            The number of components ncomp in the output is 4 if (s2 !=0 and cl is not None) else (2 if s2 else 1)


    """


    standard = cl is not None
    if standard:
        lmax = (max(len(cl), len(gl)) if s2 else len(cl)) - 1
        mode = 'STANDARD'
        nout = 4 if s2 else 1
    else:
        lmax = len(gl) - 1
        mode = 'GRAD_ONLY' if s2 else 'STANDARD'
        nout = 2 if s2 else 1

    if s1 == 0 and s2:  # Always prefer a faster spin-0 sht
        sgn_s = 1 if s2 % 2 == 0 else -1
        wig_g = wigner4pos(gl, None, theta, abs(s2), 0)[0]
        if standard:
            wig_c = wigner4pos(cl, None, theta, abs(s2), 0)[0]
            return np.stack([wig_g * sgn_s, wig_g, sgn_s * wig_c, wig_c])
        return np.stack([wig_g * sgn_s, wig_g])
    s1_pos = s1 >= 0
    sgn_s1 = 1 if s1_pos else (1 if (s1 + s2) % 2 == 0 else -1)
    ncomp = 1 + (s2 != 0) * standard

    gclm = np.empty((ncomp, lmax + 1), dtype=complex)
    prefac = np.sqrt(np.arange(1, 2 * lmax + 3, 2)) * (sgn_s1 / np.sqrt(4 * np.pi))
    gclm[0, :len(gl)] = prefac * gl[:len(gl)]
    if s2 and standard:
        gclm[1, :len(cl)] = prefac * cl[:len(cl)]
    leg = alm2leg(alm=gclm, spin=abs(s2), lmax=lmax, mval=np.array([abs(s1)], dtype=int),
                  mstart=np.array([0]), theta=theta, mode=mode).squeeze()
    wig = np.zeros((nout, theta.size), float)
    if s2:
        s_sgn = (1 if s2 % 2 == 0 else -1)
        wig[0 if s1_pos else 1] = -(leg[0].real + leg[1].imag)
        wig[1 if s1_pos else 0] = -s_sgn * (leg[0].real - leg[1].imag)
        if standard:
            wig[2 if s1_pos else 3] = -(leg[1].real - leg[0].imag)
            wig[3 if s1_pos else 2] = -s_sgn * (leg[1].real + leg[0].imag)
        return wig
    else:
        wig[0] = leg.real
        return wig


def wignerd(l: int, s1: int, s2: int, theta: np.ndarray):
    r"""Returns Wigner small-d functions

        :math:`(2l + 1) / (4pi) d^l_{s1, |s2|}(\theta)`
        :math:`(2l + 1) / (4pi) d^l_{s1,-|s2|}(\theta)`

        (or one of these if s2 is zero)

    """
    gl = np.zeros(l + 1)
    gl[-1] = 1.
    return wigner4pos(gl, None, theta, s1, s2)


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
    lmin = max(abs(s2), abs(s1))
    if s2 == 0:
        cl = leg2alm(leg=np.reshape(xis, (1, xi.size, 1)), spin=0, mval=mval, mstart=mstart, theta=theta,
                       lmax=lmax,  mode='STANDARD').squeeze().real
        sgn = 1 if s1 > 0 else (1 if abs(s1) % 2 == 0 else -1)
        cl[:lmin] = 0.
        return sgn * cl * fac
    else:
        xis = np.stack([xis, (1j * np.sign(s2)) * xis]).reshape((2, xi.size, 1))
        cl = leg2alm(leg=xis, spin=abs(s2), mval=mval, mstart=mstart, theta=theta, lmax=lmax,
                     mode='GRAD_ONLY').squeeze().real
        sgn = -1 if s2 > 0 else (-1 if abs(s2) % 2 == 0 else 1)
        cl[:lmin] = 0.
        return sgn * cl * fac


def wignerc(cl1: np.ndarray[float or complex], cl2:np.ndarray[float or complex], s1: int, t1: int, s2: int, t2: int,
            lmax_out: int = -1):
    r"""Convolution of two Wigner small-d correlation function

        Returns spectrum of

            :math:`\xi_{s_1 t_1}(\mu) \xi_{s_2 t_2}(\mu)`

        where

            :math:`\xi_{s_1 t_1}(\mu) = \sum_{\ell} C_{1,\ell} \frac{2\ell + 1}{4\pi} d^\ell_{s_1 t_1}`
            :math:`\xi_{s_2 t_2}(\mu) = \sum_{\ell} C_{2,\ell} \frac{2\ell + 1}{4\pi} d^\ell_{s_2 t_2}`

        Gauss-Legendre quadrature is used to solve this exactly.

        Args:
            cl1: spectrum of first Wigner small-d function
            cl2: spectrum of second Wigner small-d function
            s1: first spin of first function
            t1: second spin of first function
            s2: first spin of second function
            t2: second spin of second function
            lmax_out(optional): Result is provided up to lmax. Defaults to len(cl1) + len(cl2) -2


    """
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmax_out = lmax1 + lmax2 if lmax_out < 0 else lmax_out
    lmax_tot = lmax1 + lmax2 + lmax_out
    so = s1 + s2
    to = t1 + t2
    if np.any(cl1) and np.any(cl2):
        npts = (lmax_tot + 2 - lmax_tot % 2) // 2
        if not 'tht wg %s' % npts in GL_cache.keys():
            GL_cache['tht wg %s' % npts] = get_thgwg(npts)
        tht, wg = GL_cache['tht wg %s' % npts]
        if np.iscomplexobj(cl1):
            xi1 = wignerpos(np.real(cl1), tht, s1, t1) + 1j * wignerpos(np.imag(cl1), tht, s1, t1)
        else:
            xi1 = wignerpos(cl1, tht, s1, t1)
        if np.iscomplexobj(cl2):
            xi2 = wignerpos(np.real(cl2), tht, s2, t2) + 1j * wignerpos(np.imag(cl2), tht, s2, t2)
        else:
            xi2 = wignerpos(cl2, tht, s2, t2)
        xi1xi2w = xi1 * xi2 * wg
        if np.iscomplexobj(xi1xi2w):
            ret = wignercoeff(np.real(xi1xi2w), tht, so, to, lmax_out)
            ret = ret + 1j * wignercoeff(np.imag(xi1xi2w), tht, so, to, lmax_out)
            return ret
        else:
            return wignercoeff(xi1xi2w, tht, so, to, lmax_out)
    else:
        return np.zeros(lmax_out + 1, dtype=float)


def wignerdl(s1: int, s2: int, theta: float, lmax: int):
    """Returns d^l_{s1s2}(theta) for all l from 0 to lmax

    """
    assert np.isscalar(theta), 'scalar theta input here'
    return wignercoeff(np.array([1.]), np.array([theta]), s1, s2, lmax) / (2 * np.pi)


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


def get_xgwg(a: float, b: float, npts: int):
    """Gauss-Legendre points and weights for GL integration over an interval [a, b]

        Returns:
            xg: points within (a,b)
            wg: quadrature weights

    """
    tht = GL_thetas(npts)
    wg = GL_weights(npts, 1) / (2 * np.pi)
    c = 0.5 * (a + b)
    d = 0.5 * (b - a)
    return (c + np.cos(tht) * d)[::-1], (wg * d)[::-1]
