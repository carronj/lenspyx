"""Wigner small-d functions and correlation functions for spin-weighted spherical harmonics.

This module provides efficient implementations of Wigner small-d functions and their associated
correlation functions, which are fundamental for analyzing spin-weighted fields on the sphere
(e.g., CMB polarization, gravitational lensing).

The implementation uses DUCC0's `alm2leg` and `leg2alm` functions, which were not originally
optimized for Wigner transforms but achieve excellent performance through compiler optimization.

Key Concepts
------------
Wigner small-d functions :math:`d^\\ell_{s_1 s_2}(\\theta)` are the angular parts of the Wigner D-matrices,
describing the rotation of spin-weighted spherical harmonics. They appear in:

- Correlation functions of spin-weighted fields (e.g., ξ±  for CMB polarization or galaxy surveys)
- Angular power spectrum estimators for CMB polarization
- Lensing reconstruction and delensing operations
- General spin transformations on the sphere

Main Functions
--------------
wignerpos : Compute Wigner correlation function from power spectrum
    Forward transform: C_ℓ → ξ(θ)

wignercoeff : Compute power spectrum from Wigner correlation function
    Adjoint transform: ξ(θ) → C_ℓ

wigner4pos : Compute 4 correlation functions simultaneously
    Efficient for computing ξ+ and ξ- together

wignerc : Convolve two Wigner correlation functions
    Used for computing products of correlation functions

wignerd : Single Wigner d-function
    Returns d^ℓ_{s1,s2}(θ) for a specific ℓ

Utility Functions
-----------------
get_thgwg : Gauss-Legendre quadrature points and weights
    For integration over [0, π]

get_xgwg : Gauss-Legendre quadrature over arbitrary interval [a, b]

Examples
--------
Compute CMB E-mode correlation function:

>>> import numpy as np
>>> from lenspyx.wigners import wignerpos
>>> # E-mode power spectrum
>>> cl_ee = np.loadtxt('cl_ee.txt')
>>> theta = np.linspace(0, np.pi, 100)
>>> # Compute ξ_EE(θ) = Σ_ℓ (2ℓ+1)/(4π) C_ℓ^EE d^ℓ_{2,2}(θ)
>>> xi_ee = wignerpos(cl_ee, theta, s1=2, s2=2)

Compute ξ± correlation functions:

>>> from lenspyx.wigners import wigner4pos
>>> # Returns [ξ+, ξ-] where:
>>> # ξ+ = Σ_ℓ (2ℓ+1)/(4π) C_ℓ d^ℓ_{2,+2}(θ)
>>> # ξ- = Σ_ℓ (2ℓ+1)/(4π) C_ℓ d^ℓ_{2,-2}(θ)
>>> xi_plus, xi_minus = wigner4pos(cl_ee, None, theta, s1=2, s2=2)

Invert correlation function to get power spectrum:

>>> from lenspyx.wigners import wignercoeff, get_thgwg
>>> lmax = 2000
>>> npts = (lmax + 2) // 2
>>> theta, weights = get_thgwg(npts)
>>> # Measure ξ(θ) from data
>>> xi_measured = measure_correlation(theta)
>>> # Compute C_ℓ = 2π Σ_θ ξ(θ) d^ℓ_{s1,s2}(θ)
>>> cl_recovered = wignercoeff(xi_measured, theta, s1=2, s2=2, lmax=lmax)

Convolve two correlation functions:

>>> from lenspyx.wigners import wignerc
>>> cl1 = cl_ee  # E-mode spectrum
>>> cl2 = cl_bb  # B-mode spectrum
>>> # Compute spectrum of ξ_EE(θ) * ξ_BB(θ)
>>> cl_product = wignerc(cl1, cl2, s1=2, t1=2, s2=2, t2=2)

Notes
-----
- Spin values s1, s2 can be positive, negative, or zero
- The normalization includes (2ℓ+1)/(4π) factors
- Gauss-Legendre quadrature ensures exact integration for band-limited functions
- Internal caching of GL points improves performance for repeated calls

References
----------
.. [1] Varshalovich, D.A., Moskalev, A.N. and Khersonskii, V.K., 1988.
       Quantum theory of angular momentum. World Scientific.
.. [2] Hivon, E., et al., 2002. HEALPix: A Framework for High-Resolution
       Discretization and Fast Analysis of Data Distributed on the Sphere.
       ApJ, 567, 2.
.. [3] Reinecke, M. and Seljebotn, D.S., 2013. Libsharp–spherical harmonic
       transforms revisited. A&A, 554, A112.

"""
from __future__ import annotations
import numpy as np
from ducc0.sht import alm2leg, leg2alm
from ducc0.misc import GL_thetas, GL_weights

GL_cache = {}
verbose = False


def wignerpos(cl: np.ndarray[float], theta: np.ndarray[float], s1: int, s2: int):
    r"""Produces Wigner small-d transform defined by

    .. math::

        \sum_\ell \frac{2\ell + 1}{4\pi} C_\ell d^\ell_{s_1 s_2}(\theta)

    Parameters
    ----------
    cl : array_like
        Spectrum of Wigner small-d transform (power spectrum :math:`C_\ell`)
    theta : array_like
        Co-latitude in radians (in [0, π])
    s1 : int
        First spin weight
    s2 : int
        Second spin weight

    Returns
    -------
    array_like
        Real array of same size as theta containing the correlation function

    Notes
    -----
    You can use :func:`wigner4pos` instead if you also need the result for -s2
    (e.g., for computing :math:`\xi_{\pm}` simultaneously).

    See Also
    --------
    wigner4pos : Compute 4 correlation functions in one go
    wignercoeff : Adjoint transform (correlation function to spectrum)

    """
    return wigner4pos(cl, None, theta, s1, s2)[0 if s2 >= 0 else 1]


def wigner4pos(gl: np.ndarray[float], cl: np.ndarray[float] or None, theta: np.ndarray[float], s1: int, s2: int):
    r"""Compute 4 Wigner correlation functions in one go.

    Parameters
    ----------
    gl : array_like
        First spectrum (power spectrum :math:`g_\ell`)
    cl : array_like or None
        Second spectrum (power spectrum :math:`c_\ell`). Can be set to None if irrelevant,
        and is ignored if s2 is zero.
    theta : array_like
        Co-latitude in radians (in [0, π])
    s1 : int
        First spin weight
    s2 : int
        Second spin weight

    Returns
    -------
    array_like
        In the most general case, an array of shape (ncomp, ntheta) with:

        - Component 0: :math:`\sum_\ell g_\ell \frac{2\ell + 1}{4\pi} d^\ell_{s_1, |s_2|}(\theta)`
        - Component 1: :math:`\sum_\ell g_\ell \frac{2\ell + 1}{4\pi} d^\ell_{s_1,-|s_2|}(\theta)`
        - Component 2: :math:`\sum_\ell c_\ell \frac{2\ell + 1}{4\pi} d^\ell_{s_1, |s_2|}(\theta)` (if cl is not None)
        - Component 3: :math:`\sum_\ell c_\ell \frac{2\ell + 1}{4\pi} d^\ell_{s_1,-|s_2|}(\theta)` (if cl is not None)

        The number of components ncomp in the output is:

        - 4 if (s2 ≠ 0 and cl is not None)
        - 2 if (s2 ≠ 0 and cl is None)
        - 1 if s2 = 0

    Notes
    -----
    This function is more efficient than calling :func:`wignerpos` multiple times when
    you need correlation functions for both +s2 and -s2 (e.g., ξ+ and ξ-).

    See Also
    --------
    wignerpos : Compute single correlation function

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
    r"""Returns Wigner small-d functions for a specific multipole.

    Computes the normalized Wigner d-functions:

    .. math::

        \frac{2\ell + 1}{4\pi} d^\ell_{s_1, |s_2|}(\theta)

    and

    .. math::

        \frac{2\ell + 1}{4\pi} d^\ell_{s_1,-|s_2|}(\theta)

    for all θ. If s2 is zero, only the first component is returned.

    Parameters
    ----------
    l : int
        Multipole moment
    s1 : int
        First spin weight
    s2 : int
        Second spin weight
    theta : array_like
        Co-latitude angles in radians

    Returns
    -------
    array_like
        Array of shape (2, ntheta) if s2 ≠ 0, or (1, ntheta) if s2 = 0

    See Also
    --------
    wignerdl : Returns d^l_{s1,s2}(θ) without normalization for all ℓ

    """
    gl = np.zeros(l + 1)
    gl[-1] = 1.
    return wigner4pos(gl, None, theta, s1, s2)


def wignercoeff(xi: np.ndarray[float], theta: np.ndarray[float], s1: int, s2: int, lmax: int):
    r"""Computes spectrum of Wigner small-d correlation function (adjoint to wignerpos).

    This is the adjoint transform that converts a correlation function to its
    power spectrum:

    .. math::

        C_\ell = 2\pi \sum_\theta \xi(\theta) d^\ell_{s_1 s_2}(\theta)

    Parameters
    ----------
    xi : array_like
        Wigner correlation function (real array, one value per co-latitude)
    theta : array_like
        Co-latitude in radians (in [0, π])
    s1 : int
        First spin weight
    s2 : int
        Second spin weight
    lmax : int
        Maximum multipole (inclusive). Spectrum is calculated from 0 to lmax.

    Returns
    -------
    array_like
        Power spectrum :math:`C_\ell` from ℓ=0 to ℓ=lmax

    Notes
    -----
    This function implements the adjoint operation to :func:`wignerpos`. It is used
    to estimate power spectra from measured correlation functions.

    See Also
    --------
    wignerpos : Forward transform (spectrum to correlation function)

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
    r"""Convolution of two Wigner small-d correlation functions.

    Computes the power spectrum of the product of two correlation functions:

    .. math::

        \xi_{s_1 t_1}(\mu) \times \xi_{s_2 t_2}(\mu)

    where

    .. math::

        \xi_{s_1 t_1}(\mu) = \sum_{\ell} C_{1,\ell} \frac{2\ell + 1}{4\pi} d^\ell_{s_1 t_1}(\mu)

    .. math::

        \xi_{s_2 t_2}(\mu) = \sum_{\ell} C_{2,\ell} \frac{2\ell + 1}{4\pi} d^\ell_{s_2 t_2}(\mu)

    Gauss-Legendre quadrature is used to solve this exactly (up to numerical precision).

    Parameters
    ----------
    cl1 : array_like
        Spectrum of first Wigner small-d function (:math:`C_{1,\ell}`)
    cl2 : array_like
        Spectrum of second Wigner small-d function (:math:`C_{2,\ell}`)
    s1 : int
        First spin of first function
    t1 : int
        Second spin of first function
    s2 : int
        First spin of second function
    t2 : int
        Second spin of second function
    lmax_out : int, optional
        Maximum multipole of output spectrum. Defaults to len(cl1) + len(cl2) - 2.

    Returns
    -------
    array_like
        Power spectrum of the product correlation function, from ℓ=0 to ℓ=lmax_out

    Notes
    -----
    This function is useful for computing non-Gaussian covariances and higher-order statistics.
    The output has spins (s1+s2, t1+t2).

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
    r"""Returns the Wigner d-function for all multipoles.

    Computes the Wigner d-function:

    .. math::

        d^\ell_{s_1 s_2}(\theta)

    for all ℓ from 0 to lmax.

    Parameters
    ----------
    s1 : int
        First spin weight
    s2 : int
        Second spin weight
    theta : float
        Co-latitude angle in radians (scalar)
    lmax : int
        Maximum multipole moment

    Returns
    -------
    array_like
        Array of size lmax+1 containing :math:`d^\ell_{s_1 s_2}(\theta)` for ℓ=0 to lmax

    Notes
    -----
    This returns the unnormalized Wigner d-function, without the (2ℓ+1)/(4π) factor.
    For the normalized version, use :func:`wignerd`.

    See Also
    --------
    wignerd : Normalized Wigner d-function for a specific ℓ

    """
    assert np.isscalar(theta), 'scalar theta input here'
    return wignercoeff(np.array([1.]), np.array([theta]), s1, s2, lmax) / (2 * np.pi)


def get_thgwg(npts: int):
    """Gauss-Legendre integration points and weights from DUCC0.

    Provides quadrature points and weights for integration over the interval [0, π],
    optimized for integrating functions on the sphere.

    Parameters
    ----------
    npts : int
        Number of quadrature points

    Returns
    -------
    tht : array_like
        Co-latitude points in radians (array of size npts)
    wg : array_like
        Quadrature weights (array of size npts)

    Notes
    -----
    This uses DUCC0's highly optimized Gauss-Legendre quadrature implementation.
    For a band-limited function up to ℓ_max, use npts ≥ (ℓ_max + 2) / 2 for
    exact integration.

    Examples
    --------
    >>> from lenspyx.wigners import get_thgwg
    >>> theta, weights = get_thgwg(100)
    >>> # Integrate a function f(θ) over the sphere:
    >>> integral = np.sum(f(theta) * weights)

    """
    tht = GL_thetas(npts)
    wg = GL_weights(npts, 1) / (2 * np.pi)
    return tht, wg


def get_xgwg(a: float, b: float, npts: int):
    """Gauss-Legendre points and weights for integration over an arbitrary interval.

    Provides quadrature points and weights for integration over the interval [a, b].

    Parameters
    ----------
    a : float
        Lower bound of integration interval
    b : float
        Upper bound of integration interval
    npts : int
        Number of quadrature points

    Returns
    -------
    xg : array_like
        Quadrature points within (a, b) (array of size npts)
    wg : array_like
        Quadrature weights (array of size npts)

    Notes
    -----
    This function transforms the standard Gauss-Legendre quadrature on [-1, 1]
    to an arbitrary interval [a, b].

    Examples
    --------
    >>> from lenspyx.wigners import get_xgwg
    >>> x, weights = get_xgwg(0, 10, 50)
    >>> # Integrate a function f(x) over [0, 10]:
    >>> integral = np.sum(f(x) * weights)

    """
    tht = GL_thetas(npts)
    wg = GL_weights(npts, 1) / (2 * np.pi)
    c = 0.5 * (a + b)
    d = 0.5 * (b - a)
    return (c + np.cos(tht) * d)[::-1], (wg * d)[::-1]
