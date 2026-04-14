"""Experimental spherical harmonic transforms with optimizations for spherical caps.

This module provides wrapper functions and classes for spherical harmonic transforms (SHTs)
that automatically select optimized implementations based on the geometry of the data.

Key Features
------------
- Automatic selection between general SHT and optimized capped SHT
- Unified interface for different pixelization schemes (HEALPix, arbitrary locations)
- Support for both synthesis (alm → map) and adjoint synthesis (map → alm)
- Optimized implementations for data confined to spherical caps

The module wraps functions from ducc0 and capsht, providing:
- :func:`synthesis_general` : Forward transform with automatic cap detection
- :func:`adjoint_synthesis_general` : Adjoint transform with automatic cap detection
- :class:`Locations` : Unified interface for pixel locations

Notes
-----
This module requires capsht to be installed for the optimized capped transforms.
If capsht is not available, the functions will fall back to standard ducc0 implementations.

References
----------
.. [1] Carron, J. and Reinecke, M., 2026. "Fast partial-sky spherical harmonic transforms."
       arXiv:2603.17166. https://arxiv.org/abs/2603.17166
.. [2] Reinecke, M., Belkner, S., and Carron, J., 2023. "Improved cosmic microwave background
       (de-)lensing using general spherical harmonic transforms."
       arXiv:2304.10431. https://arxiv.org/abs/2304.10431

See Also
--------
ducc0.sht : Standard spherical harmonic transforms
capsht : Capped spherical harmonic transforms library

"""

import numpy as np
import ducc0
try:
    import capsht
except ImportError:
    print("capsht not found, you will not be able to use the functions in this module")
try:
    from capsht.experimental import synthesis_general_cap, synthesis_general_band, adjoint_synthesis_general_cap, adjoint_synthesis_general_band
except ImportError:
    print("synthesis_general_cap or synthesis_general_band not found in capsht.experimental, are you up to date?")
rtype = {np.dtype(np.complex128):np.dtype(np.float64), np.dtype(np.complex64):np.dtype(np.float32)}
ctype = {rtype[ctyp]:ctyp for ctyp in rtype}


def _epsapo(thtcap, epsilon, lmax, version=1, dl_7=None):
    r"""Compute the apodization parameter for capped spherical harmonic transforms.

    The apodization parameter determines the extended region beyond the cap radius
    where the transform uses a smooth window function to minimize edge effects.

    Parameters
    ----------
    thtcap : float
        Cap radius (colatitude) in radians
    epsilon : float
        Target accuracy for the transform (e.g., 1e-7)
    lmax : int
        Maximum multipole moment
    version : int, optional
        Algorithm version (default: 1). Only version 1 is currently supported.
    dl_7 : float, optional
        Scaling parameter. If None, uses default value based on version.

    Returns
    -------
    float
        Apodization parameter eps_apo, defining the fractional extension:
        tht_max = thtcap * (1 + eps_apo)

    Notes
    -----
    The apodization parameter is computed as:

    .. math::

        \varepsilon_{\text{apo}} = \sqrt{\frac{\Delta \ell}{\ell_{\max}} \frac{\pi}{\theta_{\text{cap}}}}

    where :math:`\Delta \ell` depends on the target accuracy epsilon.

    This ensures the transform achieves the requested accuracy while minimizing
    computational cost.

    """
    #dl = dl_7 * ((- np.log10(epsilon) + 1) / (7 + 1)) ** 2
    assert version == 1, 'C++ code now only implemented for version 1'
    if version == 0:
        if dl_7 is None:
            dl_7 = 15
        dl = dl_7 * ((-np.log10(epsilon) / (7 )) ** 1) ** 0.5
    elif version == 1:
        if dl_7 is None:
            dl_7 = 2*7*np.log(10.)/np.pi
        dl = dl_7 * (-np.log10(epsilon) / (7. ))
    else:
        raise ValueError('version %s not implemented'%version)
    return np.sqrt(dl / lmax * np.pi / thtcap)

def synthesis_general(alm: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float,
                      thtcap:float=None, eps_apo:float=None, tht_min:float=None, tht_max:float=None, verbose:bool=False, **kwargs):
    r"""Spherical harmonic synthesis with automatic optimization for spherical caps.

    This function automatically selects the most efficient implementation based on
    the geometry of the target locations. If data is confined to a spherical cap,
    it uses an optimized capped transform; otherwise it falls back to the general transform.

    Parameters
    ----------
    alm : array_like
        Spherical harmonic coefficients. Shape (ncomp, nalm) where ncomp is 1 for
        spin-0 fields or 2 for spin-s fields with s > 0.
    spin : int
        Spin weight of the field (0 for temperature/scalar, ±2 for polarization)
    lmax : int
        Maximum multipole moment
    loc : array_like
        Evaluation locations, shape (npix, 2) with (colatitude, longitude) in radians
    epsilon : float
        Target accuracy (e.g., 1e-7 for typical CMB applications)
    thtcap : float, optional
        Cap radius in radians. If provided, assumes all locations lie within this cap
        and uses optimized capped transform. If None, attempts automatic detection.
    eps_apo : float, optional
        Apodization parameter. If None, computed automatically from epsilon and lmax.
    tht_min : float, optional
        Minimum colatitude in the data (currently experimental)
    tht_max : float, optional
        Maximum colatitude in the data (currently experimental)
    verbose : bool, optional
        Print diagnostic information about which transform is used
    **kwargs : dict
        Additional keyword arguments passed to the underlying transform:

        - mode : str, optional (e.g., 'STANDARD', 'GRAD_ONLY')
        - map : array_like, optional (pre-allocated output array)
        - mmax : int, optional (maximum azimuthal mode number)
        - nthreads : int, optional (number of threads)

    Returns
    -------
    array_like
        Synthesized map(s). Shape (ncomp, npix) where ncomp matches the input alm.

    Notes
    -----
    The function automatically determines which implementation to use:

    - If ``thtcap`` is provided and all locations satisfy :math:`\theta \leq \theta_{\text{cap}}`,
      uses :func:`capsht.synthesis_general_cap` for optimal performance
    - Otherwise, uses :func:`ducc0.sht.synthesis_general` for arbitrary locations

    The capped transform can provide significant speedups for data
    confined to caps covering less than half the sky.

    Examples
    --------
    >>> import numpy as np
    >>> from lenspyx.experimental import synthesis_general
    >>> # Synthesize on arbitrary locations
    >>> alm = np.random.randn(1, nalm) + 1j * np.random.randn(1, nalm)
    >>> loc = np.random.rand(1000, 2) * [np.pi, 2*np.pi]  # Random locations
    >>> maps = synthesis_general(alm, spin=0, lmax=100, loc=loc, epsilon=1e-7)

    >>> # Use capped transform for polar cap
    >>> theta_cap = 30 * np.pi / 180  # 30 degree cap
    >>> loc_cap = np.random.rand(1000, 2) * [theta_cap, 2*np.pi]
    >>> maps_cap = synthesis_general(alm, spin=0, lmax=100, loc=loc_cap,
    ...                              epsilon=1e-7, thtcap=theta_cap, verbose=True)

    See Also
    --------
    adjoint_synthesis_general : Adjoint operation (map → alm)
    ducc0.sht.synthesis_general : Underlying general transform
    capsht.synthesis_general_cap : Underlying capped transform

    """
    if False and tht_min is not None and tht_max is not None: 
        # Update synthesis_general_band first
        eps_apo = eps_apo or 1.2 * _epsapo(tht_max-tht_min, epsilon, lmax)
        thta_p = tht_min - 0.5 * eps_apo * (tht_max - tht_min)
        thtb_p = tht_max + 0.5 * eps_apo * (tht_max - tht_min)
        if (thta_p >= 0.) and (thtb_p <= np.pi):
            if verbose:
                print('syng type: band %.1f deg %.1f deg' % (thta_p/np.pi*180, thtb_p/np.pi*180))
            assert 0, 'fix band to new scheme'
            return synthesis_general_band(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, 
                                      thta=tht_min, thtb=tht_max, eps_apo=eps_apo, **kwargs)
        if thta_p < 0.: # Can try synthesis_general_cap later on
            thtcap = tht_max
            eps_apo = None
    if thtcap is not None: # attempt at synthesis_general_cap
        eps_apo = eps_apo or _epsapo(thtcap, epsilon, lmax)
        epsilon_nufft = kwargs.pop('epsilon_nufft', epsilon)
        if verbose:
            print('syng type: sent to cap %.1f epsapo %.2f' % (thtcap/np.pi*180, eps_apo))
        return synthesis_general_cap(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon_nufft, thtcap=thtcap, eps_apo=eps_apo, **kwargs)
    if verbose:
        print('syng type : general')
    return ducc0.sht.synthesis_general(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon,  **kwargs)

def adjoint_synthesis_general(map: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float,
                      thtcap:float=None, eps_apo:float=None, tht_min:float=None, tht_max:float=None, verbose:bool=False, **kwargs):
    r"""Adjoint spherical harmonic synthesis with automatic optimization for spherical caps.

    This is the adjoint operation to :func:`synthesis_general`, computing spherical harmonic
    coefficients from map data. It automatically selects the most efficient implementation
    based on the geometry.

    Parameters
    ----------
    map : array_like
        Input map(s). Shape (ncomp, npix) where ncomp is 1 for spin-0 or 2 for spin-s.
    spin : int
        Spin weight of the field (0 for temperature/scalar, ±2 for polarization)
    lmax : int
        Maximum multipole moment
    loc : array_like
        Map pixel locations, shape (npix, 2) with (colatitude, longitude) in radians
    epsilon : float
        Target accuracy (e.g., 1e-7 for typical CMB applications)
    thtcap : float, optional
        Cap radius in radians. If provided, uses optimized capped transform.
    eps_apo : float, optional
        Apodization parameter. If None, computed automatically.
    tht_min : float, optional
        Minimum colatitude in the data (currently experimental)
    tht_max : float, optional
        Maximum colatitude in the data (currently experimental)
    verbose : bool, optional
        Print diagnostic information about which transform is used
    **kwargs : dict
        Additional keyword arguments:

        - mode : str, optional (e.g., 'STANDARD', 'GRAD_ONLY')
        - alm : array_like, optional (pre-allocated output array)
        - mmax : int, optional (maximum azimuthal mode number)
        - nthreads : int, optional (number of threads)

    Returns
    -------
    array_like
        Spherical harmonic coefficients. Shape (ncomp, nalm) where nalm depends on lmax and mmax.

    Notes
    -----
    This function computes the adjoint operation (not the inverse!) of the synthesis transform.
    For a forward transform :math:`\mathbf{m} = \mathbf{S} \mathbf{a}`, this computes
    :math:`\mathbf{a}' = \mathbf{S}^{\dagger} \mathbf{m}` where :math:`\dagger` denotes
    the adjoint (conjugate transpose).

    The adjoint is used in:

    - Quadratic estimators for CMB lensing and other fields
    - Iterative map-making and Wiener filtering
    - Maximum likelihood power spectrum estimation

    Like :func:`synthesis_general`, this automatically selects between general and
    capped implementations based on the data geometry.

    Examples
    --------
    >>> import numpy as np
    >>> from lenspyx.experimental import adjoint_synthesis_general
    >>> from lenspyx.utils_hp import Alm
    >>> # Compute alm from map on arbitrary locations
    >>> maps = np.random.randn(1, 1000)  # Random map
    >>> loc = np.random.rand(1000, 2) * [np.pi, 2*np.pi]
    >>> lmax = 100
    >>> alm = np.zeros((1, Alm.getsize(lmax, lmax)), dtype=complex)
    >>> alm = adjoint_synthesis_general(maps, spin=0, lmax=lmax, loc=loc,
    ...                                 epsilon=1e-7, alm=alm)

    See Also
    --------
    synthesis_general : Forward transform (alm → map)
    ducc0.sht.adjoint_synthesis_general : Underlying general adjoint transform
    capsht.adjoint_synthesis_general_cap : Underlying capped adjoint transform

    """
    if False and tht_min is not None and tht_max is not None: 
        # Update synthesis_general_band first
        eps_apo = eps_apo or 1.2 * _epsapo(tht_max-tht_min, epsilon, lmax)
        thta_p = tht_min - 0.5 * eps_apo * (tht_max - tht_min)
        thtb_p = tht_max + 0.5 * eps_apo * (tht_max - tht_min)
        if (thta_p >= 0.) and (thtb_p <= np.pi):
            if verbose:
                print('adjsyng type: band %.1f deg %.1f deg' % (thta_p/np.pi*180, thtb_p/np.pi*180))
            return adjoint_synthesis_general_band(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, 
                                      thta=tht_min, thtb=tht_max, eps_apo=eps_apo, **kwargs)
        if thta_p < 0.: # Can try synthesis_general_cap later on
            thtcap = tht_max
            eps_apo = None
    if thtcap is not None: # attempt at synthesis_general_cap
        eps_apo = eps_apo or _epsapo(thtcap, epsilon, lmax)
        if verbose:
            print('adjsyng type: sent to cap %.1f deg, eps_apo %.2f' % (thtcap/np.pi*180, eps_apo))
        return adjoint_synthesis_general_cap(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, thtcap=thtcap, eps_apo=eps_apo, **kwargs)
    if verbose:
        print('adjsyng type : general')
    return ducc0.sht.adjoint_synthesis_general(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon,  **kwargs)



class Locations(object):
    """Unified interface for locations (or pixelizations) in spherical harmonic transforms.

    This class provides a common API for working with both structured (iso-latitude rings)
    and unstructured (arbitrary locations) pixelizations. It automatically selects the
    appropriate transform implementation based on the location type.

    Parameters
    ----------
    loc : array_like, optional
        Arbitrary pixel locations, shape (npix, 2) with (colatitude, longitude) in radians.
        Use this for unstructured grids or partial sky coverage.
    geom : lenspyx.remapping.utils_geom.Geom, optional
        Iso-latitude ring geometry (e.g., HEALPix, Gauss-Legendre).
        Use this for structured, symmetric pixelizations.
    epsilon : float, optional
        Target accuracy for transforms (default: 1e-7). Only relevant when using `loc`.

    Notes
    -----
    Exactly one of `loc` or `geom` must be provided.

    This class serves as a wrapper that:

    - Provides consistent :meth:`synthesis` and :meth:`adjoint_synthesis` methods
    - Automatically routes to optimized implementations (capped vs general, structured vs unstructured)
    - Handles differences in calling conventions between backends

    Attributes
    ----------
    loc : array_like or None
        Pixel locations if using unstructured grid
    geom : Geom or None
        Geometry object if using structured grid
    thtrange : tuple
        (min_theta, max_theta) colatitude range
    epsilon : float
        Transform accuracy parameter

    Examples
    --------
    Using arbitrary locations:

    >>> import numpy as np
    >>> from lenspyx.experimental import Locations
    >>> # Create random pixel locations in a polar cap
    >>> npix = 10000
    >>> loc = np.random.rand(npix, 2) * [np.pi/4, 2*np.pi]
    >>> locs = Locations(loc=loc, epsilon=1e-7)
    >>> # Synthesize a map
    >>> alm = np.random.randn(1, nalm) + 1j * np.random.randn(1, nalm)
    >>> maps = locs.synthesis(alm, spin=0, lmax=100, mmax=100, nthreads=4)

    Using HEALPix geometry:

    >>> from lenspyx.remapping.utils_geom import Geom
    >>> geom = Geom.get_healpix_geometry(nside=512)
    >>> locs = Locations(geom=geom)
    >>> maps = locs.synthesis(alm, spin=0, lmax=1500, mmax=1500, nthreads=4)

    See Also
    --------
    lenspyx.remapping.utils_geom.Geom : Geometry objects for structured grids
    synthesis_general : Underlying synthesis function
    adjoint_synthesis_general : Underlying adjoint synthesis function

    """
    def __init__(self, loc:np.ndarray=None, geom=None, epsilon=1e-7):
        assert ((loc is None) + (geom is None)) == 1, 'one and only one of loc or geom can be set'
     

        if loc is not None:
            thtmin = np.min(loc[:, 0])
            thtmax = np.max(loc[:, 0])
            npix = loc.shape[0]
        else:
            thtmin = np.min(geom.theta)
            thtmax = np.max(geom.theta)
            npix = geom.npix()

        self.loc = loc
        self.geom = geom
        self.thtrange = (thtmin, thtmax)
        self.epsilon = epsilon
        self._npix = npix


    def npix(self):
        return self._npix

    def synthesis(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray=None, **kwargs):
        """Compute spherical harmonic synthesis (alm → map).

        Transforms spherical harmonic coefficients to a map on the pixelization.

        Parameters
        ----------
        alm : array_like
            Spherical harmonic coefficients, shape (ncomp, nalm) where:

            - ncomp = 1 for spin-0 fields (temperature, scalar)
            - ncomp = 2 for spin-s fields with s > 0 (gradient and curl components)
        spin : int
            Spin weight of the field
        lmax : int
            Maximum multipole moment in the alm array
        mmax : int
            Maximum azimuthal mode number in the alm array
        nthreads : int
            Number of threads to use for the transform
        m : array_like, optional
            Pre-allocated output array, shape (ncomp, npix). If None, allocated automatically.
        **kwargs : dict
            Additional arguments passed to the underlying transform (e.g., mode='GRAD_ONLY')

        Returns
        -------
        array_like
            Map array of shape (ncomp, npix)

        Examples
        --------
        >>> locs = Locations(loc=my_locations)
        >>> maps = locs.synthesis(alm, spin=2, lmax=2000, mmax=2000, nthreads=8)

        """
        if m is None:
            m = np.empty((1 + (spin > 0), self.npix()), dtype=rtype[alm.dtype])
        if self.loc is not None:
            return self._synthesis_loc(alm, spin, lmax, mmax, nthreads, m, **kwargs)
        return self._synthesis_geom(alm, spin, lmax, mmax, nthreads, m, **kwargs)   

    def adjoint_synthesis(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray, **kwargs):
        r"""Compute adjoint spherical harmonic synthesis (map → alm).

        Transforms a map to spherical harmonic coefficients using the adjoint operation.
        This is NOT the inverse transform, but rather the conjugate transpose.

        Parameters
        ----------
        alm : array_like
            Pre-allocated output array for coefficients, shape (ncomp, nalm)
        spin : int
            Spin weight of the field
        lmax : int
            Maximum multipole moment
        mmax : int
            Maximum azimuthal mode number
        nthreads : int
            Number of threads to use for the transform
        m : array_like
            Input map array, shape (ncomp, npix) where:

            - ncomp = 1 for spin-0 fields
            - ncomp = 2 for spin-s fields with s > 0
        **kwargs : dict
            Additional arguments passed to the underlying transform

        Returns
        -------
        array_like
            Spherical harmonic coefficients, shape (ncomp, nalm)

        Notes
        -----
        For a forward transform :math:`\mathbf{m} = \mathbf{S} \mathbf{a}`, this computes
        :math:`\mathbf{a}' = \mathbf{S}^{\dagger} \mathbf{m}` where :math:`\dagger` is the
        adjoint (conjugate transpose), not the inverse.

        The adjoint is used in quadratic estimators, iterative algorithms, and
        maximum likelihood estimation.

        Examples
        --------
        >>> locs = Locations(loc=my_locations)
        >>> alm_out = np.zeros((1, nalm), dtype=complex)
        >>> alm_out = locs.adjoint_synthesis(alm_out, spin=0, lmax=2000, mmax=2000,
        ...                                  nthreads=8, m=input_map)

        """
        if self.loc is not None:
            return self._adjoint_synthesis_loc(alm, spin, lmax, mmax, nthreads, m, **kwargs)
        return self._adjoint_synthesis_geom(alm, spin, lmax, mmax, nthreads, m, **kwargs)   

    def _synthesis_geom(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray, **kwargs):
        # relevant kwargs here: mode.
        assert self.geom is not None, 'no isolatitude geometry set'
        return self.geom.synthesis(alm, spin, lmax, mmax, nthreads, map=m, **kwargs)

    def _adjoint_synthesis_geom(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray, **kwargs):
        # relevant kwargs here: mode.
        assert self.geom is not None, 'no isolatitude geometry set'
        return self.geom.adjoint_synthesis(np.atleast_2d(m), spin, lmax, mmax, nthreads, alm=np.atleast_2d(alm), apply_weights=False, **kwargs)

    def _synthesis_loc(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray, **kwargs):
        assert self.loc is not None, 'no locations set'
        # relevant kwargs here: epsilon, mode.
        syng_params = {'tht_max': self.thtrange[1], 'tht_min': self.thtrange[0]}
        return synthesis_general(map=m, lmax=lmax, mmax=mmax, alm=alm, loc=self.loc, spin=spin, nthreads=nthreads, epsilon=self.epsilon, **syng_params, **kwargs)

    def _adjoint_synthesis_loc(self, alm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, m:np.ndarray, **kwargs):
        assert self.loc is not None, 'no locations set'
        # relevant kwargs here: epsilon, mode.
        syng_params = {'tht_max': self.thtrange[1], 'tht_min': self.thtrange[0]}
        return adjoint_synthesis_general(map=np.atleast_2d(m),lmax=lmax, mmax=mmax, alm=alm, loc=self.loc, spin=spin, nthreads=nthreads, epsilon=self.epsilon, **syng_params, **kwargs)