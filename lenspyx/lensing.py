"""CMB lensing operations: lensed map synthesis and related utilities.

This module provides functions for computing gravitationally lensed CMB maps from
spherical harmonic coefficients. Gravitational lensing remaps the CMB temperature
and polarization by deflecting photon paths according to the intervening mass distribution.

Key Functions
-------------
- :func:`alm2lenmap` : Compute lensed maps from unlensed alms and deflection field
- :func:`alm2lenmap_spin` : Compute lensed spin-weighted maps (general spin-s fields)
- :func:`dlm2angles` : Convert deflection alms to deflected angles
- :func:`synfast` : Generate lensed CMB realizations from power spectra

The lensing remapping is exact (non-perturbative) using interpolation on the sphere,
with configurable accuracy through the epsilon parameter.

Examples
--------
Generate a lensed CMB realization:

>>> from lenspyx.lensing import synfast
>>> from lenspyx.utils import get_ffp10_cls
>>> cls = get_ffp10_cls()
>>> maps = synfast(cls, lmax=2048, nside=2048)
>>> # maps contains 'T' and 'QU' keys with lensed maps

Lens unlensed alms:

>>> from lenspyx.lensing import alm2lenmap
>>> # alm_t, alm_e, alm_b: unlensed CMB alms
>>> # phi_lm: lensing potential alms
>>> import numpy as np
>>> dlm = utils_hp.almxfl(phi_lm, np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2)), None, False)
>>> t_lensed, q_lensed, u_lensed = alm2lenmap([alm_t, alm_e, alm_b], dlm)

References
----------
.. [1] Reinecke, M., Belkner, S., and Carron, J., 2023. "Improved cosmic microwave background
       (de-)lensing using general spherical harmonic transforms."
       arXiv:2304.10431. https://arxiv.org/abs/2304.10431

See Also
--------
lenspyx.remapping.deflection_029 : Low-level lensing implementation
ducc0.misc.get_deflected_angles : DUCC0 deflection angle computation

"""
from __future__ import print_function, annotations
from os import cpu_count
import numpy as np
from numpy.random import default_rng

from ducc0.misc import get_deflected_angles

from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping.deflection_029 import deflection
from lenspyx import cachers
from lenspyx.utils_hp import almxfl, Alm
from lenspyx.utils import timer


def get_geom(geometry: tuple[str, dict]=('healpix', {'nside':2048})):
    r"""Get sphere pixelization geometry instance from name and arguments.

    Parameters
    ----------
    geometry : tuple of (str, dict), optional
        Tuple containing (geometry_name, parameters_dict). Default: ('healpix', {'nside': 2048})

        Examples of supported geometries:

        - 'healpix' : HEALPix pixelization, parameters: {'nside': int}
        - 'thingauss' : Thin Gauss-Legendre rings, parameters: {'lmax': int, 'smax': int}
        - 'gl' : Gauss-Legendre, parameters: {'lmax': int}
        - Custom geometries following :class:`lenspyx.remapping.utils_geom.Geom`

    Returns
    -------
    Geom
        Geometry object with methods for synthesis, adjoint_synthesis, etc.

    Examples
    --------
    >>> from lenspyx.lensing import get_geom
    >>> # HEALPix geometry
    >>> geom = get_geom(('healpix', {'nside': 1024}))
    >>> # Gauss-Legendre geometry
    >>> geom = get_geom(('gl', {'lmax': 3000}))

    See Also
    --------
    lenspyx.remapping.utils_geom.Geom : Base geometry class

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1])


def dlm2angles(dlms:np.ndarray, geometry:Geom, mmax=None, nthreads: int=0, calc_rotation=False):
    r"""Convert lensing deflection alms to deflected angles.

    Computes the deflected pointing directions (and optionally rotation angles) from
    the deflection field spherical harmonic coefficients.

    Parameters
    ----------
    dlms : array_like
        Spin-1 deflection field in harmonic space. Can be:

        - Single array: gradient-only deflection :math:`\sqrt{\ell(\ell+1)}\phi_{\ell m}`
          where :math:`\phi` is the lensing potential
        - Two arrays: [gradient, curl] with gradient as above and curl
          :math:`\sqrt{\ell(\ell+1)}\Omega_{\ell m}` where :math:`\Omega` is the curl potential

        The curl can be omitted if zero, resulting in slightly faster execution.
    geometry : Geom
        Sphere pixelization geometry (iso-latitude ring structure).
    mmax : int, optional
        Maximum m value of dlms. If None, assumes mmax = lmax.
    nthreads : int, optional
        Number of threads to use. If 0, uses :func:`os.cpu_count()`.
    calc_rotation : bool, optional
        If True, also computes the rotation angle :math:`\gamma` by which to rotate
        non-zero spin fields after deflection:

        .. math::

            {}_s P(\hat{n}) \rightarrow e^{is\gamma(\hat{n})} {}_s P(\hat{n}')

        Default: False

    Returns
    -------
    angles : array_like
        Array of shape (npix, 2) or (npix, 3) containing:

        - Column 0: Deflected colatitude :math:`\theta'` (radians)
        - Column 1: Deflected longitude :math:`\phi'` (radians)
        - Column 2: Rotation angle :math:`\gamma` (radians, only if calc_rotation=True)

    Examples
    --------
    >>> from lenspyx.lensing import dlm2angles
    >>> from lenspyx.remapping.utils_geom import Geom
    >>> import numpy as np
    >>> # Create deflection from lensing potential
    >>> lmax = 2048
    >>> phi_lm = np.random.randn(Alm.getsize(lmax, lmax)) + \
    ...          1j * np.random.randn(Alm.getsize(lmax, lmax))
    >>> utils_hp.almxfl(phi_lm, np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2)), None, False)
    >>> # Get deflected angles
    >>> geom = Geom.get_healpix_geometry(nside=1024)
    >>> angles = dlm2angles(dlm, geom, nthreads=8)
    >>> theta_deflected = angles[:, 0]
    >>> phi_deflected = angles[:, 1]
    >>> # With rotation angles for polarization
    >>> angles_rot = dlm2angles(dlm, geom, calc_rotation=True, nthreads=8)
    >>> gamma = angles_rot[:, 2]

    Notes
    -----
    The deflection field is derived from the lensing potentials via:

    .. math::

        d_{\ell m} = \sqrt{\ell(\ell+1)} \left(\phi_{\ell m} + i \Omega_{\ell m}\right)

    See Also
    --------
    alm2lenmap : Compute lensed maps directly
    lenspyx.remapping.utils_geom.Geom : Geometry class

    """
    if nthreads <= 0:
        nthreads = cpu_count()
    dlms2d = np.atleast_2d(dlms)
    lmax = Alm.getlmax(dlms2d[0].size, mmax=mmax)
    assert dlms[0].size == Alm.getsize(lmax, mmax), ('Inconsistent input lmax and mmax', (lmax, mmax))
    tht, phi0, nph, ofs = geometry.theta, geometry.phi0, geometry.nph, geometry.ofs
    d1 = geometry.synthesis(dlms2d, 1, lmax, mmax, nthreads, mode='STANDARD' if dlms2d.shape[0] == 2 else 'GRAD_ONLY')
    return get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T, calc_rotation=calc_rotation, nthreads=nthreads)
    

def alm2lenmap(alm, dlms, geometry: tuple[str, dict]=('healpix', {'nside':2048}), epsilon=1e-7, verbose=0, nthreads: int=0, pol=True):
    r"""Compute lensed CMB maps from unlensed alms and deflection field.

    This function performs exact (non-perturbative) gravitational lensing of CMB
    temperature and polarization maps using interpolation on the deflected sphere.

    Parameters
    ----------
    alm : array_like or list of array_like
        Unlensed CMB spherical harmonic coefficients. Can be:

        - Single array: Temperature only (spin-0)
        - List of 2 arrays: [T, E] if pol=True, otherwise two spin-0 fields
        - List of 3 arrays: [T, E, B] if pol=True, otherwise three spin-0 fields
    dlms : array_like or list of array_like
        Spin-1 deflection field in harmonic space:

        - Single array: Gradient-only deflection :math:`\sqrt{\ell(\ell+1)}\phi_{\ell m}`
        - List of 2 arrays: [gradient, curl] deflections where curl is
          :math:`\sqrt{\ell(\ell+1)}\Omega_{\ell m}`

        The curl can be omitted if zero for slightly faster transforms.
    geometry : tuple of (str, dict), optional
        Sphere pixelization: (geometry_name, parameters).
        Default: ('healpix', {'nside': 2048})
    epsilon : float, optional
        Target numerical accuracy of the result. Default: 1e-7.
        Execution time has only weak dependence on this parameter.
    verbose : int, optional
        If non-zero, prints timing and diagnostic information. Default: 0.
    nthreads : int, optional
        Number of threads to use. If 0, uses :func:`os.cpu_count()`. Default: 0.
    pol : bool, optional
        If True, interprets input arrays as CMB fields (T, E, B) and returns
        lensed T, Q, U. If False, performs spin-0 transforms only. Default: True.

    Returns
    -------
    maps : tuple or array_like
        Lensed maps, each an array of size npix from the input geometry:

        - If pol=True and 2-3 input alms: Returns (T, Q, U) tuple
        - If pol=False or single alm: Returns single map or list of maps

    Examples
    --------
    Lens temperature and polarization:

    >>> from lenspyx.lensing import alm2lenmap
    >>> from lenspyx.utils_hp import Alm
    >>> import numpy as np
    >>> lmax = 3000
    >>> nalm = Alm.getsize(lmax, lmax)
    >>> # Create unlensed alms
    >>> alm_t = np.random.randn(nalm) + 1j * np.random.randn(nalm)
    >>> alm_e = np.random.randn(nalm) + 1j * np.random.randn(nalm)
    >>> alm_b = np.zeros(nalm, dtype=complex)
    >>> # Create deflection from lensing potential
    >>> phi_lm = np.random.randn(nalm) + 1j * np.random.randn(nalm)
    >>> dlm = utils_hp.almxfl(phi_lm, np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2)), None, False)
    >>> # Compute lensed maps
    >>> t_lens, q_lens, u_lens = alm2lenmap([alm_t, alm_e, alm_b], dlm,
    ...                                      geometry=('healpix', {'nside': 2048}),
    ...                                      nthreads=8)

    Lens temperature only:

    >>> t_lens = alm2lenmap(alm_t, dlm, nthreads=8)

    Notes
    -----
    The lensing operation remaps the CMB according to:

    .. math::

        X^{\text{lensed}}(\hat{n}) = X^{\text{unlensed}}(\hat{n} + \alpha(\hat{n}))

    where :math:`\alpha` is the lensing deflection vector field and :math:`X` is T, Q, or U, or another field.

    For polarization, the Stokes parameters are rotated by the lensing-induced angle.

    See Also
    --------
    alm2lenmap_spin : Lens arbitrary spin-weight fields
    synfast : Generate lensed realizations from power spectra
    dlm2angles : Get deflected angles from deflection alms

    """
    if nthreads <= 0:
        nthreads = cpu_count()
        if verbose:
            print('alm2lenmap: using %s nthreads'%nthreads)
    if isinstance(dlms, list) or dlms.ndim > 1:
        assert len(dlms) <= 2
        dglm = dlms[0]
        dclm = None if len(dlms) == 1 else dlms[1]
    else:
        dglm = dlms
        dclm = None

    defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if isinstance(alm, list) or alm.ndim == 2:
        if pol and len(alm) in [2, 3]:
            T = defl.gclm2lenmap(alm[0], None, 0, False).squeeze()
            Q, U = defl.gclm2lenmap(np.array(alm[1:]), None, 2, False)
            if verbose:
                print(defl.tim)
            return T, Q, U
        else:
            ret = [defl.gclm2lenmap(a, None, 0, False).squeeze() for a in alm]
            if verbose:
                print(defl.tim)
            return ret
    ret = defl.gclm2lenmap(alm, None, 0, False).squeeze()
    if verbose:
        print(defl.tim)
    return ret


def alm2lenmap_spin(gclm: np.ndarray or list, dlms:np.ndarray or list, spin:int, geometry: tuple[str, dict] = ('healpix', {'nside':2048}), epsilon: float=1e-7, verbose=0, nthreads: int=0):
    r"""Compute lensed spin-weighted map from gradient/curl modes and deflection field.

    This function remaps arbitrary spin-s fields. 

    Parameters
    ----------
    gclm : array_like or list of array_like
        Unlensed gradient and curl mode alms (E and B modes for spin-2).

        - Single array: Gradient-only (e.g., E-mode only)
        - List of 2 arrays: [gradient, curl] (e.g., [E, B])
    dlms : array_like or list of array_like
        Spin-1 deflection field in harmonic space:

        - Single array: Gradient-only deflection :math:`\sqrt{\ell(\ell+1)}\phi_{\ell m}`
        - List of 2 arrays: [gradient, curl] deflections
    spin : int
        Spin weight of the field to deflect (≥ 0). Examples:

        - spin=0 : Scalar field (temperature)
        - spin=2 : Polarization (most common)
        - spin≥3 : Higher-spin fields
    geometry : tuple of (str, dict), optional
        Sphere pixelization. Default: ('healpix', {'nside': 2048})
    epsilon : float, optional
        Target numerical accuracy. Default: 1e-7
    verbose : int, optional
        Print timing information if non-zero. Default: 0
    nthreads : int, optional
        Number of threads. If 0, uses :func:`os.cpu_count()`. Default: 0

    Returns
    -------
    maps : array_like
        Lensed maps with shape (2, npix) containing the real and imaginary parts
        of the spin-s field (or shape (1, npix) for spin-0).

    Notes
    -----
    For a spin-s field, the lensing operation includes both deflection and rotation:

    .. math::

        {}_s X^{\text{lensed}}(\hat{n}) = e^{is\gamma(\hat{n})} {}_s X^{\text{unlensed}}(\hat{n}')

    where :math:`\gamma` is the rotation angle induced by lensing and
    :math:`\hat{n}' = \hat{n} + \alpha(\hat{n})` is the deflected direction.

    If curl modes are zero (for either the deflection or the field alms), they can
    be omitted, resulting in slightly faster transforms.

    Examples
    --------
    >>> from lenspyx.lensing import alm2lenmap_spin
    >>> import numpy as np
    >>> # Lens polarization (spin-2)
    >>> e_lm = np.random.randn(nalm) + 1j * np.random.randn(nalm)
    >>> b_lm = np.zeros_like(e_lm)
    >>> dlm =almxfl(phi_lm, np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2), None, False)
    >>> q_u_lensed = alm2lenmap_spin([e_lm, b_lm], dlm, spin=2, nthreads=8)
    >>> q_lensed = q_u_lensed[0]
    >>> u_lensed = q_u_lensed[1]

    See Also
    --------
    alm2lenmap : same for spin-0 field
    dlm2angles : Get deflected angles and rotation

    """
    if spin == 0:
        return alm2lenmap(gclm, dlms, geometry=geometry, epsilon=epsilon, verbose=verbose, nthreads=nthreads)
    if isinstance(dlms, list) or dlms.ndim > 1:
        assert len(dlms) <= 2
        dglm = dlms[0]
        dclm = None if len(dlms) == 1 else dlms[1]
    else:
        dglm = dlms
        dclm = None

    if nthreads <= 0:
        nthreads = cpu_count()
        if verbose:
            print('alm2lenmap_spin: using %s nthreads'%nthreads)

    defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if isinstance(gclm, list) and gclm[1] is None:
        gclm = gclm[0]
    ret = defl.gclm2lenmap(gclm, None, spin, False)
    if verbose:
        print(defl.tim)
    return ret


def synfast(cls: dict, lmax=None, mmax=None, geometry=('healpix', {'nside': 2048}),
            epsilon=1e-7, nthreads=0, alm=False, seed=None, verbose=0):
    r"""Generate lensed CMB realizations from power spectra.

    Creates correlated Gaussian random fields on the sphere and applies gravitational
    lensing according to the input power spectra. This is the standard way to generate
    lensed CMB simulations.

    Parameters
    ----------
    cls : dict
        Dictionary of auto- and cross-power spectra with string keys.
        Recognized field labels (case-insensitive):

        - 'T' or 't' : CMB temperature (spin-0 intensity)
        - 'E' or 'e' : E-mode polarization
        - 'B' or 'b' : B-mode polarization
        - 'P' or 'p' : Lensing potential :math:`\phi`
        - 'O' or 'o' : Lensing curl potential :math:`\Omega`

        Keys are two-character strings like 'TT', 'TE', 'EE', 'PP', etc.
        Arrays must be :math:`C_\ell` (not :math:`D_\ell = \ell(\ell+1)C_\ell/(2\pi)`).

        **Important**:

        - If auto-spectrum 'AA' is absent, field 'A' is assumed zero
        - If neither 'PP' nor 'OO' are present, output maps are unlensed
        - All relevant cross-spectra must be provided for correlated fields
    lmax : int, optional
        Band-limit of unlensed alms. If None, inferred from length of input spectra.
    mmax : int, optional
        Maximum azimuthal mode number. If None, defaults to lmax.
    geometry : tuple of (str, dict), optional
        Pixelization: (geometry_name, parameters).
        Default: ('healpix', {'nside': 2048})
    epsilon : float, optional
        Target numerical accuracy for lensing. Default: 1e-7.
        Execution time has weak dependence on this.
    nthreads : int, optional
        Number of threads for SHTs. If 0, uses :func:`os.cpu_count()`. Default: 0.
    alm : bool, optional
        If True, also returns unlensed alms. Default: False.
    seed : int, optional
        Random number generator seed for reproducible results. Default: None (random).
    verbose : int, optional
        Print timing information if non-zero. Default: 0.

    Returns
    -------
    maps : dict
        Dictionary of lensed maps:

        - 'T' : Lensed temperature (if 'TT' was in cls and non-zero)
        - 'QU' : Lensed Q and U Stokes parameters, shape (2, npix)
          (if 'EE' or 'BB' were in cls and non-zero)
    alms : tuple, optional (if alm=True)
        Tuple of (alm_arrays, field_labels) where:

        - alm_arrays : shape (nfields, nalm) with unlensed alms
        - field_labels : string indicating field ordering (e.g., 'tebp')

    Examples
    --------
    Generate lensed CMB with temperature and polarization:

    >>> from lenspyx.lensing import synfast
    >>> from lenspyx.utils import camb_clfile
    >>> # Load power spectra
    >>> cls = camb_clfile('cosmo_params.ini')
    >>> # Generate lensed realization
    >>> maps = synfast(cls, lmax=3000, geometry=('healpix', {'nside': 2048}),
    ...                nthreads=8, seed=42)
    >>> t_lensed = maps['T']
    >>> q_lensed, u_lensed = maps['QU']

    Generate and return unlensed alms:

    >>> maps, (alms, labels) = synfast(cls, lmax=3000, alm=True, seed=42)
    >>> # labels might be 'tebp' for T, E, B, phi
    >>> if 't' in labels:
    ...     alm_t = alms[labels.index('t')]

    Generate unlensed maps (no lensing potential in cls):

    >>> cls_unlensed = {'tt': cl_tt, 'ee': cl_ee, 'te': cl_te}
    >>> maps_unlensed = synfast(cls_unlensed, lmax=2000)

    Notes
    -----
    The function:

    1. Generates correlated Gaussian random alms from the input :math:`C_\ell`
    2. If lensing potentials (P or O) are present, computes deflection field
    3. Applies exact (non-perturbative) lensing via interpolation
    4. Returns lensed maps on the specified geometry

    The lensing deflection is computed as:

    .. math::

        d_{\ell m} = \sqrt{\ell(\ell+1)} \left(\phi_{\ell m} + i \Omega_{\ell m}\right)

    See Also
    --------
    alm2lenmap : Lens pre-existing alms
    lenspyx.utils.camb_clfile : Read CAMB power spectrum files

    """
    tim = timer('synfast', False)
    lmax_cls = np.max([len(cl) - 1 for cl in cls.values()])
    lmax = lmax_cls if lmax is None else min(lmax, lmax_cls)
    if mmax is None:
        mmax = lmax
    cmb_labels = ['t', 'e', 'b', 'p', 'o']
    spec_labels = [k.lower() for k in cls.keys()]
    # First remove zero fields:
    zros = []
    for fg in spec_labels:
        assert len(fg) == 2 and fg[0] in cmb_labels and fg[1] in cmb_labels
        if fg[0] == fg[1]:
            assert np.all(cls[fg] >= 0), 'auto spectrum of %s must be >= 0' % fg
            if not np.any(cls[fg]):
                zros.append(fg[0])
    labelsf = []
    for fg in spec_labels:
        assert len(fg) == 2 and fg[0] in cmb_labels and fg[1] in cmb_labels
        if fg[0] not in zros and fg[1] not in zros:
            assert fg[0] + fg[0] in spec_labels, 'must have %s auto-spectrum' % fg[0]
            assert fg[1] + fg[1] in spec_labels, 'must have %s auto-spectrum' % fg[1]
            if fg[0] == fg[1]:
                for field in fg:
                    if field not in labelsf:
                        labelsf.append(field)
    assert len(labelsf) <= 4
    labels = ''
    for f in 'tebpo':  # This just sorts the present labels according to 'tebpx'
        labels += f * (f in labelsf)

    ncomp = len(labels)
    mat = np.empty((lmax + 1, ncomp, ncomp), dtype=float)
    for i, f in enumerate(labels):
        for j, g in enumerate(labels[i:]):
            mat[:, i + j, i] = cls.get(f + g, cls.get(g + f, np.zeros(lmax + 1, dtype=float)))[:lmax + 1]
    ts, vs = np.linalg.eigh(mat)
    assert np.all(ts >= 0.)  # Matrix not positive semidefinite
    for m, t, v in zip(mat, ts, vs):
        m[:] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))
    tim.add('cl matrix')
    # Build phases:
    alm_size = Alm.getsize(lmax, mmax)
    rng = default_rng(seed)
    phases = 1j * rng.standard_normal((ncomp, alm_size), dtype=float)
    phases += rng.standard_normal((ncomp, alm_size), dtype=float)
    phases *= np.sqrt(0.5)
    real_idcs = Alm.getidx(lmax, np.arange(lmax + 1, dtype=int), 0)
    phases[:, real_idcs] = phases[:, real_idcs].real * np.sqrt(2.)
    tim.add('phases generation')

    # Now builds alms. We might need more since we cannot handle now curl only transforms
    labels_wgrad = labels
    if 'b' in labels and 'e' not in labels:
        labels_wgrad = labels_wgrad.replace('b', 'eb')
    if 'o' in labels and 'p' not in labels:
        labels_wgrad = labels_wgrad.replace('o', 'po')
    alms = np.zeros((len(labels_wgrad), phases[0].size), dtype=complex)
    #    for L in Ls: #L @ L.T is full matrx
    for i, f in enumerate(labels):
        idx = labels_wgrad.index(f)
        alms[idx] += almxfl(phases[i], mat[:, i, i], mmax, False)
        for j in range(ncomp):
            fl = mat[:, i, j]
            if i != j and np.any(fl):
                alms[idx] += almxfl(phases[j], fl, mmax, False)
    del phases
    tim.add('alms from phases')
    maps = {}
    if 'p' in labels_wgrad or 'o' in labels_wgrad:  # There is actual lensing
        p2d = np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
        dglm, dclm = None, None
        if 'p' in labels_wgrad:
            dglm = alms[labels_wgrad.index('p')]
            almxfl(dglm, p2d, mmax, True)
        if 'o' in labels_wgrad:
            dclm = alms[labels_wgrad.index('o')]
            almxfl(dclm, p2d, mmax, True)
        if dglm is None:
            dglm = np.zeros_like(dclm)
        if dclm is None:
            dclm = np.zeros_like(dglm)
        if nthreads <= 0:
            nthreads = cpu_count()
        defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                          cacher=cachers.cacher_mem(safe=False))
        if 't' in labels_wgrad:
            maps['T'] = defl.gclm2lenmap(alms[0:1], mmax, 0, False).squeeze()
        if 'e' in labels_wgrad:
            i = labels_wgrad.index('e')
            maps['QU'] = defl.gclm2lenmap(alms[i:i + 1 + ('b' in labels_wgrad)], mmax, 2, False)
        tim.add('alm2lenmap')
        # undo deflection weighting otherwise unlensed plm array would be dlm instead
        d2p = np.zeros_like(p2d)
        d2p[1:] = 1. / p2d[1:]
        almxfl(dglm, d2p, mmax, True)
        almxfl(dclm, d2p, mmax, True)
    else:  # no lensing here
        geom = get_geom(geometry)
        if 't' in labels_wgrad:
            maps['T'] = geom.synthesis(alms[0:1], 0, lmax, mmax, nthreads)
        if 'e' in labels_wgrad:
            i = labels_wgrad.index('e')
            sht_mode = 'STANDARD' if 'b' in labels_wgrad else 'GRAD_ONLY'
            maps['QU'] = geom.synthesis(alms[i:i + 1 + ('b' in labels_wgrad)], 2, lmax, mmax, False, mode=sht_mode)
        tim.add('alm2map')
    if verbose:
        print(tim)
    return maps if not alm else (maps, (alms, labels_wgrad))
