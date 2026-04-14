===============
lenspyx.lensing
===============

CMB lensing operations: lensed map synthesis and related utilities.

This module provides functions for computing gravitationally lensed CMB maps from
spherical harmonic coefficients. Gravitational lensing remaps the CMB temperature
and polarization by deflecting photon paths according to the intervening mass distribution.

Overview
--------

Gravitational lensing modifies the observed CMB by deflecting photon paths. The lensing
operation remaps the CMB according to:

.. math::

    X^{\text{lensed}}(\hat{n}) = X^{\text{unlensed}}(\hat{n} + \nabla\phi(\hat{n}))

where :math:`\phi` is the lensing potential and :math:`X` represents T, Q, or U Stokes polarization modes.

For polarization, the Stokes parameters Q and U are additionally rotated by the
lensing-induced angle :math:`\gamma`.

The implementation uses exact (non-perturbative) lensing via interpolation on the sphere,
with configurable accuracy through the ``epsilon`` parameter.

API Reference
-------------

.. currentmodule:: lenspyx.lensing

Main Functions
~~~~~~~~~~~~~~

.. autofunction:: synfast

.. autofunction:: alm2lenmap

.. autofunction:: alm2lenmap_spin

.. autofunction:: dlm2angles

Examples
--------

Generate a lensed CMB realization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.lensing import synfast
    from lenspyx.utils import camb_clfile

    # Load CMB power spectra (including lensing potential)
    cls = camb_clfile('cosmo2017_10K_acc3_lensedCls.dat')
    # Must contain 'tt', 'ee', 'bb', 'te', 'pp' keys

    # Generate lensed realization
    maps = synfast(cls, lmax=3000,
                   geometry=('healpix', {'nside': 2048}),
                   nthreads=8, seed=42, verbose=1)

    # Access lensed maps
    t_lensed = maps['T']         # Temperature
    q_lensed = maps['QU'][0]     # Q Stokes parameter
    u_lensed = maps['QU'][1]     # U Stokes parameter

Lens existing unlensed alms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.lensing import alm2lenmap
    from lenspyx.utils_hp import Alm, almxfl
    import numpy as np

    lmax = 3000
    nalm = Alm.getsize(lmax, lmax)

    # Create or load unlensed alms
    alm_t = np.load('unlensed_alm_t.npy')
    alm_e = np.load('unlensed_alm_e.npy')
    alm_b = np.zeros(nalm, dtype=complex)  # Unlensed B=0

    # Create deflection from lensing potential
    phi_lm = np.load('phi_lm.npy')
    L = np.arange(lmax + 1, dtype=float)
    deflection_factor = np.sqrt(L * (L + 1))
    dlm = almxfl(phi_lm, deflection_factor, None, False)

    # Compute lensed maps
    t_lens, q_lens, u_lens = alm2lenmap([alm_t, alm_e, alm_b], dlm,
                                        geometry=('healpix', {'nside': 2048}),
                                        epsilon=1e-7, nthreads=8, verbose=1)

Compute deflected angles from lensing potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.lensing import dlm2angles
    from lenspyx.remapping.utils_geom import Geom
    from lenspyx.utils_hp import Alm
    import numpy as np

    # Lensing potential alms
    phi_lm = np.load('phi_lm.npy')
    lmax = Alm.getlmax(phi_lm.size, mmax=None)

    # Convert to deflection
    dlm = phi_lm * np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2))

    # Get geometry
    geom = Geom.get_healpix_geometry(nside=1024)

    # Compute deflected angles (with rotation for polarization)
    angles = dlm2angles(dlm, geom, calc_rotation=True, nthreads=8)

    theta_deflected = angles[:, 0]  # Deflected colatitude
    phi_deflected = angles[:, 1]    # Deflected longitude
    gamma_rotation = angles[:, 2]   # Rotation angle for spin fields

Lens arbitrary spin-weighted fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.lensing import alm2lenmap_spin
    from lenspyx.utils_hp import almxfl
    import numpy as np

    # For spin-2 field (e.g., polarization)
    e_lm = np.load('e_mode_alms.npy')
    b_lm = np.load('b_mode_alms.npy')

    # Deflection field
    dlm = almxfl(phi_lm, np.sqrt(np.arange(lmax+1) * np.arange(1, lmax+2)), None, False)

    # Lens the spin-2 field
    qu_lensed = alm2lenmap_spin([e_lm, b_lm], dlm, spin=2,
                                geometry=('healpix', {'nside': 2048}),
                                nthreads=8)

    q_lensed = qu_lensed[0]
    u_lensed = qu_lensed[1]

Use different geometries
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.lensing import synfast

    # HEALPix geometry (most common)
    maps_hp = synfast(cls, lmax=2000,
                     geometry=('healpix', {'nside': 1024}))

    # Gauss-Legendre geometry (for high-accuracy applications)
    maps_gl = synfast(cls, lmax=2000,
                     geometry=('gl', {'lmax': 2000}))

Notes
-----

- The lensing implementation is exact (non-perturbative), using interpolation
- The ``epsilon`` parameter controls numerical accuracy
- For reproducible results, always set the ``seed`` parameter
- The deflection field is :math:`d_{\ell m} = \sqrt{\ell(\ell+1)}\left(\phi_{\ell m} + i \Omega_{\ell m}\right)`

References
----------

.. [1] Reinecke, M., Belkner, S., and Carron, J., 2023. "Improved cosmic microwave background
       (de-)lensing using general spherical harmonic transforms."
       arXiv:2304.10431. https://arxiv.org/abs/2304.10431

See Also
--------

- Jupyter notebook: ``examples/demo_lenspyx.ipynb`` for interactive examples
- :mod:`lenspyx.remapping.deflection_029` : Low-level lensing implementation
- :mod:`lenspyx.experimental` : Optimized transforms for partial-sky data
