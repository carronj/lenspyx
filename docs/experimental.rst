====================
lenspyx.experimental
====================

Experimental spherical harmonic transforms with optimizations for spherical caps.

This module provides wrapper functions and classes for spherical harmonic transforms (SHTs)
that automatically select optimized implementations based on the geometry of the data.

Overview
--------

The module wraps functions from ducc0 and capsht, providing:

- Automatic selection between general SHT and optimized capped SHT
- Unified interface for different pixel location types (HEALPix, arbitrary locations)
- Support for both synthesis (alm → map) and adjoint synthesis (map → alm)
- Optimized implementations for data confined to spherical caps

When data is confined to a spherical cap (less than half the sky), the capped transforms
can provide significant speedups compared to full-sky transforms.

.. note::
   This module requires capsht to be installed for the optimized capped transforms.
   If capsht is not available, functions will fall back to standard ducc0 implementations.

API Reference
-------------

.. currentmodule:: lenspyx.experimental

Synthesis Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: synthesis_general

.. autofunction:: adjoint_synthesis_general

Locations Class
~~~~~~~~~~~~~~~

.. autoclass:: Locations
   :members: synthesis, adjoint_synthesis, npix
   :undoc-members:
   :show-inheritance:

Examples
--------

Using synthesis_general with automatic cap detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from lenspyx.experimental import synthesis_general
    from lenspyx.utils_hp import Alm
    
    # Create random alm coefficients
    lmax = 2000
    nalm = Alm.getsize(lmax, lmax)
    alm = np.random.randn(1, nalm) + 1j * np.random.randn(1, nalm)
    
    # Define locations in a polar cap (30 degrees)
    npix = 10000
    theta_cap = 30 * np.pi / 180
    loc = np.random.rand(npix, 2) * [theta_cap, 2*np.pi]
    
    # Synthesize with automatic cap detection
    maps = synthesis_general(alm, spin=0, lmax=lmax, loc=loc, 
                            epsilon=1e-7, thtcap=theta_cap, verbose=True)

Using Locations class with arbitrary locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from lenspyx.experimental import Locations
    from lenspyx.utils_hp import Alm

    # Create Locations object from arbitrary pixel positions
    npix = 10000
    loc = np.random.rand(npix, 2) * [np.pi, 2*np.pi]
    locs = Locations(loc=loc, epsilon=1e-7)

    # Forward transform (alm → map)
    lmax = 1000
    nalm = Alm.getsize(lmax, lmax)
    alm = np.random.randn(1, nalm) + 1j * np.random.randn(1, nalm)
    maps = locs.synthesis(alm, spin=0, lmax=lmax, mmax=lmax, nthreads=4)

    # Adjoint transform (map → alm)
    alm_out = np.zeros((1, nalm), dtype=complex)
    alm_out = locs.adjoint_synthesis(alm_out, spin=0, lmax=lmax,
                                     mmax=lmax, nthreads=4, m=maps)

Using Locations with HEALPix geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.experimental import Locations
    from lenspyx.remapping.utils_geom import Geom
    from lenspyx.utils_hp import Alm
    import numpy as np

    # Create HEALPix geometry
    nside = 512
    geom = Geom.get_healpix_geometry(nside)
    locs = Locations(geom=geom)

    # Synthesize on HEALPix grid
    lmax = 1500
    nalm = Alm.getsize(lmax, lmax)
    alm = np.random.randn(1, nalm) + 1j * np.random.randn(1, nalm)
    maps = locs.synthesis(alm, spin=0, lmax=lmax, mmax=lmax, nthreads=8)

Polarization (spin-2) example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from lenspyx.experimental import synthesis_general
    from lenspyx.utils_hp import Alm
    
    # Create E and B mode coefficients
    lmax = 2000
    nalm = Alm.getsize(lmax, lmax)
    alm_eb = np.random.randn(2, nalm) + 1j * np.random.randn(2, nalm)
    
    # Define locations
    npix = 5000
    loc = np.random.rand(npix, 2) * [np.pi/2, 2*np.pi]
    
    # Synthesize Q and U maps (spin=2)
    maps_qu = synthesis_general(alm_eb, spin=2, lmax=lmax, loc=loc, 
                                epsilon=1e-7, nthreads=8)
    # maps_qu has shape (2, npix) containing [Q, U]

Notes
-----

- The capped transforms are most beneficial when data covers much less than ~50% of the sky
- The ``Locations`` class provides a convenient unified interface regardless of geometry type
- Use ``verbose=True`` to see which transform implementation is selected

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
lenspyx.remapping.utils_geom : Geometry objects for structured grids
