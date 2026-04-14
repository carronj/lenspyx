================
lenspyx.wigners
================

Wigner small-d functions and correlation functions for spin-weighted spherical harmonics.

This module provides efficient implementations of Wigner small-d functions and their associated
correlation functions, which are fundamental for analyzing spin-weighted fields on the sphere
(e.g., CMB polarization, gravitational lensing).

Overview
--------

Wigner small-d functions :math:`d^\ell_{s_1 s_2}(\theta)` are the angular parts of the Wigner D-matrices,
describing the rotation of spin-weighted spherical harmonics. They appear in:

- Correlation functions of spin-weighted fields (e.g., ξ± for CMB polarization or galaxy surveys)
- Angular power spectrum estimators for CMB polarization
- Lensing reconstruction and delensing operations
- General spin transformations on the sphere

Key Functions
-------------

Forward Transforms
~~~~~~~~~~~~~~~~~~

.. autofunction:: lenspyx.wigners.wignerpos

.. autofunction:: lenspyx.wigners.wigner4pos

Adjoint Transforms
~~~~~~~~~~~~~~~~~~

.. autofunction:: lenspyx.wigners.wignercoeff

Convolutions
~~~~~~~~~~~~

.. autofunction:: lenspyx.wigners.wignerc

Wigner d-functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: lenspyx.wigners.wignerd

.. autofunction:: lenspyx.wigners.wignerdl

Quadrature Utilities
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lenspyx.wigners.get_thgwg

.. autofunction:: lenspyx.wigners.get_xgwg

Examples
--------

Computing CMB E-mode correlation function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from lenspyx.wigners import wignerpos
    
    # E-mode power spectrum
    cl_ee = np.loadtxt('cl_ee.txt')  
    theta = np.linspace(0, np.pi, 100)
    
    # Compute ξ_EE(θ) = Σ_ℓ (2ℓ+1)/(4π) C_ℓ^EE d^ℓ_{2,2}(θ)
    xi_ee = wignerpos(cl_ee, theta, s1=2, s2=2)

Computing ξ± correlation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.wigners import wigner4pos
    
    # Returns [ξ+, ξ-] where:
    # ξ+ = Σ_ℓ (2ℓ+1)/(4π) C_ℓ d^ℓ_{2,+2}(θ)  
    # ξ- = Σ_ℓ (2ℓ+1)/(4π) C_ℓ d^ℓ_{2,-2}(θ)
    xi_plus, xi_minus = wigner4pos(cl_ee, None, theta, s1=2, s2=2)

Inverting correlation function to power spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.wigners import wignercoeff, get_thgwg
    
    lmax = 2000
    npts = (lmax + 2) // 2
    theta, weights = get_thgwg(npts)
    
    # Measure ξ(θ) from data
    xi_measured = measure_correlation(theta)
    
    # Compute C_ℓ = 2π Σ_θ ξ(θ) d^ℓ_{s1,s2}(θ) 
    cl_recovered = wignercoeff(xi_measured, theta, s1=2, s2=2, lmax=lmax)

Convolving correlation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lenspyx.wigners import wignerc
    
    cl1 = cl_ee  # E-mode spectrum
    cl2 = cl_bb  # B-mode spectrum  
    
    # Compute spectrum of ξ_EE(θ) * ξ_BB(θ)
    cl_product = wignerc(cl1, cl2, s1=2, t1=2, s2=2, t2=2)

Notes
-----

- Spin values s1, s2 can be positive, negative, or zero
- The normalization includes (2ℓ+1)/(4π) factors
- Gauss-Legendre quadrature ensures exact integration for band-limited functions

The implementation uses DUCC0's ``alm2leg`` and ``leg2alm`` functions, which achieve
excellent performance through compiler optimization.

Module Reference
----------------

.. automodule:: lenspyx.wigners
    :members:
    :undoc-members:
    :show-inheritance:
