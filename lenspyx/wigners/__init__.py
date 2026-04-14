"""Wigner small-d functions and correlation functions.

This module provides efficient implementations of Wigner small-d functions for
spin-weighted spherical harmonic analysis.

Main Functions
--------------
- wignerpos : Forward transform C_ℓ → ξ(θ)
- wignercoeff : Adjoint transform ξ(θ) → C_ℓ
- wigner4pos : Compute 4 correlation functions simultaneously
- wignerc : Convolve two Wigner correlation functions
- wignerd : Single Wigner d-function for specific ℓ
- get_thgwg : Gauss-Legendre quadrature points and weights

See wigners.wigners module for detailed documentation and examples.
"""

from .wigners import (
    wignerpos,
    wignercoeff,
    wigner4pos,
    wignerc,
    wignerd,
    wignerdl,
    get_thgwg,
    get_xgwg
)

__all__ = [
    'wignerpos',
    'wignercoeff',
    'wigner4pos',
    'wignerc',
    'wignerd',
    'wignerdl',
    'get_thgwg',
    'get_xgwg'
]
