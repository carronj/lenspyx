"""This module contains DUCC-based fonctions related to Wigner-small d tranforms


"""
from __future__ import annotations
import numpy as np
from ducc0.sht.experimental import alm2leg


def wignerpos(cl: np.ndarray, theta: np.ndarray, s1: int, s2: int):
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
    if s1 < 0 or (s1 == 0 and s2 > 0):
        # TODO
        # second cond. branching resulting in spin-0 eval. The case 0 -2 is still not optimal
        t_cl = cl if (s1 + s2) % 2 == 0 else -cl
        if s2 < 0:
            return wignerpos(t_cl, theta, -s1, -s2)
        else:
            return wignerpos(t_cl, theta, s2, s1)
    else:
        mval = np.array([abs(s1)], dtype=int)
        lmax = len(cl) - 1
        sgn = (-1) ** (1 + (s2 if s2 < 0 else 0) + (s2 == 0))
        gl = sgn * cl * np.sqrt(np.arange(1, 2 * lmax + 3, 2)) / np.sqrt(4 * np.pi)
        glm_r = gl.astype(np.complex128)
        if s2 != 0:
            glm_i = (1j * np.sign(s2)) * gl
            glm = np.stack([glm_r, glm_i])
        else:
            glm = np.atleast_2d(glm_r)
        mstart = np.array([0], dtype=int)
        leg = alm2leg(alm=glm, spin=abs(s2), lmax=lmax, mval=mval, mstart=mstart, theta=theta)
        return leg.squeeze()[0].real if s2 != 0 else leg.squeeze().real
