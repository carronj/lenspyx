"""This module contains DUCC-based fonctions related to Wigner-small d tranforms


"""
from __future__ import annotations
import numpy as np
from ducc0.sht.experimental import alm2leg


def wignerpos(cl, theta, s1, s2):
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
