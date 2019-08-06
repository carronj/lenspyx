# quicklens/shts/util.py
# --
# this module contains utilities for working with harmonic
# coefficients. there are three different formats used here:
#
#   'vlm' = complex coefficients vlm[l*l+l+m], with l \in [0, lmax] and m \in [-l, l]
#   'alm' = complex coefficients alm[m * (2*lmax+1-m)/2 + l] with l in [0, lmax] and m in [0, l].
#                * corresponds to a field which has a real-valued spin-0 map.
#                * this is the format used by healpy for harmonic transforms
#   'rlm' = real coefficients rlm[l*l + 2*m + 0] and rlm[l*l + 2*m + 1]
#                * corresponds to the real and imaginary parts of alm, useful for matrix operations.

import numpy as np


def nlm2lmax(nlm):
    """ returns the lmax for an array of alm with length nlm. """
    lmax = int(np.floor(np.sqrt(2 * nlm) - 1))
    assert ((lmax + 2) * (lmax + 1) // 2 == nlm)
    return lmax


def lmax2nlm(lmax):
    """ returns the length of the complex alm array required for maximum multipole lmax. """
    return (lmax + 1) * (lmax + 2) // 2


def alm2vlm(glm, clm=None):
    """
    convert alm format -> vlm format coefficients. glm is gradient mode, clm is curl mode.
    For pure gradients holds vl-m = (-1) ** m vlm^*, half the array is redundant and vlm = - glm,
    with ret[l * l + l + m] = -glm
    with ret[l * l + l - m] = -(-1)^m glm^*

    For pure curls     holds vl-m =-(-1) ** m vlm^*, half the array is redundant and vlm = -i clm,
    with ret[l * l + l + m] = -i clm
    with ret[l * l + l - m] = -i(-1)^m clm^*
    """
    lmax = nlm2lmax(len(glm))
    ret = np.zeros((lmax + 1) ** 2, dtype=complex)
    for l in range(0, lmax + 1):
        ms = np.arange(1, l + 1)
        ret[l * l + l] = -glm[l]
        ret[l * l + l + ms] = -glm[ms * (2 * lmax + 1 - ms) / 2 + l]
        ret[l * l + l - ms] = -(-1) ** ms * np.conj(glm[ms * (2 * lmax + 1 - ms) / 2 + l])

    if not clm is None:
        assert (len(clm) == len(glm))
        for l in range(0, lmax + 1):
            ms = np.arange(1, l + 1)
            ret[l * l + l] += -1.j * clm[l]
            ret[l * l + l + ms] += -1.j * clm[ms * (2 * lmax + 1 - ms) / 2 + l]
            ret[l * l + l - ms] += -(-1) ** ms * 1.j * np.conj(clm[ms * (2 * lmax + 1 - ms) / 2 + l])

    return ret


def vlm2alm(vlm):
    """ convert vlm format coefficients -> alm. returns gradient and curl pair (glm, clm). """
    lmax = int(np.sqrt(len(vlm)) - 1)

    glm = np.zeros(lmax2nlm(lmax), dtype=np.complex)
    clm = np.zeros(lmax2nlm(lmax), dtype=np.complex)

    for l in range(0, lmax + 1):
        ms = np.arange(1, l + 1)

        glm[l] = -vlm[l * l + l].real
        clm[l] = -vlm[l * l + l].imag

        glm[ms * (2 * lmax + 1 - ms) / 2 + l] = -0.5 * (vlm[l * l + l + ms] + (-1) ** ms * np.conj(vlm[l * l + l - ms]))
        clm[ms * (2 * lmax + 1 - ms) / 2 + l] = 0.5j * (vlm[l * l + l + ms] - (-1) ** ms * np.conj(vlm[l * l + l - ms]))
    return glm, clm


def alm2rlm(alm):
    """ converts a complex alm to 'real harmonic' coefficients rlm. """

    lmax = nlm2lmax(len(alm))
    rlm = np.zeros((lmax + 1) ** 2)

    ls = np.arange(0, lmax + 1)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in range(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].imag * rt2
    return rlm


def rlm2alm(rlm):
    """ converts 'real harmonic' coefficients rlm to complex alm. """

    lmax = int(np.sqrt(len(rlm)) - 1)
    assert ((lmax + 1) ** 2 == len(rlm))

    alm = np.zeros(lmax2nlm(lmax), dtype=np.complex)

    ls = np.arange(0, lmax + 1, dtype=np.int64)
    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in range(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1.j * rlm[l2s[m:] + 2 * m + 0]) * ir2
    return alm
