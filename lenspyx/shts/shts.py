"""spin-weight harmonic transform module

    This module has benefited from pre-existing work by Duncan Hanson

"""
from __future__ import print_function

import os
import numpy as np
import pyfftw
from lenspyx.shts import fsht
from lenspyx import utils

def vtm2map(spin, vtm, Nphi, pfftwthreads=None, bicubic_prefilt=False, phiflip=()):
    """Longitudinal Fourier transform to an ECP grid.

        Sends vtm array to (bicubic prefiltered map) with Nphi points equidistant in [0,2pi).
        With bicubic prefiltering this uses 2lmax + 1 1d Ntheta-sized FFTs and one 2d (Ntheta x Nphi) iFFT.

        The pyFFTW.FFTW is twice as fast than the pyFFTW.interface,
        but the FFTW wisdom calculation overhead can compensate for the gain if only one map is lensed.

        vtm comes from shts.vlm2vtm which returns a farray, with contiguous vtm[:,ip].
        for spin 0 we have vtm^* = vt_-m, real filtered maps and we may use rffts. (not done)

        vtm should of size 2 * lmax + 1
        Flipping phi amounts to phi -> 2pi - phi -> The phi fft is sent to its transpose.

    """
    if pfftwthreads is None: pfftwthreads = os.environ.get('OMP_NUM_THREADS', 1)
    lmax = (vtm.shape[1] - 1) // 2
    Nt = vtm.shape[0]
    assert (Nt, 2 * lmax + 1) == vtm.shape, ((Nt, 2 * lmax + 1), vtm.shape)
    assert Nphi % 2 == 0, Nphi
    if bicubic_prefilt:
        #TODO: Could use real ffts for spin 0. For high-res interpolation this task can take about half of the total time.
        a = pyfftw.empty_aligned(Nt, dtype=complex)
        b = pyfftw.empty_aligned(Nt, dtype=complex)
        ret = pyfftw.empty_aligned((Nt, Nphi), dtype=complex)
        fftmap = pyfftw.empty_aligned((Nt, Nphi), dtype=complex)
        ifft2 = pyfftw.FFTW(fftmap, ret, axes=(0, 1), direction='FFTW_BACKWARD', threads=pfftwthreads)
        fft1d = pyfftw.FFTW(a, b, direction='FFTW_FORWARD', threads=1)
        fftmap[:] = 0. #NB: sometimes the operations above can result in nan's
        if Nphi > 2 * lmax:
            # There is unique correspondance m <-> kphi where kphi is the 2d flat map frequency
            for ip, m in enumerate(range(-lmax, lmax + 1)):
                fftmap[:, (Nphi + m if m < 0 else m)] = fft1d(vtm[:, ip])
        else:
            # The correspondance m <-> k is not unique anymore, but given by
            # (m - k) = N j for j in 0,+- 1, +- 2 ,etc. -> m = k + Nj
            for ik in range(Nphi):
                # candidates for m index
                ms = lmax + ik + Nphi * np.arange(-lmax / Nphi - 1, lmax / Nphi + 1, dtype=int)
                ms = ms[np.where((ms >= 0) & (ms <= 2 * lmax))]
                fftmap[:, ik] = fft1d(np.sum(vtm[:, ms], axis=1))
        w0 = Nphi * 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nt)) + 4.)
        w1 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(Nphi)) + 4.)
        fftmap[:] *= np.outer(w0, w1)
        retmap = ifft2().real if spin == 0 else ifft2()

    else :
        # Probably no real gain to expect here from pyfftw for the ffts.
        if Nphi > 2 * lmax + 1:
            a = np.zeros((Nt,Nphi),dtype = complex)
            a[:,:2 * lmax + 1] = vtm
            ret = np.fft.ifft(a) * (np.exp(np.arange(Nphi) * (-1j / Nphi * (2. * np.pi) * lmax)) * Nphi)
        else:
            ret = np.fft.ifft(vtm[:,lmax - Nphi/2:lmax + Nphi/2])
            ret *= (np.exp(np.arange(Nphi) * (-1j / Nphi * (2. * np.pi) * Nphi/2)) * Nphi)
        retmap = ret.real if spin == 0 else ret
    retmap[phiflip, :] = retmap[phiflip, ::-1]
    return retmap


def glm2vtm_sym(s, tht, glm):
    """This produces :math:`\sum_l _s\Lambda_{lm} v_{lm}` for pure gradient input for a range of colatitudes"""

    if s == 0:
        lmax = utils.nlm2lmax(len(glm))
        ret = np.empty((2 * len(tht), 2 * lmax + 1), dtype=complex)
        ret[:, lmax:] = fsht.glm2vtm_s0sym(lmax, tht, -glm)
        ret[:, 0:lmax] = (ret[:, slice(2 * lmax + 1, lmax, -1)]).conjugate()
        return ret
    return vlm2vtm_sym(s, tht, utils.alm2vlm(glm))


def vlm2vtm_sym(s, tht, vlm):
    """This produces :math:`\sum_l _s\Lambda_{lm} v_{lm}` for a range of colatitudes"""
    assert s >= 0
    tht = np.array(tht)
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    if s == 0:
        print("Consider using glm2vtm_sym for spin 0 for factor of 2 speed-up")
        return fsht.vlm2vtm_sym(lmax, s, tht, vlm)
    else:
        #: resolving poles, since fsht implementation does not handle them.
        north = np.where(tht <= 0.)[0]
        south = np.where(tht >= np.pi)[0]
        if len(north) == 0 and len(south) == 0:
            return fsht.vlm2vtm_sym(lmax, s, tht, vlm)
        else:
            nt = len(tht)
            ret = np.zeros( (2 * nt, 2 * lmax + 1), dtype=complex)
            if len(north) > 0:
                ret[north] = _vlm2vtm_northpole(s, vlm)
                ret[nt + north] = _vlm2vtm_southpole(s, vlm)
            if len(south) > 0:
                ret[south] = _vlm2vtm_southpole(s, vlm)
                ret[nt + south] = _vlm2vtm_northpole(s, vlm)
            if len(north) + len(south) < len(tht):
                others = np.where( (tht < np.pi) & (tht > 0.))[0]
                vtm =  fsht.vlm2vtm_sym(lmax, s, tht[others], vlm)
                ret[others] = vtm[:len(others)]
                ret[nt + others] = vtm[len(others):]
            return ret


def _vlm2vtm_northpole(s, vlm):
    """Spin-weight harmonics on the north pole

        :math: `_s\Lambda_{l,-s} (-1)^s  \sqrt{ (2l + 1) / 4\pi }`

        and zero for other m.

    """
    assert s >= 0, s
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    ret = np.zeros(2 * lmax + 1, dtype=complex)
    l = np.arange(lmax + 1)
    ret[lmax - s] =  np.sum(vlm[l * l + l - s] * np.sqrt((2 * l + 1.))) / np.sqrt(4. * np.pi)  * (-1) ** s
    return ret


def _vlm2vtm_southpole(s, vlm):
    """Spin-weight harmonics on the north pole.

        :math:`_s\Lambda_{l,s} (-1)^l  \sqrt{ (2l + 1) / 4\pi }`

        and zero for other m.

    """
    assert s >= 0, s
    lmax = int(np.sqrt(len(vlm)) - 1)
    assert (len(vlm) == (lmax + 1) ** 2)
    ret = np.zeros(2 * lmax + 1, dtype=complex)
    l = np.arange(lmax + 1)
    ret[lmax + s] =  np.sum(vlm[l * l + l + s] * (-1) ** l * np.sqrt((2 * l + 1.))) / np.sqrt(4. * np.pi)
    return ret

