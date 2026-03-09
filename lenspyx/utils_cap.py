import numpy as np
from lenspyx.remapping.utils_geom import Geom
import pylab as pl

class args_default:
    def __init__(self):
        self.lmax_len = 4000
        self.dlmax_gl = 500
        self.spin = 2
        self.nt = 4
        self.HL = 0
        self.alloc = 0
        self.tracemalloc = False
        self.epsilon = 7
        self.gonly = False
        self.dlmax = 500
        self.whiten = True # make the spectra white before interpolation
        self.apofct = 0
        self.dl=7
        self.adapt_mmax = True


def fskycap(thtcap):
    return (1. - np.cos(thtcap)) * 0.5


class transition01:
    """Helpers for transition functions equal to 1 at 0 and 0 at 1"""
    @staticmethod
    def _eval01(x, version):
        assert np.all( (x < 1) & (x > 0))
        if version in ['ES', 'es', 3]:
            # Exponential of semi-circle, with b giving 1/2 at 1/2
            beta = np.log(0.5) / (np.sqrt(3/4.) - 1.)
            return np.exp(beta * (np.sqrt(1. - x ** 2) - 1))
        if version in ['KB', 'kb', 4]: # Kaiser-bessel
            b = 5.74
            return np.i0(b * np.sqrt(1. - x ** 2)) / np.i0(b)
        if version in [0]: # Smooth typical bump function. this is 1/2 at 1/2
            fx  = np.exp(-1. / x)
            f1x = np.exp(-1. / (1 - x))
            return f1x / (fx + f1x)
        if version in ['Hann', 1]:
            return np.cos(x * np.pi * 0.5) ** 2
        if version in [2]: # another smooth bump
            return np.exp(-1. /( 1 - x ** 2) + 1.)

    @staticmethod
    def eval(x, version):
        ret = np.zeros(x.size)
        ret[np.where(x <= 0.)] = 1.
        ret[np.where(x >= 1.)] = 0.
        i = np.where( (x > 0) & (x < 1))
        ret[i] = transition01._eval01(x[i], version)
        return ret

    @staticmethod
    def eval_fft(npts, version):
        x = np.arange(npts) * ( (2 * np.pi) / npts ) - np.pi
        return np.fft.fft(transition01.eval(x, version))


    @staticmethod
    def bump(x, x1, dx, version):
        """bump, equal to 1 on [-1, 1], and 0 outside of [-1-dx, 1+dx]

        """
        assert dx >= 0, dx
        ret = np.zeros(x.size)
        ax = np.abs(x)
        ret[np.where(ax <= x1)] = 1.
        ret[np.where(ax >= (x1 + dx))] = 0.
        i = np.where( (ax > x1) & (ax < (x1 + dx)) )
        ret[i] = transition01.eval( (ax[i] - x1) / dx, version)
        return ret

    @staticmethod
    def bump_fft(npts, x1, dx, version):
        x = np.arange(npts) * ( (2 * np.pi) / npts ) - np.pi
        bp = transition01.bump(x, x1, dx, version)
        return np.fft.fft(bp)

def Nc(lmax, dl, thetacap, dtheta):
    """

    Args:
        lmax: band-limit of the alm array
        dl: Effective band-limit of the apodization window (for white spectra, dl=7 gives single precision results?)
        thetacap: co-latitude angle defining the cap
        dtheta: angular distance dedicated to apodization

        This should have a mimimum at sqrt(dl / lmax * (thetcap /pi))

    Returns:

    """
    return (lmax + np.pi * dl / dtheta) * (thetacap + dtheta) / np.pi

# parametrize interval
def eps_opti(lmax, thetacap, dl=7):
    """Guess of optimum choice of apodization length, given the alm band-limit lmax and coordinate of the cap, and the [0, pi) window bandlimit"""
    guess = np.sqrt(dl / lmax * np.pi / thetacap * np.pi)
    return min(guess, (np.pi / thetacap - 1.)*(1.-1e-13)) # Cant overshoot

def examine(ref, diff, geom:Geom, epsilon_ref, label=''):
    """Look at difference to the ref map ring by ring


    """
    rad2deg = 180 / np.pi
    # extracts rings and plot rms dev
    rms = np.zeros(geom.theta.size)
    thta = np.min(geom.theta)
    thtb = np.max(geom.theta)
    dtheta = 0.1 * (thtb - thta)
    for i, ir in enumerate(np.argsort(geom.theta)):
        pix = geom.rings2pix(geom, [ir])
        rms[i] = np.sqrt(np.mean(diff[pix] ** 2)) /np.sqrt(np.mean(ref[pix] ** 2))
    pl.semilogy(np.sort(geom.theta) * rad2deg, rms, label=label)
    pl.axhline(epsilon_ref,c='k')
    pl.axvline(thta * rad2deg, c='grey')
    pl.axvline(thtb * rad2deg, c='grey')
    pl.xlim( max((thta-1.2 * dtheta) * rad2deg, 0.), min((thtb + 1.2 * dtheta) * rad2deg, np.pi * rad2deg))
    pl.xlabel(r'$\theta$ [deg]')
    pl.ylabel(r'rel dev. (rms)')
    return rms