import time, os
from datetime import timedelta
import numpy as np
import sys
import lenspyx
from lenspyx.utils_hp import Alm
import json

def cli(cl):
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1. / cl[ii]
    return ret

class timer:
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = self.t0
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix
        self.keys = {}
        self.t0s = {}

    def __iadd__(self, othertimer):
        for k in othertimer.keys:
            if not k in self.keys:
                self.keys[k] = othertimer.keys[k]
            else:
                self.keys[k] += othertimer.keys[k]
        return self

    def reset(self):
        self.t0 = time.time()

    def reset_ti(self):
        self.ti = time.time()
        self.t0 = time.time()

    def start(self, key): # starts a new time tracker
        assert key not in self.t0s.keys()
        self.t0s[key] = time.time()

    def close(self, key): # close tracker and store result
        assert key in self.t0s.keys()
        if key not in self.keys.keys():
            self.keys[key]  = time.time() - self.t0s[key]
        else:
            self.keys[key] += time.time() - self.t0s[key]
        del self.t0s[key]

    def __str__(self):
        if len(self.keys) == 0:
            return r""
        maxlen = np.max([len(k) for k in self.keys])
        dt_tot = time.time() - self.ti
        s = ""
        ts = "\r  {0:%s}" % (str(maxlen) + "s")
        for k in self.keys:
            s += ts.format(k) + ":  [" + str(timedelta(seconds=self.keys[k])) + "] " + "(%.1f%%)  \n"%(100 * self.keys[k]/dt_tot)
        s += ts.format("Total") + ":  [" + str(timedelta(seconds=dt_tot)) + "] " + "d:h:m:s:mus"
        return s

    def add(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.t0
        self.t0 = t0

    def add_elapsed(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.ti

    def dumpjson(self, fn):
        json.dump(self.keys, open(fn, 'w'))

    def checkpoint(self, msg):
        dt = time.time() - self.t0
        self.t0 = time.time()

        if self.verbose:
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dhi = np.floor((self.t0 - self.ti) / 3600.)
            dmi = np.floor(np.mod((self.t0 - self.ti), 3600.) / 60.)
            dsi = np.floor(np.mod((self.t0 - self.ti), 60))
            sys.stdout.write("\r  %s   [" % self.prefix + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] "
                             + " (total [" + (
                                 '%02d:%02d:%02d' % (dhi, dmi, dsi)) + "]) " + msg + ' %s \n' % self.suffix)



def blm_gauss(fwhm, lmax, spin:int):
    """Computes spherical harmonic coefficients of a circular Gaussian beam
    pointing towards the North Pole

    See an example of usage
    `in the documentation <https://healpy.readthedocs.io/en/latest/blm_gauss_plot.html>`_

    Parameters
    ----------
    fwhm : float, scalar
        desired FWHM of the beam, in radians
    lmax : int, scalar
        maximum l multipole moment to compute
    spin : bool, scalar
        if True, E and B coefficients will also be computed

    Returns
    -------
    blm : array with dtype numpy.complex128
          lmax will be as specified
          mmax is equal to spin
    """
    fwhm = float(fwhm)
    lmax = int(lmax)
    mmax = spin
    ncomp = 2 if spin > 0 else 1
    nval = Alm.getsize(lmax, mmax)

    if mmax > lmax:
        raise ValueError("lmax value too small")

    blm = np.zeros((ncomp, nval), dtype=np.complex128)
    sigmasq = fwhm * fwhm / (8 * np.log(2.0))
    ls = np.arange(spin, lmax + 1)
    if spin == 0:
        blm[0, Alm.getidx(lmax, ls, spin)] = np.sqrt((2 * ls + 1) / (4.0 * np.pi)) * np.exp(-0.5 * sigmasq * ls * ls)

    if spin > 0:
        blm[0, Alm.getidx(lmax, ls, spin)] = np.sqrt((2 * ls + 1) / (32 * np.pi)) * np.exp(-0.5 * sigmasq * ls * ls)
        blm[1] = 1j * blm[0]
    return blm

def get_nphi(th1, th2, facres=0, target_amin=0.745):
    """Calculates a phi sampling density at co-latitude theta """
    # 0.66 corresponds to 2 ** 15 = 32768
    sint = max(np.sin(th1), np.sin(th2))
    for res in np.arange(15, 3, -1):
        if 2. * np.pi / (2 ** (res-1)) * 180. * 60 /np.pi * sint >= target_amin : return 2 ** (res + facres)
    assert 0


class thgrid(object):
    def __init__(self, th1, th2):
        self.th1 = th1
        self.th2 = th2

    @staticmethod
    def th2colat(th):
        ret = np.abs(th)
        ret[np.where(th > np.pi)] = 2 * np.pi - th[np.where(th > np.pi)]
        return ret

    def mktgrid(self, nt):
        return self.th1 + np.arange(nt) * ( (self.th2- self.th1) / (nt-1))

    def togridunits(self, tht, nt):
        return (tht - self.th1) / ((self.th2- self.th1) / (nt-1))


def nlm2lmax(nlm):
    """Returns the lmax for an array of alm with length nlm. """
    lmax = int(np.floor(np.sqrt(2 * nlm) - 1))
    assert ((lmax + 2) * (lmax + 1) // 2 == nlm)
    return lmax


def lmax2nlm(lmax):
    """Returns the length of the complex alm array required for maximum multipole lmax. """
    return (lmax + 1) * (lmax + 2) // 2


def alm2vlm(glm, clm=None):
    """Converts alm format -> vlm format coefficients. glm is gradient mode, clm is curl mode.

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
        ret[l * l + l + ms] = -glm[ms * (2 * lmax + 1 - ms) // 2 + l]
        ret[l * l + l - ms] = -(-1) ** ms * np.conj(glm[ms * (2 * lmax + 1 - ms) // 2 + l])

    if not clm is None:
        assert (len(clm) == len(glm))
        for l in range(0, lmax + 1):
            ms = np.arange(1, l + 1)
            ret[l * l + l] += -1.j * clm[l]
            ret[l * l + l + ms] += -1.j * clm[ms * (2 * lmax + 1 - ms) // 2 + l]
            ret[l * l + l - ms] += -(-1) ** ms * 1.j * np.conj(clm[ms * (2 * lmax + 1 - ms) // 2 + l])

    return ret


def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls, tensCls or ScalCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    with open(fname) as f:
        firstline = next(f)
    keys = [i.lower() for i in firstline.replace('\n', '').split(' ') if i.isalpha()][1:]
    cols = np.loadtxt(fname).transpose()

    ell = cols[0].astype(np.int64)
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)

    cls = {k : np.zeros(lmax + 1, dtype=float) for k in keys}

    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)

    w = lambda ell :ell * (ell + 1.) / (2. * np.pi)
    wpp = lambda ell : ell ** 2 * (ell + 1.) ** 2 / (2. * np.pi)
    wptpe = lambda ell :np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
    for i, k in enumerate(keys):
        if k == 'pp':
            we = wpp(ell)
        elif 'p' in k and ('e' in k or 't' in k):
            we = wptpe(ell)
        else:
            we = w(ell)
        cls[k][ell[idc]] = cols[i + 1][idc] / we[idc]
    return cls


class Drop:
    def __init__(self, a=0., b=1):
        """Smooth fct that drops from 1 at a to zero at b"""
        assert b > a, (a, b)
        self.a = a
        self.extent = b - a

    def x2eps(self, x):
        return (x - self.a) / self.extent

    def eval(self, x):
        eps = self.x2eps(x)
        if np.isscalar(x):
            if eps >= 1: return 0.
            if eps <= 0: return 1.
            return np.exp(1 - 1. / (1 - eps ** 2))
        ret = np.zeros_like(eps)
        ret[np.where(eps <= 0.)] = 1.
        ii = np.where( (eps > 0) & (eps < 1))
        ret[ii] = np.exp(1 - 1. / (1 - eps[ii] ** 2))
        return ret


def get_ffp10_cls(lmax=None):
    path2cls = os.path.dirname(lenspyx.__file__)
    cls_unl = camb_clfile(path2cls + '/data/cls/FFP10_wdipole_lenspotentialCls.dat', lmax=lmax)
    cls_len = camb_clfile(path2cls + '/data/cls/FFP10_wdipole_lensedCls.dat', lmax=lmax)
    cls_glen = camb_clfile(path2cls + '/data/cls/FFP10_wdipole_gradlensedCls.dat', lmax=lmax)
    assert np.all([k in cls_glen.keys() for k in ['tt', 'te', 'ee', 'bb']])
    return cls_unl, cls_len, cls_glen