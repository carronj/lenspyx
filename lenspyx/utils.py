import time
import numpy as np
import sys

class timer:
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = self.t0
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix
        self.keys = {}

    def __iadd__(self, othertimer):
        for k in othertimer.keys:
            if not k in self.keys:
                self.keys[k] = othertimer.keys[k]
            else:
                self.keys[k] += othertimer.keys[k]
        return self

    def reset(self):
        self.t0 = time.time()

    def __str__(self):
        if len(self.keys) == 0:
            return r""
        s = ""
        for k in self.keys:
            dt = self.keys[k]
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            #s +=  "%24s: %.1f"%(k, self.keys[k]) + '\n'
            s += "\r  %24s:  [" % k + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + "\n"
        dt = time.time() - self.ti
        dh = np.floor(dt / 3600.)
        dm = np.floor(np.mod(dt, 3600.) / 60.)
        ds = np.floor(np.mod(dt, 60))
        s += "\r  %24s:  [" % 'Total' + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] "
        return s

    def add(self, label):
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0  - self.t0
        self.t0 = t0

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
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls
