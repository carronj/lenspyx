from __future__ import print_function

import numpy as np
import time
import os, sys
import healpy as hp
from lenspyx.shts import shts
from lenspyx import bicubic

class timer():
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = np.copy(self.t0)
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix

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

def vtm2filtmap(spin, vtm, nphi, threads=None,phiflip=()):
    return shts.vtm2map(spin, vtm, nphi, pfftwthreads=threads, bicubic_prefilt=True, phiflip=phiflip)

def _buildangles( tht_phi, Red,Imd):
    """
    e.g.
        Redtot, Imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
        pix = hp.query_strip(nside, th1, np.max(tht_patch), inclusive=True)
        costnew,phinew = buildangles(hp.pix2ang(nside, pix),Redtot[pix],Imdtot[pix])
    """
    costnew = np.cos(tht_phi[0])
    phinew = tht_phi[1].copy()
    norm = np.sqrt(Red ** 2 + Imd ** 2)
    ii = np.where(norm > 0.)
    costnew[ii] = np.cos(norm[ii]) * costnew[ii] - np.sin(norm[ii]) * np.sin(tht_phi[0][ii]) * (Red[ii] / norm[ii])
    ii = np.where( (norm > 0.) & (costnew ** 2 < 1.))
    phinew[ii] += np.arcsin((Imd[ii] / norm[ii]) * np.sin(norm[ii]) / (np.sqrt(1. - costnew[ii] ** 2)))
    return np.arccos(costnew),phinew

def get_Nphi(th1, th2, facres=0, target_amin=0.745):
    """ Calculates a phi sampling density at co-latitude theta """
    # 0.66 corresponds to 2 ** 15 = 32768
    sint = max(np.sin(th1), np.sin(th2))
    for res in np.arange(15, 3, -1):
        if 2. * np.pi / (2 ** (res-1)) * 180. * 60 /np.pi * sint >= target_amin : return 2 ** (res + facres)
    assert 0


class thgrid():
    def __init__(self, th1, th2):
        """
        Co-latitudes th1 and th2 between 0 (N. pole) and pi (S. Pole).
        negative theta values are reflected across the north pole.
        (this allows simple padding to take care of the non periodicty of the pine in theta direction.)
        Same for south pole.
        """
        self.th1 = th1
        self.th2 = th2

    def mktgrid(self, nt):
        return self.th1 + np.arange(nt) * ( (self.th2- self.th1) / (nt-1))

    def togridunits(self, tht, nt):
        return (tht - self.th1) / ((self.th2- self.th1) / (nt-1))


def _th2colat(th):
    ret = np.abs(th)
    ret[np.where(th > np.pi)] = 2 * np.pi - th[np.where(th > np.pi)]
    return ret


def tlm2lenmap(nside, tlm, dlm, verbose=True, nband=8, facres=0):
    return lens_glm_sym_timed(0, dlm, -tlm, nside,nband=nband, facres=facres)


def lens_glm_sym_timed(spin, dlm, glm, nside, nband=8, facres=0, clm=None, rotpol=True):
    """
    Same as lens_alm but lens simultnously a North and South colatitude band,
    to make profit of the symmetries of the spherical harmonics.
    """
    assert spin >= 0,spin
    t = timer(True,suffix=' ' + __name__)
    target_nt = 3 ** 1 * 2 ** (11 + facres) # on one hemisphere
    times = {}

    #co-latitudes
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
    ret = np.zeros(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    Nt_perband = int(target_nt / nband)
    t0 = time.time()
    Redtot, Imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
    times['dx,dy (full sky)'] = time.time() - t0
    times['dx,dy band split'] = 0.
    times['pol. rot.'] = 0.
    t.checkpoint('healpy Spin 1 transform for displacement (full %s map)' % nside)
    _Npix = 0 # Total number of pixels used for interpolation
    def coadd_times(tim):
        for _k in tim.keys():
            if _k not in times :
                times[_k] = tim[_k]
            else : times[_k] += tim[_k]

    for ib, th1, th2 in zip(range(nband), th1s, th2s):
        print("BAND %s in %s :"%(ib, nband))
        t0 = time.time()
        pixN = hp.query_strip(nside, th1, th2, inclusive=True)
        pixS = hp.query_strip(nside, np.pi- th2,np.pi - th1, inclusive=True)
        tnewN,phinewN = _buildangles(hp.pix2ang(nside, pixN),Redtot[pixN],Imdtot[pixN])
        tnewS,phinewS = _buildangles(hp.pix2ang(nside, pixS), Redtot[pixS], Imdtot[pixS])

        # Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        matnewN = np.max(tnewN)
        mitnewN = np.min(tnewN)
        matnewS = np.max(tnewS)
        mitnewS = np.min(tnewS)
        buffN = 10 * (matnewN - mitnewN) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        buffS = 10 * (matnewS - mitnewS) / (Nt_perband - 1) / (1. - 2. * 10. / (Nt_perband - 1))
        _thup = min(np.pi - (matnewS + buffS),mitnewN - buffN)
        _thdown = max(np.pi - (mitnewS - buffS),matnewN + buffN)

        #print "min max tnew (degrees) in the band %.3f %.3f "%(_th1 /np.pi * 180.,_th2 /np.pi * 180.)
        #==== these are the theta and limits. It is ok to go negative or > 180
        print('input t1,t2 %.3f %.3f in degrees'%(_thup /np.pi * 180,_thdown/np.pi * 180.))
        print('North %.3f and South %.3f buffers in amin'%(buffN /np.pi * 180 * 60,buffS/np.pi * 180. * 60.))
        Nphi = get_Nphi(_thup, _thdown, facres=facres)
        dphi_patch = (2. * np.pi) / Nphi * max(np.sin(_thup),np.sin(_thdown))
        dth_patch = (_thdown - _thup) / (Nt_perband -1)
        print("cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180))
        times['dx,dy band split'] += time.time() - t0
        if spin == 0:
            lenN,lenS,tim = lens_band_sym_timed(glm,_thup,_thdown,Nt_perband, tnewN,phinewN, tnewS,phinewS,nphi=Nphi)
            ret[pixN] = lenN
            ret[pixS] = lenS
        else :
            lenNR,lenNI,lenSR,lenSI,tim = gclm2lensmap_symband_timed(spin, glm, _thup, _thdown, Nt_perband,
                                                            (tnewN, phinewN), (tnewS, phinewS), Nphi=Nphi, clm = clm)
            ret[pixN] = lenNR + 1j * lenNI
            ret[pixS] = lenSR + 1j * lenSI
            t0 = time.time()
            if rotpol and spin > 0 :
                ret[pixN] *= polrot(spin,ret[pixN], hp.pix2ang(nside, pixN)[0],Redtot[pixN],Imdtot[pixN])
                ret[pixS] *= polrot(spin,ret[pixS], hp.pix2ang(nside, pixS)[0],Redtot[pixS],Imdtot[pixS])
                times['pol. rot.'] += time.time() -t0
        coadd_times(tim)

        #coadd_times(tim)
        _Npix += 2 * Nt_perband * Nphi
    t.checkpoint('Total exec. time')

    print("STATS for lmax tlm %s lmax dlm %s"%(hp.Alm.getlmax(glm.size),hp.Alm.getlmax(dlm.size)))
    tot= 0.
    for _k in times.keys() :
        print('%20s: %.2f'%(_k, times[_k]))
        tot += times[_k]
    print("%20s: %2.f sec."%('tot',tot))
    print("%20s: %2.f sec."%('excl. defl. angle calc.',tot - times['dx,dy (full sky)']-times['dx,dy band split']))
    print('%20s: %s = %.1f^2'%('Tot. int. pix.',int(_Npix),np.sqrt(_Npix * 1.)))

    return ret


def lens_band_sym_timed(glm, th1, th2, nt, tnewN, phinewN, tnewS, phinewS, nphi=2 ** 14):
    """
    Same as lens_band_sym with some more timing info.
    """
    assert len(tnewN) == len(phinewN) and len(tnewS) == len(phinewS)

    tgrid = thgrid(th1, th2).mktgrid(nt)
    times = {}
    t0 = time.time()
    vtm = shts.glm2vtm_sym(0, _th2colat(tgrid), glm)
    times['vtm'] = time.time() - t0
    t0 = time.time()
    filtmapN = vtm2filtmap(0, vtm[0:nt], nphi, phiflip=np.where((tgrid < 0))[0])
    filtmapS = vtm2filtmap(0, vtm[slice(2 * nt - 1, nt - 1, -1)], nphi, phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    times['vtm2filtmap'] = time.time() - t0
    del vtm
    t0 = time.time()
    lenmaps = []
    for N, filtmap, (tnew, phinew) in zip([1, 0], [filtmapN, filtmapS],
                                                  [(tnewN, phinewN), (tnewS, phinewS)]):
        tnew = thgrid(th1, th2).togridunits(tnew, nt) if N else thgrid(np.pi - th2, np.pi - th1).togridunits(tnew, nt)
        phinew /= ((2. * np.pi) / nphi)
        lenmaps.append(bicubic.deflect(filtmap, tnew, phinew))
    times['interp'] = time.time() - t0
    return lenmaps[0], lenmaps[1], times
