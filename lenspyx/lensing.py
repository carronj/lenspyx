from __future__ import print_function

import numpy as np
import healpy as hp
from lenspyx.shts import shts
from lenspyx.bicubic import bicubic
from lenspyx import utils
from lenspyx import angles

def alm2lenmap(alm, dlm, nside, dclm=None, facres=-1, nband=8, verbose=True):
    """Computes a deflected spin-0 Healpix map from its alm and deflection field alm.

        Args:
            alm: undeflected map healpy alm array
            dlm: deflection field healpy alm array, gradient part. This is :math:`\sqrt{L(L+1)}\phi_{LM}`
            nside: desired Healpix resolution of the deflected map
            dclm(optional): deflection field healpy alm array, curl part.
            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ 0.7amin * 2 ** facres.
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.

        Returns:
            Deflected healpy map at resolution nside.


    """
    if dclm is not None: assert hp.Alm.getlmax(dclm) == hp.Alm.getlmax(dlm)
    return _lens_gclm_sym_timed(0, dlm, -alm, nside, dclm=dclm, nband=nband, facres=facres, verbose=verbose)

def gclm2lenmap(gclm, dlm, nside, spin, dclm=None, nband=8, facres=-1, verbose=True):
    """Computes a deflected spin-weight Healpix map from its gradient and curl modes and deflection field alm.

        Args:
            gclm: list with undeflected map healpy gradient and curl array (e.g. polarization Elm and Blm).
            dlm: deflection field healpy alm array, gradient part.
                 This is related to the usual lensing potential by :math:`\sqrt{L(L+1)}\phi_{LM}`
            nside: desired Healpix resolution of the deflected map
            spin: spin of the array to deflect.
            dclm(optional): deflection field healpy alm array, curl part.
            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ 0.7amin * 2 ** facres.
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.

        Returns:
            Deflected healpy map at resolution nside.


    """
    assert len(gclm) == 2
    if dclm is not None: assert hp.Alm.getlmax(dclm) == hp.Alm.getlmax(dlm)
    return _lens_gclm_sym_timed(spin, dlm, gclm[0], nside,
                                clm=gclm[1], dclm=dclm, nband=nband, facres=facres, verbose=verbose)

def _lens_gclm_sym_timed(spin, dlm, glm, nside, nband=8, facres=0, clm=None, dclm=None, verbose=True):
    """Performs the deflection by splitting the full latitude range into distinct bands which are done one at a time.

        See *_lens_gcband_sym* for the single band (north and south hemispheres) lensing.

    """
    assert spin >= 0,spin
    times = utils.timer(verbose, suffix=' ' + __name__)
    target_nt = 2 ** (13 + facres) # on one hemisphere

    #co-latitudes
    th1s = np.arange(nband) * (np.pi * 0.5 / nband)
    th2s = np.concatenate((th1s[1:],[np.pi * 0.5]))
    nt_perband = int(target_nt / nband)
    redtot, imdtot = hp.alm2map_spin([dlm, np.zeros_like(dlm) if dclm is None else dclm], nside, 1, hp.Alm.getlmax(dlm.size))
    times.add('defl. spin 1 transform')
    interp_pix = 0
    ret = np.empty(hp.nside2npix(nside),dtype = float if spin == 0 else complex)
    for ib, th1, th2 in zip(range(nband), th1s, th2s):
        if verbose: print("BAND %s in %s :"%(ib, nband))
        pixn = hp.query_strip(nside, th1, th2, inclusive=True)
        pixs = hp.query_strip(nside, np.pi- th2,np.pi - th1, inclusive=True)
        thtp, phipn = angles.get_angles(nside, pixn, redtot[pixn], imdtot[pixn], 'north')
        thtps, phips = angles.get_angles(nside, pixs, redtot[pixs], imdtot[pixs], 'south')

        # Adding a 10 pixels buffer for new angles to be safely inside interval.
        # th1,th2 is mapped onto pi - th2,pi -th1 so we need to make sure to cover both buffers
        mathtp = np.max(thtp); mithtp = np.min(thtp)
        mathtps = np.max(thtps); mithtps = np.min(thtps)
        buffN = 10 * (mathtp - mithtp) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        buffS = 10 * (mathtps - mithtps) / (nt_perband - 1) / (1. - 2. * 10. / (nt_perband - 1))
        th1 = min(np.pi - (mathtps + buffS),mithtp - buffN)
        th2 = max(np.pi - (mithtps - buffS),mathtp + buffN)

        #==== these are the theta and limits. It is ok to go negative or > 180
        if verbose: print('input t1,t2 %.3f %.3f in degrees'%(th1 /np.pi * 180,th2/np.pi * 180.))
        if verbose: print('North %.3f and South %.3f buffers in amin'%(buffN /np.pi * 180 * 60,buffS/np.pi * 180. * 60.))
        nphi = utils.get_nphi(th1, th2, facres=facres)
        dphi_patch = (2. * np.pi) / nphi * max(np.sin(th1),np.sin(th2))
        dth_patch = (th2 - th1) / (nt_perband -1)
        if verbose: print("cell (theta,phi) in amin (%.3f,%.3f)" % (dth_patch / np.pi * 60. * 180, dphi_patch / np.pi * 60. * 180))
        times.add('defl. angles calc.')
        len_nr, len_ni, len_sr, len_si = _lens_gcband_sym(spin, glm, th1, th2, nt_perband, nphi, thtp, phipn, thtps, phips,
                                                         clm=clm, times=times)
        if spin == 0:
            ret[pixn] = len_nr
            ret[pixs] = len_sr
        else :
            ret[pixn] = angles.rotation(nside, spin, pixn, redtot[pixn], imdtot[pixn]) * (len_nr + 1j * len_ni)
            ret[pixs] = angles.rotation(nside, spin, pixs, redtot[pixs], imdtot[pixs]) * (len_sr + 1j * len_si)
            times.add(r'pol. //-transport rot.')
        interp_pix += nphi * nt_perband * 2
    if verbose: print(times)
    if verbose: print(r"Number of interpolating pixels: %s ~ %s^2 "%(interp_pix, int(np.sqrt(interp_pix))))
    return ret


def _lens_gcband_sym(spin, glm, th1, th2, nt, nphi, thtpn, phipn, thtps, phips, clm=None, times=None):
    r"""Returns deflected maps between a co-latitude range in the North Hemisphere and its South hemisphere counterpart.

        This routine performs the deflection only, and not the rotation sourced by the local axes //-transport.

        Args:
            spin: spin of the field to deflect described by gradient glm and curl clm mode.
            glm:  gradient lm components of the field, in healpy format.
            th1:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on south hemisphere.
            th2:  input co-latitude range is (th1,th2) on North hemisphere and (pi - th2,pi -th1) on south hemisphere.
            nt:  Number of tht point to use for the interpolation.
            nphi : Number of phi points to use for the interpolation.
            thtpn: deflected colatitude on north hemisphere sector.
            phipn: deflected longitude on north hemisphere sector.
            thtps: deflected colatitude on south hemisphere sector.
            phips: deflected longitude on south hemisphere sector.

            clm(optional) : curl lm components of the field to deflect, in healpy ordering.

        Returns:
            real and imaginary parts of the north and south bands (re-n, im-n, re-s, im-s)

    """
    assert spin >= 0
    assert len(thtpn) == len(phipn) and len(thtps) == len(phips)
    if times is None: times = utils.timer(True)

    tgrid = utils.thgrid(th1, th2).mktgrid(nt)
    if spin == 0:
        vtm = shts.glm2vtm_sym(0, utils.thgrid.th2colat(tgrid), glm)
    else:
        vtm = shts.vlm2vtm_sym(spin, utils.thgrid.th2colat(tgrid), shts.util.alm2vlm(glm, clm=clm))
    times.add('vtm')

    # Band on north hemisphere
    filtmap = vtm2filtmap(spin, vtm[0:nt], nphi, phiflip=np.where((tgrid < 0))[0])
    t_grid = utils.thgrid(th1, th2).togridunits(thtpn, nt)
    p_grid = phipn / ((2. * np.pi) / nphi)
    times.add('vtm2filtmap')
    lenmapnr = bicubic.deflect(np.require(filtmap.real, float, requirements='F'), t_grid, p_grid)
    lenmapni = bicubic.deflect(np.require(filtmap.imag, float, requirements='F'), t_grid, p_grid) if spin > 0 else None
    times.add('interp')

    # Symmetric band on south hemisphere
    filtmap = vtm2filtmap(spin, vtm[slice(2 * nt - 1, nt - 1, -1)], nphi, phiflip=np.where((np.pi - tgrid) > np.pi)[0])
    t_grid = utils.thgrid(np.pi - th2, np.pi - th1).togridunits(thtps, nt)
    p_grid = phips / ((2. * np.pi) / nphi)
    times.add('vtm2filtmap')
    lenmapsr = bicubic.deflect(np.require(filtmap.real, float, requirements='F'), t_grid, p_grid)
    lenmapsi = bicubic.deflect(np.require(filtmap.imag, float, requirements='F'), t_grid, p_grid) if spin > 0 else None
    times.add('interp')
    return lenmapnr, lenmapni, lenmapsr, lenmapsi

def vtm2filtmap(spin, vtm, nphi, threads=None, phiflip=()):
    return shts.vtm2map(spin, vtm, nphi, pfftwthreads=threads, bicubic_prefilt=True, phiflip=phiflip)

