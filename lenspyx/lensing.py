from __future__ import print_function, annotations
from os import cpu_count
import numpy as np
from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping.deflection_029 import deflection
from lenspyx import cachers


def alm2lenmap(alm, dlms, geometry:tuple[str, dict]=('healpix', {'nside':2048}), epsilon=1e-7, verbose=True, nthreads:int=0):
    r"""Computes a deflected spin-0 healpix map from its alm and deflection field alm.

        Args:
            alm: undeflected map healpy alm array
            dlms: The spin-1 deflection, in the form of a list or akin of two healpy alm arrays.

                    The two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.

            epsilon(optional): target accuracy of the result (defaults to 1e-7)
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.
            nthreads(optional): number of threads to use (defaults to os.cpu_count())

        Returns:
            lensed map, array of size given by the number of pixels of input geometry


    """
    assert len(dlms) == 2, len(dlms)
    if nthreads <= 0:
        nthreads = cpu_count()
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']))(**(geometry[1]))
    defl = deflection(geo, dlms[0], None, dclm=dlms[1], epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    ret = defl.gclm2lenmap(alm, None, 0, False)
    if verbose:
        print(defl.tim)
    return ret


def alm2lenmap_spin(gclm, dlms, spin:int, geometry:tuple[str, dict]=('healpix', {'nside':2048}), epsilon:float=1e-7, verbose=True, nthreads:int=0):
    r"""Computes a deflected spin-weight Healpix map from its gradient and curl modes and deflection field alm.

        Args:
            gclm: list with undeflected map healpy gradient and curl array (e.g. polarization Elm and Blm).
                    The curl mode can be set to None if irrelevant.

            dlms: The spin-1 deflection, in the form of a list or akin of two healpy alm arrays.

                    The two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.


            spin: spin-weight of the maps to deflect (e.g. 2 for polarization).
            epsilon(optional): target accuracy of the result (defaults to 1e-7)
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.
            nthreads(optional): number of threads to use (defaults to os.cpu_count())


        Returns:
            lensed maps for input geometry (real and imaginary parts
            array of size given by the number of pixels of input geometry


    """
    if spin == 0:
        return alm2lenmap(gclm, dlms, geometry=geometry, epsilon=epsilon, verbose=verbose, nthreads=nthreads)
    assert len(gclm) == 2, len(gclm)
    assert len(dlms) == 2, len(dlms)
    if nthreads <= 0:
        nthreads = cpu_count()

    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']))(**(geometry[1]))
    defl = deflection(geo, dlms[0], None, dclm=dlms[1], epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if gclm[1] is None:
        gclm[1] = np.zeros_like(gclm[0])
    ret = defl.gclm2lenmap(gclm, None, spin, False)
    if verbose:
        print(defl.tim)
    return ret

