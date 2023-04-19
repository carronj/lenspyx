from __future__ import print_function, annotations
import numpy as np
from os import cpu_count
from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping.deflection_029 import deflection
from lenspyx import cachers

def alm2lenmap(alm, dlms, geom:tuple[str, dict]=('healpix', {'nside':2048}), epsilon=1e-7, verbose=True, nthreads:int=0):
    r"""Computes a deflected spin-0 healpix map from its alm and deflection field alm.

        Args:
            alm: undeflected map healpy alm array
            dlms: The spin-1 deflection, in the form of a list or akin of two healpy alm arrays.

                    The two arrays are the gradient and curl deflection healpy alms.
                    (e.g. :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential)
                    The curl can be set to None if irrelevant.

            facres(optional): the deflected map is constructed by interpolation of the undeflected map,
                              built at target res. ~ :math:`0.7 * 2^{-\rm facres}.` arcmin
            nband(optional): To avoid dealing with too many large maps in memory, the operations is split in bands.
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to true.
            experimental(optional): well, that's experimental
            nthreads(optional): number of threads to use (defaults to os.cpu_count())

        Returns:
            Deflected healpy map at resolution nside.


    """
    assert len(dlms) == 2
    if nthreads <= 0:
        nthreads = cpu_count()
    geo = getattr(Geom, '_'.join(['get', geom[0], 'geometry']))(**(geom[1]))
    defl = deflection(geo, dlms[0], None, dclm=dlms[1], epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    ret = defl.gclm2lenmap(alm, None, 0, False)
    if verbose:
        print(defl.tim)
    return ret

def alm2lenmap_spin(gclm, dlms, spin:int, epsilon:float=1e-7, verbose=True, nthreads:int=0):
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
            Deflected healpy maps at resolution nside (real and imaginary parts).


    """
    if spin == 0:
        return alm2lenmap(gclm, dlms, nside, epsilon=epsilon, nband=nband, facres=facres, verbose=verbose,
                          experimental=experimental, nthreads=nthreads)
    assert len(gclm) == 2
    assert len(dlms) == 2
    if nthreads <= 0:
        nthreads = cpu_count()

    geom = Geom.get_healpix_geometry(nside)
    defl = deflection(geom, dlms[0], None, dclm=dlms[1], epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if gclm[1] is None:
        gclm[1] = np.zeros_like(gclm[0])
    ret = defl.gclm2lenmap(gclm, None, spin, False)
    if verbose:
        print(defl.tim)
    return ret

