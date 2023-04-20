from __future__ import print_function, annotations
from os import cpu_count
import numpy as np
from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping.deflection_029 import deflection
from lenspyx import cachers


def get_geom(geometry:tuple[str, dict]=('healpix', {'nside':2048})):
    r"""Returns sphere pixelization geometry instance from name and arguments

        Note:
            Custom geometries can be defined following lenspyx.remapping.utils_geom.Geom

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1])


def alm2lenmap(alm, dlms, geometry:tuple[str, dict]=('healpix', {'nside':2048}), epsilon=1e-7, verbose=0, nthreads:int=0, pol=True):
    r"""Computes lensed CMB maps from their alm's and deflection field alm's.

        Args:
            alm: undeflected map healpy alm array or sequence of arrays
            dlms: The spin-1 deflection, in the form of one or two arrays.

                    The two arrays are the gradient and curl deflection healpy alms:

                    :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential

                    :math:`\sqrt{L(L+1)}\Omega_{LM}` with :math:`\Omega` the lensing curl potential


                    The curl can be omitted if zero, resulting in principle in slightly faster transforms

            geometry(optional): sphere pixelization, tuple with geometry name and argument dictionary,
                                defaults to Healpix with nside 2048
            epsilon(optional): target accuracy of the result (defaults to 1e-7)
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to 0.
            nthreads(optional): number of threads to use (defaults to os.cpu_count())
            pol: if True, input arrays are interpreted as T and E if there are two, T E B if there are 3, otherwise performs only spin-0 transforms.


        Returns:
            lensed maps, each an array of size given by the number of pixels of input geometry.
            T, Q, U if pol and there 2 or 3 input arrays, otherwise spin-0 maps



    """
    if nthreads <= 0:
        nthreads = cpu_count()
        if verbose:
            print('alm2lenmap: using %s nthreads'%nthreads)
    if isinstance(dlms, list) or dlms.ndim > 1:
        assert len(dlms) <= 2
        dglm = dlms[0]
        dclm = None if len(dlms) == 1 else dlms[1]
    else:
        dglm = dlms
        dclm = None

    defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if isinstance(alm, list) or alm.ndim == 2:
        if pol and len(alm) in [2, 3]:
            T = defl.gclm2lenmap(alm[0], None, 0, False).squeeze()
            Q, U = defl.gclm2lenmap(np.array(alm[1:]), None, 2, False)
            if verbose:
                print(defl.tim)
            return T, Q, U
        else:
            ret = [defl.gclm2lenmap(a, None, 0, False).squeeze() for a in alm]
            if verbose:
                print(defl.tim)
            return ret
    ret = defl.gclm2lenmap(alm, None, 0, False).squeeze()
    if verbose:
        print(defl.tim)
    return ret


def alm2lenmap_spin(gclm:np.ndarray or list, dlms:np.ndarray or list, spin:int, geometry:tuple[str, dict]=('healpix', {'nside':2048}), epsilon:float=1e-7, verbose=0, nthreads:int=0):
    r"""Computes a deflected spin-weight lensed CMB map from its gradient and curl modes and deflection field alm.

        Args:
            gclm:  undeflected map healpy gradient (and curl, if relevant) modes
                    (e.g. polarization Elm and Blm).

            dlms: The spin-1 deflection, in the form of one or two arrays.

                    The two arrays are the gradient and curl deflection healpy alms:

                    :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential

                    :math:`\sqrt{L(L+1)}\Omega_{LM}` with :math:`\Omega` the lensing curl potential


                    The curl can be omitted if zero, resulting in principle in slightly faster transforms


            spin(int >= 0): spin-weight of the maps to deflect (e.g. 2 for polarization).
            geometry(optional): sphere pixelization, tuple with geometry name and argument dictionary,
                                defaults to Healpix with nside 2048
            epsilon(optional): target accuracy of the result (defaults to 1e-7)
            verbose(optional): If set, prints a bunch of timing and other info. Defaults to 0.
            nthreads(optional): number of threads to use (defaults to os.cpu_count())


        Returns:
            lensed maps for input geometry (real and imaginary parts),
            arrays of size given by the number of pixels of input geometry

        Note:

            If curl modes are zero (deflection and/or alm's to lens), they can be omitted, which can result in slightly faster transforms


    """
    if spin == 0:
        return alm2lenmap(gclm, dlms, geometry=geometry, epsilon=epsilon, verbose=verbose, nthreads=nthreads)
    if isinstance(dlms, list) or dlms.ndim > 1:
        assert len(dlms) <= 2
        dglm = dlms[0]
        dclm = None if len(dlms) == 1 else dlms[1]
    else:
        dglm = dlms
        dclm = None

    if nthreads <= 0:
        nthreads = cpu_count()
        if verbose:
            print('alm2lenmap_spin: using %s nthreads'%nthreads)

    defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                      cacher=cachers.cacher_mem(safe=False))
    if isinstance(gclm, list) and gclm[1] is None:
        gclm = gclm[0]
    ret = defl.gclm2lenmap(gclm, None, spin, False)
    if verbose:
        print(defl.tim)
    return ret

