from __future__ import print_function, annotations
from os import cpu_count
import numpy as np
from lenspyx.remapping.utils_geom import Geom
from lenspyx.remapping.deflection_029 import deflection
from lenspyx import cachers
from lenspyx.utils_hp import almxfl, Alm
from lenspyx.utils import timer
from numpy.random import default_rng


def get_geom(geometry: tuple[str, dict]=('healpix', {'nside':2048})):
    r"""Returns sphere pixelization geometry instance from name and arguments

        Note:
            Custom geometries can be defined following lenspyx.remapping.utils_geom.Geom

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1])


def alm2lenmap(alm, dlms, geometry: tuple[str, dict]=('healpix', {'nside':2048}), epsilon=1e-7, verbose=0, nthreads: int=0, pol=True):
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
            pol(optional): if True, input arrays are interpreted as T and E if there are two, T E B if there are 3, otherwise performs only spin-0 transforms.
                           Defaults to True.


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


def alm2lenmap_spin(gclm: np.ndarray or list, dlms:np.ndarray or list, spin:int, geometry: tuple[str, dict] = ('healpix', {'nside':2048}), epsilon: float=1e-7, verbose=0, nthreads: int=0):
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


def synfast(cls: dict, lmax=None, mmax=None, geometry=('healpix', {'nside': 2048}),
            epsilon=1e-7, nthreads=0, alm=False, seed=None, verbose=0):
    r"""Generate a set of lensed maps according to input spectra

        Args:
            cls(dict): dict of spectra and cross-spectra with keys of the form 'TT', 'TE', 'EE',  etc.
                       Accepted keys are:

                             'T' (or 't'): spin-0 intensity
                             'E': E-polarization
                             'B': B-polarization
                             'P': lensing potential
                             'O': lensing curl potential

                       The array must be the :math:`C_\ell`, not :math:`D_\ell`

                       If the auto-spectrum 'AA' is not present the 'A' field is assumed to be zero
                       (as a consequence, if neither 'P' and 'O' are present then the output maps are not lensed)

            lmax(int, optional): band-limit of the unlensed alms, infered from length of cls by default
            mmax(int, optional): maximum m of the unlensed alms, defaults to lmax
            geometry(tuple, optional): tuple of geometry name and parameters (defaults to healpix at nside 2048)
            epsilon(float, optional): desired accuracy of the output map (exec. time only has a weak dependence on this)
            nthreads(int, optional): number of threads used for non-uniform SHTs, defaults to os.cpu_count
            alm(bool, optional): returns also unlensed alms if True
            seed(int, optional): random generator seed for reproducible results, defaults to None
            verbose(bool, optional): some timing info if set, defaults to zero

        Returns:
            A dictionary with lensed maps, which contains
                'T' if 'TT' were present in the input cls and non-zero
                'QU  if 'EE' or 'BB' were present and non-zero
            if alm is set to True, returns the unlensed alms, together with a string indicating the ordering

    """
    tim = timer('synfast', False)
    lmax_cls = np.max([len(cl) - 1 for cl in cls.values()])
    if lmax is None:
        lmax = lmax_cls
    if mmax is None:
        mmax = lmax
    lmax = min(lmax, lmax_cls)
    cmb_labels = ['t', 'e', 'b', 'p', 'x']
    spec_labels = [k.lower() for k in cls.keys()]
    # First remove zero fields:
    zros = []
    for fg in spec_labels:
        assert len(fg) == 2 and fg[0] in cmb_labels and fg[1] in cmb_labels
        if fg[0] == fg[1]:
            assert np.all(cls[fg] >= 0), 'auto spectrum of %s must be >= 0' % fg
            if not np.any(cls[fg]):
                zros.append(fg[0])
    labelsf = []
    for fg in spec_labels:
        assert len(fg) == 2 and fg[0] in cmb_labels and fg[1] in cmb_labels
        if fg[0] not in zros and fg[1] not in zros:
            assert fg[0] + fg[0] in spec_labels, 'must have %s auto-spectrum' % fg[0]
            assert fg[1] + fg[1] in spec_labels, 'must have %s auto-spectrum' % fg[1]
            if fg[0] == fg[1]:
                for field in fg:
                    if field not in labelsf:
                        labelsf.append(field)
    assert len(labelsf) <= 4
    labels = ''
    for f in 'tebpx':  # This just sorts the present labels according to 'tebpx'
        labels += f * (f in labelsf)

    ncomp = len(labels)
    mat = np.empty((lmax + 1, ncomp, ncomp), dtype=float)
    for i, f in enumerate(labels):
        for j, g in enumerate(labels[i:]):
            mat[:, i + j, i] = cls.get(f + g, cls.get(g + f, np.zeros(lmax + 1, dtype=float)))[:lmax + 1]
    ts, vs = np.linalg.eigh(mat)
    assert np.all(ts >= 0.)  # Matrix not positive semidefinite
    for m, t, v in zip(mat, ts, vs):
        m[:] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))
    tim.add('cl matrix')
    # Build phases:
    alm_size = Alm.getsize(lmax, mmax)
    rng = default_rng(seed)
    phases = 1j * rng.standard_normal((ncomp, alm_size), dtype=float)
    phases += rng.standard_normal((ncomp, alm_size), dtype=float)
    phases *= np.sqrt(0.5)
    real_idcs = Alm.getidx(lmax, np.arange(lmax + 1, dtype=int), 0)
    phases[:, real_idcs] = phases[:, real_idcs].real * np.sqrt(2.)
    tim.add('phases generation')

    # Now builds alms. We might need more since we cannot handle now curl only transforms
    labels_wgrad = labels
    if 'b' in labels and 'e' not in labels:
        labels_wgrad = labels_wgrad.replace('b', 'eb')
    if 'o' in labels and 'p' not in labels:
        labels_wgrad = labels_wgrad.replace('o', 'po')
    alms = np.zeros((len(labels_wgrad), phases[0].size), dtype=complex)
    #    for L in Ls: #L @ L.T is full matrx
    for i, f in enumerate(labels):
        idx = labels_wgrad.index(f)
        alms[idx] += almxfl(phases[i], mat[:, i, i], mmax, False)
        for j in range(ncomp):
            fl = mat[:, i, j]
            if i != j and np.any(fl):
                alms[idx] += almxfl(phases[j], fl, mmax, False)
    del phases
    tim.add('alms from phases')
    maps = {}
    if 'p' in labels_wgrad or 'o' in labels_wgrad:  # There is actual lensing
        p2d = np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
        dglm, dclm = None, None
        if 'p' in labels_wgrad:
            dglm = alms[labels_wgrad.index('p')]
            almxfl(dglm, p2d, mmax, True)
        if 'o' in labels_wgrad:
            dclm = alms[labels_wgrad.index('o')]
            almxfl(dclm, p2d, mmax, True)
        if dglm is None:
            dglm = np.zeros_like(dclm)
        if dclm is None:
            dclm = np.zeros_like(dglm)
        if nthreads <= 0:
            nthreads = cpu_count()
        defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                          cacher=cachers.cacher_mem(safe=False))
        if 't' in labels_wgrad:
            maps['T'] = defl.gclm2lenmap(alms[0:1], mmax, 0, False).squeeze()
        if 'e' in labels_wgrad:
            i = labels_wgrad.index('e')
            maps['QU'] = defl.gclm2lenmap(alms[i:i + 1 + ('b' in labels_wgrad)], mmax, 2, False)
        tim.add('alm2lenmap')
    else:  # no lensing here
        geom = get_geom(geometry)
        if 't' in labels_wgrad:
            maps['T'] = geom.synthesis(alms[0:1], 0, lmax, mmax, nthreads)
        if 'e' in labels_wgrad:
            i = labels_wgrad.index('e')
            sht_mode = 'STANDARD' if 'b' in labels_wgrad else 'GRAD_ONLY'
            maps['QU'] = geom.synthesis(alms[i:i + 1 + ('b' in labels_wgrad)], 2, lmax, mmax, False, mode=sht_mode)
        tim.add('alm2map')
    if verbose:
        print(tim)
    return maps if not alm else (maps, (alms, labels_wgrad))
