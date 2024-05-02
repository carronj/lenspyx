import ducc0
import numpy as np
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.utils import timer
from lenspyx.remapping.utils_geom import Geom, st2mmax
from lenspyx.remapping.deflection_028 import ducc_sht_mode, ctype, rtype
from psutil import cpu_count


def _build_cc_geom(ringw, nphi):
    """Builds CC geometry discarding rings with zero weights, but same offset indices


    """
    ntheta = len(ringw)
    cc_geom = Geom.get_cc_geometry(ntheta, nphi)
    assert np.all(np.argsort(cc_geom.theta) == np.arange(cc_geom.theta.size))
    nzro = np.where(ringw != 0)[0]
    ofs = np.insert(np.cumsum(cc_geom.nph[nzro][:-1]), 0, 0)
    return nzro, Geom(cc_geom.theta[nzro],cc_geom.phi0[nzro], cc_geom.nph[nzro], ofs, ringw[nzro])

def gclm2dfs(gclm, mmax, spin, ringw=None, ntheta=None, numthreads=0, verbose=0, scale=1):
    """

    Args:
        gclm: healpy-like alm array
        mmax: mmax of input array
        spin: spin of the transform
        ringw:   optional set of weights ot apply on each ring
        ntheta:  optional number of rings in dfs map (defaults to good size of lmax_unl + 2)
        numthreads: optional number of threads (defaults to cpu_count())
        verbose: print execution time information
        scale: theta point in spherical map is set to theta * scale on DFS map, on a grid with 1 / scale less points

    Returns:
        doubled fourier sphere map and their fourier coefficients

    """
    tim = timer(True, prefix='gclm2dfs')
    gclm = np.atleast_2d(gclm)
    lmax_unl = Alm.getlmax(gclm[0].size, mmax)
    if numthreads <= 0:
        numthreads = cpu_count(logical=False)

    # convert gclm to dfs
    if ntheta is None:
        ntheta = ducc0.fft.good_size(lmax_unl + 2)
    if ringw is None:
        ringw = np.ones(ntheta, dtype=float)

    thtmax = np.max(np.linspace(0, np.pi, ntheta) * (ringw > 0))
    mmax_eff = int(np.ceil(st2mmax(spin, thtmax, lmax_unl)))
    nphihalf = ducc0.fft.good_size(mmax_eff + 1)
    nphi = 2 * nphihalf
    assert ntheta >= (lmax_unl + 2), ('this is weird', lmax_unl, ntheta)

    # Build geometry for which we must compute the spherical map
    nzro_rings, cc_geom = _build_cc_geom(ringw, nphi)
    if scale * thtmax > np.pi:
        print('****** it does not make much sense to send points outside the sphere...')
    if scale * thtmax >= np.pi - thtmax:
        print("It seems a bit odd to send theta points past their reflected points...")

    ntheta_dfs = ducc0.fft.good_size(int(np.round(ntheta / scale)))  # number of rings in CC DFS map
    print("Setting up DFS grid Nt Np %s %s"%(ntheta_dfs, nphi))

    # Is this any different to scarf wraps ?
    # NB: type of map, map_df, and FFTs will follow that of input gclm
    # relevant map values:
    map = cc_geom.synthesis(gclm, spin, lmax_unl, mmax, numthreads,  mode=ducc_sht_mode(gclm, spin))
    # we must now patch them onto the doubled fourier sphere
    map_dfs = np.zeros((2 * ntheta_dfs - 2, nphi), dtype=map.dtype if spin == 0 else ctype[map.dtype])
    # :Use 1d exluding rings for non trivial weights
    #map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi,
    #                                          spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=numthreads,
    #                                          mode=mode)
    tim.add('experimental.synthesis')
    # extend map to double Fourier sphere map
    #FIXME: this assumes contiguous values in nzrorings

    if spin == 0:
        map_dfs[nzro_rings, :] = map[0].reshape(cc_geom.theta.size, nphi)
    else:
        map_dfs[nzro_rings, :] = (map[0] + 1j * map[1]).reshape(cc_geom.theta.size, nphi)
    del map
    assert ringw.size == ntheta
    for ir, w in zip(nzro_rings, ringw[nzro_rings]):
        map_dfs[ir, :] *= w
    map_dfs[ntheta_dfs:, :nphihalf] = map_dfs[ntheta_dfs - 2:0:-1, nphihalf:]
    map_dfs[ntheta_dfs:, nphihalf:] = map_dfs[ntheta_dfs - 2:0:-1, :nphihalf]
    if (spin % 2) != 0:
        map_dfs[ntheta_dfs:, :] *= -1
    tim.add('map_dfs build')

    # go to Fourier space
    map_dfs_r = map_dfs.copy() #just saving this in case we want it
    if spin == 0:
        tmp = np.empty(map_dfs.shape, dtype=ctype[map_dfs.dtype])
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=numthreads, out=tmp)
        del tmp
    else:
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=numthreads, out=map_dfs)
    tim.add('map_dfs 2DFFT')
    if verbose:
        print(tim)
    tht_dfs = np.linspace(0, 2 * np.pi, ntheta_dfs * 2 - 2, endpoint=False)
    thtfac = (ntheta - 1.) / (ntheta_dfs -1.) # factor to convert theta of spherical map to theta of dfs
    return tht_dfs, map_dfs_r, map_dfs, thtfac