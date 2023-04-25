"""This module covers quadratic-estimation tools.


    It is essentially the same as plancklens but more flexible, faster and platform independent


"""
from os import cpu_count
import numpy as np
from lenspyx.qest import utils_qe as uqe
from lenspyx import utils_hp
from lenspyx.remapping.utils_geom import Geom


def eval_qe(qe_key, lmax_ivf, cls_weight, get_alm, lmax_qlm, verbose=True, get_alm2=None, geometry=None):
    """Evaluates a quadratic estimator gradient and curl terms.

        (see 'library' below for QE estimation coupled to CMB inverse-variance filtered simulation libraries,
        whose implementation can be faster for some estimators.)

        Args:
            qe_key: QE key defining the estimator (as defined in the qresp module), e.g. 'ptt' for lensing TT estimator
            lmax_ivf: CMB multipoles up to lmax are used in the QE
            cls_weight: set of CMB spectra entering the QE estimator weights
            get_alm: callable with 't', 'e', 'b' arguments, returning the corresponding inverse-variance filtered CMB map
            nside: the estimator are calculated in position space at healpy resolution nside.
            lmax_qlm: gradient and curl terms are obtained up to multipole lmax_qlm.
            verbose(optional): some printout if set
            get_alm2(optional): maps for second leg if different from first. The estimator is symmetrized
            geometry(optional): intermediate sphere pixelization (defaults to thinned-GL)

        Returns:
            glm and clm healpy arrays (gradient and curl terms of the QE estimate)

    """
    qe_list = _get_qes(qe_key, lmax_ivf, cls_weight)
    if geometry is None:
        qe_spin = np.max([qe[0].spin_ou + qe[1].spin_ou for qe in uqe.qe_compress(qe_list)])
        geometry = Geom.get_thingauss_geometry((2 * lmax_ivf + lmax_qlm) // 2 + 1, qe_spin)
    return _eval_qe(qe_list, get_alm, lmax_qlm, verbose=verbose, get_alm2=get_alm2, geo=geometry)


def _eval_qe(qe_list, get_alm, lmax_qlm, geo:Geom,
             verbose=True, get_alm2=None, mmax_qlm:int or None=None, nthreads=0):
    """Evaluation of a QE from its list of leg definitions.

        Args:
            qe_list: list of qe instances
            get_alm: callable with 't', 'e', 'b' arguments, giving the corresponding inverse-variance filtered CMB maps
            lmax_qlm: outputs are given up to multipole lmax_qlm
            get_alm2 : callable for second leg if different from the first (symmetrizes estimator by default)

        Returns:
            glm and clm healpy arrays (gradient and curl terms of the QE estimate)

    """
    if nthreads <= 0:
        nthreads = cpu_count()
    if get_alm2 is None:
        get_alm2 = get_alm
    if mmax_qlm is None:
        mmax_qlm = lmax_qlm

    symmetrize = not (get_alm2 is get_alm)
    qes = uqe.qe_compress(qe_list, verbose=verbose)
    qe_spin = qes[0][0].spin_ou + qes[0][1].spin_ou
    cL_out = qes[0][-1](np.arange(lmax_qlm + 1))
    assert qe_spin >= 0, qe_spin
    for q in qes[1:]:
        assert np.all(q[-1](np.arange(lmax_qlm + 1)) == cL_out)
        assert q[0].spin_ou + q[1].spin_ou == qe_spin
    ncomp, npix = 1 + (qe_spin != 0), geo.npix()
    if qe_spin != 0:
        dc = np.zeros((1, npix,), dtype=complex)
        dr = dc.view(float).reshape((npix, 2)).T # Real view onto complex array
    else:
        dr = np.zeros((1, npix,), dtype=float)
        dc = dr
    for i, q in enumerate(qes):
        if verbose:
            print("QE %s out of %s :"%(i + 1, len(qes)))
            print("in-spins 1st leg and out-spin", q[0].spins_in, q[0].spin_ou)
            print("in-spins 2nd leg and out-spin", q[1].spins_in, q[1].spin_ou)
        dc += q[0](get_alm, geo) * q[1](get_alm2, geo)
        if symmetrize:
            dc += q[0](get_alm2, geo) * q[1](get_alm, geo)
    gclm = geo.adjoint_synthesis(m=dr, spin=qe_spin, lmax=lmax_qlm, mmax=mmax_qlm, nthreads=nthreads)
    if symmetrize:
        gclm *= 0.5
    if qe_spin == 0:
        gclm *= -1 # We use here different spin-0 gradient convention than ducc
    assert gclm.ndim == 2 and len(gclm) == ncomp
    utils_hp.almxfl(gclm[0], cL_out, mmax_qlm, inplace=True)
    if ncomp == 2:
        utils_hp.almxfl(gclm[1], cL_out, mmax_qlm, inplace=True)
    return gclm


def _get_qes(qe_key: str, lmax: int, cls_weight:dict):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_weight (dict): CMB spectra entering the weights (when relevant).

    The weights are defined by their action on the inverse-variance filtered spin-weight $ _{s}\bar X_{lm}$.

    """
    lmax2 = lmax
    if qe_key[0] in ['p', 'x', 'a', 'f', 's']:
        if qe_key in ['ptt', 'xtt', 'att', 'ftt', 'stt']:
            s_lefts= [0]
        elif qe_key in ['p_p', 'x_p', 'a_p', 'f_p']:
            s_lefts= [-2, 2]
        else:
            s_lefts = [0, -2, 2]
        qes = []
        s_rights_in = s_lefts
        for s_left in s_lefts:
            for sin in s_rights_in:
                sout = -s_left
                s_qe, irr1, cl_sosi, cL_out =  _get_covresp(qe_key[0], sout, sin, cls_weight, lmax2)
                if np.any(cl_sosi):
                    lega = uqe.qeleg(s_left, s_left, 0.5 *(1. + (s_left == 0)) * np.ones(lmax + 1, dtype=float))
                    legb = uqe.qeleg(sin, sout + s_qe, 0.5 * (1. + (sin == 0)) * 2 * cl_sosi)
                    qes.append(uqe.qe(lega, legb, cL_out))
        if len(qe_key) == 1 or qe_key[1:] in ['tt', '_p']:
            return uqe.qe_simplify(qes)
        elif qe_key[1:] in ['te', 'et', 'tb', 'bt', 'ee', 'eb', 'be', 'bb']:
            return uqe.qe_simplify(uqe.qe_proj(qes, qe_key[1], qe_key[2]))
        elif qe_key[1:] in ['_te', '_tb', '_eb']:
            return uqe.qe_simplify(uqe.qe_proj(qes, qe_key[2], qe_key[3]) + uqe.qe_proj(qes, qe_key[3], qe_key[2]))
        else:
            assert 0, 'qe key %s  not recognized'%qe_key
    else:
        assert 0, qe_key + ' not implemented'

def _get_resp_legs(source, lmax):
    r"""Defines the responses terms for a CMB map anisotropy source.

    Args:
        source (str): anisotropy source (e.g. 'p', 'f', ...).
        lmax (int): responses are given up to lmax.

    Returns:
        4-tuple (r, rR, -rR, cL):  source spin response *r* (positive or zero),
        the harmonic responses for +r and -r (2 1d-arrays), and the scaling between the G/C modes
        and the potentials of interest. (for lensing, cL is given by :math:`L\sqrt{L (L + 1)}`).

    """
    if source in ['p', 'x']:
        # lensing (gradient and curl): _sX -> _sX -  1/2 alpha_1 \eth _sX - 1/2 \alpha_{-1} \bar \eth _sX
        return {s : (1, -0.5 * uqe.get_spin_lower(s, lmax), -0.5 * uqe.get_spin_raise(s, lmax),
                     lambda ell : uqe.get_spin_raise(0, np.max(ell))[ell]) for s in [0, -2, 2]}
    if source == 'f': # Modulation: _sX -> _sX + f _sX.
        return {s : (0, 0.5 * np.ones(lmax + 1, dtype=float), 0.5 * np.ones(lmax + 1, dtype=float),
                        lambda ell: np.ones(len(ell), dtype=float)) for s in [0, -2, 2]}
    if source in ['a', 'a_p']: # Polarisation rotation _\pm 2 X ->  _\pm 2 X + \mp 2 i a _\pm 2 X
        ret = {s: (0,  -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                       -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                        lambda ell: np.ones(len(ell), dtype=float)) for s in [-2, 2]}
        ret[0]=(0, np.zeros(lmax + 1, dtype=float),
                   np.zeros(lmax + 1, dtype=float),
                   lambda ell: np.ones(len(ell), dtype=float))
        return ret

    assert 0, source + ' response legs not implemented'

def _get_covresp(source, s1, s2, cls, lmax):
    r"""Defines the responses terms for a CMB covariance anisotropy source.

        \delta < s_d(n) _td^*(n')> \equiv
        _r\alpha(n) W^{r, st}_l _{s - r}Y_{lm}(n) _tY^*_{lm}(n') +
        _r\alpha^*(n') W^{r, ts}_l _{s}Y_{lm}(n) _{t-r}Y^*_{lm}(n')

    """
    if source in ['p','x', 'f', 'a', 'a_p']:
        # Lensing, modulation, or pol. rotation field from the field representation
        s_source, prR, mrR, cL_scal = _get_resp_legs(source, lmax)[s1]
        coupl = uqe.spin_cls(s1, s2, cls)[:lmax + 1]
        return s_source, prR * coupl, mrR * coupl, cL_scal
    elif source in ['stt', 's']: # Point sources
        cond = s1 == 0 and s2 == 0
        s_source = 0
        prR = 0.25 * cond * np.ones(lmax + 1, dtype=float)
        mrR = 0.25 * cond * np.ones(lmax + 1, dtype=float)
        cL_scal =  lambda ell : np.ones(len(ell), dtype=float)
        return s_source, prR, mrR, cL_scal
    elif source in ['ntt', 'n']:
        assert 0, 'dont think this parametrization works here'
    else:
        assert 0, 'source ' + source + ' cov. response not implemented'