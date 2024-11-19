from __future__ import annotations
import numpy as np
from os import cpu_count
from lenspyx.remapping.utils_geom import Geom
from lenspyx.utils_hp import almxfl, alm_copy, Alm
from ducc0.misc import wigner3j_int

class qeleg:
    def __init__(self, spin_in: int, spin_out:int, cl:np.ndarray[float]):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def __eq__(self, leg):
        if self.spin_in != leg.spin_in or self.spin_ou != leg.spin_ou or self.get_lmax() != self.get_lmax():
            return False
        return np.all(self.cl == leg.cl)

    def __mul__(self, other):
        return qeleg(self.spin_in, self.spin_ou, self.cl * other)

    def __add__(self, other):
        assert self.spin_in == other.spin_in and self.spin_ou == other.spin_ou
        lmax = max(self.get_lmax(), other.get_lmax())
        cl = np.zeros(lmax + 1, dtype=float)
        cl[:len(self.cl)] += self.cl
        cl[:len(other.cl)] += other.cl
        return qeleg(self.spin_in, self.spin_ou, cl)

    def copy(self):
        return qeleg(self.spin_in, self.spin_ou, np.copy(self.cl))

    def get_lmax(self):
        return len(self.cl) - 1


class qeleg_multi:
    def __init__(self, spins_in, spin_out, cls):
        assert isinstance(spins_in, list) and isinstance(cls, list) and len(spins_in) == len(cls)
        self.spins_in = spins_in
        self.cls = cls
        self.spin_ou = spin_out

    def __iadd__(self, other_qe: qeleg):
        """Adds one spin_in/cl tuple.

        """
        assert other_qe.spin_ou == self.spin_ou, (other_qe.spin_ou, self.spin_ou)
        self.spins_in.append(other_qe.spin_in)
        self.cls.append(np.copy(other_qe.cl))
        return self

    def __call__(self, get_alm: callable, geometry: Geom, nthreads: int = 0):
        """Returns the spin-weighted real-space map of the estimator.

            We first build X_lm in the wanted _{si}X_lm _{so}Y_lm and then convert this alm2map_spin conventions.

        """
        if nthreads <= 0:
            nthreads = cpu_count()
        lmax = self.get_lmax()
        mmax = lmax
        ncomp, npix = 1 + (self.spin_ou != 0), geometry.npix()
        alm_size = Alm.getsize(lmax, mmax)
        gclm = np.zeros((ncomp, alm_size), dtype=complex)
        for i, (si, cl) in enumerate(zip(self.spins_in, self.cls)):
            assert si in [0, -2, 2], str(si) + ' input spin not implemented'
            alms = [get_alm('e'), get_alm('b')] if abs(si) == 2 else [-get_alm('t'), 0]
            sgn_g = (-1) ** si if si < 0 else 1
            gclm[0] += almxfl(alm_copy(alms[0], None, lmax, mmax), sgn_g * cl, mmax, False)
            if np.any(alms[1]) and ncomp > 1:
                sgn_c = (-1) ** si if si < 0 else -1
                gclm[1] += almxfl(alm_copy(alms[1], None, lmax, mmax), sgn_c * cl, mmax, False)
        if self.spin_ou > 0:
            gclm[1] *= -1
        elif self.spin_ou == 0:
            gclm *= -1
        if self.spin_ou == 0:
            return geometry.synthesis(gclm, abs(self.spin_ou), lmax, mmax, nthreads=nthreads)
        else:
            mc = np.empty((geometry.npix(),), dtype=complex)
            mr = mc.view(float).reshape((npix, 2)).T
            geometry.synthesis(gclm, abs(self.spin_ou), lmax, mmax, nthreads=nthreads, map=mr)
            if self.spin_ou < 0:
                assert len(mr) == 2
                if self.spin_ou % 2 == 1:
                    mr[0] *= -1
                if self.spin_ou % 2 == 0:
                    mr[1] *= -1
            return mc

    def get_lmax(self):
        return np.max([len(cl) for cl in self.cls]) - 1

    def is_conjugate(self, leg2: qeleg_multi, rtol=1e-14, atol=0.):
        """Tests whether leg2 is complex conjugate of leg1


        """
        cond =  self.spin_ou == -leg2.spin_ou
        cond *= (len(self.spins_in) == len(self.spins_in))
        cond *= (self.get_lmax() == leg2.get_lmax())
        if cond:
            ix = np.argsort(self.spins_in)
            jx = np.argsort(self.spins_in)[::-1]
            for i, j in zip(ix, jx): # Should cover relevant cases
                cond *= (self.spins_in[i] == -leg2.spins_in[j])
                sgn = 1 if (self.spin_ou + self.spins_in[i]) % 2 == 0 else -1
                cond *= np.allclose(self.cls[i], sgn * leg2.cls[j], rtol=rtol, atol=atol)
            return cond
        else:
            return False


class qe:
    def __init__(self, leg_a: qeleg, leg_b: qeleg, cL: callable):
        assert leg_a.spin_ou + leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def get_lmax_a(self):
        return self.leg_a.get_lmax()

    def get_lmax_b(self):
        return self.leg_b.get_lmax()


def qe_proj(qe_list:list[qe], a, b):
    """Projection of a list of QEs onto another QE using only a subset of maps.

        Args:
            qe_list: list of qe instances
            a: (in 't', 'e', 'b') The 1st leg of the output qes will only use this field
            b: (in 't', 'e', 'b') The 2nd leg of the output qes will only use this field
    """
    assert a in ['t', 'e', 'b'] and b in ['t', 'e', 'b']
    l_in = [0] if a == 't' else [-2, 2]
    r_in = [0] if b == 't' else [-2, 2]
    qes_ret = []
    for q in qe_list:
        si, ri = (q.leg_a.spin_in, q.leg_b.spin_in)
        if si in l_in and ri in r_in:
            leg_a = q.leg_a.copy()
            leg_b = q.leg_b.copy()
            if si == 0 and ri == 0:
                qes_ret.append(qe(leg_a, leg_b, q.cL))
            elif si == 0 and abs(ri) > 0:
                sgn = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a, leg_b * 0.5 * sgn, q.cL))
            elif ri == 0 and abs(si) > 0:
                sgn = 1 if a == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgn, leg_b, q.cL))
            elif abs(ri) > 0 and abs(si) > 0:
                sgna = 1 if a == 'e' else -1
                sgnb = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5 * sgnb, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5 * sgnb, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5, q.cL))
            else:
                assert 0, (si, ri)
    return qe_simplify(qes_ret)


def qe_simplify(qe_list: list[qe], _swap=False, verbose=False):
    """Simplifies a list of QE estimators by co-adding terms when possible.


    """
    skip = []
    qes_ret = []
    qes = [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qe_list] if _swap else qe_list
    for i, qe1 in enumerate(qes):
        if i not in skip:
            leg_a = qe1.leg_a.copy()
            leg_b = qe1.leg_b.copy()
            for j, qe2 in enumerate(qes[i + 1:]):
                if qe2.leg_a == leg_a:
                    if qe2.leg_b.spin_in == qe1.leg_b.spin_in and qe2.leg_b.spin_ou == qe1.leg_b.spin_ou:
                        Ls = np.arange(max(qe1.leg_b.get_lmax(), qe2.leg_b.get_lmax()) + 1)
                        if np.all(qe1.cL(Ls) == qe2.cL(Ls)):
                            leg_b += qe2.leg_b
                            skip.append(j + i + 1)
            if np.any(leg_a.cl) and np.any(leg_b.cl):
                qes_ret.append(qe(leg_a, leg_b, qe1.cL))
    if verbose and len(skip) > 0:
        print("%s terms down from %s" % (len(qes_ret), len(qes)))
    if not _swap:
        return qe_simplify(qes_ret, _swap=True, verbose=verbose)
    return [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qes_ret]


def qe_compress(qes: list[qe], verbose=False):
    """This combines pairs of estimators with identical 1st leg to reduce the number of spin transform in its evaluation


        Return:
            a list of tuples (qeleg_multi, qeleg_multi, cl) with 1st leg, 2nd legs and weights.

        Note:
            The 1st leg always only have a single component

    """
    # NB: this only compares first legs.
    skip = []
    # First removes zero weight QEs
    for i, qi in enumerate(qes):
        not_zero = np.any(qi.leg_a.cl) and np.any(qi.leg_b.cl)
        if not not_zero:
            skip.append(i)
    # Then combines remaining QEs
    qes_compressed = []
    for i, qi in enumerate(qes):
        if i not in skip:
            lega = qi.leg_a
            lega_m = qeleg_multi([qi.leg_a.spin_in], qi.leg_a.spin_ou, [qi.leg_a.cl])
            legb_m = qeleg_multi([qi.leg_b.spin_in], qi.leg_b.spin_ou, [qi.leg_b.cl])
            for j, qj in enumerate(qes[i + 1:]):
                if qj.leg_a == lega and legb_m.spin_ou == qj.leg_b.spin_ou:
                    legb_m += qj.leg_b
                    skip.append(i + 1 + j)
            qes_compressed.append( (lega_m, legb_m, qi.cL))
    if len(skip) > 0 and verbose:
        print("%s alm2map_spin transforms now required, down from %s"%(2 * (len(qes) - len(skip)) , 2 * len(qes)) )
    return qes_compressed


def _dict_transpose(cls):
    ret = {}
    for k in cls.keys():
        if len(k) == 1:
            ret[k + k] = np.copy(cls[k])
        else:
            assert len(k) == 2
            ret[k[1] + k[0]] = np.copy(cls[k])
    return ret


def spin_cls(s1, s2, cls):
    r"""Spin-weighted power spectrum :math:`_{s1}X_{lm} _{s2}X^{*}_{lm}`

        The output is real unless necessary.


    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(spin_cls(-s1, -s2, _dict_transpose(cls)))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, 'not implemented')
    if s1 == 0:
        if s2 == 0:
            return cls['tt']
        tb = cls.get('tb', None)
        assert 'te' in cls.keys() or 'et' in cls.keys()
        te = cls.get('te', cls.get('et'))
        return -te if tb is None else  (-te + 1j * np.sign(s2) * tb)
    elif s1 == 2:
        if s2 == 0:
            assert 'te' in cls.keys() or 'et' in cls.keys()
            tb = cls.get('bt', cls.get('tb', None))
            et = cls.get('et', cls.get('te'))
            return -et if tb is None else (-et - 1j * tb)
        elif s2 == 2:
            return cls['ee'] + cls['bb']
        elif s2 == -2:
            eb = cls.get('be', cls.get('eb', None))
            return  (cls['ee'] - cls['bb']) if eb is None else (cls['ee'] - cls['bb'] + 2j * eb)
        else:
            assert 0


def get_spin_matrix(sout, sin, cls):
    r"""Spin-space matrix R^{-1} cls[T, E, B] R where R is the mapping from _{0, \pm 2}X to T, E, B.

        cls is dictionary with keys 'tt', 'te', 'ee', 'bb'.
        If a key is not present the corresponding spectrum is assumed to be zero.
        ('t' 'e' and 'b' keys also works in place of 'tt' 'ee', 'bb'.)

        Output is complex only when necessary (that is, TB and/or EB present and relevant).

    """
    assert sin in [0, 2, -2] and sout in [0, 2, -2], (sin, sout)
    if sin == 0:
        if sout == 0:
            return cls.get('tt', cls.get('t', 0.))
        tb = cls.get('tb', None)
        return (-cls.get('te', 0.) - 1j * np.sign(sout) * tb) if tb is not None else -cls.get('te', 0.)
    if sin == 2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return (-0.5 * (te - 1j * tb)) if tb is not None else (-0.5 * te)
        if sout == 2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
        if sout == -2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return (ret - 1j * eb) if eb is not None else ret
    if sin == -2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return -0.5 * (te + 1j * tb) if tb is not None else -0.5 * te
        if sout == 2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return (ret + 1j * eb) if eb is not None else ret
        if sout == -2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
    assert 0, (sin, sout)


def get_spin_raise(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret


def get_spin_lower(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def sF(s, l2, L):
    """Spherical geometry trigonometric weightings

        Returns :math:`{}_{s}F^{+} and {}_{s}F^{-}`

        See C10 of spherical bispectrum expansion paper

        Returns values for all possible l1

        Returns:
            l1_min, {}_{s}F^{+}, {}_{s}F^{-}

    """
    l1minp, w3p = wigner3j_int(l2, L,  s, 0)
    l1minm, w3m = wigner3j_int(l2, L, -s, 0)
    assert l1minp == l1minm
    return l1minp, (w3p + w3m) * 0.5, (w3p - w3m) * 0.5
def joincls(cls_list):
    lmaxp1 = np.min([len(cl) for cl in cls_list])
    return np.prod(np.array([cl[:lmaxp1] for cl in cls_list]), axis=0)