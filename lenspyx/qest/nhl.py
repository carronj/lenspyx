from __future__ import annotations

import numpy as np
from lenspyx.qest import utils_qe as ut, qest
from lenspyx.wigners.wigners import wignerc
from lenspyx.wigners.utils_wigners import WignerAccumulator
uspin = ut

def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivf1, lmax_ivf2, lmax_qlm=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

        Args:
            qe_key1: QE key 1
            qe_key2: QE key 2
            cls_weights: dictionary with the CMB spectra entering the QE weights.
                        (expected are 'tt', 'te', 'ee' when/if relevant)
            cls_ivfs: dictionary with the inverse-variance filtered CMB spectra.
                        (expected are 'tt', 'te', 'ee', 'bb', 'tb', 'eb' when/if relevant)
            lmax_ivf1: QE 1 uses CMB multipoles down to lmax_ivf1.
            lmax_ivf2: QE 2 uses CMB multipoles down to lmax_ivf2.
            lmax_qlm(optional): outputs are calculated down to lmax_out. Defaults to (lmax_ivf1 + lmax_ivf2) // 2

        Outputs:
            4-tuple of gradient (G) and curl (C) mode Gaussian noise co-variances GG, CC, GC, CG.

    """
    qes1 = ut.qe_compress(qest._get_qes(qe_key1, lmax_ivf1, cls_weights))
    qes2 = qes1 if ((lmax_ivf2 == lmax_ivf1) and (qe_key1 == qe_key2)) else ut.qe_compress(qest._get_qes(qe_key2, lmax_ivf2, cls_weights))
    if lmax_qlm is None:
        lmax_qlm = (lmax_ivf1 + lmax_ivf2) // 2
    return  _get_nhl(qes1, qes2, cls_ivfs, lmax_qlm)
def _get_nhl_pl(qes1, qes2, cls_ivfs, lmax_qlm, cls_ivfs_bb=None, cls_ivfs_ab=None, ret_terms=False):
    GG_N0 = np.zeros(lmax_qlm + 1, dtype=float)
    CC_N0 = np.zeros(lmax_qlm + 1, dtype=float)
    GC_N0 = np.zeros(lmax_qlm + 1, dtype=float)
    CG_N0 = np.zeros(lmax_qlm + 1, dtype=float)

    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs_ab
    if ret_terms:
        terms = []
    for qe1 in qes1:
        cL1 = qe1.cL(np.arange(lmax_qlm + 1))
        for qe2 in qes2:
            cL2 = qe2.cL(np.arange(lmax_qlm + 1))
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)

            clsu = uspin.joincls([qe1.leg_a.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(si, ui, cls_ivfs_aa)])
            cltv = uspin.joincls([qe1.leg_b.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(ti, vi, cls_ivfs_bb)])
            R_sutv = uspin.joincls([wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_qlm), cL1, cL2])

            clsv = uspin.joincls([qe1.leg_a.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(si, vi, cls_ivfs_ab)])
            cltu = uspin.joincls([qe1.leg_b.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(ti, ui, cls_ivfs_ba)])
            R_sutv = R_sutv + uspin.joincls([wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_qlm), cL1, cL2])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = uspin.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-si, ui, cls_ivfs_aa)])
            cltv = uspin.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = uspin.joincls([wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_qlm), cL1, cL2])

            clsv = uspin.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-si, vi, cls_ivfs_ab)])
            cltu = uspin.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-ti, ui, cls_ivfs_ba)])
            R_msmtuv = R_msmtuv + uspin.joincls([wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_qlm), cL1, cL2])

            GG_N0 +=  0.5 * R_sutv.real
            GG_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv.real

            CC_N0 += 0.5 * R_sutv.real
            CC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.real

            GC_N0 -= 0.5 * R_sutv.imag
            GC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

            CG_N0 += 0.5 * R_sutv.imag
            CG_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag
            if ret_terms:
                terms += [0.5 * R_sutv, 0.5 * (-1) ** (to + so) * R_msmtuv]
    return (GG_N0, CC_N0, GC_N0, CG_N0) if not ret_terms else (GG_N0, CC_N0, GC_N0, CG_N0, terms)

def _get_nhl_pl2(qes1, qes2, cls_ivfs, lmax_qlm, ret_terms=False, verbose=True):


    npts = (3 * lmax_qlm) // 2 + 1 # FIXME
    Ls_test = np.arange(lmax_qlm, lmax_qlm+1)
    qespin_1, qespin_2, postfac = None, None, None
    rp_sutv_acc = WignerAccumulator(npts)
    rm_sutv_acc = WignerAccumulator(npts)
    tht = rp_sutv_acc.tht
    sym = qes1 == qes2

    if ret_terms:
        terms = []
    for i1, qe1 in enumerate(qes1):
        cl_1a, cl_1b, cL_1 = qe1.leg_a.cl, qe1.leg_b.cl, qe1.cL
        si, ti = qe1.leg_a.spin_in, qe1.leg_b.spin_in
        so, to = qe1.leg_a.spin_ou, qe1.leg_b.spin_ou
        if qespin_1 is None: qespin_1 = so + to
        assert qespin_1 == so + to
        sgnms = 1 if (si + so) % 2 == 0 else -1
        sgnmt = 1 if (ti + to) % 2 == 0 else -1
        for i2, qe2 in enumerate(qes2[i1 if sym else 0:]):
            fac = 1 if ( (not sym) or (i2 == 0)) else 2 # could make custom q1 is q2
            cl_2a, cl_2b, cL_2 = qe2.leg_a.cl, qe2.leg_b.cl, qe2.cL
            ui, vi = qe2.leg_a.spin_in, qe2.leg_b.spin_in
            uo, vo = qe2.leg_a.spin_ou, qe2.leg_b.spin_ou
            if qespin_2 is None: qespin_2 = uo + vo
            assert qespin_2 == uo + vo
            clsu_g = uspin.joincls([cl_1a, cl_2a.conj(), uspin.spin_cls(si, ui, cls_ivfs)])
            cltv_g = uspin.joincls([cl_1b, cl_2b.conj(), uspin.spin_cls(ti, vi, cls_ivfs)])
            clsv_g = uspin.joincls([cl_1a, cl_2b.conj(), uspin.spin_cls(si, vi, cls_ivfs)])
            cltu_g = uspin.joincls([cl_1b, cl_2a.conj(), uspin.spin_cls(ti, ui, cls_ivfs)])

            clsu_c = uspin.joincls([sgnms * cl_1a.conj(), cl_2a.conj(), uspin.spin_cls(-si, ui, cls_ivfs)])
            cltv_c = uspin.joincls([sgnmt * cl_1b.conj(), cl_2b.conj(), uspin.spin_cls(-ti, vi, cls_ivfs)])
            clsv_c = uspin.joincls([sgnms * cl_1a.conj(), cl_2b.conj(), uspin.spin_cls(-si, vi, cls_ivfs)])
            cltu_c = uspin.joincls([sgnmt * cl_1b.conj(), cl_2a.conj(), uspin.spin_cls(-ti, ui, cls_ivfs)])

            #if uo:
            #    xi_su = wigner4pos(fac * clsu_g, fac * clsu_c, tht, so, uo)
            #    xi_tu = wigner4pos(fac * cltu_g, fac * cltu_c, tht, to, uo)
            #else:

            #xi_tv = wigner4pos(fac * cltv_g, fac * cltv_c, tht, to, vo)
            #xi_sv = wigner4pos(fac * clsv_g, fac * clsv_c, tht, so, vo)
            #xi_tu = wigner4pos(fac * cltu_g, fac * cltu_c, tht, to, uo)
            #rp_sutv_acc.add_xi12(xi_su[0 if uo >= 0 else 2],  so, uo, xitv[],  to, vo)
            #rm_sutv_acc.add_xi12(xi_su[], -so, uo, xitv[], -to, vo)

            rp_sutv_acc.add(fac * clsu_g, cltv_g, so, uo, to, vo) # can accelerate this with wigner4
            rm_sutv_acc.add(fac * clsu_c, cltv_c, -so, uo, -to, vo)



            rp_sutv_acc.add(fac * clsv_g, cltu_g, so, vo, to, uo)
            rm_sutv_acc.add(fac * clsv_c, cltu_c, -so, vo, -to, uo)

    postfac = 0.5 * cL_2(np.arange(lmax_qlm + 1)) * cL_1(np.arange(lmax_qlm + 1))
    sgn = 1 if qespin_1 % 2 == 0 else -1
    rp_sutv = rp_sutv_acc.flush(qespin_1, qespin_2, lmax_qlm)
    rm_sutv = rm_sutv_acc.flush(qespin_1, qespin_2, lmax_qlm)
    GG_N0 = (rp_sutv + sgn * rm_sutv) * postfac
    CC_N0 = (rp_sutv - sgn * rm_sutv) * postfac

    return (GG_N0, CC_N0) if not ret_terms else (GG_N0, CC_N0, terms)

def _get_nhl(qes1:list[(ut.qeleg_multi, ut.qeleg_multi, callable)], qes2:list[(ut.qeleg_multi, ut.qeleg_multi, callable)], cls_ivfs, lmax_qlm,
             cls_ivfs_bb=None, cls_ivfs_ab=None, ret_terms=False):

    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs_ab
    lmax_ivfa = np.max([len(cls_ivfs_aa[k]) for k in cls_ivfs_aa]) - 1
    lmax_ivfb = np.max([len(cls_ivfs_bb[k]) for k in cls_ivfs_bb]) - 1
    lmax_ivf = max(lmax_ivfa, lmax_ivfb)

    npts = (2 * lmax_ivf + lmax_qlm) // 2 + 1
    Ls_test = np.arange(lmax_qlm, lmax_qlm+1)
    qespin_1, qespin_2, postfac = None, None, None
    rp_sutv_acc = WignerAccumulator(npts)
    rm_sutv_acc = WignerAccumulator(npts)


    if ret_terms:
        terms = []
    for i1, qe1 in enumerate(qes1):
        legs_1a, legs_1b, cL_1 = qe1
        so, to  = (legs_1a.spin_ou, legs_1b.spin_ou)
        if qespin_1 is None:
            qespin_1 = so + to
        assert qespin_1 == so + to
        assert len(legs_1a.spins_in) == 1, (legs_1a.spins_in, legs_1b.spins_in)  # could allow also first leg having one comp.
        ncomp_1 = len(legs_1b.spins_in)
        sis, tis = (legs_1a.spins_in * ncomp_1, legs_1b.spins_in)
        if qes1 is qes2: # symmetrize if same QE
            this_qes2 = qes2[i1:]
            facs = [0.5 * 2] * len(this_qes2)
            facs[0] *= 0.5
        else:
            this_qes2 = qes2
            facs = [0.5] * len(this_qes2)
        for i2, (fac, qe2) in enumerate(zip(facs, this_qes2)):
            legs_2a, legs_2b, cL_2 = qe2
            assert len(legs_2a.spins_in) == 1  # could allow also first leg having one comp.
            ncomp_2 = len(legs_2b.spins_in)
            uis, vis = (legs_2a.spins_in * ncomp_2, legs_2b.spins_in)
            uo, vo  = (legs_2a.spin_ou, legs_2b.spin_ou)
            if postfac is None:
                postfac = cL_2(np.arange(lmax_qlm + 1)) * cL_1(np.arange(lmax_qlm + 1))
            assert postfac[Ls_test] == cL_1(Ls_test) * cL_1(Ls_test)
            if qespin_2 is None:
                qespin_2 = uo + vo
            assert qespin_2 == uo + vo
            for si, ti, cl_1a, cl_1b in zip(sis, tis, legs_1a.cls * ncomp_1, legs_1b.cls):
                for ui, vi, cl_2a, cl_2b in zip(uis, vis, legs_2a.cls * ncomp_2, legs_2b.cls):
                    clsu = uspin.joincls([cl_1a, cl_2a.conj(), uspin.spin_cls(si, ui, cls_ivfs_aa)])
                    cltv = uspin.joincls([cl_1b, cl_2b.conj(), uspin.spin_cls(ti, vi, cls_ivfs_bb)])
                    clsv = uspin.joincls([cl_1a, cl_2b.conj(), uspin.spin_cls(si, vi, cls_ivfs_ab)])
                    cltu = uspin.joincls([cl_1b, cl_2a.conj(), uspin.spin_cls(ti, ui, cls_ivfs_ba)])
                    rp_sutv_acc.add(clsu * fac, cltv, so, uo, to, vo) # can parallelize m's here
                    rp_sutv_acc.add(clsv * fac, cltu, so, vo, to, uo)

                    sgnms = (-1) ** (si + so)
                    sgnmt = (-1) ** (ti + to)
                    clsu = uspin.joincls([sgnms * cl_1a.conj(), cl_2a.conj(), uspin.spin_cls(-si, ui, cls_ivfs_aa)])
                    cltv = uspin.joincls([sgnmt * cl_1b.conj(), cl_2b.conj(), uspin.spin_cls(-ti, vi, cls_ivfs_bb)])
                    clsv = uspin.joincls([sgnms * cl_1a.conj(), cl_2b.conj(), uspin.spin_cls(-si, vi, cls_ivfs_ab)])
                    cltu = uspin.joincls([sgnmt * cl_1b.conj(), cl_2a.conj(), uspin.spin_cls(-ti, ui, cls_ivfs_ba)])
                    rm_sutv_acc.add(clsu * fac, cltv, -so, uo, -to, vo)
                    rm_sutv_acc.add(clsv * fac, cltu, -so, vo, -to, uo)

    sgn = 1 if qespin_1 % 2 == 0 else -1
    rp_sutv = rp_sutv_acc.flush(qespin_1, qespin_2, lmax_qlm)
    rm_sutv = rm_sutv_acc.flush(qespin_1, qespin_2, lmax_qlm)
    GG_N0 = (rp_sutv + sgn * rm_sutv) * postfac
    CC_N0 = (rp_sutv - sgn * rm_sutv) * postfac
    if ret_terms:
        terms += [0.5 * rp_sutv, 0.5 * (-1) ** (to + so) * rm_sutv]
    return (GG_N0, CC_N0) if not ret_terms else (GG_N0, CC_N0, terms)