import numpy as np
from lenspyx.qest.qest import _get_qes, _get_covresp
from lenspyx.qest import utils_qe as ut
from lenspyx.wigners.wigners import wignerc

uspin = ut

def get_response(qe_key, lmax_ivf, source, cls_weight, cls_cmb, fal, fal_leg2=None, lmax_ivf2=None, lmax_qlm=None):
    r"""QE response calculation

        Args:
            qe_key: Quadratic estimator key (see this module docstring for descriptions)
            lmax_ivf: max. CMB multipole used in the QE
            source: anisotropy source key
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            cls_cmb(dict): CMB spectra entering the CMB response (in principle lensed spectra, or grad-lensed spectra)
            fal(dict): (isotropic approximation to the) filtering cls. e.g. fal['tt'] :math:`= \frac {1} {C^{\rm TT}_\ell  +  N^{\rm TT}_\ell / b^2_\ell}` for temperature if filtered independently from polarization.
            fal_leg2(dict): Same as *fal* but for the second leg, if different.
            lmax_ivf2(optional): max. CMB multipole used in the QE on the second leg (if different to lmax_ivf)
            lmax_qlm(optional): responses are calculated up to this multipole. Defaults to lmax_ivf + lmax_ivf2

        Note:
            The result is *not* symmetrized with respect to the 'fals', if not the same on the two legs.
            In this case you probably want to run this twice swapping the fals in the second run.

    """
    if lmax_ivf2 is None: lmax_ivf2 = lmax_ivf
    if lmax_qlm is None : lmax_qlm = lmax_ivf + lmax_ivf2
    if '_bh_' in qe_key: # Bias-hardened estimators:
        k, hsource = qe_key.split('_bh_') # kQE hardened against hsource
        assert len(hsource) == 1, hsource
        h = hsource[0]
        RGG_ks, RCC_ks, RGC_ks, RCG_ks = get_response(k, lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm)
        RGG_hs, RCC_hs, RGC_hs, RCG_hs = get_response(h + k[1:], lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm)
        RGG_kh, RCC_kh, RGC_kh, RCG_kh = get_response(k, lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm)
        RGG_hh, RCC_hh, RGC_hh, RCG_hh = get_response(h + k[1:], lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm)
        RGG = RGG_ks - (RGG_kh * RGG_hs * cli(RGG_hh) + RGC_kh * RCG_hs * cli(RCC_hh))
        RCC = RCC_ks - (RCG_kh * RGC_hs * cli(RGG_hh) + RCC_kh * RCC_hs * cli(RCC_hh))
        RGC = RGC_ks - (RGG_kh * RGC_hs * cli(RGG_hh) + RGC_kh * RCC_hs * cli(RCC_hh))
        RCG = RCG_ks - (RCG_kh * RGG_hs * cli(RGG_hh) + RCC_kh * RCG_hs * cli(RCC_hh))
        return RGG, RCC, RGC, RCG

    qes = _get_qes(qe_key, lmax_ivf, cls_weight)
    return _get_response(qes, source, cls_cmb, fal, lmax_qlm, fal_leg2=fal_leg2)

def _get_response(qes, source, cls_cmb, fal_leg1, lmax_qlm, fal_leg2=None):
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    RGG = np.zeros(lmax_qlm + 1, dtype=float)
    RCC = np.zeros(lmax_qlm + 1, dtype=float)
    RGC = np.zeros(lmax_qlm + 1, dtype=float)
    RCG = np.zeros(lmax_qlm + 1, dtype=float)
    Ls = np.arange(lmax_qlm + 1, dtype=int)
    for qe in qes:
        si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
        so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
        for s2 in ([0, -2, 2]):
            FA = uspin.get_spin_matrix(si, s2, fal_leg1)
            if np.any(FA):
                for t2 in ([0, -2, 2]):
                    FB = uspin.get_spin_matrix(ti, t2, fal_leg2)
                    if np.any(FB):
                        rW_st, prW_st, mrW_st, s_cL_st = _get_covresp(source, -s2, t2, cls_cmb, len(FB) - 1)
                        clA = ut.joincls([qe.leg_a.cl, FA])
                        clB = ut.joincls([qe.leg_b.cl, FB, mrW_st.conj()])
                        Rpr_st = wignerc(clA, clB, so, s2, to, -s2 + rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                        rW_ts, prW_ts, mrW_ts, s_cL_ts = _get_covresp(source, -t2, s2, cls_cmb, len(FA) - 1)
                        clA = ut.joincls([qe.leg_a.cl, FA, mrW_ts.conj()])
                        clB = ut.joincls([qe.leg_b.cl, FB])
                        Rpr_st = Rpr_st + wignerc(clA, clB, so, -t2 + rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        assert rW_st == rW_ts and rW_st >= 0, (rW_st, rW_ts)
                        if rW_st > 0:
                            clA = ut.joincls([qe.leg_a.cl, FA])
                            clB = ut.joincls([qe.leg_b.cl, FB, prW_st.conj()])
                            Rmr_st = wignerc(clA, clB, so, s2, to, -s2 - rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                            clA = ut.joincls([qe.leg_a.cl, FA, prW_ts.conj()])
                            clB = ut.joincls([qe.leg_b.cl, FB])
                            Rmr_st = Rmr_st + wignerc(clA, clB, so, -t2 - rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        else:
                            Rmr_st = Rpr_st
                        prefac = qe.cL(Ls)
                        RGG += prefac * ( Rpr_st.real + Rmr_st.real * (-1) ** rW_st)
                        RCC += prefac * ( Rpr_st.real - Rmr_st.real * (-1) ** rW_st)
                        RGC += prefac * (-Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)
                        RCG += prefac * ( Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)

    return RGG, RCC, RGC, RCG

def cli(cl):
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1. / cl[ii]
    return ret