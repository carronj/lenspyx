from __future__ import annotations
import numpy as np
from lenspyx.utils import timer
from lenspyx.qest import qest
from lenspyx.qest import utils_qe as ut
from lenspyx.wigners.wigners import wignerc, wignerpos, wigner4pos
from lenspyx import utils_hp
from lenspyx.wigners.utils_wigners import WignerAccumulator

uspin = ut


def get_response(qe_key: str, lmax_ivf: int, source: str, cls_weight: dict, cls_cmb:dict, fal:dict, fal_leg2=None, lmax_qlm=None):
    r"""QE response calculation

        Args:
            qe_key: Quadratic estimator key (see this module docstring for descriptions)
            lmax_ivf: max. CMB multipole used in the QE
            source: anisotropy source key
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            cls_cmb(dict): CMB spectra entering the CMB response (in principle lensed spectra, or grad-lensed spectra)
            fal(dict): (isotropic approximation to the) filtering cls. e.g.

                      fal['tt']   :math:`= \frac {1} {C^{\rm TT}_\ell  +  N^{\rm TT}_\ell / b^2_\ell}`

                    for temperature if filtered independently from polarization.

            fal_leg2(dict): Same as *fal* but for the second leg, if different.
            lmax_qlm(optional): responses are calculated up to this multipole. Defaults to lmax_ivf


        Note:
            The result is *not* symmetrized with respect to the 'fals', if not the same on the two legs.
            In this case you probably want to run this twice swapping the fals in the second run.


    """
    if lmax_qlm is None:
        lmax_qlm = lmax_ivf
    if '_bh_' in qe_key: # Bias-hardened estimators:
        k, hsource = qe_key.split('_bh_')
        assert len(hsource) == 1, hsource
        h = hsource[0]
        RGG_ks, RCC_ks, RGC_ks, RCG_ks = get_response(k, lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_qlm=lmax_qlm)
        RGG_hs, RCC_hs, RGC_hs, RCG_hs = get_response(h + k[1:], lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_qlm=lmax_qlm)
        RGG_kh, RCC_kh, RGC_kh, RCG_kh = get_response(k, lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_qlm=lmax_qlm)
        RGG_hh, RCC_hh, RGC_hh, RCG_hh = get_response(h + k[1:], lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_qlm=lmax_qlm)
        RGG = RGG_ks - (RGG_kh * RGG_hs * cli(RGG_hh) + RGC_kh * RCG_hs * cli(RCC_hh))
        RCC = RCC_ks - (RCG_kh * RGC_hs * cli(RGG_hh) + RCC_kh * RCC_hs * cli(RCC_hh))
        RGC = RGC_ks - (RGG_kh * RGC_hs * cli(RGG_hh) + RGC_kh * RCC_hs * cli(RCC_hh))
        RCG = RCG_ks - (RCG_kh * RGG_hs * cli(RGG_hh) + RCC_kh * RCG_hs * cli(RCC_hh))
        return RGG, RCC, RGC, RCG

    qes = ut.qe_compress(qest._get_qes(qe_key, lmax_ivf, cls_weight))
    return _get_response(qes, source, cls_cmb, fal, lmax_ivf, lmax_qlm, fal_leg2=fal_leg2)


def get_dresponse_dlncl(qe_key: str, cmb_l: int, cl_key: str, lmax_ivf: int, source: str, cls_weight: dict, cls_cmb: dict, fal_leg1: dict,
                        fal_leg2=None, lmax_qlm=None):
    r"""QE isotropic response derivative function

            :math:`\frac{dR_L} { d\ln C_l}`

            for each L up to lmax_qlm and where l is the input cmb_l


    """
    if lmax_qlm is None : lmax_qlm = lmax_ivf
    dcls_cmb = {k: np.zeros_like(cls_cmb[k]) for k in cls_cmb.keys()}
    dcls_cmb[cl_key][cmb_l] = cls_cmb[cl_key][cmb_l]
    qes = ut.qe_compress(qest._get_qes(qe_key, lmax_ivf, cls_weight))
    return _get_response(qes, source, dcls_cmb, fal_leg1, lmax_ivf, lmax_qlm, fal_leg2=fal_leg2)


def _accumulate_f(spin_ins, s2, fal, cls):
    assert len(spin_ins) == len(cls), (len(spin_ins), len(cls))
    f = 0
    for i, s in enumerate(spin_ins):
        fs = uspin.get_spin_matrix(s, s2, fal)
        if np.any(fs):
            f = f + ut.joincls([cls[i], fs])
    return f


def _get_response(qes:list[(ut.qeleg_multi, ut.qeleg_multi, callable)], source, cls_cmb, fal_leg1, lmax_ivf, lmax_qlm,
                  fal_leg2=None, verbose=False):
    tim = timer('', False)
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2

    Ls = np.arange(lmax_qlm + 1, dtype=int)
    Ls_test = Ls[lmax_qlm:lmax_qlm+1]

    qe_spin, source_spin, prefac = None, None, None
    Rpr_st_acc = WignerAccumulator((2 * lmax_ivf + lmax_qlm) // 2 + 1)
    Rmr_st_acc = WignerAccumulator((2 * lmax_ivf + lmax_qlm) // 2 + 1)
    tht = Rmr_st_acc.tht
    tim.add('setup')
    for qe in qes:
        tim.start('Wigner setup')
        legs_a, legs_b, cL = qe
        sis, tis = (legs_a.spins_in, legs_b.spins_in)
        so, to  = (legs_a.spin_ou, legs_b.spin_ou)
        assert (len(sis) == 1) or (len(tis) == 1)
        if qe_spin is None:
            qe_spin = so + to
        assert qe_spin == so + to
        # Prepare filters
        f_as, f_bs = {}, {}
        for spin in [0, -2, 2]:
            f_a = _accumulate_f(sis, spin, fal_leg1, legs_a.cls)
            f_b = _accumulate_f(tis, spin, fal_leg2, legs_b.cls)
            if np.any(f_a):
                f_as[spin] = f_a
            if np.any(f_b):
                f_bs[spin] = f_b
        cls2_ms, cls2_ps, cls1_ms, cls1_ps = {}, {}, {}, {}
        for s2 in f_as:
            assert -s2 in f_as
            cls2_ms[s2] = np.zeros_like(f_as[s2])
            cls2_ps[s2] = np.zeros_like(f_as[s2])
        for t2 in f_bs:
            assert -t2 in f_bs
            cls1_ms[t2] = np.zeros_like(f_bs[t2])
            cls1_ps[t2] = np.zeros_like(f_bs[t2])
        for s2 in f_as:
            for t2 in f_bs:
                rW_st, prW_st, mrW_st, s_cL_st = qest._get_covresp(source, -s2, t2, cls_cmb, len(f_bs[t2]) - 1)
                rW_ts, prW_ts, mrW_ts, s_cL_ts = qest._get_covresp(source, -t2, s2, cls_cmb, len(f_as[s2]) - 1)
                assert rW_st == rW_ts
                cls2_ms[s2] += f_bs[t2] * mrW_st.conj()
                cls1_ms[t2] += f_as[s2] * mrW_ts.conj()
                if rW_st:
                    cls2_ps[s2] += f_bs[t2] * prW_st.conj()
                    cls1_ps[t2] += f_as[s2] * prW_ts.conj()
                if source_spin is None:
                    source_spin = rW_st
                assert source_spin == rW_st
                if prefac is None:
                    prefac = cL(Ls) * s_cL_st(Ls)  # Fix s_cL_ts
                assert prefac[Ls_test] == cL(Ls_test) * s_cL_st(Ls_test)
        assert source_spin >= 0, source_spin
        tim.close('Wigner setup')
        for i, (f_s, cls_ms, cls_ps) in enumerate(zip([f_as, f_bs], [cls2_ms, cls1_ms], [cls2_ps, cls1_ps])):
            for s in f_s:
                so, to = (legs_a.spin_ou, legs_b.spin_ou) if i == 0 else (legs_b.spin_ou, legs_a.spin_ou)
                if s > 0:
                    xi1_pms = wigner4pos(f_s[+s], f_s[-s], tht, so, s)
                    if source_spin:  # pr and mr have identical first leg
                        Rpr_st_acc.add_xi1(xi1_pms[0], so, +s, cls_ms[+s], to, -s + source_spin)
                        Rmr_st_acc.add_xi1(xi1_pms[0], so, +s, cls_ps[+s], to, -s - source_spin)
                        Rpr_st_acc.add_xi1(xi1_pms[3], so, -s, cls_ms[-s], to, +s + source_spin)
                        Rmr_st_acc.add_xi1(xi1_pms[3], so, -s, cls_ps[-s], to, +s - source_spin)
                    else:
                        xi2_pms = wigner4pos(cls_ms[-s], cls_ms[+s], tht, to, s)
                        Rpr_st_acc.add_xi12(xi1_pms[0], so, +s, xi2_pms[1], to, -s)
                        Rpr_st_acc.add_xi12(xi1_pms[3], so, -s, xi2_pms[0], to, +s)
                elif s == 0:
                    xi1 = wignerpos(f_s[0], tht, so, 0)  # Could get the -s2 at the same time
                    if source_spin:  # pr and mr have identical first leg
                        xi2_pms = wigner4pos(cls_ms[0], cls_ps[0], tht, to, source_spin)
                        Rpr_st_acc.add_xi12(xi1, so, 0, xi2_pms[0], to, + source_spin)
                        Rmr_st_acc.add_xi12(xi1, so, 0, xi2_pms[3], to, - source_spin)
                    else:
                        Rpr_st_acc.add(f_s[s], cls_ms[s], so, s, to, -s + source_spin)
            tim.add('Wigner, leg %s'%i)

    if source_spin:
        sgn = 1 if source_spin % 2 == 0 else -1
        rpr_st = Rpr_st_acc.flush(qe_spin,  source_spin, lmax_qlm)
        rmr_st = Rmr_st_acc.flush(qe_spin, -source_spin, lmax_qlm) * sgn
        rgg = prefac * (rpr_st + rmr_st)
        rcc = prefac * (rpr_st - rmr_st)
    else:
        rgg = 2 * prefac * Rpr_st_acc.flush(qe_spin, source_spin, lmax_qlm)
        rcc = np.zeros_like(rgg)
    tim.add('spin2GC')
    if verbose:
        print(tim)
    return np.stack([rgg, rcc])


def cli(cl):
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1. / cl[ii]
    return ret


def get_mf_response(qe_key:str, nlev_t:float, beam:float, lmax_ivf:int, lmax_sky:int, cls_unl:dict,
                lmin_ivf = 1, nlev_p=None, inoise_cls:dict or None=None, lmax_qlm=None):
    """Delensed-noise mean field perturbative response

        Args:
            qe_key: quadratic estimator key (e.g. 'ptt' for temperature lensing estimator)
            nlev_t: noise level of T-map in uK amin
            beam: beam FWHM in amin
            lmax_ivf: maximum multipole included in the quadratic estimator
            lmax_sky: maximum multipole of the CMB sky
                    (as used in the likelihood model. output at L can feel lsky up to lmax_ivf + L)
            cls_unl: dictionary of unlensed CMB spectra
            lmin_ivf(optional): sets the inverse noise to zero below this if set
            inoise_cls(optional): Can sets this to customized inverse noise spectra.
                                  In this case 'beam', 'nlev' are ignored
            nlev_p(optional): polarization noise level (defaults to root 2 nlev_t)
            lmax_qlm: maximal output QE multipole (defaults to lmax_sky + lmax_ivf, after which the output should vanish)

        Returns:
            lensing gradient potential and curl potential perturbative response R

            :math:`<g^{\phi, QD}>_{LM} \sim R_L \phi_{LM}`

            and similarly for the curl


    """
    assert lmax_sky >= lmax_ivf, (lmax_ivf, lmax_sky, 'inconsistent inputs')
    lmax_qlm = lmax_qlm or lmax_sky + lmax_ivf
    if inoise_cls is None:
        inoise_cls = dict()
    if 'tt' not in inoise_cls and qe_key[1:] in ['tt', '', ]:
        inoise_cls['tt'] = np.zeros(lmax_sky + 1, dtype=float)
        nlev_rad = {'tt': (nlev_t / 60 / 180 * np.pi)}
        inoise_cls['tt'][:lmax_ivf + 1] = (utils_hp.gauss_beam(beam / 60 / 180 * np.pi, lmax=lmax_ivf) /  nlev_rad['tt']) ** 2
        inoise_cls['tt'][:lmax_ivf + 1] *= (cls_unl['tt'][:lmax_ivf + 1] > 0)
        inoise_cls['tt'][:lmin_ivf] *= 0.
    for spec in ['ee', 'bb']:
        if spec in inoise_cls.keys():
            continue
        if nlev_p is None: nlev_p = np.sqrt(2) * nlev_t
        inoise_cls[spec] = np.zeros(lmax_sky + 1, dtype=float)
        nlev_rad = {spec: (nlev_p / 60 / 180 * np.pi)}
        inoise_cls[spec][:lmax_ivf + 1] = (utils_hp.gauss_beam(beam / 60 / 180 * np.pi, lmax=lmax_ivf) /  nlev_rad[spec]) ** 2
        inoise_cls[spec][:lmax_ivf + 1] *= (cls_unl['ee'][:lmax_ivf + 1] > 0)
        inoise_cls[spec][:lmin_ivf] *= 0.

    for k in inoise_cls.keys():
        inoise_cls[k][lmax_ivf+1:] *= 0
        inoise_cls[k][:lmin_ivf] *= 0.

    cls_noise = {spec: cli(inoise_cls[spec]) for spec in inoise_cls.keys()}
    # Adding non-zero big value
    for spec in list(inoise_cls.keys()):
        cls_noise[spec][np.where(cls_noise[spec] == 0)] = np.max(cls_noise[spec]) * 1e8
    #cls_noise['tt'][np.where(cls_noise['tt'] == 0)] = np.max(cls_noise['tt']) * 1e8
    fal = {spec: np.zeros(lmax_sky + 1, dtype=float) for spec in inoise_cls.keys()}
    for spec in list(inoise_cls.keys()):
        sli = slice(0, lmax_sky + 1)
        fal[spec][sli] = cli(cls_unl[spec][sli] + cls_noise[spec][sli])
        #fal[spec][:lmin_ivf] *= 0
    return get_response(qe_key, lmax_sky, 'p', cls_unl, cls_noise, fal, lmax_qlm=lmax_qlm)