from __future__ import annotations
from os import cpu_count
import numpy as np
from lenspyx import utils_hp
from copy import deepcopy
from lenspyx.qest import qest, qresp
from lenspyx import get_geom
from lenspyx.remapping.utils_geom import Geom


class OpFilt:
    def __init__(self, job: str, cls_filt: dict[str], transfs: dict, inoise: dict, geom: str, nthreads: int = 0):
        """CMB inverse-variance filtering instance

            This may be used to inverse-variance filter a set of maps

            Args:
                job(str): task to perform, e.g. 't' for T-only filtering, 'eb' for Pol, or 'teb'.
                     Other combinations are also possible

                cls_filt(dict): dictionary of fiducial set of spectra used in the filtering
                transfs(dict): dictionary of transfer functions. Here only diagonal in harmonic space is supported
                inoise(dict): inverse noise variance maps (or inverse noise cls if in harmonic space)
                geom(str): sky geometry (use 'harmonic' for an idealized situation where maps are given in harmonic space)
                nthreads(str): SHT's will use this number of threads when relevant


        """
        # Example geom input: 'harmonic', 'healpix_2048'
        harmonic = geom.lower() == 'harmonic'
        for i, f in enumerate(job):
            assert (f + f) in cls_filt
            assert f in transfs
            if harmonic:
                assert f + f in inoise or f in inoise
                if f in inoise:
                    inoise[f + f] = inoise[f]
            for g in job[i + 1:]:  # always want f + g in the keys rather
                for cl in [cls_filt, inoise]:
                    if g + f in cl:
                        cl[f + g] = cl[g + f]

        if nthreads <= 0:
            nthreads = cpu_count()
        lmaxs_wted = {field: len(transfs[field]) - 1 for field in job}
        lmaxs_ivfs = {field: len(transfs[field]) - 1 for field in job}

        if harmonic:
            self.maps_labels = job # FIXME: this need not be
            self.nmaps = len(job)
        else:
            assert len(job) <= 3 and np.all([f in ['t', 'e', 'b'] for f in job])
            self.maps_labels = ['t'] * ('t' in job) + ['q', 'u'] * (('e' in job) or ('b' in job))
            self.nmaps = (1 * ('t' in job) + 2 * ('e' in job or 'b' in job))
            if 'e' in job and 'b' in job:
                lmaxs_ivfs['e'] = max(lmaxs_ivfs['e'], lmaxs_ivfs['b'])
                lmaxs_ivfs['b'] = max(lmaxs_ivfs['e'], lmaxs_ivfs['b'])

        self.job = job
        self.cls = cls_filt
        self.inoise = inoise
        self.geom = geom

        self.lmax_wted = lmaxs_wted
        self.lmax_ivfs = lmaxs_ivfs
        self.mmax_ivfs = lmaxs_ivfs # TODO: not necessary

        self.nalm  = len(job)


        self.transfs = transfs


        self.geom = Geom.get_healpix_geometry(2048) #FIXME

        self.nthreads = nthreads

        self.harmonic = harmonic
        self._fal = None
        self._build_fal()
        print('OpFilt setup')
        for f in job:
            print(f + ' lmax %s' % lmaxs_ivfs[f])
        print()

    def get_fal(self):
        self._build_fal()
        return deepcopy(self._fal)

    def _build_fal(self):
        if self._fal is None:
            if self.harmonic:
                # (S + N)^{-1} = Ni - Ni (Si + Ni)i Ni
                assert self.nalm == len(self.job)
                lmax = np.max([lmax for lmax in self.lmax_ivfs.values()])
                ni = np.zeros((lmax + 1, self.nalm, self.nalm), dtype=float)
                s = np.zeros((lmax + 1, self.nalm, self.nalm), dtype=float)
                for i, f in enumerate(self.job):
                    transf_f = self.transfs[f]
                    for j, g in enumerate(self.job):
                        if f == g:
                            assert f + g in self.cls and f + g in self.inoise
                        transf_g = self.transfs[g]
                        if f + g in self.inoise:
                            ino = self.inoise[f + g]
                            ni[:, i, j] = self._joincls([ino, transf_f, transf_g], lmax)
                            ni[:, j, i] = self._joincls([ino, transf_f, transf_g], lmax)
                        if f + g in self.cls:
                            s[:, i, j] = self._joincls([self.cls[f + g]], lmax)
                            s[:, j, i] = self._joincls([self.cls[f + g]], lmax)
                sipni_il = np.linalg.pinv(np.linalg.pinv(s) + ni)
                for tni, sipni_i in zip(ni, sipni_il):
                    tni -= np.dot(tni, np.dot(sipni_i, tni))

                fal = dict()
                for i, f in enumerate(self.job):
                    for j, g in enumerate(self.job[i:]):
                        if np.any(ni[:self.lmax_ivfs[f] + 1, i, i + j]):
                            fal[f + g] = ni[:self.lmax_ivfs[f] + 1, i, i + j]
                self._fal = fal
            else:
                assert 0, 'not implemented yet'

    @staticmethod
    def _joincls(cls: list[np.ndarray or float], lmax: int):
        """Multiplies inputs arrays up to lmax. Outputs has shape lmax + 1"""
        ret = np.ones(lmax + 1, dtype=float)
        for cl in cls:
            if np.isscalar(cl):
                ret *= cl
            else:
                this_lmax = min(len(cl) - 1, lmax)
                ret[:this_lmax + 1] *= cl[:this_lmax + 1]
        return ret

    def _almxflcopy(self, f: str, fl: np.ndarray[float], alm: np.ndarray[complex], mmax=None):
        assert alm.ndim == 1
        assert fl.ndim == 1 and fl.size > self.lmax_ivfs[f], fl.shape
        if mmax is None:
            mmax = utils_hp.Alm.getlmax(alm.size, mmax)
        retlm = utils_hp.alm_copy(alm, mmax, self.lmax_ivfs[f], self.mmax_ivfs[f])
        utils_hp.almxfl(retlm, fl, self.mmax_ivfs[f], True)
        return retlm

    def __call__(self, maps: dict):
        ivf_alms = dict()
        if self.harmonic:
            for f in self.job:
                assert f in maps, maps.keys()
                assert maps[f].ndim == 1
                ivf_alms[f] = self._almxflcopy(f, self._fal[f+f], maps[f])
            for fg in self._fal:
                if fg[0] != fg[1]:  # off-diagonals, explicitly assuming symmetry
                    assert fg[1] + fg[0] not in self._fal
                    ivf_alms[fg[0]] += self._almxflcopy(fg[0], 2 * self._fal[fg], maps[fg[1]])
            return ivf_alms
        else:
            assert 0, 'implement this'

    @staticmethod
    def _get_shtmode(fg):
        """Returns ducc0 sht_mode, spin and Stokes labels of transform from labels"""
        if fg == 't':
            return 'STANDARD', 0, 't'
        elif fg in ['e', 'eb']:
            return 'GRAD_ONLY', 2, 'qu'
        elif fg == 'b':
            return 'CURL_ONLY', 2, 'qu'
        else:
            assert 0, 'dont know what to make of ' + fg

    def _apply_alm(self, gclms: dict):
        """This calculates the operation

            :math:`B^t N^{-1} B`

            to the input set of alms

            The operation is performed inplace

        """
        assert not self.harmonic, 'what are you doing here'
        for i, fg in gclms:
            lmax, mmax = self.lmax_ivfs[fg[0]], self.mmax_ivfs[fg[0]]
            if len(fg) == 2:
                assert lmax == self.lmax_ivfs[fg[1]]
                assert mmax == self.lmax_ivfs[fg[1]]
            assert gclms[fg].ndim == 2 and gclms[fg].shape[1] == utils_hp.Alm.getsize(lmax, mmax), \
                (lmax, mmax,  utils_hp.Alm.getsize(lmax, mmax))
            utils_hp.almxfl(gclms[fg][0], self.transfs[fg[0]], mmax, True)
            if len(fg) == 2:
                utils_hp.almxfl(gclms[fg][1], self.transfs[fg[1]], mmax, True)
            sht_mode, spin, st = self._get_shtmode(fg)
            assert len(st) == 2 if spin else 1
            m = self.geom.synthesis(gclms[fg], spin, lmax, mmax, self.nthreads, mode=sht_mode)
            # apply N^{-1}. Now this assumes different gclmshave independent noise
            if spin:
                assert len(st) == 2 and st not in self.inoise, 'correlated noise not implemented yet, but easy'
                s, t = st  # real-space map labels (e.g. q u)
                m[1] *= self.inoise[t + t]
            else:
                assert len(st) == 1
                s = st
                m[0] *= self.inoise[s + s]
            # synthesis back onto gclm
            self.geom.adjoint_synthesis(m, spin, lmax, mmax, self.nthreads, alm=gclms[fg], mode=sht_mode)
            utils_hp.almxfl(gclms[fg][0], self.transfs[fg[0]], mmax, True)
            if len(fg) == 2:
                utils_hp.almxfl(gclms[fg][1], self.transfs[fg[1]], mmax, True)


class Qlms:
    def __init__(self, opfilt_1: OpFilt, opfilt_2: OpFilt, cls_weight: dict, lmax_qlm: int):
        """Calculator of quadratic estimator from CMB inverse-variance filtering instance

                Args:
                    opfilt_1(OpFilt): filtering instance for the first leg
                    opfilt_2(OpFilt): filtering instance for the second leg (can be the same)
                    cls_weight(dict): spectra used as weights when buildingt the QE (e.g. lensed CMB spectra)
                    lmax_qlm(int): QE's are computed down to this


        """
        # explicitly allow different lmaxes?
        lmax_ivf1 = np.max([lmax for lmax in opfilt_1.lmax_ivfs.values()])
        lmax_ivf2 = np.max([lmax for lmax in opfilt_2.lmax_ivfs.values()])

        self.opfilt_1 = opfilt_1
        self.opfilt_2 = opfilt_2
        self.lmax_ivf = max(lmax_ivf1, lmax_ivf2)

        self.cls_weight = cls_weight
        self.lmax_qlm = lmax_qlm

    def get_response(self, qe_key: str, source_key: str, cls_cmb: dict):
        """Calculate response of QE to anisotropy source

            Args:
                qe_key: label of the QE (e.g. 'ptt' for lensing TT QE, see Plancklens doc for this)
                source_key: label of the anisotropy source (e.g. 'p' for lensing)
                cls_cmb: dictionary of cls describing the sky response to the anisotropy
                         (typically lensed cls or better grad cls for lensing)

            Returns:
                Response of gradient and curl modes to gradient and curl modes


        """
        fal1 = self.opfilt_1.get_fal()
        fal2 = self.opfilt_2.get_fal()
        qresp.get_response(qe_key, self.lmax_ivf, source_key, self.cls_weight, cls_cmb, fal1, fal_leg2=fal2, lmax_qlm=self.lmax_qlm)

    def get_qlms(self, qe_key: str, maps: dict, verbose=False):
        """Calculates a quadratic estimator

            Args:
                qe_key: label of the QE (e.g. 'ptt' for lensing TT QE, see Plancklens doc for this)
                maps: input maps, as a dictionary. Must be consistent with what the filter expects to see
                verbose: some printout if set

            Returns:
                gradient and curl mode of the estimator, array of shape (1 if spin==0 else 2, qlm_size)


        """
        for f in self.opfilt_1.maps_labels + self.opfilt_2.maps_labels:
            assert f in maps, 'missing input: ' + f

        if self.opfilt_1 is self.opfilt_2:
            alms_1 = self.opfilt_1(maps).get
            alms_2 = alms_1
        else:
            alms_1 = self.opfilt_1(maps).get
            alms_2 = self.opfilt_2(maps).get

        lmax_ivf = self.lmax_ivf
        return qest.eval_qe(qe_key, lmax_ivf, self.cls_weight, alms_1, self.lmax_qlm, verbose=verbose, get_alm2=alms_2)
