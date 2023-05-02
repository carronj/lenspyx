"""Module with simple inverse-variance filtering instances


"""
from __future__ import annotations
import numpy as np
from copy import deepcopy
from lenspyx import utils_hp


class OpFilt:
    def __init__(self, cls_filt: dict[str: np.ndarray], transfs: dict[str: np.ndarray], inoise: dict[str: np.ndarray or float]):
        """CMB Harmonic-space inverse-variance filtering instance

            This may be used to inverse-variance filter a set of maps

            Args:
                cls_filt(dict): dictionary of fiducial set of spectra used in the filtering
                transfs(dict): dictionary of transfer functions. Here only diagonal in harmonic space is supported
                inoise(dict):  Inverse noise cls of transfer-convovled maps. May be scalars if white.


        """
        job = []
        for f in transfs:
            if f + f in cls_filt:
                job.append(f)

        for i, f in enumerate(job):
            assert (f + f) in cls_filt
            assert f in transfs
            assert f + f in inoise or f in inoise
            if f in inoise:
                inoise[f + f] = inoise[f]
            for g in job:  # always want f + g in the keys
                if g != f:
                    for cl in [cls_filt, inoise]:
                        if g + f in cl:
                            cl[f + g] = cl[g + f]
                        if f + g in cl:
                            cl[g + f] = cl[f + g]

        lmaxs_wted = {field: len(transfs[field]) - 1 for field in job}
        lmaxs_ivfs = {field: len(transfs[field]) - 1 for field in job}

        maps_labels = job  # input map labels
        alms_labels = job  # filtered solutions labels

        self.job = job
        self.maps_labels = maps_labels

        self.cls = cls_filt
        self.inoise = inoise


        self.lmax_wted = lmaxs_wted
        self.lmax_ivfs = lmaxs_ivfs
        self.mmax_ivfs = lmaxs_ivfs  # not necessary

        self.nalm = len(job)

        self.transfs = transfs

        print('OpFilt setup')
        for f in self.job:
            print(f + ' lmax %s' % lmaxs_ivfs[f])
        print('expected input maps: ' + ' '.join(maps_labels))
        print('filtered alms      : ' + ' '.join(alms_labels))

        self._fal = None
        self._fal = self._build_fal()

    def get_fal(self):
        self._build_fal()
        return deepcopy(self._fal)

    def _build_fal(self):
        if self._fal is None:
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
                    if i == j:
                        ni[:, i, i] *= (self.cls[f + f][:lmax+1] > 0)
                    if f + g in self.cls:
                        s[:, i, j] = self._joincls([self.cls[f + g]], lmax)
                        s[:, j, i] = self._joincls([self.cls[f + g]], lmax)
            sipni_il = np.linalg.pinv(np.linalg.pinv(s, hermitian=True) + ni, hermitian=True)
            for tni, sipni_i in zip(ni, sipni_il):
                tni -= np.dot(tni, np.dot(sipni_i, tni))

            fal = dict()
            for i, f in enumerate(self.job):
                for j, g in enumerate(self.job[i:]):
                    if np.any(ni[:self.lmax_ivfs[f] + 1, i, i + j]):
                        fal[f + g] = ni[:self.lmax_ivfs[f] + 1, i, i + j]
                        fal[g + f] = fal[f + g]
            self._fal = fal
        return self._fal

    @staticmethod
    def _joincls(cls: list[np.ndarray or float], lmax: int):
        """Multiplies inputs arrays up to lmax.

                Output has shape lmax + 1

        """
        ret = np.ones(lmax + 1, dtype=float)
        for cl in cls:
            if np.isscalar(cl):
                ret *= cl
            else:
                this_lmax = min(len(cl) - 1, lmax)
                ret[:this_lmax + 1] *= cl[:this_lmax + 1]
        return ret

    def _almxflcopy(self, f: str, fl: np.ndarray[float], alm: np.ndarray[complex], mmax=None):
        assert alm.ndim == 1, alm.shape
        assert fl.ndim == 1 and fl.size > self.lmax_ivfs[f], fl.shape
        if mmax is None:
            mmax = utils_hp.Alm.getlmax(alm.size, mmax)
        retlm = utils_hp.alm_copy(alm, mmax, self.lmax_ivfs[f], self.mmax_ivfs[f])
        utils_hp.almxfl(retlm, fl, self.mmax_ivfs[f], True)
        return retlm

    def __call__(self, alms: dict[str: np.ndarray]):
        """Applies inverse-variance filtering to input alm arrays


        """
        ivf_alms = dict()
        for f in self.maps_labels:
            assert f in alms, alms.keys()
            assert alms[f].ndim == 1
            ivf_alms[f] = self._almxflcopy(f, self._fal[f+f], alms[f])
        self._build_fal()
        for fg in self._fal:
            assert len(fg) % 2 == 0, fg
            f, g = fg[:len(fg) // 2], fg[len(fg) // 2:]
            if f != g:  # off-diagonals, checking explicitly symmetry
                fac = 1 if (g + f) in self._fal else 2
                assert (g + f not in self._fal) or (self._fal[g + f] is self._fal[f + g])
                ivf_alms[f] += self._almxflcopy(f, fac * self._fal[fg], alms[g])
        return ivf_alms
