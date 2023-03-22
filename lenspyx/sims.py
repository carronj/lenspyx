"""Generic cmb-only sims module

"""

import numpy as np
from plancklens.sims import cmbs
from lenspyx import utils_hp
from lenspyx.remapping.deflection import deflection
from lenspyx.remapping.utils_geom import Geom
from lenscarf import cachers

class sims_cmb_len(object):
    """Lensed CMB skies simulation library.

        Args:
            lmax_len: lensed cmbs are produced up to lmax
            cmb_unl:  plancklens unlensed cmbs library instance
            offsets_plm: offset lensing plm simulation index (useful e.g. for MCN1), tuple with block_size and offsets
            offsets_cmbunl: offset unlensed cmb (useful e.g. for MCN1), tuple with block_size and offsets
            dlmax(defaults to 1024): unlensed cmbs are produced up to lmax + dlmax, for accurate lensing at lmax


        Note:
            These sims do not contain aberration or modulation


    """
    def __init__(self, lmax_len:int, cmb_unl:cmbs.sims_cmb_unl,
                 cache:cachers.cacher or None=None, offsets_plm:tuple or None=None, offsets_cmbunl:tuple or None=None,
                 dlmax:int=1024, dlmax_gl:int=1024, epsilon:float=1e-5, verbosity=0):

        if cache is None:  # Will not save the lensed unless this is set
            cache = cachers.cacher_none()

        self.lmax = lmax_len
        self.dlmax = dlmax
        self.lmax_unl = lmax_len + dlmax
        self.dlmax_gl = dlmax_gl

        # ducc0 parameters:
        self.epsilon = epsilon
        self.verbosity = verbosity

        self.unlcmbs = cmb_unl
        self.fields = cmb_unl.fields

        self.offset_plm = offsets_plm if offsets_plm is not None else (1, 0)
        self.offset_cmb = offsets_cmbunl if offsets_cmbunl is not None else (1, 0)

        self.cacher = cache


    @staticmethod
    def offset_index(idx, block_size, offset):
        """Offset index by amount 'offset' cyclically within blocks of size block_size

        """
        return (idx // block_size) * block_size + (idx % block_size + offset) % block_size

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,
                'offset_plm':self.offset_plm, 'offset_cmb':self.offset_cmb,
                'epsilon':self.epsilon, 'ofactor':1.5}

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'e':
            return self.get_sim_eblm(idx)[0]
        elif field == 'b':
            return self.get_sim_eblm(idx)[1]
        elif field == 'p':
            return self.get_sim_plm(idx)
        elif field == 'o':
            return self.get_sim_olm(idx)
        else :
            assert 0, (field,self.fields)

    def get_sim_plm(self, idx):
        return self.unlcmbs.get_sim_plm(self.offset_index(idx, self.offset_plm[0], self.offset_plm[1]))

    def get_sim_olm(self, idx):
        if 'o' in self.fields:
            return self.unlcmbs.get_sim_olm(idx)
        else:
            return np.zeros_like(self.get_sim_plm(idx))

    def _get_dlm(self, idx):
        dlm = self.get_sim_plm(idx)
        dclm = self.get_sim_olm(idx) # curl mode
        lmax_dlm = utils_hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        # potentials to deflection
        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        utils_hp.almxfl(dlm,  p2d, mmax_dlm, True)
        utils_hp.almxfl(dclm, p2d, mmax_dlm, True)
        return dlm, dclm, lmax_dlm, mmax_dlm

    def _get_f(self, idx):
        dlm, dclm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
        lenjob_geometry = Geom.get_thingauss_geometry(self.lmax_unl + self.dlmax_gl, 2)
        f = deflection(lenjob_geometry, dlm, mmax_dlm, cacher=cachers.cacher_mem(safe=False), dclm=dclm,
                       epsilon=self.epsilon, verbosity=self.verbosity)
        return f

    def get_sim_tlm(self, idx):
        fname ='sim_%04d_tlm'%idx
        if not self.cacher.is_cached(fname):
            tlm= self.unlcmbs.get_sim_tlm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
            f = self._get_f(idx)
            tlm = f.lensgclm(tlm, self.lmax_unl, 0, self.lmax, self.lmax)
            self.cacher.cache(fname, tlm)
            return tlm
        return self.cacher.load(fname)

    def get_sim_eblm(self, idx):
        fneb ='sim_%04d_eblm'%idx
        if not self.cacher.is_cached(fneb):
            f = self._get_f(idx)
            if 'b' not in self.fields:
                elm = self.unlcmbs.get_sim_elm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
                eblm = f.lensgclm(np.atleast_2d(elm), self.lmax_unl, 2, self.lmax, self.lmax)
            else:
                eblm = np.empty((2, utils_hp.Alm.getsize(self.lmax_unl, self.lmax_unl)), dtype=complex)
                eblm[0] = self.unlcmbs.get_sim_elm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
                eblm[1] = self.unlcmbs.get_sim_blm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
                eblm = f.lensgclm(eblm, self.lmax_unl, 2, self.lmax, self.lmax)
            self.cacher.cache('sim_%04d_eblm' % idx, eblm)
            return eblm
        return self.cacher.load(fneb)

    def get_sim_elm(self, idx):
        return self.get_sim_eblm(idx)[0]

    def get_sim_blm(self, idx):
        return self.get_sim_eblm(idx)[1]
