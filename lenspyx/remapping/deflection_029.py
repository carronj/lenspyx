from __future__ import annotations


import numpy as np
from lenspyx.utils_hp import Alm
from lenspyx import cachers
from lenspyx.remapping import deflection_028 as deflection_28
from ducc0.sht.experimental import adjoint_synthesis_general, synthesis_general
import ducc0
try:
    from lenspyx.fortran.remapping import remapping as fremap
    HAS_FORTRAN = True
except:
    HAS_FORTRAN = False

HAS_DUCCGRADONLY = 'mode:' in synthesis_general.__doc__
HAS_DUCCROTATE = 'lensing_rotate' in ducc0.misc.__dict__
if not HAS_DUCCGRADONLY or not HAS_DUCCROTATE:
    print("You might need to update to ducc0 latest version")
# some helper functions

class deflection(deflection_28.deflection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cis =  False # Testing this

    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin:int, backwards:bool, polrot=True, ptg=None):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        self.tim.start('gclm2lenmap')
        self.tim.reset()
        if self.single_prec and gclm.dtype != np.complex64:
            print('** gclm2lenmap: inconsistent input dtype !')
            gclm = gclm.astype(np.complex64)
        gclm = np.atleast_2d(gclm)
        sht_mode = deflection_28.ducc_sht_mode(gclm, spin)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if ptg is None:
            ptg = self._get_ptg()
        assert ptg.shape[-1] == 2, ptg.shape
        assert ptg.dtype == np.float64, 'synthesis_general only accepts float here'
        if spin == 0:
            values = synthesis_general(lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg, spin=spin, epsilon=self.epsilon,
                                       nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)
            self.tim.add('synthesis general (%s)' % sht_mode)
        else:
            npix = self.geom.npix()
            # This is a trick with two views of the same array to get complex values as output to multiply by the phase
            valuesc = np.empty((npix,), dtype=np.complex64 if self.single_prec else np.complex128)
            values = valuesc.view(np.float32 if self.single_prec else np.float64).reshape((npix, 2)).T
            synthesis_general(map=values, lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg, spin=spin, epsilon=self.epsilon,
                              nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)
            self.tim.add('synthesis general (%s)' % sht_mode)
            if spin and polrot:
                if HAS_DUCCROTATE:
                    ducc0.misc.lensing_rotate(valuesc, self._get_gamma(), spin, self.sht_tr)
                    self.tim.add('polrot (ducc)')
                else:
                    func = fremap.apply_inplace if valuesc.dtype == np.complex128 else fremap.apply_inplacef
                    func(valuesc, self._get_gamma(), spin, self.sht_tr)
                    self.tim.add('polrot (fortran)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        return values

    def lenmap2gclm(self, points:np.ndarray[float], spin:int, lmax:int, mmax:int, gclm_out=None, sht_mode='STANDARD'):
        assert points.ndim == 2, points.ndim
        assert not np.iscomplexobj(points), (spin, points.ndim, points.dtype)
        self.tim.start('lenmap2gclm')
        self.tim.reset()
        ptg = self._get_ptg()
        self.tim.add('get_pointing')
        if gclm_out is not None:
            assert deflection_28.rtype[gclm_out.dtype] == points.dtype, 'precision must match'
        ret = adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points, loc=ptg, spin=spin, epsilon=self.epsilon,
                                            nthreads=self.sht_tr, mode=sht_mode, alm=gclm_out, verbose=self.verbosity)
        self.tim.add('adjoint_synthesis_general (%s)'%sht_mode)
        self.tim.close('lenmap2gclm')
        return ret.squeeze()

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cachers.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        return deflection(self.geom, dlm[0], mmax_dlm, self.sht_tr, cacher, dlm[1],
                          verbosity=self.verbosity, epsilon=self.epsilon, single_prec=self.single_prec)
