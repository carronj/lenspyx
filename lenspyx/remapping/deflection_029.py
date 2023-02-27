from __future__ import annotations


import numpy as np
from lenscarf.utils_hp import Alm
from lenspyx.remapping import deflection as deflection_28
import ducc0

try:
    from lenscarf.fortran import remapping as fremap
    HAS_FORTRAN = True
except:
    HAS_FORTRAN = False

try:
    import numexpr
    HAS_NUMEXPR = True
except:
    HAS_NUMEXPR = False
    print("deflection.py::could not load numexpr, falling back on python impl.")

# some helper functions

ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longfloat): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}
rtype = {np.dtype(np.complex64): np.float32,
         np.dtype(np.complex128): np.float64,
         np.dtype(np.longcomplex): np.longfloat,
         np.complex64: np.float32,
         np.complex128: np.float64,
         np.longcomplex: np.longfloat}

class deflection(deflection_28.deflection):
    def __int__(self, *args, **kwargs):
        super().init(*args, **kwargs)


    def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        self.tim.start('gclm2lenmap')
        self.tim.reset()
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if ptg is None:
            ptg = self._get_ptg()
        assert ptg.shape[-1] == 2, ptg.shape
        if ptg.dtype != np.float64: #FIXMEL synthesis general only accepts float
            ptg = ptg.astype(np.float64)
            self.tim.add('float type conversion')
        if spin == 0:
            values = ducc0.sht.experimental.synthesis_general(lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg, spin=spin,
                                                          epsilon=self.epsilon, nthreads=self.sht_tr)
            self.tim.add('synthesis general')
        else:
            npix = self.geom.npix()
            # This is a trick with two views of the same array to get complex values as output to multiply by the phase
            valuesc = np.empty((npix,), dtype=np.complex64 if self.single_prec else np.complex128)
            values = valuesc.view(np.float32 if self.single_prec else np.float64).reshape((npix, 2)).T
            ducc0.sht.experimental.synthesis_general(map=values, lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg,
                                                              spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr)
            self.tim.add('synthesis general')

            if polrot:
                if HAS_NUMEXPR:
                    mg = self._get_mgamma()
                    js = - 1j * spin
                    valuesc *= numexpr.evaluate("exp(js * mg)")
                    self.tim.add('polrot (numexpr)')
                else:
                    valuesc *= np.exp((-1j * spin) * self._get_mgamma())  # polrot. last entry is -gamma
                    self.tim.add('polrot (python)')
        self.tim.close('gclm2lenmap')
        if self.verbosity:
            print(self.tim)
        return values

    def lenmap2gclm(self, points:np.ndarray[complex], spin:int, lmax:int, mmax:int):
        assert points.ndim == 1 and np.iscomplexobj(points), (points.ndim, points.dtype)
        self.tim.start('lenmap2gclm')
        self.tim.reset()
        ptg = self._get_ptg()
        self.tim.add('_get_ptg')
        #points2 = np.empty((1 + (spin > 0), points.size), dtype=deflection_28.rtype[points.dtype])
        #points2[0, :] = points.real
        #if spin > 0:
        #    points2[1, :] = points.imag
        # Use a view instead to turn complex array into real:
        points2 = points.view(rtype[points.dtype]).reshape(points.size, 2).T
        self.tim.add('_refactoring to real')
        ret = ducc0.sht.experimental.adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points2, loc=ptg, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr)
        self.tim.add('adjoint_synthesis_general')
        self.tim.close('lenmap2gclm')
        return ret
