import time
import healpy as hp
from lenspyx import lensing
import numpy as np
from plancklens import utils

#  33   Megapix /s (excl. prefiltering) if the map already prefiltered for fortran ordering
#  13   Megapix /s (excl. prefiltering) if the map already prefiltered for C ordering

lmax = 2048 + 1024
nside = 2048
cls_unl = utils.camb_clfile('/Users/jcarron/PycharmProjects/Plancklens2018/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat')
tunl = hp.synalm(cls_unl['tt'][:lmax + 1])
dunl = hp.synalm(cls_unl['pp'][:lmax + 1])
hp.almxfl(dunl, np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)), inplace=True)
lenmap = lensing.tlm2lenmap(nside, tunl, dunl, verbose=True, nband=8, facres=-1)
