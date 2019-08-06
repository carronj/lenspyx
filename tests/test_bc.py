import time
from lenspyx import bicubic as t, filter as ftl
import numpy as np

order = 'F'
shape = (46, 46)
m = np.random.random(shape)
m_ftl = ftl.prefilter(m)
fx = np.outer(np.ones(shape[0]), np.arange(shape[1]))
fy = np.outer(np.arange(shape[0]), np.ones(shape[1]))

fx = np.copy(fx, order=order)
fy = np.copy(fy, order=order)
m_ftl = np.copy(m_ftl, order=order)

t0 = time.time()
m_len = t.deflect(m_ftl, fx, fy)
dt = time.time() - t0
print("Crude remp. speed %.1f Mpix / sec"%(np.prod(shape) / 1e6 / dt))
print(np.max(np.abs(m_len / m - 1.)))
print(m_len.flags)

#  33   Megapix /s (excl. prefiltering) if the map already prefiltered for fortran ordering
#  13   Megapix /s (excl. prefiltering) if the map already prefiltered for C ordering
