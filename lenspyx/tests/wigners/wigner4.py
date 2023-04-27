from __future__ import  annotations
try:
    from plancklens.wigners import wigners as wigners_pl
except:
    assert 0, 'cant do this test witout plancklens'

import numpy as np
from ducc0.misc import GL_thetas
from lenspyx.wigners import wigners

if __name__ == '__main__':
    lmax = 5
    cl = np.random.standard_normal(lmax + 1)
    s2 = 2
    npts = (3 * lmax) // 2 + 1

    theta = GL_thetas(npts)
    xg = np.cos(theta)
    maxdev = 0.
    for s1 in np.arange(-3, 4):
        for s2 in np.arange(-3, 4):
            gps = wigners.wigner4pos(cl, None, theta, s1, s2)
            ref = wigners_pl.wignerpos(cl, xg, s1, abs(s2))
            this_dev = np.max(np.abs(gps[0] - ref))
            maxdev = max(maxdev, this_dev)
            if this_dev > 1e-9:
                print(s1, s2)
            if s2:
                ref = wigners_pl.wignerpos(cl, xg, s1, -abs(s2))
                maxdev = max(maxdev, np.max(np.abs(gps[1] - ref)))

    print(maxdev)
    assert maxdev < 1e-9
    gl = np.random.standard_normal(lmax + 1)
    cl = np.random.standard_normal(lmax + 1)

    maxdev = 0.
    for s1 in np.arange(-3, 4):
        for s2 in np.arange(-3, 4):
            gps = wigners.wigner4pos(gl, cl, theta, s1, s2)
            ref = wigners_pl.wignerpos(gl, xg, s1, abs(s2))
            maxdev = max(maxdev, np.max(np.abs(gps[0] - ref)))
            if s2:
                ref = wigners_pl.wignerpos(gl, xg, s1, -abs(s2))
                maxdev = max(maxdev, np.max(np.abs(gps[1] - ref)))
                ref = wigners_pl.wignerpos(cl, xg, s1,  abs(s2))
                maxdev = max(maxdev, np.max(np.abs(gps[2] - ref)))
                ref = wigners_pl.wignerpos(cl, xg, s1, -abs(s2))
                maxdev = max(maxdev, np.max(np.abs(gps[3] - ref)))
    print(maxdev)

    assert maxdev < 1e-9