"""Test pointing calculation on z-axis rotation

"""
import numpy as np
import ducc0


nrings = 100
nph = 20
phi0 = np.zeros(nrings, dtype=float)
nphi = np.ones(nrings, dtype=np.uint64) * nph
ofs = np.arange(nrings) * nph
d1 = np.zeros((nrings * nph, 2), dtype=float)
t = np.linspace(0.0001, np.pi / 2 - 0.0001, nrings)
tmp = np.pi - t
print('All these numbers should be zero')
for sgn in [1, -1]: # north or south
    d0 = -2 * t if sgn > 0 else 2 * t
    tht = t if sgn > 0 else tmp
    for ir in range(nrings):
        d1[ofs[ir]:ofs[ir] + nph, 0] = d0[ir] # deflection field that produces rotation of angle pi around z

    ptg = ducc0.misc.get_deflected_angles(theta=tht, phi0=phi0, nphi=nphi.astype(np.uint64),
                                          ringstart=ofs.astype(np.uint64), deflect=d1, calc_rotation=True, nthreads=1).T
    for ir in [0, 3, 10, 40]:
        phi_pred = phi0[ir] + np.arange(nph) * (2 * np.pi / nphi[ir]) + np.pi
        phi_pred[np.where(phi_pred >= 2 * np.pi)] -= 2 * np.pi
        sli = slice(ir * nph, ir * nph + nph)
        print('%.1f :'%(tht[ir] / np.pi * 180), np.max(np.abs(ptg[0, sli] / tht[ir] - 1.)), np.max(np.abs(ptg[1, sli] - phi_pred)),(np.max(np.abs(ptg[2, sli]) - np.pi)))