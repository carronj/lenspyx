import numpy as np
import healpy as hp

def _sind_d_m1(d, deriv=False):
    """Approximation to sind / d  - 1"""
    assert np.max(d) <= 0.01, (np.max(d), 'CMB Lensing deflections should never be that big')
    d2 = d * d
    if not deriv:
        return np.poly1d([0., -1 / 6., 1. / 120., -1. / 5040.][::-1])(d2)
    else:
        return - 1. / 3. * (1. -  d2 / 10. * ( 1.  - d2 / 28.))


def resolve_phi_poles(red, imd, cost, sint, cap, verbose=False):
    """Hack to numerically resolve the poles deflection ambiguities """
    d = np.sqrt(red ** 2 + imd ** 2)
    sind_d = 1. + _sind_d_m1(d)
    costp = np.cos(d) * cost - red *  sind_d * sint
    sintp = np.sqrt(1. - costp ** 2)
    s0 = np.sign(imd)
    s1 = np.sign(imd * np.cos(d) / sintp + imd * sind_d / sintp ** 3 * costp * (-np.sin(d) * cost * d - red * np.cos(d) * sint))
    criticals = np.where(s0 != s1)[0]
    if verbose:
        print("resolve_poles: I have flipped %s signs out of %s pixels on %s pole"%(len(criticals), len(red), cap))
    #: sol is dphi = pi - asin(Im / sintp * sind /d) instead of asin
    return criticals

def get_angles(nside, pix, red, imd, cap, verbose=True):
    """Builds deflected positions according to deflection field red, imd and undeflected coord. cost and phi.

        Very close to the poles, the relation sin(p' - p) = Im[d] / sint' (sin d / d) can be ambiguous.
        We resolve this through the *resolve_phi_poles* function for a number of candidate pixels.

        Returns:
            deflected tht and phi coordinates

    """
    assert len(pix) == len(red) and len(pix) == len(imd)
    tht, phi = hp.pix2ang(nside, pix)
    cost = np.cos(tht)
    sint = np.sqrt(1. - cost ** 2)
    d = np.sqrt(red ** 2 + imd ** 2)
    cosd = np.cos(d)
    sind_d = _sind_d_m1(d) + 1.
    costp = cost * cosd - red * sind_d * sint

    dphip = np.arcsin(imd / np.sqrt(1. - costp ** 2) * sind_d)
    if cap == 'north':
        crit = np.where((cosd <=  cost) & (red <= 0.))[0]
    elif cap == 'south':
        crit = np.where((cosd <= -cost) & (red >= 0.))[0]
    else:
        assert 0
    #: candidates for ambiguous relation sin dphi = imd / sintp sind / d.
    #: This is either arcsin(...) or pi - arcsin(...) (if > 0) or -(pi - |arcsin|) (if < 0)
    if len(crit) > 0:
        if np.isscalar(cost) and np.isscalar(sint): #for ring-only calc.
            criticals = resolve_phi_poles(red[crit], imd[crit], cost, sint, cap, verbose=verbose)
        else:
            criticals = resolve_phi_poles(red[crit], imd[crit], cost[crit], sint[crit], cap, verbose=verbose)

        sgn = np.sign(dphip[crit[criticals]])
        dphip[crit[criticals]] = sgn * (np.pi - np.abs(dphip[crit[criticals]]))
    return np.arccos(costp), phi + dphip

def rotation(nside, spin, pix, redi, imdi):
    """Complex rotation of the deflected spin-weight field from local axes //-transport

    """
    assert spin > 0, spin
    assert len(pix) == len(redi) and len(pix) == len(imdi)
    d = np.sqrt(redi ** 2 + imdi ** 2)
    tht, phi = hp.pix2ang(nside, pix)
    if np.min(d) > 0:
        # tanap = imdi / (d * np.sin(d) * (np.cos(tht) / np.sin(tht)) + redi * np.cos(d))

        gamma = np.arctan2(imdi, redi) - np.arctan2(imdi, d * np.sin(d) * (np.cos(tht) / np.sin(tht)) + redi * np.cos(d))
    else:
        gamma = np.zeros(len(pix), dtype=float)
        i = np.where(d > 0.)
        # tanap = imdi[i] / (d[i] * np.sin(d[i]) * (np.cos(tht[i]) / np.sin(tht[i])) + redi[i] * np.cos(d[i]))
        gamma[i] = np.arctan2(imdi[i], redi[i]) - np.arctan2(imdi[i],
                                                             d[i] * np.sin(d[i]) * (np.cos(tht[i]) / np.sin(tht[i])) +
                                                             redi[i] * np.cos(d[i]))
    return np.exp(1j * spin * gamma)
