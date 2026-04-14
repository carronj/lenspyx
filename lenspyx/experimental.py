import numpy as np
from ducc0.sht import synthesis_general as syngducc, adjoint_synthesis_general as adjsyngducc
try:
    import capsht
except ImportError:
    print("capsht not found, you will not be able to use the functions in this module")
try:
    from capsht.experimental import synthesis_general_cap, synthesis_general_band, adjoint_synthesis_general_cap, adjoint_synthesis_general_band
except ImportError:
    print("synthesis_general_cap or synthesis_general_band not found in capsht.experimental, are you up to date?")

def _epsapo(thtcap, epsilon, lmax, version=1, dl_7=None):
    #dl = dl_7 * ((- np.log10(epsilon) + 1) / (7 + 1)) ** 2
    assert version == 1, 'C++ code now only implemented for version 1'
    if version == 0:
        if dl_7 is None:
            dl_7 = 15
        dl = dl_7 * ((-np.log10(epsilon) / (7 )) ** 1) ** 0.5
    elif version == 1: 
        if dl_7 is None:
            dl_7 = 2*7*np.log(10.)/np.pi
        dl = dl_7 * (-np.log10(epsilon) / (7. ))
    else:
        raise ValueError('version %s not implemented'%version)
    return np.sqrt(dl / lmax * np.pi / thtcap)

def synthesis_general(alm: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float, 
                      thtcap:float=None, eps_apo:float=None, tht_min:float=None, tht_max:float=None, verbose:bool=False, **kwargs):
    """Wrapper to synthesis_general function, including a version tuned to spherical caps
    
        See ducc0.sht.synthesis_general for most arguments, optional arguments and outputs

        if thtcap is set, all points of interest are assumed to lie within a spherical cap, and a faster version is used

        relevant keywords can include *mode*, *map*, *mmax*
    
    """
    if False and tht_min is not None and tht_max is not None: 
        # Update synthesis_general_band first
        eps_apo = eps_apo or 1.2 * _epsapo(tht_max-tht_min, epsilon, lmax)
        thta_p = tht_min - 0.5 * eps_apo * (tht_max - tht_min)
        thtb_p = tht_max + 0.5 * eps_apo * (tht_max - tht_min)
        if (thta_p >= 0.) and (thtb_p <= np.pi):
            if verbose:
                print('syng type: band %.1f deg %.1f deg' % (thta_p/np.pi*180, thtb_p/np.pi*180))
            assert 0, 'fix band to new scheme'
            return synthesis_general_band(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, 
                                      thta=tht_min, thtb=tht_max, eps_apo=eps_apo, **kwargs)
        if thta_p < 0.: # Can try synthesis_general_cap later on
            thtcap = tht_max
            eps_apo = None
    if thtcap is not None: # attempt at synthesis_general_cap
        eps_apo = eps_apo or _epsapo(thtcap, epsilon, lmax)
        epsilon_nufft = kwargs.pop('epsilon_nufft', epsilon)
        if verbose:
            print('syng type: sent to cap %.1f epsapo %.2f' % (thtcap/np.pi*180, eps_apo))
        return synthesis_general_cap(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon_nufft, thtcap=thtcap, eps_apo=eps_apo, **kwargs)
    if verbose:
        print('syng type : general')
    return syngducc(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon,  **kwargs)

def adjoint_synthesis_general(map: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float, 
                      thtcap:float=None, eps_apo:float=None, tht_min:float=None, tht_max:float=None, verbose:bool=False, **kwargs):
    """Wrapper to capsht synthesis_general function, hiding the choice of eps_apo
    

        See ducc0.sht.synthesis_general for arguments, optional arguments and outputs

        relevant keyword, *mode*, *alm*, *mmax*

    
    """
    if False and tht_min is not None and tht_max is not None: 
        # Update synthesis_general_band first
        eps_apo = eps_apo or 1.2 * _epsapo(tht_max-tht_min, epsilon, lmax)
        thta_p = tht_min - 0.5 * eps_apo * (tht_max - tht_min)
        thtb_p = tht_max + 0.5 * eps_apo * (tht_max - tht_min)
        if (thta_p >= 0.) and (thtb_p <= np.pi):
            if verbose:
                print('adjsyng type: band %.1f deg %.1f deg' % (thta_p/np.pi*180, thtb_p/np.pi*180))
            return adjoint_synthesis_general_band(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, 
                                      thta=tht_min, thtb=tht_max, eps_apo=eps_apo, **kwargs)
        if thta_p < 0.: # Can try synthesis_general_cap later on
            thtcap = tht_max
            eps_apo = None
    if thtcap is not None: # attempt at synthesis_general_cap
        eps_apo = eps_apo or _epsapo(thtcap, epsilon, lmax)
        if verbose:
            print('adjsyng type: sent to cap %.1f deg, eps_apo %.2f' % (thtcap/np.pi*180, eps_apo))
        return adjoint_synthesis_general_cap(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, thtcap=thtcap, eps_apo=eps_apo, **kwargs)
    if verbose:
        print('adjsyng type : general')
    return syngducc(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon,  **kwargs)