import numpy as np
try:
    import capsht
except ImportError:
    print("capsht not found, you will not be able to use the functions in this module")
try:
    from capsht.experimental import synthesis_general_cap
except ImportError:
    print("synthesis_general_cap not found in capsht.experimental, are you up to date?")

def _epsapo(thtcap, epsilon, lmax, dl_7=20):
    dl = dl_7 * ((- np.log10(epsilon) + 1) / (7 + 1)) ** 2
    return np.sqrt(dl / lmax * np.pi / thtcap)

def synthesis_general_cap(alm: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float, **kwargs):
    """Wrapper to capsht synthesis_general_cap function, hiding the choice of eps_apo
    
        If thtcap is not specified, it is set to np.max(loc[:, 0])

        See ducc0.sht.synthesis_general for arguments, optional arguments and outputs
    
    """
    thtcap  = kwargs.pop('thtcap', None)
    eps_apo = kwargs.pop('eps_apo', None)
    if thtcap  is None: thtcap  = np.max(loc[:, 0])
    if eps_apo is None: eps_apo = _epsapo(thtcap, epsilon, lmax)
    return capsht.experimental.synthesis_general_cap(alm=alm, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, thtcap=thtcap, eps_apo=eps_apo, **kwargs)

def adjoint_synthesis_general_cap(map: np.ndarray, spin: int, lmax: int, loc: np.ndarray, epsilon: float, **kwargs):
    """Wrapper to capsht adjoint_synthesis_general_cap function, hiding the choice of eps_apo
    
        If thtcap is not specified, it is set to np.max(loc[:, 0])

        See ducc0.sht.adjoint_synthesis_general for arguments, optional arguments and outputs
    
    """
    thtcap  = kwargs.pop('thtcap', None)
    eps_apo = kwargs.pop('eps_apo', None)
    if thtcap  is None: thtcap  = np.max(loc[:, 0])
    if eps_apo is None: eps_apo = _epsapo(thtcap, epsilon, lmax)
    return capsht.experimental.adjoint_synthesis_general_cap(map=map, spin=spin, lmax=lmax, loc=loc, epsilon=epsilon, thtcap=thtcap, eps_apo=eps_apo, **kwargs)