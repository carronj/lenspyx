from __future__ import annotations
import numpy as np
from lenspyx.wigners.wigners import get_thgwg, wignerpos, wignercoeff
from lenspyx.utils import timer
verbose = False


class WignerAccumulator:
    """Helper class to compute sums of products of Wigner small-d correlation functions


    """
    def __init__(self, npts: int):
        tht, wg = get_thgwg(npts)
        self.tht = tht
        self.wg = wg

        self.data_wflip = {}
        self.data_nflip = {}
        self.tim = timer('', False)


    def add(self, cl1: np.ndarray[float], cl2: np.ndarray[float], s1: int, t1: int, s2: int, t2: int):
        self.tim.reset()
        tht = self.tht
        so, to = s1 + s2, t1 + t2  # out-spins
        idx, flip_parity, global_sgn, L_sign = self._spinids(so, to)
        if verbose:
            print('add ', so, to, (s1,  t1), (s2, t2), global_sgn, flip_parity)

        xi12  = wignerpos(cl1 * global_sgn, tht, s1, t1)
        xi12 *= wignerpos(cl2,              tht, s2, t2)
        storage = self.data_wflip if flip_parity else self.data_nflip
        if idx not in storage.keys():
            storage[idx] = np.zeros(tht.size, dtype=float)
        storage[idx] += xi12
        self.tim.add('add')

    def add_xi1(self, xi1:np.ndarray, s1: int, t1: int, cl2:np.ndarray, s2: int, t2: int):
        self.tim.reset()
        tht = self.tht
        so, to = s1 + s2, t1 + t2  # out-spins
        idx, flip_parity, global_sgn, L_sign = self._spinids(so, to)
        if verbose:
            print('add xi1', so, to, (s1,  t1), (s2, t2), global_sgn, flip_parity)
        storage = self.data_wflip if flip_parity else self.data_nflip
        if idx not in storage.keys():
            storage[idx] = np.zeros(tht.size, dtype=float)
        storage[idx] += xi1 * wignerpos(cl2 * global_sgn, tht, s2, t2)
        self.tim.add('add xi')

    def add_xi2(self, cl1:np.ndarray, s1: int, t1: int, xi2:np.ndarray, s2: int, t2: int):
        self.tim.reset()
        tht = self.tht
        so, to = s1 + s2, t1 + t2  # out-spins
        idx, flip_parity, global_sgn, L_sign = self._spinids(so, to)
        if verbose:
            print('add xi', so, to, (s1,  t1), (s2, t2), global_sgn, flip_parity)
        storage = self.data_wflip if flip_parity else self.data_nflip
        if idx not in storage.keys():
            storage[idx] = np.zeros(tht.size, dtype=float)
        storage[idx] += wignerpos(cl1 * global_sgn, tht, s1, t1) * xi2
        self.tim.add('add xi')

    def accumulate2(self, cl1:np.ndarray, cls2:list[np.ndarray], s1: int, t1: int, s2ts:list[int], t2s: list[int]):
        assert len(s2ts) == len(t2s) == len(cls2)
        self.tim.reset()
        tht = self.tht
        xi1 = wignerpos(cl1, tht, s1, t1)
        for s2, t2, cl2 in zip(s2ts, t2s, cls2):
            so, to = s1 + s2, t1 + t2  # out-spins
            idx, flip_parity, global_sgn, L_sign = self._spinids(so, to)
            storage = self.data_wflip if flip_parity else self.data_nflip
            if idx not in storage.keys():
                storage[idx] = np.zeros(tht.size, dtype=float)
            storage[idx] += xi1 * wignerpos(cl2 * global_sgn, tht, s2, t2)
        self.tim.add('accumulate')

    def accumulate1(self, cl1s: list[np.ndarray], cl2: np.ndarray, s1s: list[int], t1s: list[int], s2: int, t2: int):
        self.accumulate2(cl2, cl1s, s2, t2, s1s, t1s)

    def flush(self, so:int, to:int, lmax: int):
        self.tim.reset()
        ret = np.zeros(lmax + 1, dtype=float)
        spin_idx = self._spinids(so, to)[0]
        storage = self.data_wflip
        if spin_idx in storage:
            s, t = spin_idx
            ret += wignercoeff(storage[spin_idx][::-1] * self.wg, self.tht, s, t, lmax)
        ret *= np.where(np.arange(lmax + 1) % 2 == 0, 1, -1)
        storage = self.data_nflip
        if spin_idx in storage:
            s, t = spin_idx
            ret += wignercoeff(storage[spin_idx] * self.wg, self.tht, s, t, lmax)
        self.tim.add('flush')
        if verbose:
            print(self.tim)
        return ret
    @staticmethod
    def _spinids(so, to):  # Can always make both spins positive and  so >= to
        L_sign = (-1 if so < 0 else 1) * (-1 if to < 0 else 1)  # do we need to add a (-1)^L sign?
        so_sgn = 1 if so % 2 == 0 else -1  # (-1) ** s0
        to_sgn = 1 if to % 2 == 0 else -1  # (-1) ** t0
        global_sgn = (so_sgn if so < 0 else 1) * (to_sgn if to < 0 else 1) * ((so_sgn * to_sgn) if to > so else 1)
        # sorting procedure forcing so > to
        flip_parity = (L_sign < 0)  # Do we need to flip the array?
        idx = (max(abs(so), abs(to)), min(abs(so), abs(to)))
        return idx, flip_parity, global_sgn, L_sign
