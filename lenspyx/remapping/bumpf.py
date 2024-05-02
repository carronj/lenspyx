import numpy as np
class bump:
    """Simple smooth bump function (first thing on wikipedia)

        Infinitely derivable and with compact support


    """
    @staticmethod
    def f(x):
        ret = np.zeros_like(x)
        ret[np.where(x > 0)] = np.exp(-1./x[np.where(x > 0)])
        return ret
    @staticmethod
    def g(x):
        return bump.f(x) / ( bump.f(x) + bump.f(1-x) )

    @staticmethod
    def bump(thmin, thmax, dtht, tht):
        """Smooth function equal to 1 on [thmin, thmax], and zero outside (thmin-dtht, thmax + dtht)


        """
        a,b,c,d = thmin-dtht, thmin, thmax, thmax + dtht
        assert a < b < c < d
        return bump.g( (tht-a) / (b-a) ) * bump.g( (d-tht) / (d-c))