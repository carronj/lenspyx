import numpy as np

def prefilter(m):
    assert m.ndim == 2
    ny, nx = m.shape
    w0 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(ny)) + 4.)
    w1 = 6. / (2. * np.cos(2. * np.pi * np.fft.fftfreq(nx)) + 4.)
    return np.fft.ifft2(np.fft.fft2(m) * np.outer(w0, w1)).real
