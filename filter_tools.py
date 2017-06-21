"""
Utility file for temporal filters
"""

from skimage.filters import gaussian
from scipy.signal import firwin
from pyfftw.interfaces.scipy_fftpack import fft, ifft, ifftshift
from numpy import tile, real, min, zeros

def amplitude_weighted_blur(x, weight, sigma):
    if sigma != 0:
        return gaussian(x*weight, sigma, mode="wrap") / gaussian(weight, sigma, mode="wrap")
    return x

def difference_of_iir(delta, rl, rh):
    """
    difference of infinite impulse responses
    """
    lowpass_1 = delta[0].copy()
    lowpass_2 = lowpass_1.copy()
    out = zeros(delta.shape, dtype=delta.dtype)
    for i in range(1, delta.shape[0]):
        lowpass_1 = (1-rh)*lowpass_1 + rh*delta[i]
        lowpass_2 = (1-rl)*lowpass_2 + rl*delta[i]
        out[i] = lowpass_1 - lowpass_2
    return out

def fir_window_bp(delta, fl, fh):
    """
    Finite impulse response, bandpass.
    This filter doesn't work exactly like the matlab version due to some fourier transform imprecisions.
    Consider replacing the transform calls to the FFTW versions.
    """
    b = firwin(delta.shape[0]+1, (fl*2, fh*2), pass_zero=False)[:-1]
    m = delta.shape[1]
    batches = 20
    batch_size = int(m / batches) + 1
    temp = fft(ifftshift(b))
    out = zeros(delta.shape, dtype=delta.dtype)
    for i in range(batches):
        indexes = (batch_size*i, min((batch_size*(i+1), m)))
        freq = fft(delta[:,indexes[0]:indexes[1]], axis=0)*tile(temp, (delta.shape[2],indexes[1]-indexes[0], 1)).swapaxes(0,2)
        out[:, indexes[0]:indexes[1]] = real(ifft(freq, axis=0))
    return out