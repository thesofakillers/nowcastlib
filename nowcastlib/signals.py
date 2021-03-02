"""
Utils for working with signal processing
"""
import numpy as np
from nowcastlib.utils import find_closest_factor


def normalize_signal(input_signal):
    """
    Rescales a signal such that it is 0-centered and has a max amplitude of 1.

    Parameters
    ----------
    input_signal : numpy.ndarray
        The signal we wish to normalize.

    Returns
    -------
    numpy.ndarray
        The resulting normalized signal
    """
    # centering it about 0
    max_signal = np.max(input_signal)
    min_signal = np.min(input_signal)
    signal_range = max_signal - min_signal
    mid_signal = max_signal - signal_range / 2
    zero_c_signal = input_signal - mid_signal
    # setting max = 1, min = -1
    normed_signal = zero_c_signal / max(zero_c_signal)
    return normed_signal


def gen_composite_red_noise(noise_length, component_length, normed=True):
    """
    Generates a signal made of sub-signals of red noise.

    Parameters
    ----------
    noise_length : int
        The desired output signal length
    component_length : int
        The desired length of each sub-signal red noise.
        If `component_length` is not a factor of `noise_length`, the factor of
        `noise_length` closest to `component_length` will be used
    normed : bool, default True
        Whether each sub-signal should be normalized, i.e. centered about 0 with a max
        amplitude of 1.

    Returns
    -------
    numpy.ndarray
        A signal made of sub-signals of red noise.
    """
    white_noise = np.random.randn(noise_length)
    # reshape white noise into comps, then cumsum each component individually
    white_noise = white_noise.reshape(
        (-1, find_closest_factor(noise_length, component_length))
    )
    red_noise_comps = np.cumsum(white_noise, axis=1)
    if normed:
        red_noise_comps = np.apply_along_axis(normalize_signal, 1, red_noise_comps)
    # remember to flatten before returning since we're actually working in 1D
    return red_noise_comps.flatten()
