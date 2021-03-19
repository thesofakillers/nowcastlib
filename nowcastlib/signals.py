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


def scale_noise(input_signal, noise_signal, snr_db):
    """
    Appropriately scales a noise signal so that it can be applied to an input signal
    as additive noise with a desired signal-to-noise ratio

    Parameters
    ----------
    input_signal : numpy.ndarray
        The input signal to which we wish to add noise
    noise_signal : numpy.ndarray
        The noise signal that we wish to add to `input_signal`
    snr_db : float
        The desired signal-to-noise ratio in decibels (dB)

    Returns
    -------
    numpy.ndarray
        The appropriately scaled noise signal. Adding this signal to `input_signal`
        will cause the resulting signal to have a SNR roughly equivalent to `snr_db`
    """
    snr = 10 ** (snr_db / 10)
    signal_energy = np.sum(input_signal ** 2)
    noise_energy = np.sum(noise_signal ** 2)
    noise_scalar = np.sqrt(signal_energy / (snr * noise_energy))
    return noise_scalar * noise_signal


def add_noise(input_signal, noise_signal, snr_db):
    """
    Adds a noise signal to an input signal at a given signal-to-noise ratio.

    Parameters
    ----------
    input_signal : numpy.ndarray
        The input signal to which we wish to add noise
    noise_signal : numpy.ndarray
        The noise signal that we wish to add to `input_signal`
    snr_db : float
        The desired signal-to-noise ratio in decibels (dB)

    Returns
    -------
    numpy.ndarray
        The original signal with the addition of noise at the desired SNR
    """
    additive_noise = scale_noise(input_signal, noise_signal, snr_db)
    return input_signal + additive_noise
