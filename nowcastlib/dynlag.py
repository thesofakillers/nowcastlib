"""
Functions for generating dynamically lagged time series.
"""
import numpy as np
import nowcastlib.signals as signals


def simulate_perturbations(input_sig, snr_db, rn_comp_len=0, allow_neg=True):
    """
    Simulates the perturbations in turbulence moving geographically between sites by
    adding red noise and white noise to the input signal.

    Parameters
    ----------
    input_sig : numpy.ndarray
        The input signal we wish to perturb
    snr_db : float
        The desired signal to noise ratio in dB. Must be greater than 1
    rn_comp_len : int, default 0
        In case composite red noise is required, the desired length of each red noise
        sub-signal. Non-composite red noise is generated if 0.
    allow_neg: bool, default True
        Whether the perturbations are allowed to cause negative values in the resulting
        signal.

    Returns
    -------
    numpy.ndarray
        The original signal, now perturbed
    tuple of numpy.ndarray
        (red_noise, white_noise) -- Tuple containing the red noise and
        white noise that were generated before being applied to the input signal
    """
    desired_length = len(input_sig)
    if rn_comp_len == 0:
        red_noise = signals.normalize_signal(np.random.randn(desired_length).cumsum())
    else:
        red_noise = signals.gen_composite_red_noise(desired_length, rn_comp_len)
    white_noise = np.random.randn(desired_length)
    # here we scale the noise such that adding it will match desired SNR.
    additive_noise = signals.scale_noise(input_sig, red_noise + white_noise, snr_db)
    output_signal = input_sig + additive_noise
    # flip sign on noise values that cause output to be negative
    if not allow_neg:
        output_signal = np.where(
            output_signal < 0,
            input_sig + np.abs(additive_noise),
            output_signal,
        )
    return (output_signal, (red_noise, white_noise))


def dynamically_lag(input_signal, wind_speed, wind_alignment, distance):
    """
    Dynamically lags a time-series to simulate the transport via wind of a phenomenon
    between two geo-spatially distanced sensors (A and B). Incorporation of noise is
    deferred to a separate function.

    Parameters
    ----------
    input_signal : pandas.core.series.Series
        Time-indexed series representing the signal measured at sensor A
    wind_speed : pandas.core.series.Series
        Time-indexed series representing the wind speed [m/s] measured at sensor A
    wind_alignment : pandas.core.seres.Series
        Time-indexed series containing the dot product between the initial bearing from
        sensor A to sensor B and the direction the wind is blowing at sensor A
    distance : float
        The distance in meters between target and source locations

    Returns
    -------
    pandas.core.series.Series
        The dynamically lagged `input_series`.
    """
    spacing_secs = input_signal.index.freq.delta.seconds
    # how many time steps in the past is the target series?
    time_secs = distance / (wind_alignment * wind_speed)
    time_steps = (
        np.floor((time_secs / spacing_secs))
        .replace([np.inf, -np.inf], np.nan)
        .astype("Int64")
    )
    total_steps = len(input_signal)
    target_idx = np.arange(total_steps) + time_steps
    # keep track of invalid indices
    nan_mask = (target_idx < 0) | (target_idx >= total_steps) | target_idx.isna()
    # temporarily set nan indices to 0, to be able to use them.
    target_idx[nan_mask] = 0
    target_series = np.where(nan_mask, np.nan, input_signal[target_idx.values])
    return target_series
