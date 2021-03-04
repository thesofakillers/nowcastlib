"""
Functions for generating dynamically lagged time series.
"""
import numpy as np
from nowcastlib.signals import gen_composite_red_noise


def simulate_perturbations(unperturbed_quantity, rn_weight, wn_weight):
    """
    Simulates the perturbations in turbulence moving geographically between sites by
    adding red noise and white noise as weighted percentage errors to the input signal.

    Parameters
    ----------
    unperturbed_quantity : numpy.ndarray
        The input signal we wish to perturb
    rn_weight : float
        Factor to scale the effect of red noise
    wn_weight : float
        Factor to scale the effect of white noise

    Returns
    -------
    numpy.ndarray
        The original signal, now perturbed
    tuple of numpy.ndarray
        (red_noise, white_noise) -- Tuple containing the red noise and white noise
        signals that were generated before being applied to the array
    """
    red_noise = gen_composite_red_noise(len(unperturbed_quantity), 600)
    white_noise = np.random.randn(len(unperturbed_quantity))
    return (
        (
            unperturbed_quantity
            + unperturbed_quantity * (rn_weight * red_noise)
            + unperturbed_quantity * (wn_weight * white_noise)
        ),
        (red_noise, white_noise),
    )


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
