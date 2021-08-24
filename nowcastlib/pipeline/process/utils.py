"""
Shared functionality across pre and postprocessing
"""
import logging
from typing import Union
import pandas as pd
from nowcastlib.pipeline.structs import config


logger = logging.getLogger(__name__)


def drop_outliers(input_series: pd.core.series.Series, options: config.OutlierOptions):
    """
    drops 'outliers' from a given pandas input series
    given inclusive thresholds specified in the input
    config object

    Parameters
    ----------
    input_series: pandas.core.series.Series
    options : nowcastlib.pipeline.structs.config.OutlierOptions

    Returns
    -------
    pandas.core.series.Series
        the filtered series
    """

    if options.quantile_based:
        return input_series[
            (input_series.quantile(options.lower) <= input_series)
            & (input_series <= input_series.quantile(options.upper))
        ]
    else:
        return input_series[
            (options.lower <= input_series) & (input_series <= options.upper)
        ]


def handle_periodic(
    input_series: pd.core.series.Series, options: config.PeriodicOptions
):
    """
    Normalizes a periodic series such that its values lies in
    the range [0, T-1] where T is the period length, as defined
    in the input config object

    Parameters
    ----------
    input_series: pandas.core.series.Series
    options : nowcastlib.pipeline.structs.config.PeriodicOptions

    Returns
    -------
    pandas.core.series.Series
        the normalized series
    """
    return input_series % options.period_length


def handle_smoothing(
    input_series: pd.core.series.Series, options: config.SmoothOptions
):
    """
    Applies a moving average calculation to an input time series
    so to achieve some form of smoothing

    Parameters
    ----------
    input_series: pandas.core.series.Series
    options : nowcastlib.pipeline.structs.config.SmoothOptions

    Returns
    -------
    pandas.core.series.Series
        the smoothed series
    """
    data_series = input_series.copy()
    window_size = options.window_size
    shift_size = int((window_size + 1) / 2)
    units = options.units
    window: Union[str, int]
    if units is not None:
        window = str(window_size) + units
    else:
        window = window_size
    return (
        data_series.rolling(
            window=window,
            closed="both",
        )
        .mean()
        .shift(-shift_size, freq=units)
    )


def process_field(
    input_series: pd.core.series.Series,
    options: config.ProcessingOptions,
    preproc_flag: bool = True,
):
    """
    (Pre/Post)-processes a field

    Parameters
    ----------
    input_series : pandas.core.series.Series
        The data of the field to process
    options : nowcastlib.pipeline.structs.config.ProcessingOptions
        Configuration options for specifying how to process
    preproc_flag : bool, default `True`
        Whether this is for preprocessing. If `False`,
        postprocessing is assumed.

    Returns
    -------
    pandas.core.series.Series
        The resulting processed field
    """
    data_series = input_series.copy()
    if options.outlier_options is not None:
        if preproc_flag is False:
            logger.warning("Outlier removal may be better suited for preprocessing")
        logger.debug("Dropping outliers...")
        data_series = drop_outliers(data_series, options.outlier_options)
    if options.periodic_options is not None:
        if preproc_flag is False:
            logger.warning(
                "Periodic normalizations may be better suited for preprocessing"
            )
        logger.debug("Normalizing periodic ranges...")
        data_series = handle_periodic(data_series, options.periodic_options)
    if options.conversion_options is not None:
        if preproc_flag is False:
            logger.warning("Unit conversions may be better suited for preprocessing")
        logger.debug("Converting units...")
        data_series = options.conversion_options.conv_func(data_series)
    if options.smooth_options is not None:
        if preproc_flag is True:
            logger.warning("Smoothing may be better suited for postprocessing")
        logger.debug("Applying moving average for smoothing...")
        data_series = handle_smoothing(data_series, options.smooth_options)
    return data_series
