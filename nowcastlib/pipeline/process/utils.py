"""
Shared functionality across pre and postprocessing
"""
import logging
from typing import Union
import pandas as pd
from nowcastlib.pipeline import structs


logger = logging.getLogger(__name__)


def drop_outliers(input_series: pd.core.series.Series, config: structs.OutlierOptions):
    """
    drops 'outliers' from a given pandas input series
    given inclusive thresholds specified in the input
    config object

    Parameters
    ----------
    input_series: pandas.core.series.Series
    config : nowcastlib.pipeline.structs.OutlierOptions

    Returns
    -------
    pandas.core.series.Series
        the filtered series
    """

    if config.quantile_based:
        return input_series[
            (input_series.quantile(config.lower) <= input_series)
            & (input_series <= input_series.quantile(config.upper))
        ]
    else:
        return input_series[
            (config.lower <= input_series) & (input_series <= config.upper)
        ]


def handle_periodic(
    input_series: pd.core.series.Series, config: structs.PeriodicOptions
):
    """
    Normalizes a periodic series such that its values lies in
    the range [0, T-1] where T is the period length, as defined
    in the input config object

    Parameters
    ----------
    input_series: pandas.core.series.Series
    config : nowcastlib.pipeline.structs.PeriodicOptions

    Returns
    -------
    pandas.core.series.Series
        the normalized series
    """
    return input_series % config.period_length


def handle_smoothing(
    input_series: pd.core.series.Series, config: structs.SmoothOptions
):
    """
    Applies a moving average calculation to an input time series
    so to achieve some form of smoothing

    Parameters
    ----------
    input_series: pandas.core.series.Series
    config : nowcastlib.pipeline.structs.SmoothOptions

    Returns
    -------
    pandas.core.series.Series
        the smoothed series
    """
    data_series = input_series.copy()
    window_size = config.window_size
    shift_size = int((window_size + 1) / 2)
    units = config.units
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


def build_field_name(config: structs.ProcessingOptions, field_name: str):
    """
    Builds the appropriate field name depending on whether
    the user wishes to overwrite the current field or not

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.ProcessingOptions
    field_name : str
        the name of the current field we are acting on

    Returns
    -------
    str
        the resulting string
    """
    if config.overwrite:
        computed_field_name = field_name
    else:
        computed_field_name = "processed_{}".format(field_name)
    return computed_field_name


def process_field(
    input_series: pd.core.series.Series,
    options: structs.ProcessingOptions,
    preproc_flag: bool = True,
):
    """
    (Pre/Post)-processes a field

    Parameters
    ----------
    input_series : pandas.core.series.Series
        The data of the field to process
    options : nowcastlib.pipeline.structs.ProcessingOptions
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
