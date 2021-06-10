"""
Functions for pre-processing of data.
"""
from typing import Union
import pandas as pd
from nowcastlib.pipeline import structs


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
    window_size = config.window_size
    shift_size = int((window_size + 1) / 2)
    units = config.units
    window: Union[str, int]
    if units is not None:
        window = str(window_size) + units
    else:
        window = window_size
    return (
        input_series.rolling(
            window=window,
            closed="both",
        )
        .mean()
        .shift(-shift_size, freq=units)
    )


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
            & (input_series <= input_series.quantile(config.higher))
        ]
    else:
        return input_series[
            (config.lower <= input_series) & (input_series <= config.higher)
        ]


def preprocess_dataset(config: structs.DataSet):
    """
    Runs preprocessing on a given set of data sources given options outlined
    in the input DataSet instance.

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.DataSet

    Returns
    -------
    list[pandas.core.frame.DataFrame]
        list containing each of the resulting processed dataframes
    """
    processed_dfs = []
    for ds_config in config.data_sources:
        processed_dfs.append(preprocess_datasource(ds_config))
    return processed_dfs


def preprocess_datasource(config: structs.DataSource):
    """
    Runs preprocessing on a given data source given options outlined
    in the input DataSource instance.

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.DataSource

    Returns
    -------
    pandas.core.frame.DataFrame
        the resulting processed dataframe
    """
    index_field = next(field for field in config.fields if field.is_date)
    data_df = pd.read_csv(
        config.path,
        usecols=[field.field_name for field in config.fields],
        index_col=index_field.field_name,
        parse_dates=False,
        comment=config.comment_format,
    )
    data_df.index = pd.to_datetime(data_df.index, format=index_field.date_format)
    data_df = data_df[~data_df.index.duplicated(keep="last")]
    data_df.sort_index(inplace=True)
    for field in config.fields:
        if field.outlier_options is not None:
            data_df[field.field_name] = drop_outliers(
                data_df[field.field_name], field.outlier_options
            )
        if field.periodic_options is not None:
            data_df[field.field_name] = handle_periodic(
                data_df[field.field_name], field.periodic_options
            )
        if field.conversion is not None:
            data_df[field.field_name] = structs.CONV_MAP[field.conversion](
                data_df[field.field_name]
            )
        if field.smooth_options is not None:
            data_df[field.field_name] = handle_smoothing(
                data_df[field.field_name], field.smooth_options
            )
    # drop all rows with NaNs, as final step
    data_df = data_df.dropna()
    return data_df
