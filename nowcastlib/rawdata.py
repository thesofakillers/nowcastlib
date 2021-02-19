"""
Functions for processing multiple raw datasets.
"""
import pandas as pd
import numpy as np


def compute_trig_fields(input_df, fields_to_compute=[]):
    """For a time series pandas dataframe, computes the Cosine and Sine of the seconds of the
    day of each data point. Also optionally computes the Cosine and Sine equivalent
    additional fields if requested. The computed fields are added to the dataframe which
    is returned with the new fields included.

    Parameters
    ----------
    input_df : pandas.core.frame.DataFrame
        The dataframe to process
    fields_to_compute : list of string, default []
        List of the names of the additional fields for which we wish to compute Cosine
        and Sine equivalents

    Returns
    -------
    pandas.core.frame.DataFrame
        The original dataframe with the addition of the newly computed fields
    """
    # get at which second of the day each data point occured
    datetime = input_df.index.to_series()
    day_seconds = (datetime - datetime.dt.normalize()).dt.total_seconds()

    for func, func_name in zip([np.cos, np.sin], ["Cosine", "Sine"]):
        # first, compute trig _time_ equivalents
        trig_day_name = "{} Day".format(func_name)
        input_df[trig_day_name] = func((2 * np.pi * day_seconds.values) / 86400.0)
        # we can then tackle custom requested fields if any
        for field_name in fields_to_compute:
            new_field_name = "{} {}".format(func_name, field_name)
            field_data = input_df[field_name]
            if "deg" in field_name:
                field_data = np.radians(field_data)
            input_df[new_field_name] = func(field_data)

    return input_df


def bfill_nan(input_array):
    """Backward-fills NaNs in numpy array

    Parameters
    ----------
    input_array : numpy.ndarray

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    NaNs at the end of the array will remain untouched

    Examples
    --------
    >>> example_array = np.array([np.nan, np.nan, 4, 0, 0, np.nan, 7, 0])
    >>> bfill_nan(example_array)
    array([4., 4., 4., 0., 0., 7., 7., 0.])
    """
    mask = np.isnan(input_array)
    # get index array, but mark the NaNs with a very large number
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
    # backfill minima
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    # can now use this backfilled index array as a map on our original
    return input_array[idx]


def compute_large_gap_mask(data_array, max_gap):
    """Computes a mask (boolean NumPy array) outlining where there aren't large gaps
    in the data, as defined by `max_gap`

    Parameters
    ----------
    data_array : numpy.ndarray
        The array on which we want to compute the mask
    max_gap : int
        the maximum number of consecutive NaNs before defining the section to be a
        large gap

    Returns
    -------
    numpy.ndarray
        The mask, a boolean numpy.ndarray, where False indicates that the current
        index is part of a large gap.

    Notes
    -----
    This is an adaption of [this StackOverflow post](https://stackoverflow.com/a/54512613/9889508)
    """
    # where are the NaNs?
    isnan = np.isnan(data_array)
    # how many NaNs so far?
    cumsum = np.cumsum(isnan).astype("int")
    # for each non-nan indices, find the cum sum of nans since the last non-nan index
    diff = np.zeros_like(data_array)
    diff[~isnan] = np.diff(cumsum[~isnan], prepend=0)
    # set the nan indices to nan
    diff[isnan] = np.nan
    # backfill nan blocks by setting each nan index to the cum. sum of nans for that block
    diff = bfill_nan(diff)
    # handle NaN end
    final_nan_check = np.isnan(diff)
    if final_nan_check.any():
        if np.isnan(diff[-(max_gap + 1) :]).all():
            diff[final_nan_check] = max_gap + 1
        else:
            diff[final_nan_check] = 0
    # finally compute mask: False where large gaps - True elsewhere
    return (diff < max_gap) | ~isnan


def compute_dataframe_mask(input_df, max_gap, additional_cols_n=0, column_names=None):
    """Computes a mask (numpy.ndarray with dtype=boolean) shaped like `input_df` outlining
    where _all_ columns overlap (i.e. are not NaN), ignoring data gaps smaller than
    `max_gap`

    Parameters
    ----------
    input_df : pandas.core.frame.DataFrame
        The dataframe upon which to compute the mask
    max_gap : int
        The maximum number of consecutive NaNs that we ignore before considering this
        a gap
    additional_cols_n : int, default 0
        The number of additional columns that may be computed before applying the mask,
        and therefore need to be considered such that the mask shape matches the
        (future) dataframe shape
    column_names : list of string, default None
        The list of column names to use for checking overlap. If not specified, all
        columns will be checked.

    Returns
    -------
    numpy.ndarray
        2-dimensional (tiled) numpy array of the same shape as data_df, to be used as
        an argument to pandas.core.frame.DataFrame.where() or .mask()

    Notes
    -----
    It is highly recommended to specify `column_names` if your dataframe is made of
    multiple data sources each with multiple columns. In this case you only need to
    check one column from each data source
    """
    if column_names is None:
        column_names = input_df.columns
    # collect gap masks for each specified column
    gap_masks = []
    for col_name in column_names:
        mask = compute_large_gap_mask(input_df[col_name].values, max_gap)
        gap_masks.append(mask)
    # find the intersection of all these masks, to only keep overlapping points
    computed_mask = np.logical_and.reduce(gap_masks)
    # we need to reshape our final_mask such that it matches our data_df's shape.
    computed_mask = np.tile(
        computed_mask, (len(input_df.columns) + additional_cols_n, 1)
    ).transpose()
    return computed_mask


def make_chunks(input_df, min_length):
    """Given a sparse pandas DataFrame (i.e. data interrupted by NaNs), splits the
    DataFrame into the non-sparse chunks.

    Parameters
    ----------
    input_df : pandas.core.frame.DataFrame
        The sparse dataframe we wish to process
    min_length : int
        The minimum length a portion of data must be to be considered a chunk

    Returns
    -------
    list of pandas.core.frame.DataFrame
        A list, where each element is a DataFrame corresponding to a chunk of the
        original
    """
    sparse_ts = input_df.iloc[:, 0].astype(pd.SparseDtype("float"))
    # extract block length and locations
    block_locs = zip(
        sparse_ts.values.sp_index.to_block_index().blocs,
        sparse_ts.values.sp_index.to_block_index().blengths,
    )
    # use these to index our dataframe and populate our chunk list
    blocks = [
        input_df.iloc[start : (start + length - 1)]
        for (start, length) in block_locs
        if length >= min_length
    ]
    return blocks
