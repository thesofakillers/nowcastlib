"""
Functions for processing multiple raw datasets.
"""
import pandas as pd
import numpy as np


def compute_trig_fields(input_df, fields_to_compute=[]):
    """
    For a time series pandas dataframe, computes the Cosine and Sine of the seconds of the
    day of each data point. Also optionally computes the Cosine and Sine equivalent
    additional fields if requested. The computed fields are added to the dataframe which
    is returned with the new fields included.

    :param pandas.core.frame.DataFrane input_df: The dataframe to process
    :param list[string] fields_to_compute: default []. List of the names of the additional
        fields for which we wish to compute Cosine and Sine equivalents
    :return: The original dataframe with the addition of the newly computed fields
    :rtype: pandas.core.frame.DataFrame
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
    """
    Backward-fills NaNs in array
    Consider the array        [NaN, NaN, 4, 0, 0, NaN, 7, 0]
    The function will out put [4, 4, 4, 0, 0, 7, 7, 0]

    Note, NaNs at the end of the array will remain untouched
    """
    mask = np.isnan(input_array)
    # get index array, but mark the NaNs with a very large number
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
    # backfill minima
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    # can now use this backfilled index array as a map on our original
    return input_array[idx]


# https://stackoverflow.com/a/54512613/9889508
def compute_large_gap_mask(data_array, max_gap):
    """
    Computes a mask (boolean NumPy array) outlining where there aren't large gaps in the
    data, as defined by `max_gap`

    :param numpy.ndarray data_array: The array on which we want to compute the mask
    :param int max_gap: the maximum number of consecutive NaNs before defining the
                        section to be a large gap
    :return: The mask, a boolean numpy.ndarray, where False indicates that the current
             index is part of a large gap.
    :rtype: numpy.ndarray
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
    """
    Computes a mask (numpy.ndarray with dtype=boolean) shaped like `data_df` outlining
    where _all_ columns overlap (i.e. are not NaN), ignoring data gaps smaller than
    `max_gap`

    :param pandas.core.frame.DataFrame input_df: The dataframe upon which to compute the
        mask
    :param int max_gap: The maximum number of consecutive NaNs that we ignore before
        considering this a gap
    :param int additional_cols_n: default 0. The number of additional columns that may
        be computed before applying the mask, and therefore need to be considered such
        that the mask shape matches the (future) dataframe shape
    :param list[string] column_names: default None. The list of column names to use for
        checking overlap. If not specified, all columns will be checked. It is highly
        recommended to specify `column_names` if your dataframe is made of multiple data
        sources each with multiple columns. In this case you only need to check one column
        from each data source

    :return: 2-dimensional (tiled) numpy array of the same shape as data_df, to be used as
        an argument to pandas.core.frame.DataFrame.where() or .mask()
    :rtype: numpy.ndarray
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
    """
    Given a sparse pandas DataFrame (i.e. data interrupted by NaNs), splits the
    DataFrame into the non-sparse chunks.

    :param pandas.core.frame.DataFrame input_df: the sparse dataframe we wish to process
    :param int min_length: the minimum length a portion of data must be to be
        considered a chunk
    :return: A list, where each element is a DataFrame corresponding to a chunk of the
        original
    :rtype: list[pandas.core.frame.DataFrame]
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
