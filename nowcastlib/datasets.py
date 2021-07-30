"""
Functions for syncing and chunking multiple datasets.
"""
from typing import Union, Optional
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
    # finally compute mask: False where large gaps, True elsewhere
    return (diff < max_gap) | ~isnan


def contiguous_locs_array(input_array):
    """
    Finds the start and end indices of contiguous `True`
    regions in the input array.

    Parameters
    ----------
    input_array : numpy.ndarray
        1D input boolean numpy array

    Returns
    -------
    numpy.ndarray
        2D numpy array containing the start and end
        indices of the contiguous regions of `True`
        in the input array. Shape is (-1, 2).

    Notes
    -----
    Credit: https://stackoverflow.com/a/4495197/9889508
    """
    # Find the indices of changes
    (idx,) = np.diff(input_array).nonzero()
    # need to shift indices to the right since we are interested in _after_ changes
    idx += 1
    if input_array[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
    if input_array[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, input_array.size]  # Edit
    # Reshape the result into two colums
    idx.shape = (-1, 2)
    return idx


def contiguous_locs_df(input_df):
    """
    Produces a 2D numpy array of start and end indices
    of the contiguous chunks of a sparse pandas dataframe

    Returns
    -------
    numpy.ndarray
        2D numpy array containing the start and end
        indices of the contiguous regions of data
        (i.e. where no column is Nan). Shape is (-1, 2).
    """
    sparse_ts = input_df.iloc[:, 0].astype(pd.SparseDtype("float"))
    # extract block length and locations
    starts = sparse_ts.values.sp_index.to_block_index().blocs
    lengths = sparse_ts.values.sp_index.to_block_index().blengths
    ends = starts + lengths
    block_locs = np.array((starts, ends)).T
    return block_locs


def contiguous_locs(
    data: Union[pd.core.frame.DataFrame, np.ndarray],
    chunk_locations: Optional[np.ndarray] = None,
):
    """
    Wrapper function for finding start and end indices
    of contiguous data either in an array or a dataframe
    """
    if chunk_locations is None:
        if isinstance(data, np.ndarray):
            block_locs = contiguous_locs_array(data)
        elif isinstance(data, pd.core.frame.DataFrame):
            block_locs = contiguous_locs_df(data)
    else:
        # if they are already provided, just use them obvs
        block_locs = chunk_locations.copy()
    return block_locs


def fill_start_end(start, end):
    """
    Given NumPy arrays of start and end indices
    returns an array containing all indices between
    each of the start and end indices, including the
    start indices.

    Examples
    --------
    >>> starts = np.array([1,7,20])
    >>> ends = np.array([3,10,25])
    >>> fill_start_end(starts, ends)
    array([ 1,  2,  7,  8,  9, 20, 21, 22, 23, 24])

    Notes
    -----
    Credit: https://stackoverflow.com/a/4708737/9889508
    """
    cml_lens = (end - start).cumsum()
    # initialize indices; resulting array will be length of cumulative sum of lengths
    idx = np.ones(cml_lens[-1], dtype=int)
    idx[0] = start[0]
    # computing 'break' indices
    idx[cml_lens[:-1]] += start[1:] - end[:-1]
    # finally take cumulative sum to compute indices in between
    idx = idx.cumsum()
    return idx


def filter_contiguous_regions(bool_array, true_locs, min_length):
    """
    Given a boolean array, removes contiguous `True`
    regions too short (by setting them to `False`)

    Parameters
    ----------
    bool_array : numpy.ndarray
        1D NumPy array of booleans
    true_locs : numpy.ndarray
        (-1, 2) NumPy array containing the start and end
        indices of each contiguous region of `True`s
    min_length : int
        The minimum length necessary for a contiguous
        region of `True`s to be considered one

    Returns
    -------
    numpy.ndarray
        resulting 1D filtered boolean array
    numpy.ndarray
        (-1, 2) numpy array containing the filtered
        contiguous region start and end indices
    """
    lengths = true_locs[:, 1] - true_locs[:, 0]
    # mask for which starts and stops to keep
    mask = lengths >= min_length
    remove_starts, remove_ends = true_locs[~mask].T
    # get all indices between the start and stops to remove, so to set these as False
    additional_false_idxs = fill_start_end(remove_starts, remove_ends)
    # "filter" the boolean array by setting the computed indices to False
    filtered_array = bool_array.copy()
    filtered_array[additional_false_idxs] = False
    # use previously computed mask to filter the location array too
    filtered_locs = true_locs[mask]
    return filtered_array, filtered_locs


def compute_dataframe_mask(
    input_df, max_gap, min_length, additional_cols_n=0, column_names=None
):
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
    min_length : int
        The minimum length for a contiguous chunk of data to be considered one
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
    numpy.ndarray
        (-1, 2) numpy array containing the filtered
        contiguous region start and end indices

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
    # where are the contiguous chunks in the mask?
    chunk_locations = contiguous_locs_array(computed_mask)
    # filter mask and chunk_locs: get rid of contiguous chunks that are too short
    computed_mask, chunk_locations = filter_contiguous_regions(
        computed_mask, chunk_locations, min_length
    )
    # we need to reshape our final_mask such that it matches our data_df's shape.
    computed_mask = np.tile(
        computed_mask, (len(input_df.columns) + additional_cols_n, 1)
    ).transpose()
    # include chunk_locations in return in case we need to re-use them
    return computed_mask, chunk_locations


def make_chunks(input_df, chunk_locations=None):
    """
    Given a sparse pandas DataFrame (i.e. data interrupted by NaNs),
    splits the DataFrame into the non-sparse chunks.

    Parameters
    ----------
    input_df : pandas.core.frame.DataFrame
        The sparse dataframe we wish to process
    chunk_locations : np.ndarray, default None
        2D numpy array with the pre-computed chunk start
        and end indices. Shape is (-1, 2). Optional.

    Returns
    -------
    list of pandas.core.frame.DataFrame
        A list, where each element is a DataFrame corresponding to a chunk of the
        original
    """
    # need to compute chunk_locations if not provided
    if chunk_locations is None:
        block_locs = contiguous_locs_df(input_df)
    else:
        block_locs = chunk_locations.copy()
    # use these to index our dataframe and populate our chunk list
    blocks = [input_df.iloc[start:end] for (start, end) in block_locs]
    return blocks


def serialize_as_chunks(chunked_df, path):
    """
    Serializes a sparse dataframe as contiguous
    chunks into an hdf5 file

    Parameters
    ----------
    chunked_df : pandas.core.frame.DataFrame
        the dataframe we wish to serialize as chunks
    path : pathlib.PosixPath
        the path to which we wish to save our hdf5 file,
        including the file name and extension
    """
    chunks = make_chunks(chunked_df)
    hdfs = pd.HDFStore(str(path))
    for i, chunk in enumerate(chunks):
        chunk.to_hdf(hdfs, "chunk_{:d}".format(i), format="table")
