"""
Functions for splitting data
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from nowcastlib.pipeline import structs
import nowcastlib.datasets as datasets


def train_test_split_sparse(
    sparse_df: pd.core.frame.DataFrame,
    config: structs.EvaluationOptions,
    chunk_locations: Optional[np.ndarray] = None,
) -> Tuple:
    """
    Splits a sparse dataframe in train and test sets

    Parameters
    ---------
    sparse_df : pandas.core.frame.DataFrame
        The DataFrame we wish to split into train and test sets
    config : structs.EvaluationOptions
        config options for determining where to split our datasets
    block_locs : numpy.ndarray, default None

    Returns
    -------
    train_data : tuple
        Tuple of length 2 containing the resulting training data.
        The first element is the dataframe, the second element
        is the updated accompanying `block_locs`.
    test_data : tuple
        Tuple of length 2 containing the resulting testing data.
        The first element is the dataframe, the second element
        is the updated accompanying `block_locs`.
    """
    # pylint: disable=too-many-locals
    block_locs = datasets.contiguous_locs(sparse_df, chunk_locations)
    starts, ends = block_locs.T
    # find the desired index
    configured_split = config.train_split
    if isinstance(configured_split, int):
        desired_index = configured_split
    elif isinstance(configured_split, str):
        desired_index = len(sparse_df.truncate(after=configured_split)) - 1
    elif isinstance(configured_split, float):
        idx_to_count = datasets.fill_start_end(starts, ends)
        total_valid = len(idx_to_count)
        desired_index = idx_to_count[int(configured_split * total_valid)]
    # find the closest relevant nan edges to the desired index
    closest_edge_idx = np.abs(ends - desired_index).argmin()
    train_end_index = ends[closest_edge_idx]
    test_start_index = starts[closest_edge_idx + 1]
    # finally split
    train_df = sparse_df.iloc[:train_end_index]
    test_df = sparse_df.iloc[test_start_index:]
    # update block_locs
    train_block_locs = block_locs[: (closest_edge_idx + 1)]
    test_block_locs = block_locs[(closest_edge_idx + 1) :] - test_start_index

    return (train_df, train_block_locs), (test_df, test_block_locs)


def rep_holdout_split_sparse(
    sparse_df: pd.core.frame.DataFrame,
    config: structs.ValidationOptions,
    chunk_locations: Optional[np.ndarray] = None,
):
    """
    Splits a sparse dataframe in k train and validation sets
    for repeated holdout
    """
    block_locs = datasets.contiguous_locs(sparse_df, chunk_locations)
    starts, ends = block_locs.T
    # not counting NaNs when taking percentages
    idx_to_count = datasets.fill_start_end(starts, ends)
    total_valid = len(idx_to_count)
    # determine the implicitly requested start and end edge indices
    train_samples = int(config.train_extent * total_valid)
    val_samples = int((config.val_extent) * total_valid)
    req_val_max_start = idx_to_count[-val_samples]
    req_train_min_end = idx_to_count[train_samples]
    # get the indices where the closest edges occur in the starts,ends arrays
    max_val_start_idx = np.abs(starts - req_val_max_start).argmin()
    min_train_end_idx = np.abs(ends - req_train_min_end).argmin()
    # min end of training: window start; max start of val: window end
    # get previous index of max start since we are concerned with ends
    window_train_ends = ends[min_train_end_idx:max_val_start_idx]
    window_val_starts = starts[min_train_end_idx + 1 : max_val_start_idx + 1]
    # randomly sample n times without replacement to get split breakpoints
    split_idxs = np.random.choice(window_train_ends.size, config.iterations, False)
    split_train_ends = window_train_ends[split_idxs]
    split_val_starts = window_val_starts[split_idxs]
    # find where training should start based on splitting index
    ideal_train_starts = split_train_ends - train_samples
    # fix so it coincides to a chunk start
    train_starts = starts[
        [np.abs(starts - ideal_start).argmin() for ideal_start in ideal_train_starts]
    ]
    # find where validation should end based on splitting index
    ideal_val_ends = split_val_starts + val_samples
    # fix so it coincides to a chunk end
    val_ends = ends[[np.abs(ends - ideal_end).argmin() for ideal_end in ideal_val_ends]]
    # finally build our dataframes
    train_dfs = [
        sparse_df.iloc[start:end] for start, end in zip(train_starts, split_train_ends)
    ]
    test_dfs = [
        sparse_df.iloc[start:end] for start, end in zip(split_val_starts, val_ends)
    ]
    return train_dfs, test_dfs
