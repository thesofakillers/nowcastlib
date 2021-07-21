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
    # get start and end indices of contiguous blocks of data
    if chunk_locations is None:
        block_locs = datasets.contiguous_locs_df(sparse_df)
    else:
        block_locs = chunk_locations.copy()
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


# def rep_holdout_split_sparse(sparse_df, block_locs=None):
#     """
#     Splits a sparse dataframe in k train and validation sets
#     for repeated holdout
#     """
