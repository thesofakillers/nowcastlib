"""
Functions for splitting data
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from nowcastlib.pipeline import structs
import nowcastlib.datasets as datasets


def train_test_split_sparse(
    sparse_df: pd.core.frame.DataFrame,
    config: structs.SplitOptions,
    chunk_locations: Optional[np.ndarray] = None,
) -> Tuple[
    Tuple[pd.core.frame.DataFrame, np.ndarray],
    Tuple[pd.core.frame.DataFrame, np.ndarray],
]:
    """
    Splits a sparse dataframe in train and test sets

    Parameters
    ---------
    sparse_df : pandas.core.frame.DataFrame
        The DataFrame we wish to split into train and test sets
    config : structs.SplitOptions
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
    for repeated holdout. Split is approximate due to sparse
    'chunked' nature of the input dataframe.
    """
    n_samples = len(sparse_df)
    tscv = TimeSeriesSplit(
        n_splits=config.iterations,
        max_train_size=int(config.train_extent * n_samples),
        test_size=int(config.val_extent * n_samples),
    )
    train_dfs = []
    val_dfs = []
    for train_idxs, val_idxs in tscv.split(sparse_df):
        train_dfs.append(sparse_df.iloc[train_idxs])
        val_dfs.append(sparse_df.iloc[val_idxs])
    return train_dfs, val_dfs
    # TODO update and return chunk_locations
