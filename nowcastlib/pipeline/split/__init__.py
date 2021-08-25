"""
Functions for splitting data
"""
import logging
import pathlib
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from nowcastlib.pipeline import structs
from nowcastlib.pipeline import sync
from nowcastlib import datasets


logger = logging.getLogger(__name__)


def serialize_splits(
    config,
    outer_split_data: structs.TrainTestSplit,
    inner_split_data: structs.IteratedSplit,
):
    """
    Creates directory structure and serializes
    the dataframes as chunks to hdf5 files
    """
    logger.info("Serializing splits...")
    parent_dir = pathlib.Path(config.parent_path)
    parent_dir.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    # outer split
    main_split = parent_dir / "main_split"
    main_split.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    # destructuring
    train_data, test_data = outer_split_data
    datasets.serialize_as_chunks(main_split / "train_data.hdf5", *train_data)
    datasets.serialize_as_chunks(main_split / "test_data.hdf5", *test_data)
    # inner splits
    cv_split = parent_dir / "cv_split"
    cv_split.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    # destructuring
    val_train_data, val_test_data = inner_split_data
    for i, (train_tuple, test_tuple) in enumerate(zip(val_train_data, val_test_data)):
        # TODO pass chunk locations so we can do *train_tuple instead of train_tuple[0]
        datasets.serialize_as_chunks(
            cv_split / "train_data_{}.hdf5".format(i + 1), train_tuple[0]
        )
        datasets.serialize_as_chunks(
            cv_split / "val_data_{}.hdf5".format(i + 1), test_tuple[0]
        )
    logger.info("Split serialization complete.")


def train_test_split_sparse(
    sparse_df: pd.core.frame.DataFrame,
    config: structs.config.SplitOptions,
    chunk_locations: Optional[np.ndarray] = None,
) -> structs.TrainTestSplit:
    """
    Splits a sparse dataframe in train and test sets

    Parameters
    ---------
    sparse_df : pandas.core.frame.DataFrame
        The DataFrame we wish to split into train and test sets
    config : structs.config.SplitOptions
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
    config: structs.config.ValidationOptions,
    sparse_df: pd.core.frame.DataFrame,
    chunk_locations: Optional[np.ndarray] = None,
) -> structs.IteratedSplit:
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
    train_data: List[structs.SparseData] = []
    val_data: List[structs.SparseData] = []
    for train_idxs, val_idxs in tscv.split(sparse_df):
        train_data.append((sparse_df.iloc[train_idxs], np.empty((1, 2))))
        val_data.append((sparse_df.iloc[val_idxs], np.empty((1, 2))))
    return train_data, val_data
    # TODO update and return chunk_locations, instead of np.empty()


def split_dataset(
    config: structs.config.DataSet,
    sparse_data: Optional[structs.SparseData] = None,
) -> structs.SplitDataSet:
    """
    Splits dataset into train and test sets. Then splits train set into
    train and validation sets for cross validation.

    Parameters
    ----------
    config: structs.config.DataSet
        data set configuration options instance
    sparse_data: tuple of [pandas.core.frame.DataFrame, numpy.ndarray], default `None`
        The sparse dataframe to split and the array of start and end
        indices of the contiguous chunks of data in the dataframe.
        \nIf `None`, DataSet synchronization will be performed to obtain
        the sparse dataframe.

    Returns
    -------
    outer_split : structs.TrainTestSplit
        Tuple of length 2 containing the outer split data.
        \nThe first element is a tuple with the training data (df and chunk locations),
        \nthe second element is a tuple with the testing data (df and chunk locations)
    inner_split : structs.IteratedSplit
        Tuple of length 2 containing the inner split data.
        \nThe first element is a list of tuples with the training data
        (df and chunk locations),
        \nthe second element is a list of tuples with the validation data
        (df and chunk locations)
        \nThe lists will have as many elements as the number of CV folds.

    """
    assert (
        config.split_options is not None
    ), "`config.split_options` must be defined to perform splitting"
    # will perform dataset synchronization if sparse_data is not provided
    if sparse_data is None:
        chunked_df, chunk_locs = sync.synchronize_dataset(config)
    else:
        chunked_df = sparse_data[0].copy()
        chunk_locs = sparse_data[1].copy()
    logger.info("Splitting dataset...")
    logger.debug("Performing outer split...")
    # outer train and test split
    outer_train_data, outer_test_data = train_test_split_sparse(
        chunked_df, config.split_options, chunk_locs
    )
    logger.debug("Performing inner split...")
    # inner train and validation split(s)
    inner_train_data, inner_val_data = rep_holdout_split_sparse(
        config.split_options.validation, *outer_train_data
    )
    logger.info("Dataset splitting complete...")
    return (outer_train_data, outer_test_data), (inner_train_data, inner_val_data)
