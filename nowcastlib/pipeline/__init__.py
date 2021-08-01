"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: PIPELINE.md
"""
import pathlib
import pandas as pd
from nowcastlib.datasets import serialize_as_chunks
from . import structs
from . import process
from . import sync
from . import split


def pipe_dataset(config: structs.DataSet):
    """
    Runs all configured steps of the Nowcast Library Pipeline
    on a set of data described by the input DataSet instance
    """
    # preprocessing
    preprocessed_dfs = process.preprocess.preprocess_dataset(config)
    # syncing
    chunked_df, chunk_locs = sync.synchronize_dataset(config, preprocessed_dfs)
    # splitting
    if config.split_options is not None:
        # outer train and test split
        (train_data, _), (test_data, _) = split.train_test_split_sparse(
            chunked_df, config.split_options, chunk_locs
        )
        # postprocessing
        [proc_train_data], [proc_test_data] = process.postprocess.postprocess_splits(
            config, [train_data], [test_data]
        )
        # inner train and validation split(s)
        train_dfs, val_dfs = split.rep_holdout_split_sparse(
            train_data, config.split_options.validation
        )
        # postprocessing
        (
            proc_val_train_dfs,
            proc_val_test_dfs,
        ) = process.postprocess.postprocess_splits(config, train_dfs, val_dfs)
        # serialize
        if config.split_options.output_options is not None:
            serialize_splits(
                config.split_options.output_options,
                proc_train_data,
                proc_test_data,
                proc_val_train_dfs,
                proc_val_test_dfs,
            )


def serialize_splits(config, train_data, test_data, val_train_dfs, val_test_dfs):
    """
    Creates directory structure and serializes
    the dataframes as chunks to hdf5 files
    """
    parent_dir = pathlib.Path(config.parent_path)
    parent_dir.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    main_split = parent_dir / "main_split"
    main_split.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    serialize_as_chunks(train_data, main_split / "train_data.hdf5")
    serialize_as_chunks(test_data, main_split / "test_data.hdf5")
    cv_split = parent_dir / "cv_split"
    cv_split.mkdir(parents=config.create_parents, exist_ok=config.overwrite)
    for i, (train_df, test_df) in enumerate(val_train_dfs, val_test_dfs):
        serialize_as_chunks(train_df, cv_split / "train_data_{}.hdf5".format(i))
        serialize_as_chunks(test_df, cv_split / "val_data_{}.hdf5".format(i))
