"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: PIPELINE.md
"""
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
    # outer train and test split
    if config.split_options is not None:
        train_data, test_data = split.train_test_split_sparse(
            chunked_df, config.split_options, chunk_locs
        )
        # postprocessing
        [proc_train_data], [proc_test_data] = process.postprocess.postprocess_dataset(
            config, [train_data], [test_data]
        )
        # inner train and validation split(s)
        if config.split_options.validation is not None:
            train_dfs, val_dfs = split.rep_holdout_split_sparse(
                train_data[0], config.split_options.validation
            )
            # postprocessing
            (
                proc_val_train_dfs,
                proc_val_test_dfs,
            ) = process.postprocess.postprocess_dataset(config, train_dfs, val_dfs)
    # serialize?
    # return
