"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: PIPELINE.md
"""
from . import structs
from . import preprocess
from . import sync
from . import split
from . import postprocess


def pipe_dataset(config: structs.DataSet):
    """
    Runs all configured steps of the Nowcast Library Pipeline
    on a set of data described by the input DataSet instance
    """
    # preprocessing
    preprocessed_dfs = preprocess.preprocess_dataset(config)
    # syncing
    chunked_df, chunk_locs = sync.synchronize_dataset(config, preprocessed_dfs)
    # splitting
    # outer train and test split
    if config.eval_options is not None:
        train_data, test_data = split.train_test_split_sparse(
            chunked_df, config.eval_options, chunk_locs
        )
        # inner train and validation split(s)
        if config.eval_options.validation is not None:
            train_dfs, val_dfs = split.rep_holdout_split_sparse(
                train_data[0], config.eval_options.validation
            )
    # # postprocessing
    # proc_val_train, proc_val_test = postprocess.postprocess_dataset(
    #     config,
    #     val_train[0],
    #     val_test_data[0],
    # )
