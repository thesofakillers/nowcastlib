"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: README.md
"""
import pathlib
import pandas as pd
from . import structs
from . import process
from . import sync
from . import split


def pipe_dataset(config: structs.DataSet):
    """
    Runs all configured data-wrangling steps of the
    Nowcast Library Pipeline on a set of data described
    by the input DataSet instance
    """
    # preprocessing
    preprocessed_dfs = process.preprocess.preprocess_dataset(config)
    # syncing
    chunked_df, chunk_locs = sync.synchronize_dataset(config, preprocessed_dfs)
    # splitting and postprocessing
    if config.split_options is not None:
        process.postprocess.postprocess_dataset(config, (chunked_df, chunk_locs))
