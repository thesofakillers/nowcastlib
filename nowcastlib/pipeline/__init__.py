"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: README.md
"""
import pathlib
import pandas as pd
from . import structs
from . import process
from . import sync
from . import features
from . import split
from . import standardize


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
    # postprocessing
    postprocessed_df = process.postprocess.postprocess_dataset(config, chunked_df)
    # add generated fields if necessary
    if config.generated_fields is not None:
        postprocessed_df = features.generate_fields(config, postprocessed_df)
    # splitting
    if config.split_options is not None:
        outer_split, inner_split = split.split_dataset(
            config, (postprocessed_df, chunk_locs)
        )
        # standardization (requires splits)
        outer_split, inner_split = standardize.standardize_dataset(
            config, outer_split, inner_split
        )
        # serialization
        split.serialize_splits(
            config.split_options.output_options, outer_split, inner_split
        )
