"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: README.md
"""
import logging
import pathlib
import pandas as pd
from .structs import config
from . import process
from . import sync
from . import features
from . import split
from . import standardize
from . import utils

logger = logging.getLogger(__name__)


def pipe_dataset(options: config.DataSet):
    """
    Runs all configured data-wrangling steps of the
    Nowcast Library Pipeline on a set of data described
    by the input DataSet instance
    """
    # preprocessing
    preprocessed_dfs = process.preprocess.preprocess_dataset(options)
    # syncing
    chunked_df, chunk_locs = sync.synchronize_dataset(options, preprocessed_dfs)
    # postprocessing
    postprocessed_df = process.postprocess.postprocess_dataset(options, chunked_df)
    # add generated fields if necessary
    if options.generated_fields is not None:
        postprocessed_df = features.generate_fields(options, postprocessed_df)
    # serializing postprocessing if necessary
    if options.postprocessing_output is not None:
        logger.info("Serializing postprocessing results...")
        utils.handle_serialization(postprocessed_df, options.postprocessing_output)
        logger.info("Serialization complete.")
    # splitting
    if options.split_options is not None:
        outer_split, inner_split = split.split_dataset(
            options, (postprocessed_df, chunk_locs)
        )
        # standardization (requires splits)
        outer_split, inner_split = standardize.standardize_dataset(
            options, outer_split, inner_split
        )
        # serialization
        split.serialize_splits(
            options.split_options.output_options, outer_split, inner_split
        )
