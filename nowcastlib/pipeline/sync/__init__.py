"""
Functions for synchronizing data
"""
import sys
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nowcastlib.pipeline import structs
from nowcastlib.pipeline.process import preprocess
from nowcastlib.pipeline import utils
from nowcastlib import datasets

plt.ion()
logger = logging.getLogger(__name__)


def handle_diag_plots(
    config: structs.SyncOptions, dataframes: List[pd.core.frame.DataFrame]
):
    """
    Produces plots that can aid the user in noticing mistakes
    in their configuration before proceeding with the
    synchronization process

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.SyncOptions
    dataframes : list[pandas.core.frame.DataFrame]
        The set of dataframes one wishes to synchronize.

    Returns
    -------
    bool
        whether the user wishes to continue with
        running the pipeline or not
    """
    n_samples = 10000
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    for i, data_df in enumerate(dataframes):
        sample_spacing = data_df.index.to_series().diff()
        data = sample_spacing.sample(n_samples).astype("timedelta64[s]")
        weights = np.ones_like(data) / n_samples
        ax1.hist(
            data,
            bins=np.arange(150, step=1),
            weights=weights,
            label="DataSource {} sample spacing".format(i),
            histtype="step",
            linewidth=1.5,
        )
    ax1.axvline(config.sample_spacing, color="black", label="Selected Sample Spacing")
    ax1.set_xlabel("Sample Spacing [s]")
    ax1.set_ylabel("Prevalence")
    ax1.set_title(
        "Sample spacing of {} random samples across the input Data Sources".format(
            n_samples
        )
    )
    ax1.legend()
    fig.set_tight_layout(True)
    logger.info("Press any button to exit. Use mouse to zoom and resize")
    while True:
        plt.draw()
        if plt.waitforbuttonpress():
            break
    return utils.yes_or_no("Are you satisfied with the target sample rate?")


def handle_chunking(
    data_df: pd.core.frame.DataFrame,
    config: structs.ChunkOptions,
    column_names: Optional[List[str]] = None,
):
    """
    Finds overlapping chunks of data across dataframe columns,
    taking into account gap size preferences.

    Parameters
    ----------
    data_df: pandas.core.frame.DataFrame
        the sparse dataframe to perform chunking on
    config : nowcastlib.pipeline.structs.ChunkOptions
        chunking configuration options
    column_names : list[str], default None
        the names of the columns to check for overlaps.
        If `None`, all columns will be used.

    Returns
    -------
    pandas.core.frame.DataFrame
        The resulting dataframe, rows where not
        all columns contained data now contain NaN
        across all columns, depending on gap preferences.
    numpy.ndarray
        2D numpy array containing the start and end
        integer indices of the contiguous chunks of data
        in the input dataframe. Shape is (-1, 2).
    """
    # find overlapping data, ignoring small gaps
    sample_spacing_secs = data_df.index.freq.delta.seconds
    max_spacing_steps = np.floor((config.max_gap_size / sample_spacing_secs)).astype(
        int
    )
    min_chunk_length = int(config.min_chunk_size / sample_spacing_secs)
    final_mask, chunk_locs = datasets.compute_dataframe_mask(
        data_df, max_spacing_steps, min_chunk_length, 0, column_names
    )
    # imputing gaps, restoring large gaps
    interpolated_df = data_df.interpolate("linear", limit_direction="both")
    chunked_df = interpolated_df.where(final_mask)  # type: ignore

    return chunked_df, chunk_locs


def synchronize_dataset(
    config: structs.DataSet, dataset: Optional[List[pd.core.frame.DataFrame]] = None
):
    """
    Synchronizes a set of data sources given options outlined
    in the input DataSet config instance. Optionally writes the
    results to disk.

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.DataSet
    dataset : list[pandas.core.frame.DataFrame], default None
        The set of dataframes one wishes to synchronize.
        If `None`, the preprocessing output produced by the
        config options will be synchronized.

    Returns
    -------
    pandas.core.frame.DataFrame
        A single dataframe containing the
        synchronized data. The dataframe is sparse,
        with individual rows comprising solely of NaNs.
    numpy.ndarray
        2D numpy array containing the start and end
        integer indices of the contiguous chunks of data
        in the input dataframe. Shape is (-1, 2).
    """
    sync_config = config.sync_options
    assert (
        sync_config is not None
    ), "`config.sync_options` must be defined to perform synchronization"
    # avoid preprocessing if datasets are passed directly
    if dataset is None:
        data_dfs = preprocess.preprocess_dataset(config)
    else:
        data_dfs = dataset
    logger.info("Synchronizing dataset...")

    if sync_config.diagnostic_plots is not False:
        continue_processing = handle_diag_plots(sync_config, data_dfs)
        if continue_processing is False:
            logger.info(
                "Closing program prematurely to allow for configuration changes"
            )
            sys.exit()

    total_dfs = len(data_dfs)
    resampled_dfs = []
    for i, data_df in enumerate(data_dfs):
        logger.debug("Resampling DataSource %d of %d...", i + 1, total_dfs)
        data_df.index.name = None
        offset_str = "{}S".format(sync_config.sample_spacing)
        resampled_dfs.append(
            data_df.resample(
                offset_str,
                origin=data_df.index[0].floor(offset_str),
            ).mean()
        )
    logger.debug("Finding overlapping range and joining into single dataframe...")
    synced_df = pd.concat(resampled_dfs, axis=1, join="inner")
    logger.debug("Splitting data into contiguous chunks...")
    chunked_df, chunk_locs = handle_chunking(
        synced_df, sync_config.chunk_options, [df.columns[0] for df in data_dfs]
    )
    if sync_config.data_output is not None:
        logger.debug("Serializing chunked dataframe...")
        utils.handle_serialization(chunked_df, sync_config.data_output)
    if sync_config.chunks_output is not None:
        logger.debug("Serializing chunk locations...")
        utils.handle_serialization(chunk_locs, sync_config.chunks_output)
    return chunked_df, chunk_locs
