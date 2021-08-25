"""
Module for standardization functionality
"""
import sys
from typing import List, Tuple
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklearn_pproc
from nowcastlib.pipeline import structs
from nowcastlib.pipeline import utils

plt.ion()

logger = logging.getLogger(__name__)


def handle_diag_plots(
    input_series: pd.core.series.Series,
    configured_method: structs.config.StandardizationMethod,
):
    """
    Plots different rescalings of the input series,
    asking the user for confirmation
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
    pwr_trnsformer = sklearn_pproc.PowerTransformer()
    robust_trnsfrmr = sklearn_pproc.RobustScaler()
    ax1.hist(input_series, bins=200, color="black")
    ax1.set_title("Original Series")
    ax2.hist(
        pwr_trnsformer.fit_transform(input_series.to_numpy().reshape(-1, 1)),
        bins=200,
        color="darkblue"
        if configured_method == structs.config.StandardizationMethod.POWER
        else "darkgrey",
    )
    ax2.set_title("Power Transform")
    ax3.hist(
        robust_trnsfrmr.fit_transform(input_series.to_numpy().reshape(-1, 1)),
        bins=200,
        color="darkblue"
        if configured_method == structs.config.StandardizationMethod.ROBUST
        else "darkgrey",
    )
    ax3.set_title("Robust Scaling")
    ax4.hist(
        np.log(1 + (input_series - input_series.min())),
        bins=200,
        color="darkblue"
        if configured_method == structs.config.StandardizationMethod.LOGNORM
        else "darkgrey",
    )
    ax4.set_title("log(1 + (input_series - input_series.min()))")
    fig.suptitle("Configured method shown in Blue")
    fig.set_tight_layout(True)
    logger.info("Press any button to exit. Use mouse to zoom and resize")
    while True:
        plt.draw()
        if plt.waitforbuttonpress():
            break
    return utils.yes_or_no(
        "Are you satisfied with the selected Standardization Method?"
    )


def standardize_dataset(
    options: structs.config.DataSet,
    outer_split: structs.TrainTestSplit,
    inner_split: structs.IteratedSplit,
) -> structs.SplitDataSet:
    """
    Standardizes a DataSet, accounting for train/test nuances
    """
    # destructure outer split, so we have access to locs
    (train_df, train_locs), (test_df, test_locs) = outer_split
    # and standardize
    [proc_train_data], [proc_test_data] = standardize_splits(
        options, [train_df], [test_df]
    )
    # destructure inner split, so we have access to locs
    inner_train_data, inner_val_data = inner_split
    train_dfs, inner_train_locs = [[*x] for x in zip(*inner_train_data)]
    val_dfs, inner_val_locs = [[*x] for x in zip(*inner_val_data)]
    (
        proc_val_train_dfs,
        proc_val_test_dfs,
    ) = standardize_splits(options, train_dfs, val_dfs)
    # return in correct format
    return (
        ((proc_train_data, train_locs), (proc_test_data, test_locs)),
        (
            list(zip(proc_val_train_dfs, inner_train_locs)),
            list(zip(proc_val_test_dfs, inner_val_locs)),
        ),
    )


def standardize_splits(
    options: structs.config.DataSet,
    train_dfs: List[pd.core.frame.DataFrame],
    test_dfs: List[pd.core.frame.DataFrame],
) -> Tuple[List[pd.core.frame.DataFrame], List[pd.core.frame.DataFrame]]:
    """
    Standardizes a set of train-test splits given options outlined
    in the input DataSet config instance.

    Returns
    -------
    std_train_dfs : List[pandas.core.frame.DataFrame]
        List of the newly standardized train dataframes
    std_test_dfs : List[pandas.core.frame.DataFrame]
        List of the newly standardized test dataframes
    """
    logger.info("Standardizing splits...")
    # instantiate standardized dfs
    std_train_dfs = train_dfs.copy()
    std_test_dfs = test_dfs.copy()
    # gather which fields to process into single list
    raw_fields: List[structs.config.RawField] = [
        field for source in options.data_sources for field in source.fields
    ]
    # rename overwrite-protected fields so to avoid acting on the original field
    fields_to_process = [utils.rename_protected_field(field) for field in raw_fields]
    # proceed with standardization iteratively
    for i, _ in enumerate(zip(train_dfs, test_dfs)):
        # standardize new fields if necessary
        if options.generated_fields is not None:
            for new_field in options.generated_fields:
                if new_field.std_options is not None:
                    logger.debug("Standardizing field %s...", new_field.target_name)
                    (
                        std_train_dfs[i][new_field.target_name],
                        std_test_dfs[i][new_field.target_name],
                    ) = standardize_field(
                        std_train_dfs[i][new_field.target_name],
                        std_test_dfs[i][new_field.target_name],
                        new_field.std_options,
                    )
        # standardize processed raw fields at the end
        for field in fields_to_process:
            if field.std_options is not None:
                logger.debug("Standardizing field %s...", field.field_name)
                (
                    std_train_dfs[i][field.field_name],
                    std_test_dfs[i][field.field_name],
                ) = standardize_field(
                    std_train_dfs[i][field.field_name],
                    std_test_dfs[i][field.field_name],
                    field.std_options,
                )
    logger.info("Standardization complete.")
    return std_train_dfs, std_test_dfs


def standardize_field(
    train_data: pd.core.series.Series,
    test_data: pd.core.series.Series,
    options: structs.config.StandardizationOptions,
):
    """
    Standardizes a field based on config options,
    taking care not to leak information from the training set
    to the testing set
    """
    if options.diagnostic_plots is True:
        continue_processing = handle_diag_plots(train_data, options.method)
        if continue_processing is False:
            logger.info(
                "Closing program prematurely to allow for configuration changes"
            )
            sys.exit()
    if options.method == structs.config.StandardizationMethod.LOGNORM:
        return (
            np.log(1 + (train_data - train_data.min())),
            np.log(1 + (test_data - test_data.min())),
        )
    # handle transformer based methods
    elif options.method == structs.config.StandardizationMethod.POWER:
        transformer = sklearn_pproc.PowerTransformer()
    elif options.method == structs.config.StandardizationMethod.ROBUST:
        transformer = sklearn_pproc.RobustScaler()
    # fit only on training data, to avoid information leakage
    fitted_trnsfrmr = transformer.fit(train_data.to_numpy().reshape(-1, 1))
    # use the fitted transformer for transforming both train and test data
    return (
        fitted_trnsfrmr.transform(train_data.to_numpy().reshape(-1, 1)),
        fitted_trnsfrmr.transform(test_data.to_numpy().reshape(-1, 1)),
    )
