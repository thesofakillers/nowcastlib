"""
Functions for post-processing of data
"""
import sys
import logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import attr
import sklearn.preprocessing as sklearn_pproc
import matplotlib.pyplot as plt
from nowcastlib.pipeline import structs
from nowcastlib.pipeline import utils
from nowcastlib.pipeline.process import utils as process_utils
from . import generate

plt.ion()
logger = logging.getLogger(__name__)


def generate_field(
    data_df: pd.core.frame.DataFrame,
    field_config: structs.GeneratedField,
):
    """
    generates a new field by applying the relevant generator function

    Parameters
    ----------
    data_df : pandas.core.frame.DataFrame
        the dataframe holding the data we can access
    field_config : nowcastlib.pipeline.structs.GeneratedField
    Returns
    -------
    pandas.core.series.Series
        the resulting dataseries
    """
    if (
        field_config.gen_func == structs.GeneratorFunction.CUSTOM
        or field_config.func_path is not None
    ):
        # TODO handle custom case
        raise NotImplementedError("Custom generator functions are not yet supported")
    else:
        if "index" in field_config.input_fields:
            input_df = data_df[
                [field for field in field_config.input_fields if field != "index"]
            ].assign(index=data_df.index)
            # ensure column order matches input_fields order
            input_df = input_df[field_config.input_fields]
        else:
            input_df = data_df[field_config.input_fields].copy()
        # use the function_map dictionary to select the right generator function
        func = generate.function_map[field_config.gen_func]
    # prepare additional kwargs appropriately
    additional_args = (
        {} if field_config.additional_kwargs is None else field_config.additional_kwargs
    )
    # finally, generate the field with the function we picked earlier
    return func(*[input_df[col] for col in input_df], **additional_args)


def handle_diag_plots(
    input_series: pd.core.series.Series,
    configured_method: structs.StandardizationMethod,
):
    """
    Plots different rescalings of the input series,
    asking the user for confirmation
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    pwr_trnsformer = sklearn_pproc.PowerTransformer()
    robust_trnsfrmr = sklearn_pproc.RobustScaler()
    ax1.hist(input_series, bins=200, color="black")
    ax1.set_title("Original Series")
    ax2.hist(
        pwr_trnsformer.fit_transform(input_series.to_numpy().reshape(-1, 1)),
        bins=200,
        color="darkblue"
        if configured_method == structs.StandardizationMethod.POWER
        else "darkgrey",
    )
    ax2.set_title("Power Transform")
    ax3.hist(
        robust_trnsfrmr.fit_transform(input_series.to_numpy().reshape(-1, 1)),
        bins=200,
        color="darkblue"
        if configured_method == structs.StandardizationMethod.ROBUST
        else "darkgrey",
    )
    ax3.set_title("Robust Scaling")
    ax4.hist(
        np.log(1 + (input_series - input_series.min())),
        bins=200,
        color="darkblue"
        if configured_method == structs.StandardizationMethod.LOGNORM
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
    return utils.yes_or_no("Are you satisfied with the target sample rate?")


def standardize_field(
    train_data: pd.core.series.Series,
    test_data: pd.core.series.Series,
    config: structs.StandardizationOptions,
):
    """
    Standardizes a field based on config options,
    taking care not to leak information from the training set
    to the testing set
    """
    if config.diagnostic_plots is True:
        continue_processing = handle_diag_plots(
            train_data.to_numpy().reshape(-1, 1), config.method
        )
        if continue_processing is False:
            logger.info(
                "Closing program prematurely to allow for configuration changes"
            )
            sys.exit()
    if config.method == structs.StandardizationMethod.LOGNORM:
        return (
            np.log(1 + (train_data - train_data.min())),
            np.log(1 + (test_data - test_data.min())),
        )
    # handle transformer based methods
    elif config.method == structs.StandardizationMethod.POWER:
        transformer = sklearn_pproc.PowerTransformer()
    elif config.method == structs.StandardizationMethod.ROBUST:
        transformer = sklearn_pproc.RobustScaler()
    # fit only on training data, to avoid information leakage
    fitted_trnsfrmr = transformer.fit(train_data.to_numpy().reshape(-1, 1))
    # use the fitted transformer for transforming both train and test data
    return (
        fitted_trnsfrmr.transform(train_data.to_numpy().reshape(-1, 1)),
        fitted_trnsfrmr.transform(test_data.to_numpy().reshape(-1, 1)),
    )


def rename_protected_field(field: structs.RawField) -> structs.RawField:
    """
    Renames overwrite-protected fields so to obtain a list of fields that
    are overwrite-able
    """
    if field.preprocessing_options is not None:
        if field.preprocessing_options.overwrite is False:
            correct_name = process_utils.build_field_name(
                field.preprocessing_options, field.field_name
            )
            return structs.RawField(
                field_name=correct_name,
                **attr.asdict(
                    field,
                    filter=lambda attrib, _: attrib.name != "field_name",
                ),
            )
        else:
            return field
    else:
        return field


def postprocess_splits(
    config: structs.DataSet,
    train_dfs: List[pd.core.frame.DataFrame],
    test_dfs: List[pd.core.frame.DataFrame],
):
    """
    Postprocesses a set of data sources given options outlined
    in the input DataSet config instance.
    """
    logger.info("Postprocessing dataset...")
    raw_fields: List[structs.RawField] = [
        field for source in config.data_sources for field in source.fields
    ]
    # rename overwrite-protected fields so to avoid acting on the original field
    fields_to_process = [rename_protected_field(field) for field in raw_fields]
    # start processing with processes that act the same on test and train dfs
    for train_df, test_df in zip(train_dfs, test_dfs):
        for field in fields_to_process:
            logger.debug("Processing field %s...", field.field_name)
            if field.postprocessing_options is not None:
                for dataframe in [train_df, test_df]:
                    dataframe[field.field_name] = process_utils.process_field(
                        dataframe[field.field_name], field.postprocessing_options, False
                    )
        # compute and process new fields if necessary
        if config.generated_fields is not None:
            for new_field in config.generated_fields:
                logger.debug("Generating field %s...", new_field.target_name)
                for dataframe in [train_df, test_df]:
                    # generate
                    dataframe[new_field.target_name] = generate_field(
                        dataframe, new_field
                    )
                # standardize
                if new_field.std_options is not None:

                    logger.debug("Standardizing field %s...", new_field.target_name)
                    (
                        train_df[new_field.target_name],
                        test_df[new_field.target_name],
                    ) = standardize_field(
                        train_df[new_field.target_name],
                        test_df[new_field.target_name],
                        new_field.std_options,
                    )
        # standardize processed raw fields _after_ using them for computing gen fields
        for field in fields_to_process:
            if field.std_options is not None:
                logger.debug("Standardizing field %s...", field.field_name)
                (
                    train_df[field.field_name],
                    test_df[field.field_name],
                ) = standardize_field(
                    train_df[field.field_name],
                    test_df[field.field_name],
                    field.std_options,
                )
        return train_dfs, test_dfs
