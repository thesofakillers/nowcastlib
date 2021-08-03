"""
Functions for post-processing of data
"""
import sys
import logging
import pathlib
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import attr
import cattr
import sklearn.preprocessing as sklearn_pproc
import matplotlib.pyplot as plt
from nowcastlib.datasets import serialize_as_chunks
from nowcastlib.pipeline import structs
from nowcastlib.pipeline import utils
from nowcastlib.pipeline import sync
from nowcastlib.pipeline import split
from nowcastlib.pipeline.process import utils as process_utils
from . import generate

plt.ion()
logger = logging.getLogger(__name__)
# disable SettingWithCopy warning since it was catching False Positives
pd.set_option("chained_assignment", None)

cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)


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
    # convert tuple to list, safely
    input_fields = [element for element in field_config.input_fields]
    if (
        field_config.gen_func == structs.GeneratorFunction.CUSTOM
        or field_config.func_path is not None
    ):
        # TODO handle custom case
        raise NotImplementedError("Custom generator functions are not yet supported")
    else:
        if "index" in input_fields:
            input_df = data_df[
                [field for field in input_fields if field != "index"]
            ].assign(index=data_df.index)
            # ensure column order matches input_fields order
            input_df = input_df[input_fields]
        else:
            input_df = data_df[input_fields].copy()
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))
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
    return utils.yes_or_no(
        "Are you satisfied with the selected Standardization Method?"
    )


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
        continue_processing = handle_diag_plots(train_data, config.method)
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
            return cattr.structure(
                {
                    "field_name": correct_name,
                    **attr.asdict(
                        field,
                        filter=lambda attrib, _: attrib.name != "field_name",
                    ),
                },
                structs.RawField,
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
    Postprocesses a set of train-test splits given options outlined
    in the input DataSet config instance.
    """
    logger.info("Postprocessing dataset...")
    # instantiate processed lists
    proc_train_dfs = train_dfs.copy()
    proc_test_dfs = test_dfs.copy()
    # gather which fields to process into single list
    raw_fields: List[structs.RawField] = [
        field for source in config.data_sources for field in source.fields
    ]
    # rename overwrite-protected fields so to avoid acting on the original field
    fields_to_process = [rename_protected_field(field) for field in raw_fields]
    # start processing with processes that act the same on test and train dfs
    for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        for field in fields_to_process:
            logger.debug("Processing field %s...", field.field_name)
            if field.postprocessing_options is not None:
                proc_train_dfs[i][field.field_name] = process_utils.process_field(
                    train_df[field.field_name],
                    field.postprocessing_options,
                    False,
                )
                proc_test_dfs[i][field.field_name] = process_utils.process_field(
                    test_df[field.field_name],
                    field.postprocessing_options,
                    False,
                )
        # compute and process new fields if necessary
        if config.generated_fields is not None:
            for new_field in config.generated_fields:
                logger.debug("Generating field %s...", new_field.target_name)
                proc_train_dfs[i][new_field.target_name] = generate_field(
                    train_df, new_field
                )
                proc_test_dfs[i][new_field.target_name] = generate_field(
                    test_df, new_field
                )
                # standardize
                if new_field.std_options is not None:

                    logger.debug("Standardizing field %s...", new_field.target_name)
                    (
                        proc_train_dfs[i][new_field.target_name],
                        proc_test_dfs[i][new_field.target_name],
                    ) = standardize_field(
                        proc_train_dfs[i][new_field.target_name],
                        proc_test_dfs[i][new_field.target_name],
                        new_field.std_options,
                    )
        # standardize processed raw fields _after_ using them for computing gen fields
        for field in fields_to_process:
            if field.std_options is not None:
                logger.debug("Standardizing field %s...", field.field_name)
                (
                    proc_train_dfs[i][field.field_name],
                    proc_test_dfs[i][field.field_name],
                ) = standardize_field(
                    proc_train_dfs[i][field.field_name],
                    proc_test_dfs[i][field.field_name],
                    field.std_options,
                )
    return proc_train_dfs, proc_test_dfs


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
    for i, (train_df, test_df) in enumerate(zip(val_train_dfs, val_test_dfs)):
        serialize_as_chunks(train_df, cv_split / "train_data_{}.hdf5".format(i + 1))
        serialize_as_chunks(test_df, cv_split / "val_data_{}.hdf5".format(i + 1))


def postprocess_dataset(
    config: structs.DataSet,
    synced_data: Optional[Tuple[pd.core.frame.DataFrame, np.ndarray]] = None,
):
    """
    Postprocesses a dataset given options outlined
    in the input DataSet config instance.
    """
    assert (
        config.split_options is not None
    ), "`config.split_options` must be defined to perform postprocessing"
    # avoid syncing if synced info (dataframe and block locs) are provided
    if synced_data is None:
        chunked_df, chunk_locs = sync.synchronize_dataset(config)
    else:
        chunked_df = synced_data[0].copy()
        chunk_locs = synced_data[1].copy()

    # outer train and test split
    (train_data, _), (test_data, _) = split.train_test_split_sparse(
        chunked_df, config.split_options, chunk_locs
    )
    # postprocessing
    [proc_train_data], [proc_test_data] = postprocess_splits(
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
    ) = postprocess_splits(config, train_dfs, val_dfs)
    # serialize
    if config.split_options.output_options is not None:
        logger.info("Serializing postprocessing output as hdf5 chunks...")
        serialize_splits(
            config.split_options.output_options,
            proc_train_data,
            proc_test_data,
            proc_val_train_dfs,
            proc_val_test_dfs,
        )
