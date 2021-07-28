"""
Functions for post-processing of data
"""
import logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import attr
from nowcastlib.pipeline import structs
from nowcastlib.pipeline.process import utils as process_utils
from . import generate

logger = logging.getLogger(__name__)


def generate_field(
    data_df: pd.core.frame.DataFrame,
    gen_func: structs.GeneratorFunction,
    input_fields: Tuple[str],
    additional_args: Optional[dict] = None,
    func_path: Optional[str] = None,
):
    """
    generates a new field by applying the relevant generator function

    Parameters
    ----------
    data_df : pandas.core.frame.DataFrame
        the dataframe holding the data we can access
    gen_func: nowcastlib.pipeline.structs.GeneratorFunction
        which generator function to use for generating our field
    input_fields: tuple of str
        the names of the fields that will be passed to the
        generator function as *args
    additional_args: dict, default `None`
        a dictionary where keys correspond to additional
        key-word arguments to pass to the generator function
        as **kwargs
    func_path: str, default `None`
        the path to the file containing the custom function
        implementation in case gen_func is set to
        nowcastlib.pipeline.structs.GeneratorFunction.CUSTOM
    Returns
    -------
    pandas.core.series.Series
        the resulting dataseries
    """
    if gen_func == structs.GeneratorFunction.CUSTOM or func_path is not None:
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
        func = generate.function_map[gen_func]

    # prepare additional kwargs appropriately
    additional_args = {} if additional_args is None else additional_args

    return func(*[input_df[col] for col in input_df], **additional_args)


def postprocess_dataset(
    config: structs.DataSet,
    train_dfs: List[pd.core.frame.DataFrame],
    test_dfs: List[pd.core.frame.DataFrame],
):
    """
    Postprocesses a set of data sources given options outlined
    in the input DataSet config instance.
    """
    logger.info("Postprocessing dataset...")
    fields_to_process: List[structs.DataField] = [
        field for source in config.data_sources for field in source.fields
    ]
    # rename overwrite-protected fields so to avoid acting on the original field
    fields_to_process = list(map(rename_protected_fields, fields_to_process))
    # start processing with processes that act the same on test and train dfs
    for train_df, test_df in zip(train_dfs, test_dfs):
        for field in fields_to_process:
            options = field.postprocessing_options
            if options is not None:
                for dataframe in [train_df, test_df]:
                    dataframe[field.field_name] = process_utils.process_field(
                        dataframe[field.field_name], options, False
                    )
        if config.extra_postprocessing is not None:
            # compute new fields if necessary
            new_fields = config.extra_postprocessing.new_fields
            if new_fields is not None:
                for new_field in new_fields:
                    for dataframe in [train_df, test_df]:
                        dataframe[new_field.target_name] = generate_field(
                            dataframe,
                            **attr.asdict(
                                new_field,
                                # get rid of the target_name attr as this isnt a kwarg
                                filter=lambda attrib, _: attrib.name != "target_name",
                            ),
                        )
            std_config = config.extra_postprocessing.standardize_fields
            if std_config is not None:
                # standardize
                # TODO
                pass
        # serialize
        # TODO
        return [None], [None]


def rename_protected_fields(field: structs.DataField) -> structs.DataField:
    """
    Renames overwrite-protected fields so to obtain a list of fields that
    are overwrite-able
    """
    if field.preprocessing_options is not None:
        if field.preprocessing_options.overwrite is False:
            correct_name = process_utils.build_field_name(
                field.preprocessing_options, field.field_name
            )
            return structs.DataField(
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
