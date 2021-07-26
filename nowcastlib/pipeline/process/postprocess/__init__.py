"""
Functions for post-processing of data
"""
import logging
from typing import Optional, List
import pandas as pd
import attr
from nowcastlib.pipeline import structs
from nowcastlib.pipeline.process import utils as process_utils

logger = logging.getLogger(__name__)


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
    # compute new fields if necessary
    # TODO
    # standardize
    # TODO
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
                )
            )
        else:
            return field
    else:
        return field
