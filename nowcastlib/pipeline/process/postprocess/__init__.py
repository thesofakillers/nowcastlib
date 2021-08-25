"""
Functions for post-processing of data
"""
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
from nowcastlib.pipeline.structs import config
from nowcastlib.pipeline import utils
from nowcastlib.pipeline import sync
from nowcastlib.pipeline.process import utils as process_utils

logger = logging.getLogger(__name__)
# disable SettingWithCopy warning since it was catching False Positives
pd.set_option("chained_assignment", None)


def postprocess_dataset(
    options: config.DataSet, data_df: Optional[pd.core.frame.DataFrame] = None
) -> pd.core.frame.DataFrame:
    """
    Postprocesses a dataset given options outlined
    in the input DataSet config instance.
    """
    # need to get data_df from syncing process if not provided
    if data_df is None:
        chunked_df, _ = sync.synchronize_dataset(options)
    else:
        chunked_df = data_df.copy()
    logger.info("Postprocessing dataset...")
    # instantiate our processed result
    proc_df = chunked_df.copy()
    # gather which fields to process into single list
    raw_fields: List[config.RawField] = [
        field for source in options.data_sources for field in source.fields
    ]
    # rename overwrite-protected fields so to avoid acting on the original field
    fields_to_process = [utils.rename_protected_field(field) for field in raw_fields]
    # finally we may perform postprocessing
    for field in fields_to_process:
        logger.debug("Processing field %s...", field.field_name)
        if field.postprocessing_options is not None:
            proc_df[field.field_name] = process_utils.process_field(
                chunked_df[field.field_name],
                field.postprocessing_options,
                False,
            )
    logger.info("Dataset postprocessing complete.")
    return proc_df
