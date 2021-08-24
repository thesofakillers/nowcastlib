"""
Functions for pre-processing of data.
"""
import logging
from typing import Union
import pandas as pd
from nowcastlib.pipeline.structs import config
from nowcastlib.pipeline import utils as pipeline_utils
from nowcastlib.pipeline.process import utils as process_utils

logger = logging.getLogger(__name__)


def preprocess_datasource(options: config.DataSource):
    """
    Runs preprocessing on a given data source given options outlined
    in the input DataSource instance.

    Parameters
    ----------
    options : nowcastlib.pipeline.structs.config.DataSource

    Returns
    -------
    pandas.core.frame.DataFrame
        the resulting processed dataframe
    """
    logger.debug("Preprocessing %s...", options.name)
    index_field = next(field for field in options.fields if field.is_date)
    logger.debug("Reading file...")
    data_df = pd.read_csv(
        options.path,
        usecols=[field.field_name for field in options.fields],
        index_col=index_field.field_name,
        parse_dates=False,
        comment=options.comment_format,
    )
    data_df.index = pd.to_datetime(data_df.index, format=index_field.date_format)
    data_df = data_df[  # pylint: disable=unsubscriptable-object
        ~data_df.index.duplicated(keep="last")
    ]
    data_df.sort_index(inplace=True)
    for field in options.fields:
        logger.debug("Processing field %s of %s...", field.field_name, options.name)
        proc_options = field.preprocessing_options
        if proc_options is not None:
            # next two lines handle whether user wishes to overwrite field or not
            computed_field_name = pipeline_utils.build_field_name(
                proc_options, field.field_name
            )
            data_df[computed_field_name] = data_df[field.field_name].copy()
            data_df[computed_field_name] = process_utils.process_field(
                data_df[computed_field_name], proc_options, True
            )
    logger.debug("Dropping NaNs...")
    data_df = data_df.dropna()
    if options.preprocessing_output is not None:
        logger.debug("Serializing preprocessing output...")
        pipeline_utils.handle_serialization(data_df, options.preprocessing_output)
    return data_df


def preprocess_dataset(options: config.DataSet):
    """
    Runs preprocessing on a given set of data sources given options outlined
    in the input DataSet instance.

    Parameters
    ----------
    options : nowcastlib.pipeline.structs.config.DataSet

    Returns
    -------
    list[pandas.core.frame.DataFrame]
        list containing each of the resulting processed dataframes
    """
    logger.info("Preprocessing dataset...")
    processed_dfs = []
    for ds_config in options.data_sources:
        processed_dfs.append(preprocess_datasource(ds_config))
    logger.info("Dataset preprocessing complete.")
    return processed_dfs
