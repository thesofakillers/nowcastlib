"""module containing functionality related to feature engineering and selection"""
import logging
import pandas as pd
from nowcastlib.pipeline.structs import config
from . import generate


logger = logging.getLogger(__name__)


def generate_field(
    data_df: pd.core.frame.DataFrame,
    field_config: config.GeneratedField,
):
    """
    generates a new field by applying the relevant generator function

    Parameters
    ----------
    data_df : pandas.core.frame.DataFrame
        the dataframe holding the data we can access
    field_config : nowcastlib.pipeline.structs.config.GeneratedField
    Returns
    -------
    pandas.core.series.Series
        the resulting dataseries
    """
    # convert tuple to list, safely
    input_fields = [element for element in field_config.input_fields]
    if (
        field_config.gen_func == config.GeneratorFunction.CUSTOM
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


def generate_fields(options: config.DataSet, data_df: pd.core.frame.DataFrame):
    """
    Augments an input dataframe with additional fields
    generated from the existing fields and auxiliary data
    """
    proc_df = data_df.copy()
    if options.generated_fields is not None:
        logger.info("Generating additional fields...")
        for new_field in options.generated_fields:
            logger.debug("Generating field %s...", new_field.target_name)
            proc_df[new_field.target_name] = generate_field(data_df, new_field)
        logger.info("Field Generation complete.")
    return proc_df
