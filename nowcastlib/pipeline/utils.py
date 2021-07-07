"""Shared functionality across the Nowcast Library Pipeline submodule"""
import pandas as pd
from nowcastlib.pipeline import structs

def handle_serialization(
    data_df: pd.core.frame.DataFrame, config: structs.SerializationOptions
):
    """
    Serializes a given dataframe to disk in the appropriate format
    """
    if config.output_format == "csv":
        data_df.to_csv(config.output_path, float_format="%g")
    elif config.output_format == "pickle":
        data_df.to_pickle(config.output_path)
