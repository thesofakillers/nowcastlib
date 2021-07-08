"""Shared functionality across the Nowcast Library Pipeline submodule"""
from typing import Union
import pandas as pd
import numpy as np
from nowcastlib.pipeline import structs


def handle_serialization(
    data: Union[pd.core.frame.DataFrame, np.ndarray],
    config: structs.SerializationOptions,
):
    """
    Serializes a given dataframe or numpy array
    to disk in the appropriate format
    """
    if isinstance(data, pd.core.frame.DataFrame):
        if config.output_format == "csv":
            data.to_csv(config.output_path, float_format="%g")
        elif config.output_format == "pickle":
            data.to_pickle(config.output_path)
    else:
        if config.output_format == "npy":
            np.save(config.output_path, data)
