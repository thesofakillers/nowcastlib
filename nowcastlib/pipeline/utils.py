"""Shared functionality across the Nowcast Library Pipeline submodule"""
from typing import Union, Type, Any
import pandas as pd
import numpy as np
from nowcastlib.pipeline import structs


def disambiguate_intfloatstr(train_split: Any, _klass: Type) -> Union[int, float, str]:
    """Disambiguates Union of int, float and str for cattrs"""
    if isinstance(train_split, int):
        return int(train_split)
    elif isinstance(train_split, float):
        return float(train_split)
    elif isinstance(train_split, str):
        return str(train_split)
    else:
        raise ValueError("Cannot disambiguate Union[int, flaot, str]")


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


def yes_or_no(question):
    """
    Asks the user a yes or no question, parsing the answer
    accordingly.
    """
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False
        else:
            print("Please enter either 'y'/'Y' or 'n'/'N'")
