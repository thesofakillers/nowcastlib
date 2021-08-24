"""Shared functionality across the Nowcast Library Pipeline submodule"""
from typing import Union, Type, Any
import pandas as pd
import numpy as np
import attr
import cattr
from nowcastlib.pipeline.structs import config

cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)


def build_field_name(options: config.ProcessingOptions, field_name: str):
    """
    Builds the appropriate field name depending on whether
    the user wishes to overwrite the current field or not

    Parameters
    ----------
    options : nowcastlib.pipeline.structs.config.ProcessingOptions
    field_name : str
        the name of the current field we are acting on

    Returns
    -------
    str
        the resulting string
    """
    if options.overwrite:
        computed_field_name = field_name
    else:
        computed_field_name = "processed_{}".format(field_name)
    return computed_field_name


def rename_protected_field(field: config.RawField) -> config.RawField:
    """
    Renames overwrite-protected fields so to obtain a list of fields that
    are overwrite-able
    """
    if field.preprocessing_options is not None:
        if field.preprocessing_options.overwrite is False:
            correct_name = build_field_name(
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
                config.RawField,
            )
        else:
            return field
    else:
        return field


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
    options: config.SerializationOptions,
):
    """
    Serializes a given dataframe or numpy array
    to disk in the appropriate format
    """
    if isinstance(data, pd.core.frame.DataFrame):
        if options.output_format == "csv":
            data.to_csv(options.output_path, float_format="%g")
        elif options.output_format == "pickle":
            data.to_pickle(options.output_path)
    else:
        if options.output_format == "npy":
            np.save(options.output_path, data)


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
