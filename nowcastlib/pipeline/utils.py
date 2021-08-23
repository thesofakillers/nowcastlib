"""Shared functionality across the Nowcast Library Pipeline submodule"""
from typing import Union
import pandas as pd
import numpy as np
import attr
import cattr
from nowcastlib.pipeline import structs

cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)


def build_field_name(config: structs.ProcessingOptions, field_name: str):
    """
    Builds the appropriate field name depending on whether
    the user wishes to overwrite the current field or not

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.ProcessingOptions
    field_name : str
        the name of the current field we are acting on

    Returns
    -------
    str
        the resulting string
    """
    if config.overwrite:
        computed_field_name = field_name
    else:
        computed_field_name = "processed_{}".format(field_name)
    return computed_field_name


def rename_protected_field(field: structs.RawField) -> structs.RawField:
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
                structs.RawField,
            )
        else:
            return field
    else:
        return field


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
