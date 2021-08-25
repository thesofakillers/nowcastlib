"""
Command-Line interface functionality for splitting and related processes
"""
import json
import argparse
from typing import Union
import cattr
from nowcastlib.pipeline import split, standardize, utils
from nowcastlib.pipeline.structs import config


def configure_parser(action_object):
    """Configures the subparser for our preprocess command"""
    sparser = action_object.add_parser(
        "splitprocess",
        description="Perform dataset splitting and related processes",
        help="Run `nowcastlib splitprocess -h` for further help",
        formatter_class=argparse.HelpFormatter,
    )
    sparser.add(
        "-c",
        "--config",
        required=True,
        help="path to JSON file following the DataSet format. See docs for available fields",
    )


def run(args):
    """runs appropriate function based on provided cli args"""
    with open(args.config) as json_file:
        options = json.load(json_file)
    cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)
    cattr_cnvrtr.register_structure_hook(
        Union[int, float, str], utils.disambiguate_intfloatstr
    )
    dataset_config = cattr_cnvrtr.structure(options, config.DataSet)
    # splitting
    if dataset_config.split_options is not None:
        outer_split, inner_split = split.split_dataset(dataset_config)
        # standardization (requires splits)
        outer_split, inner_split = standardize.standardize_dataset(
            dataset_config, outer_split, inner_split
        )
        # serialization
        split.serialize_splits(
            dataset_config.split_options.output_options, outer_split, inner_split
        )
