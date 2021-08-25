"""
Command-Line interface functionality for preprocessing
"""
import json
from typing import Union
import argparse
import cattr
from nowcastlib.pipeline.structs import config
from nowcastlib.pipeline.utils import disambiguate_intfloatstr
from nowcastlib.pipeline.process import postprocess


def configure_parser(action_object):
    """Configures the subparser for our preprocess command"""
    pparser = action_object.add_parser(
        "postprocess",
        description="Postprocess dataset",
        help="Run `nowcastlib postprocess -h` for further help",
        formatter_class=argparse.HelpFormatter,
    )
    pparser.add(
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
        Union[int, float, str], disambiguate_intfloatstr
    )
    dataset_config = cattr_cnvrtr.structure(options, config.DataSet)
    return postprocess.postprocess_dataset(dataset_config)
