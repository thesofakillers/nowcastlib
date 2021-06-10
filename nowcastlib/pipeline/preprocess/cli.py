"""
DataSource processing
"""
import json
import argparse
import cattr
from nowcastlib.pipeline import structs
import nowcastlib.pipeline.preprocess as preprocess


def configure_parser(action_object):
    """Configures the subparser for our preprocess command"""
    pparser = action_object.add_parser(
        "preprocess",
        description="Preprocess datasets",
        help="Run `nowcastlib preprocess -h` for further help",
        formatter_class=argparse.HelpFormatter,
    )
    pparser.add(
        "-c",
        "--config",
        required=True,
        is_config_file=True,
        help="path to JSON file following the DataSet format. See docs for available fields)",
    )


def run(args):
    """runs appropriate function based on provided cli args"""
    with open(args.config) as json_file:
        config = json.load(json_file)
    cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)
    dataset_config = cattr_cnvrtr.structure(config, structs.DataSet)
    return preprocess.preprocess_dataset(dataset_config)
