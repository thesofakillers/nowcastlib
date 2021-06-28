"""
Command-Line interface functionality for synchronization
"""
import json
import argparse
import cattr
from nowcastlib.pipeline import structs
import nowcastlib.pipeline.sync as sync


def configure_parser(action_object):
    """Configures the subparser for our preprocess command"""
    sparser = action_object.add_parser(
        "sync",
        description="Synchronize datasets",
        help="Run `nowcastlib sync -h` for further help",
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
        config = json.load(json_file)
    cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)
    dataset_config = cattr_cnvrtr.structure(config, structs.DataSet)
    return sync.synchronize_dataset(dataset_config)
