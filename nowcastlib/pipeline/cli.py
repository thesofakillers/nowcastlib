"""
Command-Line interface functionality for preprocessing
"""
import json
from typing import Union, Any, Type
import argparse
import cattr
from nowcastlib.pipeline import structs
import nowcastlib.pipeline as pipeline


def configure_parser(action_object):
    """Configures the subparser for our preprocess command"""
    pparser = action_object.add_parser(
        "datapipe",
        description=(
            "Run the data-wrangling pipeline step: "
            "Preprocessing, Synchronization and Postprocessing"
        ),
        help="Run `nowcastlib datapipe -h` for further help",
        formatter_class=argparse.HelpFormatter,
    )
    pparser.add(
        "-c",
        "--config",
        required=True,
        help="path to JSON file following the DataSet format. See docs for available fields",
    )


def run_datapipe(args):
    """runs appropriate function based on provided cli args"""
    with open(args.config) as json_file:
        config = json.load(json_file)
    cattr_cnvrtr = cattr.GenConverter(forbid_extra_keys=True)
    cattr_cnvrtr.register_structure_hook(
        Union[int, float, str], _disambiguate_intfloatstr
    )
    dataset_config = cattr_cnvrtr.structure(config, structs.DataSet)
    return pipeline.pipe_dataset(dataset_config)


def _disambiguate_intfloatstr(train_split: Any, _klass: Type) -> Union[int, float, str]:
    """Disambiguates Union of int, float and str for cattrs"""
    if isinstance(train_split, int):
        return int(train_split)
    elif isinstance(train_split, float):
        return float(train_split)
    elif isinstance(train_split, str):
        return str(train_split)
    else:
        raise ValueError("Cannot disambiguate Union[int, flaot, str]")
