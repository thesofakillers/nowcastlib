"""
Command-Line interface functionality for preprocessing
"""
import logging
import json
from typing import Union
import argparse
import cattr
from nowcastlib.pipeline.structs import config
from nowcastlib.pipeline import utils
from nowcastlib.pipeline.process import postprocess
from nowcastlib.pipeline import features


logger = logging.getLogger(__name__)


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
        Union[int, float, str], utils.disambiguate_intfloatstr
    )
    dataset_config = cattr_cnvrtr.structure(options, config.DataSet)
    proc_df = postprocess.postprocess_dataset(dataset_config)
    # add generated fields if necessary
    if dataset_config.generated_fields is not None:
        proc_df = features.generate_fields(dataset_config, proc_df)
    if dataset_config.postprocessing_output is not None:
        logger.info("Serializing postprocessing results...")
        utils.handle_serialization(proc_df, dataset_config.postprocessing_output)
        logger.info("Serialization complete.")
    return proc_df
