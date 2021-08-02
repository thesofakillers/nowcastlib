"""
Command-Line interfaces for the Nowcast Library
"""
import logging
import argparse
import configargparse
from nowcastlib.pipeline.process.preprocess import cli as preprocess_cli
from nowcastlib.pipeline.sync import cli as sync_cli
from nowcastlib.pipeline.process.postprocess import cli as postprocess_cli
from nowcastlib.pipeline import cli as pipeline_cli
from . import triangulate


def main():
    """
    Function for organizing subcommands and providing help to user.
    """
    parser = configargparse.ArgParser()
    parser.add(
        "-v",
        "--verbose",
        action="store_true",
        help="increase verbosity level from INFO to DEBUG",
    )

    command_parsers = parser.add_subparsers(dest="command", help="available commands")

    triangulate.configure_parser(command_parsers)
    preprocess_cli.configure_parser(command_parsers)
    sync_cli.configure_parser(command_parsers)
    postprocess_cli.configure_parser(command_parsers)
    pipeline_cli.configure_parser(command_parsers)

    args = parser.parse_args()

    if args.verbose:
        logger = logging.getLogger("nowcastlib")
        logger.setLevel(logging.DEBUG)
        logger.handlers[0].setLevel(logging.DEBUG)

    command = args.command
    if command is None:
        parser.print_help()
    elif command == "triangulate":
        triangulate.triangulate(args)
    elif command == "preprocess":
        preprocess_cli.run(args)
    elif command == "sync":
        sync_cli.run(args)
    elif command == "postprocess":
        postprocess_cli.run(args)
    elif command == "datapipe":
        pipeline_cli.run_datapipe(args)
