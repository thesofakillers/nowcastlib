"""
Command-Line interfaces for the Nowcast Library
"""
import argparse
import configargparse
from . import triangulate
from . import chunksync


def main():
    """
    Function for organizing subcommands and providing help to user.
    """
    parser = configargparse.ArgParser()
    command_parsers = parser.add_subparsers(dest="command", help="available commands")

    triangulate.configure_parser(command_parsers)
    chunksync.configure_parser(command_parsers)

    args = parser.parse_args()
    command = args.command
    if command is None:
        parser.print_help()
    elif command == "triangulate":
        triangulate.triangulate(args)
    elif command == "chunksync":
        chunksync.chunksync(args)
