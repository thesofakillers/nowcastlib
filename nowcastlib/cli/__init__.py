"""
Command-Line interfaces for the Nowcast Library
"""
import configargparse
from .triangulate import triangulate


def main():
    """
    Function for organizing subcommands and providing help to user.
    """
    parser = configargparse.ArgParser()
    command_parsers = parser.add_subparsers(dest="command", help="available commands")
    triangulate_parser = command_parsers.add_parser(
        "triangulate",
        help="Run `nowcastlib triangulate -h` for further help",
        default_config_files=["./.config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        add_help=False,
    )
    command_args, _ = parser.parse_known_args()

    command = command_args.command
    if command is None:
        parser.print_help()
    elif command == "triangulate":
        triangulate(triangulate_parser)
