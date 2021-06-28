"""
.. include:: ../README.md
"""
import logging
from importlib_metadata import version
import nowcastlib.datasets
import nowcastlib.gis
import nowcastlib.signals
import nowcastlib.utils
import nowcastlib.dynlag
import nowcastlib.pipeline

__version__ = version(__package__)

__pdoc__ = {
    "cli": False,
}


# logging config {{{
root_logger = logging.getLogger(__name__)
root_logger.setLevel(logging.INFO)

logger_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logger_formatter)

root_logger.addHandler(console_handler)
# }}}
