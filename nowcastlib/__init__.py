"""
.. include:: ../README.md
"""
import logging
import nowcastlib.datasets
import nowcastlib.gis
import nowcastlib.signals
import nowcastlib.utils
import nowcastlib.dynlag
import nowcastlib.pipeline

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)

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
