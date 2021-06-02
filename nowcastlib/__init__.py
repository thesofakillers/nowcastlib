"""
.. include:: ../README.md
"""
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
