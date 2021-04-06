"""
.. include:: ../README.md
"""
from importlib_metadata import version
import nowcastlib.rawdata
import nowcastlib.gis
import nowcastlib.signals
import nowcastlib.utils
import nowcastlib.dynlag

__version__ = version(__package__)

__pdoc__ = {
    "cli": False,
}
