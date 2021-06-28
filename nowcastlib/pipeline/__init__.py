"""
Data processing and Model evaluation pipeline for the Nowcast Library
.. include:: PIPELINE.md
"""
import logging
from . import preprocess
from . import sync
from . import structs

logging.basicConfig(level=logging.DEBUG)
