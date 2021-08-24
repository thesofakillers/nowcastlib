"""
Module containing custom structures used throughout the pipeline
submodule.
"""
from typing import Tuple, TypeVar, List, Optional
import pandas as pd
import numpy as np
from . import config


SparseData = Tuple[pd.core.frame.DataFrame, np.ndarray]

TrainTestSplit = Tuple[SparseData, SparseData]
"""The dataframe and block locations of the train and test sets"""

IteratedSplit = Tuple[List[SparseData], List[SparseData]]
"""List of train data and list of validation data"""

SplitDataSet = Tuple[TrainTestSplit, IteratedSplit]
"""
The result of splitting a dataset for evaluation and validation:
"""
