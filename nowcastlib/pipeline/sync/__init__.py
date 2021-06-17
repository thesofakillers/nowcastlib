"""
Functions for synchronizing data
"""
from typing import Optional, List
from nowcastlib.pipeline import structs
from nowcastlib.pipeline import preprocess
import pandas as pd


def synchronize_dataset(
    config: structs.DataSet, dataset: Optional[List[pd.core.frame.DataFrame]] = None
):
    """
    Synchronizes a set of data sources given options outlined
    in the input DataSet config instance.

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.DataSet
    dataset : list[pandas.core.frame.DataFrame], default None
        The set of dataframes one wishes to synchronize.
        If `None`, the preprocessing output produced by the
        config options will be synchronized.


    Returns
    -------
    pandas.core.frame.DataFrame
        the resulting dataframe containing the synchronized data
    """
    # avoid preprocessing if datasets are passed directly
    if dataset is None:
        data_dfs = preprocess.preprocess_dataset(config)
    else:
        data_dfs = dataset

    # resample
    for data_df in data_dfs:
        data_df.index.name = None
        data_df = data_df.resample(
            config.sample_spacing, origin=data_df.index[0].floor(config.sample_spacing)
        ).mean()

    synced_df = pd.concat(data_dfs, axis=1, join="inner")
