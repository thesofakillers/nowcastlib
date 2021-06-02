"""
Functions for pre- and post-processing of data.
"""


def preprocess(data_df, sample_spacing):
    """
    Pre-processes a dataframe. Particularly:
    - drops rows with missing values
    - drops duplicates
    - sorts by data by date and time
    - resamples the data to a target sample frequency

    Parameters
    ----------
    data_df : pandas.core.frame.DataFrame
        the pandas dataframe we wish to process
    sample_spacing : string
        string specifying desired spacing between samples

    Returns
    -------
    pandas.core.frame.DataFrame
        the resulting pre-processed datetime indexed pandas dataframe
    """
    # drop NaNs and duplicates, sort by time ascending
    data_df.dropna()
    data_df = data_df[~data_df.index.duplicated(keep="last")]
    data_df.sort_index(inplace=True)
    # resample, ensuring to floor to the nearest `sample_spacing` to ensure overlap
    data_df = data_df.resample(
        sample_spacing, origin=data_df.index[0].floor(sample_spacing)
    ).mean()
    return data_df
