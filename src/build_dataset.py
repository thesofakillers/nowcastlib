"""
Processes raw CSV files into chunks, which are saved as HDF5 files.
Run with
`python build_dataset.py path/to/config.json`
"""
import os
import sys
import json
import pandas
import numpy


def get_cont_chunks(csv_df, date_col, max_step, min_duration):
    """
    Finds contiguous blocks of time in a given array.

    :param pandas.core.frame.DataFrame csv_df: pandas dataframe containing the
        csv time series data
    :param str date_col: The name of the column in the dataframe containing the
        datetime information
    :param int max_step: The maximum amount of time in seconds allowed between
        datapoints of the same chunk
    :param int min_duration: The minimum amount of time in seconds for that
        a contiguous chunk of data should be
    :return: a Pandas IntervalArray containing the pairs of start and stop
        times defining each chunk.
    :rtype: pandas.core.arrays.interval.IntervalArray
    """
    time_array = csv_df[date_col]
    # indexes where a discontinuity in time occurs
    idxs = numpy.where(numpy.diff(time_array) > max_step)[0]
    interval_list = list()
    total_disc_idxs = len(idxs)
    print("found {} discontinuities".format(total_disc_idxs))
    if total_disc_idxs == 0:
        # in this case, our entire dataset is a continuous chunk
        interval_list.append(pandas.Interval(time_array[0], time_array[-1]))
    else:
        # in this case, we need to add our chunks to interval_list one by one
        left_idx = 0
        right_idx = -1
        for idx in idxs:
            right_idx = idx
            duration = time_array[right_idx] - time_array[left_idx]
            if duration > min_duration:
                interv = pandas.Interval(time_array[left_idx], time_array[right_idx])
                interval_list.append(interv)
            left_idx = right_idx + 1
    intervals = pandas.arrays.IntervalArray(interval_list)
    return intervals


def process_input_sources(source_config, chunk_config):
    """
    Processes a series of input CSV data sources.
    * date columns are parsed
    * gaps are dropped
    * cosine and sine fields are calculated if specified
    * contiguous chunks of data are recognized

    :param dict source_config: dictionary where each key corresponds to configuration
        for an input data source.
    :param dict chunk_config: dictionary where we specify parameters for defining
        contiguous blocks of data.
    :return: (data_dfs, contiguous_chunks_list)
        * dfs is a list of dataframes containing the data, each corresponding to an
          input source
        * contiguous_chunks_list is a list of pandas IntervalArrays, specifying the
          contiguous chunks found for each data source.
    :rtype: tuple
    """
    # will be [df1, df2, df3, ...,dfN] for N data sources
    data_dfs = list()
    # will be [chunks1, chunks2, chunk3, ..., chunksN] for N data sources
    # chunks_i itself is an array of pandas Intervals outlining contig blocks
    contiguous_chunks_list = list()
    for source_name, source_info in source_config.items():
        print("[INFO] loading source {}".format(source_name))
        date_col_name = source_info.get("date_time_column_name", "Date time")
        date_fmt = source_info.get("date_time_column_format", "%Y-%m-%dT%H:%M:%S")
        # read data, dropping NaN rows and parsing the date column.
        data = pandas.read_csv(source_info["file"])
        data = data.dropna()
        data = data.reset_index(drop=True)
        data["master_datetime"] = pandas.to_datetime(
            data[date_col_name], format=date_fmt
        )
        data = data.drop([date_col_name], axis=1)
        # transform fields to cos/sin if requested in config
        cos_sin_fields = source_info.get("cos_sin_fields", None)
        if cos_sin_fields is not None:
            for func, pname in zip([numpy.cos, numpy.sin], ["Cosine", "Sine"]):
                for field_name in cos_sin_fields:
                    new_field_name = pname + " " + field_name
                    field_data = data[field_name]
                    if "deg" in field_name:
                        field_data = numpy.radians(field_data)
                    data[new_field_name] = func(field_data)
        # find contiguous chunks of data
        chunks = get_cont_chunks(
            data,
            "master_datetime",
            pandas.Timedelta(chunk_config["max_delta_between_blocks_sec"], unit="s"),
            pandas.Timedelta(chunk_config["min_chunk_duration_sec"], unit="s"),
        )
        data_dfs.append(data)
        contiguous_chunks_list.append(chunks)

    return data_dfs, contiguous_chunks_list


if __name__ == "__main__":

    # read in positional parameters: TODO, change by actual parse_args!
    config_path = sys.argv[1]

    # ugly configuration loading code
    config_file = open(config_path, "r")
    ds_config = json.loads(config_file.read())
    data_config = ds_config["dataset"]
    config_file.close()
    chunk_config = data_config.get("chunk_config")
    training_config = data_config.get("training_config")
    source_configs = data_config.get("data_sources")
    hdf5_path = os.path.join(chunk_config["path_to_hdf5"], chunk_config["tag"])
    sample_spacing_min = chunk_config["sample_spacing"]
    min_date = pandas.to_datetime(chunk_config.get("min_date", "1900-01-01T00:00:00"))
    max_date = pandas.to_datetime(chunk_config.get("max_date", "2100-12-31T00:00:00"))

    # process each data source by computing cos/sin fields and finding contiguous chunks.
    dfs, cont_date_intervals = process_input_sources(source_configs, chunk_config)

    # find overlapping contiguous chunks and merge datasets.
    hdfs = pandas.HDFStore(hdf5_path)
    n_samples = 0
    # use the first source as base reference
    base_intervals = cont_date_intervals[0]
    # compare each interval to all other intervals in all other sources in search of overlaps
    for i, base_interval in enumerate(base_intervals):
        n_sources_overlapping = 0
        src_overlaps = list()
        for src_intervals in cont_date_intervals[1:]:
            overlap_intervals = list()
            found_intervals = src_intervals[src_intervals.overlaps(base_interval)]
            if len(found_intervals) > 0:
                # this source overlaps at least once, so count it
                n_sources_overlapping += 1
                # define shared intervals based on overlaps
                for interval in found_intervals:
                    o_left = max(base_interval.left, interval.left)
                    o_right = min(base_interval.right, interval.right)
                    overlap_intervals.append(
                        pandas.Interval(o_left, o_right, closed="neither")
                    )
            src_overlaps.append(overlap_intervals)
        src_overlaps.insert(0, src_overlaps[0])
        # skip this interval if it doesn't overlap with ALL sources.
        if n_sources_overlapping != len(source_configs.keys()) - 1:
            continue
        # build dataframe with overlapping data from all sources
        resample_ok = True
        all_slices = list()
        for j, df in enumerate(dfs):
            # TODO: address that we always only use the first interval, see issue #5
            indexing_interval = src_overlaps[j][0]
            indexed_df = df[
                (df["master_datetime"] > indexing_interval.left)
                & (df["master_datetime"] < indexing_interval.right)
            ]
            indexed_df = indexed_df.set_index(
                pandas.DatetimeIndex(indexed_df["master_datetime"])
            )
            indexed_df = indexed_df.drop(["master_datetime"], axis=1)
            indexed_df = indexed_df.dropna()
            try:
                # resample to and impute to make the data regular
                indexed_df = indexed_df.resample(sample_spacing_min).bfill()
                all_slices.append(indexed_df)
            except:
                resample_ok = False
                print("fail")
                break
        if resample_ok:
            # concatenate the slices horizontally, inner join only keeps overlap rows
            synced_df = pandas.concat(all_slices, axis="columns", join="inner")
            # we need to check once again whether our minimum chunk length is respected
            if (
                synced_df.index.max() - synced_df.index.min()
            ).total_seconds() > chunk_config["min_chunk_duration_sec"]:
                # transform datetime to cos/sin day
                datetime = synced_df.index.to_series()
                if datetime.iloc[0] < min_date:
                    continue
                if datetime.iloc[-1] > max_date:
                    continue
                print(datetime.iloc[0], "-", datetime.iloc[-1])
                sec_day = (datetime - datetime.dt.normalize()) / pandas.Timedelta(
                    seconds=1
                )
                cos_sec_day = numpy.cos(2 * numpy.pi * sec_day.values / 86400.0)
                sin_sec_day = numpy.sin(2 * numpy.pi * sec_day.values / 86400.0)
                synced_df["Cosine Day"] = cos_sec_day
                synced_df["Sine Day"] = sin_sec_day
                # finally, append to hdf
                synced_df.to_hdf(hdfs, "chunk_{:d}".format(i), format="table")
                n_samples += len(synced_df)
                print(n_samples)
        else:
            print("resampled failed, ignoring chunk")
