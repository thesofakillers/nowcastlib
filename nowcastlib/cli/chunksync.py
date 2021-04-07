"""
Dataset synchronization and chunking Command-Line Interface
"""
import os
import argparse
import json
import pandas as pd
import numpy as np
import nowcastlib.rawdata as rawdata


def configure_parser(action_object):
    """Configures the subparser for our chunksync command"""
    cparser = action_object.add_parser(
        "chunksync",
        description="Synchronize datasets, find overlapping chunks and save them to hdf5",
        help="Run `nowcastlib chunksync -h` for further help",
        formatter_class=argparse.HelpFormatter,
    )
    cparser.add("-c", "--config", required=True, help="config file path")
    cparser.add(
        "-do",
        nargs=3,
        action="append",
        metavar=("source-name", "key", "value"),
        help="override config option under the data_sources key",
    )
    cparser.add(
        "-co",
        nargs=2,
        action="append",
        metavar=("key", "value"),
        help="override config option under the chunk_config key",
    )


def chunksync(args):
    """
    Synchronizes data from different datasets and finds overlapping chunks
    given configuration.
    """
    with open(args.config) as json_file:
        config = json.load(json_file)

    # override config file from command line if required
    if args.do is not None:
        for override in args.do:
            source_name = override[0]
            field_key = override[1]
            new_field_value = override[2]
            config["dataset"]["data_sources"][source_name][field_key] = new_field_value
    if args.co is not None:
        for override in args.do:
            field_key = override[0]
            new_field_value = override[1]
            config["dataset"]["chunk_config"][field_key] = new_field_value

    dataset_config = config["dataset"]
    source_config = dataset_config["data_sources"]
    chunk_config = dataset_config["chunk_config"]
    min_chunk_duration_sec = chunk_config["min_chunk_duration_sec"]
    max_spacing_secs = chunk_config["max_delta_between_blocks_sec"]
    sample_spacing = chunk_config["sample_spacing"]

    data_dfs = []
    req_trig_fields = []
    for source_name, source_info in source_config.items():
        # check if this source requests trig fields, so we can compute them later
        cos_sin_fields = source_info.get("cos_sin_fields", None)
        if cos_sin_fields is not None:
            for field in cos_sin_fields:
                req_trig_fields.append(field)
        print("[INFO] loading source {}".format(source_name))
        data_df = pd.read_csv(
            source_info["file"],
            index_col=source_info["date_time_column_name"],
            parse_dates=False,
        )
        data_df.index = pd.to_datetime(
            data_df.index, format=source_info["date_time_column_format"]
        )
        data_df.index.name = None
        # only keep requested fields, and drop NaNs
        data_df = data_df[source_info["field_list"]]
        data_df.dropna()
        # resample, ensuring to floor to the nearest `sample_spacing` to ensure overlap
        data_df = data_df.resample(
            sample_spacing, origin=data_df.index[0].floor(sample_spacing)
        ).mean()
        data_dfs.append(data_df)

    print("[INFO] Synchronizing data sources")
    synced_df = pd.concat(data_dfs, axis=1, join="inner")

    print("[INFO] Finding overlapping chunks and large gaps")
    sample_spacing_secs = synced_df.index.freq.delta.seconds
    max_spacing_steps = np.floor((max_spacing_secs / sample_spacing_secs)).astype(int)
    final_mask = rawdata.compute_dataframe_mask(
        synced_df,
        max_spacing_steps,
        2 * len(req_trig_fields) + 2,
        [df.columns[0] for df in data_dfs],
    )

    print("[INFO] Imputing small gaps")
    interpolated_df = synced_df.interpolate("linear", limit_direction="both")
    interpolated_df = rawdata.compute_trig_fields(interpolated_df, req_trig_fields)

    chunked_df = interpolated_df.where(final_mask)

    print("[INFO] Splitting data into chunks")
    min_chunk_length = int(min_chunk_duration_sec / sample_spacing_secs)
    chunks = rawdata.make_chunks(chunked_df, min_chunk_length)

    print("[INFO] Saving chunks to HDF5")
    hdf5_path = os.path.join(chunk_config["path_to_hdf5"], chunk_config["tag"])
    hdfs = pd.HDFStore(hdf5_path)
    for i, chunk in enumerate(chunks):
        chunk.to_hdf(hdfs, "chunk_{:d}".format(i), format="table")
