"""
Command-Line interfaces for the Nowcast Library
"""
import sys
import numpy as np
import pandas as pd
import configargparse
import nowcastlib as ncl


def triangulate():
    """
    Subcommand for generating triangulated data
    """
    # {{{ args/config parsing (triple curly brackets to allow vim section fold/collapse)
    parser = configargparse.ArgParser(
        default_config_files=[".config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add(
        "-c", "--config", required=True, is_config_file=True, help="config file path"
    )
    # wind data
    parser.add("-i", "--input-path", required=True, help="path to data")
    parser.add(
        "-x",
        "--index",
        required=True,
        help="column to use for indexing data",
    )
    parser.add(
        "-m",
        "--mask-path",
        help="path to .npy file containing mask to apply to data after processing, if desired",
    )
    parser.add(
        "-ws",
        "--wind-speed",
        required=True,
        help="column containing wind speed data",
    )
    parser.add(
        "-wd",
        "--wind-direction",
        required=True,
        help="column containing wind direction data in radians",
    )
    parser.add(
        "-s",
        "--source-data",
        required=True,
        help="the column containing the data we will dynamically lag",
    )
    parser.add(
        "-acs",
        "--additional-cols",
        action="append",
        help="specify additional column to read from the data file",
    )
    # geo information
    parser.add(
        "--site-a-lat", type=float, required=True, help="latitude coordinate of site A"
    )
    parser.add(
        "--site-a-lon", type=float, required=True, help="longitude coordinate of site A"
    )
    parser.add(
        "--site-b-lat", type=float, required=True, help="latitude coordinate of site B"
    )
    parser.add(
        "--site-b-lon", type=float, required=True, help="longitude coordinate of site B"
    )
    parser.add(
        "-pr",
        "--planet-radius",
        type=float,
        required=True,
        help="radius in meters of the planet",
    )
    # noise config
    parser.add(
        "-sp",
        "--skip-perturbations",
        default=False,
        help="whether to skip simulating perturbations",
    )
    parser.add(
        "-snr",
        "--snr-db",
        type=float,
        required=True,
        help="the desired signal to noise ratio in dB",
    )
    parser.add(
        "-rnl",
        "--rn-comp-length",
        type=int,
        help="The desired component length if composite red noise is sought",
    )
    # output config
    parser.add(
        "-t",
        "--target-name",
        required=True,
        help="what to name column containing dynamically lagged data",
    )
    parser.add(
        "-oc",
        "--output-cols",
        action="append",
        help="specify specific columns to output",
    )
    parser.add("-o", "--output-path", required=True, help="where to save the output")
    # finally parse }}}
    args = parser.parse_args()

    print("[INFO] Reading data")
    data_df = pd.read_csv(
        args.input_path,
        usecols=[args.index, args.wind_speed, args.wind_direction, args.source_data]
        + args.additional_cols,
        parse_dates=True,
        index_col=args.index,
    )
    data_df.index.freq = pd.infer_freq(data_df.index)

    print("[INFO] Processing geographical data")
    source_lat_lon = np.array([args.site_a_lat, args.site_a_lon])
    target_lat_lon = np.array([args.site_b_lat, args.site_b_lon])
    coords_df = pd.DataFrame(
        np.array([source_lat_lon, target_lat_lon]),
        columns=["Latitude", "Longitude"],
        index=["Source", "Target"],
    )
    # Calculate great circle distance to Target
    orientation_df = coords_df.copy()
    orientation_df["GCD to Target [m]"] = coords_df.apply(
        lambda x: ncl.gis.great_circle_distance(
            x, coords_df.loc["Target"][["Latitude", "Longitude"]], args.planet_radius
        ),
        axis=1,
    )
    # Calculate initial bearing to Target
    orientation_df["Initial Bearing to Target [radians]"] = coords_df.apply(
        lambda x: ncl.gis.initial_bearing(
            x, coords_df.loc["Target"][["Latitude", "Longitude"]]
        ),
        axis=1,
    )
    # Get Unit vector from bearing. Use inverted trig because bearing is measured from North
    orientation_df["bearing_i"] = [
        np.sin(orientation_df.loc["Source"]["Initial Bearing to Target [radians]"]),
        0,
    ]
    orientation_df["bearing_j"] = [
        np.cos(orientation_df.loc["Source"]["Initial Bearing to Target [radians]"]),
        0,
    ]

    print("[INFO] Calculating wind contribution")
    # Unit vector components of wind. Use inverted trig because bearing is measured from North
    data_df["wind_vector_i"] = np.sin(data_df[args.wind_direction])
    data_df["wind_vector_j"] = np.cos(data_df[args.wind_direction])
    # dot product
    data_df["wind_alignment"] = np.multiply(
        orientation_df.loc["Source"][["bearing_i", "bearing_j"]].values,
        data_df[["wind_vector_i", "wind_vector_j"]].values,
    ).sum(axis=1)

    print("[INFO] Dynamically lagging data")
    data_df[args.target_name] = ncl.dynlag.dynamically_lag(
        data_df[args.source_data],
        data_df[args.wind_speed],
        data_df["wind_alignment"],
        orientation_df.loc["Source"]["GCD to Target [m]"],
    )

    if not args.skip_perturbations:
        print("[INFO] Simulating perturbations")
        data_df[args.target_name] = ncl.dynlag.simulate_perturbations(
            data_df[args.target_name], args.snr_db, args.rn_comp_length
        )[0]

    if args.mask_path:
        print("[INFO] Applying mask")
        mask_array = np.load(args.mask_path)
        data_df = data_df.where(mask_array)

    print("[INFO] Saving to disk")
    data_df[args.output_cols].dropna().to_csv(
        args.output_path,
        float_format="%g",
        index_label=args.index,
    )
    print("[INFO] Done.")


def run_help():
    print("usage: nowcastlib [-h|triangulate] [subcommand *args]")


def main():
    """
    Function for organizing subcommands and providing help to user.
    """
    command = None
    if len(sys.argv) > 1:
        command = sys.argv[1]
    if command == None or command == "-h" or command == "--help":
        run_help()
    if command == "triangulate":
        triangulate()
