"""
Module containing custom structures used throughout the pipeline submodule.
Classes listed here should be viewed as dictionaries, with class variables
being treated analogous to dictionary keys.
"""
from typing import Tuple, Optional, Dict, Callable
from attr import attrs, attrib, validators
import numpy as np
import pandas as pd


def _normed_val(instance, attribute, value):
    """Checks whether a given value is between 0 and 1"""
    if not 0 <= value <= 1:
        raise ValueError(
            "'{0}' of the '{1}' instance needs to be in the range [0, 1]."
            " A value of {2} was passed instead.".format(
                attribute.name, instance.__class__.__name__, value
            )
        )


def _normed_outlier_val(instance, attribute, value):
    """Runs normed_validator if the outlier is quantile based"""
    if instance.quantile_based:
        _normed_val(instance, attribute, value)


@attrs(kw_only=True, frozen=True)
class ConversionOptions:
    """
    Struct containing configuration options for the unit
    conversion of a given data field
    """

    _conv_map: Dict[str, Callable] = {
        "mph2ms": (lambda x: 0.44704 * x),
        "deg2rad": np.deg2rad,
        "rad2deg": np.rad2deg,
    }
    key: str = attrib(validator=validators.in_([*_conv_map.keys()]))
    """
    One of 'mph2ms', 'deg2rad' or 'rad2deg' to specify what unit
    conversion to perform
    """

    def conv_func(self, input_series):
        """Function to use for converting the series as set by the key attribute"""
        return self._conv_map[self.key](input_series)


@attrs(kw_only=True, frozen=True)
class PeriodicOptions:
    """
    Struct containing configuration options for the scaling of a
    given data field
    """

    period_length: int = attrib()
    """
    The sample number at which the signal starts repeating
    """


@attrs(kw_only=True, frozen=True)
class OutlierOptions:
    """
    Struct containing outlier handling configuration options
    of a given data field
    """

    lower: float = attrib(default=0, validator=_normed_outlier_val)
    """Lower inclusive (percentile) threshold, eliminating numbers lower than it"""
    upper: float = attrib(default=1, validator=_normed_outlier_val)
    """Upper inclusive (percentile) threshold, eliminating numbers greater than it"""
    quantile_based: bool = attrib(default=True)
    """
    Whether the lower and higher attributes are referring to quantiles.
    If `False`, `lower` and `higher` are treated as absolute thresholds.
    """

    @upper.validator
    def _upper_gt_lower(self, attribute, value):
        """validates whether higher > lower"""
        if value <= self.lower:
            raise ValueError(
                "{0} of the {1} instance must be greater than the instance's"
                " 'lower' attribute".format(attribute.name, self.__class__.__name__)
            )


@attrs(kw_only=True, frozen=True)
class SmoothOptions:
    """
    Struct containing data smoothing configuration options
    of a given data field, achieved with a moving average operation.
    """

    window_size: int = attrib()
    """How large the window should be for a moving average operation"""
    units: Optional[str] = attrib(default=None)
    """
    What units `window_size` is given in. Should be compatible with
    [pandas offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
    If `None`, `window_size` refers to the number of samples comprising a window.
    """

    @units.validator
    def _check_pd_offset_alias(self, attribute, value):
        """checks whether the unit attribute is a valid pandas Offset alias"""
        if value is not None:
            try:
                pd.tseries.frequencies.to_offset(value)
            except ValueError as invalid_freq:
                error_string = (
                    "{0} of the {1} instance must be an Offset Alias string"
                    " as specified at"
                    " https://pandas.pydata.org/pandas-docs"
                    "/stable/user_guide/timeseries.html#offset-aliases".format(
                        attribute.name, self.__class__.__name__
                    )
                )
                raise ValueError(error_string) from invalid_freq


@attrs(kw_only=True, frozen=True)
class ProcessingOptions:
    """
    Struct containing configuration attributes for processing
    a given field of a given data source
    """

    overwrite: bool = attrib(default=False)
    """
    If `True`, overwrites the input field in the input dataframe.
    Otherwise appends a new field to the dataframe.
    """
    outlier_options: Optional[OutlierOptions] = attrib(default=None)
    """
    Configuration options for specifying which outliers to drop.
    Is performed before any unit conversion.
    If `None`, no outlier removal is performed.
    """
    periodic_options: Optional[PeriodicOptions] = attrib(default=None)
    """
    Configuration options for treating data that is periodic in nature,
    such as normalizing the desired range of values.
    Is performed before any unit conversion.
    If `None`, no processing in this regard is performed.
    """
    conversion_options: Optional[ConversionOptions] = attrib(
        default=None,
    )
    """
    Configuration options for converting a field from one unit to another
    If `None`, no conversion is performed.
    """
    smooth_options: Optional[SmoothOptions] = attrib(default=None)
    """
    Configuration options for smoothing the field.
    Is performed at the end of all other processing.
    If `None`, no smoothing is performed.
    """


@attrs(kw_only=True, frozen=True)
class DataField:
    """
    Struct containing configuration attributes for a given field
    of a given DataSource
    """

    field_name: str = attrib()
    """The name of the field as specified in the input file"""
    is_date: bool = attrib(default=False)
    """Whether the field is a date and therefore is the index of the DataSource"""
    date_format: str = attrib(default="%Y-%m-%dT%H:%M:%S")
    """What format the date is presented in if the field is a date"""
    preprocessing_options: Optional[ProcessingOptions] = attrib(default=None)
    """
    Configuration options for how to pre-process the field
    If `None`, no preprocessing will be performed.
    """
    postprocessing_options: Optional[ProcessingOptions] = attrib(default=None)
    """
    Configuration options for how to post-process the field
    If `None`, no post-processing will be performed.
    """


@attrs(kw_only=True, frozen=True)
class SerializationOptions:
    """
    Struct containing configuration attributes for
    serializing a given DataSource to disk
    """

    output_format: str = attrib(validator=validators.in_(["csv", "pickle"]))
    """
    One of 'csv', or 'pickle' to specify what format
    to save the DataSource as
    """
    output_path: str = attrib()
    """
    The desired path to the output file, including the name.
    Folders containing the output file should exist before running.
    """


@attrs(kw_only=True, frozen=True)
class DataSource:
    """
    Struct containing configuration attributes for processing
    an individual Data Source
    """

    name: str = attrib()
    """The name of the DataSource. Somewhat arbitrary but useful for legibility"""
    path: str = attrib()
    """The path to the csv file from which to read the data"""
    fields: Tuple[DataField, ...] = attrib()
    """Configuration options for each field the user is interested in"""
    comment_format: str = attrib(default="#")
    """Prefix used in csv file to signal comments, that will be dropped when reading"""
    preprocessing_output: Optional[SerializationOptions] = attrib(default=None)
    """
    Configuration options for saving the preprocessing results to disk.
    If `None`, no serialization of the preprocessing results will be performed.
    """

    @fields.validator
    def _exactly_one_date(self, attribute, value):
        """checks whether maximum one of the fields contains date information"""
        date_flags = [field.is_date for field in value if field.is_date]
        if len(date_flags) > 1:
            raise ValueError(
                "{0} of the {1} instance must contain exactly one DataField with"
                " is_date=True".format(attribute.name, self.__class__.__name__)
            )


@attrs(kw_only=True, frozen=True)
class ChunkOptions:
    """
    Struct containing configuration attributes for chunking
    a partially synchronized DataSet
    """

    min_gap_size: int = attrib()
    """
    The minimum amount of time in seconds for a gap to be considered
    large enough to not be ignored.
    """
    min_chunk_size: int = attrib()
    """
    The minimum length in seconds for contiguous block of data
    to be considered.
    """
    output_path: Optional[str] = attrib(default=None)
    """
    The path where to save resulting chunks as an hdf5 file.
    If `None`, no serialization will be performed.
    """


@attrs(kw_only=True, frozen=True)
class SyncOptions:
    """
    Struct containing configuration attributes for synchronizing
    a DataSet
    """

    sample_spacing: int = attrib()
    """
    The desired amount of time in seconds between each sample.
    If `None`, no re-sampling will be performed.
    """
    chunk_options: Optional[ChunkOptions] = attrib(default=None)
    """
    Configuration options necessary for handling chunking operations.
    If `None`, no chunking will be performed.
    """


@attrs(kw_only=True, frozen=True)
class DataSet:
    """
    Struct containing configuration attributes for processing
    a set of Data Sources
    """

    data_sources: Tuple[DataSource, ...] = attrib()
    """
    Configuration options for each of the sources of data we wish
    to process, each originating from a different file.
    """
    sync_options: Optional[SyncOptions] = attrib(default=None)
    """
    Configurations options for synchronizing the `data_sources`.
    If `None`, no synchronization will be performed
    """
