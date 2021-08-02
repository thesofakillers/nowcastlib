"""
Module containing custom structures used throughout the pipeline
submodule to aid with configuration.

A `Union` type annotation indicates that the variable
can be of any of the types listed in the `Union` type.
Variables whose type is `Union` with `NoneType` are optional.

Tuples are used instead of lists so to allow for hashing
of the struct instances. These should be treated as lists
when using .json files for specifying configuration.
"""
from typing import Union, Tuple, Optional, Dict, Callable
from enum import Enum
from attr import attrs, attrib, validators
import numpy as np
import pandas as pd


def _enforce_npy(instance, attribute, value):
    """ensures that the `output_format` key is `npy`"""
    if value is not None and value.output_format != "npy":
        raise ValueError(
            "'{0}'.output_path of the '{1}' instance needs to be `npy`"
            " A value of `{2}` was passed instead.".format(
                attribute.name, instance.__class__.__name__, value.output_format
            )
        )


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
    \nIf `None`, `window_size` refers to the number of samples comprising a window.
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


brainstorming_dataset = {
    "additional_postprocessing": {
        "new_fields": [{"generator": "time_since_sunset", "placeholder": "TODO"}],
        "standardize-fields": [{"field_name": "jkadshfla", "std_options": {}}],
    }
}


class GeneratorFunction(Enum):
    """Enumeration of the available Generator Functions"""

    T_SINCE_SUNSET = "t_since_sunset"
    """seconds elapsed since the last sunset"""
    SIN_T_SINCE_SUNSET = "sin_t_since_sunset"
    """sine of seconds elapsed since the last sunset out of 86400"""
    COS_T_SINCE_SUNSET = "cos_t_since_sunset"
    """cosine of seconds elapsed since the last sunset out of 86400"""
    SUN_ELEVATION = "sun_elevation"
    """the sun's current elevation"""
    SIN_SEC = "sin_sec"
    """sine of the second number in the current day out of 86400"""
    COS_SEC = "cos_sec"
    """cosine of the second number in the current day out of 86400"""
    SIN_DAY_YEAR = "sin_day_year"
    """sine of the day number out of 365 in the current year"""
    COS_DAY_YEAR = "cos_day_year"
    """cosine of the day number out of 365 in the current year"""
    SIN_DAY_WEEK = "sin_day_week"
    """sine of the day number out of 7 in the current week"""
    COS_DAY_WEEK = "cos_day_week"
    """cosine of the day number out of 7 in the current week"""
    SIN_MONTH_YEAR = "sin_month_year"
    """sine of the month number out of 12 in the current year"""
    COS_MONTH_YEAR = "cos_month_year"
    """cosine of the month number out of 12 in the current year"""
    IS_WEEKEND = "is_weekend"
    """whether the current day is a friday, saturday or sunday"""
    CUSTOM = "custom"
    """indicates the user will provide their own function"""


class StandardizationMethod(Enum):
    """Enumeration of the available standardization methods"""

    POWER = "power"
    """A power transform of the data (Yeo-Johnson)"""
    ROBUST = "robust"
    """
    Rescales the data making use of its interquartile range.
    \nSee https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    """
    LOGNORM = "lognorm"
    """Takes the logarithm of the data"""


@attrs(kw_only=True, frozen=True)
class StandardizationOptions:
    """
    Struct containing configuration options
    for standardizing a given field.
    """

    method: StandardizationMethod = attrib()
    """
    Which of the available methods to use. Specify as the
    Enum lowercase string value when configuring via JSON
    """
    diagnostic_plots: bool = attrib(default=True)
    """
    Whether or not to show diagnostic plots, intended to help
    the user in configuration evaluation and decision making.
    """


@attrs(kw_only=True, frozen=True)
class BaseField:
    """
    Struct containing configuration options
    shared by both Raw and Generated fields.
    """

    std_options: Optional[StandardizationOptions] = attrib(default=None)
    """
    Configuration options for standardizing
    (scaling or normalizing) the field
    """


@attrs(kw_only=True, frozen=True)
class GeneratedField(BaseField):
    """
    Struct containing configuration for specifying
    how the pipeline should generate a new field of data
    """

    target_name: str = attrib()
    """
    What the new field should be named.
    """
    input_fields: Tuple[str] = attrib()
    """
    The names of the input fields to pass to the
    generator function as *args.
    "index", to specify the index
    """
    gen_func: GeneratorFunction = attrib()
    """
    The name of the generator function to use
    for generating the new data. Specify as the
    Enum lowercase string value when configuring via JSON.
    """
    additional_kwargs: Optional[dict] = attrib(default=None)
    """
    a dictionary containing additional keyword arguments to
    to be passed to the function if necessary.
    """
    func_path: Optional[str] = attrib(default=None)
    """
    The path to the file implementing a custom
    generator function. To be specified if `gen_func` is `custom`
    """

    @func_path.validator
    def only_if_custom(self, attribute, value):
        """func_path should be defined only if the function is set to custom"""
        if value is not None:
            if self.gen_func != GeneratorFunction.CUSTOM:
                raise ValueError(
                    "'{0}' of the '{1}' instance should only be defined when the "
                    " instance's `gen_func` is set to `GeneratorFunction.CUSTOM`."
                    " A value of {2} was passed instead.".format(
                        attribute.name, self.__class__.__name__, self.gen_func
                    )
                )


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
    Does not do anything when postprocessing.
    """

    @overwrite.validator
    def prevent_overwrite(self, attribute, value):
        """cannot overwrite if performing smoothing"""
        if value is True:
            if self.smooth_options is not None:
                raise ValueError(
                    "'{0}' of the '{1}' instance needs to be `False`"
                    " to perform smoothing. A value of {2} was passed instead.".format(
                        attribute.name, self.__class__.__name__, value
                    )
                )

    outlier_options: Optional[OutlierOptions] = attrib(default=None)
    """
    Configuration options for specifying which outliers to drop.
    Is performed before any unit conversion.
    \nIf `None`, no outlier removal is performed.
    """
    periodic_options: Optional[PeriodicOptions] = attrib(default=None)
    """
    Configuration options for treating data that is periodic in nature,
    such as normalizing the desired range of values.
    Is performed before any unit conversion.
    \nIf `None`, no processing in this regard is performed.
    """
    conversion_options: Optional[ConversionOptions] = attrib(
        default=None,
    )
    """
    Configuration options for converting a field from one unit to another
    \nIf `None`, no conversion is performed.
    """
    smooth_options: Optional[SmoothOptions] = attrib(default=None)
    """
    Configuration options for smoothing the field.
    Is performed at the end of all other processing.
    \nIf `None`, no smoothing is performed.
    """


@attrs(kw_only=True, frozen=True)
class RawField(BaseField):
    """
    Struct containing configuration attributes for a raw field
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
    \nIf `None`, no preprocessing will be performed.
    """
    postprocessing_options: Optional[ProcessingOptions] = attrib(default=None)
    """
    Configuration options for how to post-process the field
    \nIf `None`, no post-processing will be performed.
    """


@attrs(kw_only=True, frozen=True)
class SerializationOptions:
    """
    Struct containing configuration attributes for
    serializing a given DataSource to disk
    """

    output_format: str = attrib(validator=validators.in_(["csv", "pickle", "npy"]))
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
    fields: Tuple[RawField, ...] = attrib()
    """Configuration options for each field the user is interested in"""
    comment_format: str = attrib(default="#")
    """Prefix used in csv file to signal comments, that will be dropped when reading"""
    preprocessing_output: Optional[SerializationOptions] = attrib(default=None)
    """
    Configuration options for saving the preprocessing results to disk.
    \nIf `None`, no serialization of the preprocessing results will be performed.
    """

    @fields.validator
    def _exactly_one_date(self, attribute, value):
        """checks whether maximum one of the fields contains date information"""
        date_flags = [field.is_date for field in value if field.is_date]
        if len(date_flags) > 1:
            raise ValueError(
                "{0} of the {1} instance must contain exactly one RawField with"
                " is_date=True".format(attribute.name, self.__class__.__name__)
            )


@attrs(kw_only=True, frozen=True)
class ChunkOptions:
    """
    Struct containing configuration attributes for chunking
    a partially synchronized DataSet
    """

    max_gap_size: int = attrib()
    """
    The maximum amount of time in seconds for a gap to be ignored
    """
    min_chunk_size: int = attrib()
    """
    The minimum length in seconds for contiguous block of data
    to be considered.
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
    \nIf `None`, no re-sampling will be performed.
    """
    chunk_options: ChunkOptions = attrib()
    """
    Configuration options necessary for handling chunking operations.
    """
    data_output: Optional[SerializationOptions] = attrib(default=None)
    """
    Configuration options for saving the resulting
    synchronized dataframe to disk.
    \nIf `None`, no serialization of the preprocessing results will be performed.
    """
    chunks_output: Optional[SerializationOptions] = attrib(
        default=None, validator=_enforce_npy
    )
    """
    Configuration options for saving the detected
    chunk locations to disk. Only 'npy' output_format
    is accepted.
    \nIf `None`, no serialization of the preprocessing results will be performed.
    """
    diagnostic_plots: bool = attrib(default=True)
    """
    Whether or not to show diagnostic plots, intended to help
    the user in configuration evaluation and decision making.
    """


@attrs(kw_only=True, frozen=True)
class ValidationOptions:
    """
    Struct containing configuration attributes for
    configuring model validation
    """

    train_extent: float = attrib(default=0.6, validator=[_normed_val])
    """
    Percentage of the data to allocate to the training set.
    """
    val_extent: float = attrib(default=0.1, validator=[_normed_val])
    """
    Percentage of the data to allocate to the validation set.
    """
    iterations: int = attrib(default=5)
    """
    How many splits to make. Must be at least 3.
    """

    @iterations.validator
    def at_least_3(self, attribute, value):
        """ensures at least 3 iterations are used for validation"""
        if value < 3:
            raise ValueError(
                "'{0}' of the '{1}' instance needs to be at least 3."
                " A value of {2} was passed instead.".format(
                    attribute.name, self.__class__.__name__, value
                )
            )


@attrs(kw_only=True, frozen=True)
class DirSerializationOptions:
    """
    Struct configuration attributes for serializing
    to specific directories, used for organizing splits
    """

    parent_path: str = attrib()
    """
    The path to the directory where to serialize the resulting splits.
    Within the directory, 2 subdirectories will be created: `main_split/`
    and `cv_split`, respectively storing the main split and the cross
    validation split.
    """
    overwrite: bool = attrib(default=False)
    """
    Whether to overwrite existing directories and files if they exist
    already. Default `False`
    """
    create_parents: bool = attrib(default=False)
    """
    Whether parent directories of `parent_path` should be created
    if they do not exist
    """


@attrs(kw_only=True, frozen=True)
class SplitOptions:
    """
    Struct containing configuration attributes for
    model evaluation
    """

    train_split: Union[int, float, str] = attrib()
    """
    The index, percentage or date to use as the final
    point in the training set. The closest non-nan row will be used.
    """
    validation: ValidationOptions = attrib()
    """
    Configuration options for further splits of the data for validation.
    """
    output_options: Optional[DirSerializationOptions] = attrib(default=None)
    """
    Configuration options for serializing the resulting splits
    in organized directories.
    \nIf `None`, no serialization will be performed.
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
    \nIf `None`, no synchronization will be performed
    """
    split_options: Optional[SplitOptions] = attrib(default=None)
    """
    Configurations options for handling data splitting.
    \nIf `None`, no splitting will be performed
    """
    generated_fields: Optional[Tuple[GeneratedField]] = attrib(default=None)
    """
    Configuration options for adding new fields to the data.
    \nIf `None`, no new fields will be computed.
    """
