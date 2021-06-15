"""
module containing custom structures used throughout the pipeline submodule
"""
from typing import Tuple, Optional, Dict, Callable
from attr import attrs, attrib, validators
import numpy as np
import pandas as pd


def normed_val(instance, attribute, value):
    """Checks whether a given value is between 0 and 1"""
    if not 0 <= value <= 1:
        raise ValueError(
            "'{0}' of the '{1}' instance needs to be in the range [0, 1]."
            " A value of {2} was passed instead.".format(
                attribute.name, instance.__class__.__name__, value
            )
        )


def normed_outlier_val(instance, attribute, value):
    """Runs normed_validator if the outlier is quantile based"""
    if instance.quantile_based:
        normed_val(instance, attribute, value)


@attrs(kw_only=True, frozen=True)
class ConversionOptions:
    """
    Struct containing configuration options for the unit
    conversion of a given data field
    """

    CONV_MAP: Dict[str, Callable] = {
        "mph2ms": (lambda x: 0.44704 * x),
        "deg2rad": np.deg2rad,
        "rad2deg": np.rad2deg,
    }
    """
    dictionary for mapping conversion keys to conversion functions.
    Currently supports 'mph2ms', 'deg2rad' and 'rad2deg'
    """
    key: str = attrib(validator=validators.in_([*CONV_MAP.keys()]))

    def conv_func(self, input_series):
        """will be rewritten upon initialization"""
        return self.CONV_MAP[self.key](input_series)


@attrs(kw_only=True, frozen=True)
class PeriodicOptions:
    """
    Struct containing configuration options for the scaling of a
    given data field
    """

    period_length: int = attrib()


@attrs(kw_only=True, frozen=True)
class OutlierOptions:
    """
    Struct containing outlier handling configuration options
    of a given data field
    """

    quantile_based: bool = attrib(default=True)
    lower: float = attrib(default=0, validator=normed_outlier_val)
    higher: float = attrib(default=1, validator=normed_outlier_val)

    @higher.validator
    def higher_gt_lower(self, attribute, value):
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
    of a given data field
    """

    window_size: int = attrib()
    units: Optional[str] = attrib(default=None)

    @units.validator
    def check_pd_offset_alias(self, attribute, value):
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
    outlier_options: Optional[OutlierOptions] = attrib(default=None)
    periodic_options: Optional[PeriodicOptions] = attrib(default=None)
    conversion_options: Optional[ConversionOptions] = attrib(
        default=None,
    )
    smooth_options: Optional[SmoothOptions] = attrib(default=None)


@attrs(kw_only=True, frozen=True)
class DataField:
    """
    Struct containing configuration attributes for a given field
    of a given data source
    """

    # pylint: disable=too-many-instance-attributes

    field_name: str = attrib()
    is_date: bool = attrib(default=False)
    date_format: str = attrib(default="%Y-%m-%dT%H:%M:%S")
    preprocessing_options: Optional[ProcessingOptions] = attrib(default=None)
    postprocessing_options: Optional[ProcessingOptions] = attrib(default=None)


@attrs(kw_only=True, frozen=True)
class SerializationOptions:
    """
    Struct containing configuration attributes for
    serializing a given DataSource
    """

    output_format: str = attrib(validator=validators.in_(["csv", "pickle"]))
    output_path: str = attrib()


@attrs(kw_only=True, frozen=True)
class DataSource:
    """
    Struct containing configuration attributes for processing
    an individual Data Source
    """

    name: str = attrib()
    path: str = attrib()
    fields: Tuple[DataField, ...] = attrib()
    comment_format: str = attrib(default="#")
    preprocessing_output: Optional[SerializationOptions] = attrib(default=None)

    @fields.validator
    def exactly_one_date(self, attribute, value):
        """checks whether maximum one of the fields contains date information"""
        date_flags = [field.is_date for field in value if field.is_date]
        if len(date_flags) > 1:
            raise ValueError(
                "{0} of the {1} instance must contain exactly one DataField with"
                " is_date=True".format(attribute.name, self.__class__.__name__)
            )


@attrs(kw_only=True, frozen=True)
class DataSet:
    """
    Struct containing configuration attributes for processing
    a set of Data Sources
    """

    data_sources: Tuple[DataSource, ...] = attrib()
