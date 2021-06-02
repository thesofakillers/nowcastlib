"""
module containing custom structures used throughout the library
"""
from typing import List, Optional
from attr import attrs, attrib, validators


def normed_validator(instance, attribute, value):
    """Checks whether a given value is between 0 and 1"""
    if not 0 <= value <= 1:
        raise ValueError(
            "'{0}' of the '{1}' instance needs to be in the range [0, 1]."
            " A value of {2} was passed instead.".format(
                attribute.name, instance.__class__.__name__, value
            )
        )


@attrs
class ScaleOptions:
    """
    Struct containing configuration options for the scaling of a
    given data column

    TODO
    """


@attrs
class PeriodicOptions:
    """
    Struct containing configuration options for the scaling of a
    given data column

    TODO
    """


@attrs
class OutlierOptions:
    """
    Struct containing outlier handling configuration options
    of a given data field

    WIP
    """

    lower: float = attrib(default=0, validator=normed_validator)
    higher: float = attrib(default=1, validator=normed_validator)

    @higher.validator
    def higher_gt_lower(self, attribute, value):
        """validates whether higher > lower"""
        if value <= self.lower:
            raise ValueError(
                "{0} of the {1} instance must be greater than the instance's"
                " 'lower' attribute".format(attribute.name, self.__class__.__name__)
            )


@attrs
class DataField:
    """
    Struct containing configuration attributes for a given field
    of a given data source

    WIP
    """

    # pylint: disable=too-many-instance-attributes

    field_name: str = attrib()
    is_date: bool = attrib(default=False)
    date_format: str = attrib(default="%Y-%m-%dT%H:%M:%S")
    smooth_window: Optional[int] = attrib(default=None)
    scale_options: Optional[ScaleOptions] = attrib(default=None)
    periodic_options: Optional[PeriodicOptions] = attrib(default=None)
    outlier_options: Optional[OutlierOptions] = attrib(default=None)
    conversion: Optional[str] = attrib(
        default=None, validator=validators.in_([None, "mph2ms", "deg2rad", "rad2deg"])
    )


@attrs
class DataSourceConfig:
    """
    Struct containing configuration attributes for processing
    an individual Data Source

    WIP
    """

    name: str = attrib()
    path: str = attrib()
    fields: List[DataField] = attrib()
    comment_format: str = attrib(default="#")

    @fields.validator
    def exactly_one_date(self, attribute, value):
        """checks whether maximum one of the fields contains date information"""
        date_flags = [field.is_date for field in value if field.is_date]
        if len(date_flags) > 1:
            raise ValueError(
                "{0} of the {1} instance must contain exactly one DataField with"
                " is_date=True".format(attribute.name, self.__class__.__name__)
            )
