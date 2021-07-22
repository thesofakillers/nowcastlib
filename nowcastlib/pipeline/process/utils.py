"""
Shared functionality across pre and postprocessing
"""
from nowcastlib.pipeline import structs


def build_field_name(config: structs.ProcessingOptions, field_name: str):
    """
    Builds the appropriate field name depending on whether
    the user wishes to overwrite the current field or not

    Parameters
    ----------
    config : nowcastlib.pipeline.structs.ProcessingOptions
    field_name : str
        the name of the current field we are acting on

    Returns
    -------
    str
        the resulting string
    """
    if config.overwrite:
        computed_field_name = field_name
    else:
        computed_field_name = "processed_{}".format(field_name)
    return computed_field_name
