"""WRF helpers with lazy loading for optional scientific dependencies."""

from importlib import import_module


name = "WRF"

_ORGANIZATION_FUNCTIONS = {
    "build_time_df",
    "generate_wrf_filenames",
    "get_timeidx",
    "get_timeidx_and_wrf_file",
    "parse_filename_datetime_wrf",
    "parse_wrfout_time",
    "round_to_nearest_5_minutes",
}
_WRF_FUNCTIONS = {
    "add_cartopy_features",
    "create_gif",
    "find_closest_radar_file",
    "format_gridlines",
    "get_LMA_flash_data",
    "get_nws_cmap_norm",
    "make_contour_levels",
}

__all__ = sorted(_ORGANIZATION_FUNCTIONS | _WRF_FUNCTIONS)


def __getattr__(attribute_name):
    """Load only the module that owns the requested WRF helper."""
    if attribute_name in _ORGANIZATION_FUNCTIONS:
        module = import_module(".organization", __name__)
    elif attribute_name in _WRF_FUNCTIONS:
        module = import_module(".wrffuncs", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attribute_name!r}")

    attribute = getattr(module, attribute_name)
    globals()[attribute_name] = attribute
    return attribute


def __dir__():
    return sorted(set(globals()) | set(__all__))
