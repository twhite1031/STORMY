"""STORMY's public API, loaded on demand.

Importing :mod:`STORMY` itself stays lightweight. Optional dependencies are loaded
only when a downloader, radar helper, or plotting helper is actually requested.
"""

from importlib import import_module


name = "STORMY"

_DOWNLOAD_FUNCTIONS = {
    "STORMY_downloader",
    "download_ASOS",
    "download_ASOS_STATES",
    "download_ERA5PRESSURE",
    "download_ERA5SINGLE",
    "download_ERA5_SINGLE",
    "download_GOES",
    "download_LMA",
    "download_MRMS",
    "download_NWSSOUNDING",
    "download_NWS_SOUNDING",
    "download_SENTINEL",
    "download_WSR88D",
}
_RADAR_FUNCTIONS = {"parse_filename_datetime_obs"}
_WRF_ORGANIZATION_FUNCTIONS = {
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

# Preserve the package's documented star-import surface while resolving each name
# lazily instead of importing the entire dependency tree at package import time.
__all__ = [
    "STORMY_downloader",
    "download_GOES",
    "download_WSR88D",
    "download_LMA",
    "download_NWSSOUNDING",
    "download_NWS_SOUNDING",
    "download_MRMS",
    "download_ASOS",
    "download_ASOS_STATES",
    "download_ERA5SINGLE",
    "download_ERA5_SINGLE",
    "download_ERA5PRESSURE",
    "download_SENTINEL",
]


def __getattr__(attribute_name):
    """Load a public helper only when code first accesses it."""
    if attribute_name in _DOWNLOAD_FUNCTIONS:
        module = import_module(".downloads.download_data_V2", __name__)
    elif attribute_name in _RADAR_FUNCTIONS:
        module = import_module(".radar.radarfuncs", __name__)
    elif attribute_name in _WRF_ORGANIZATION_FUNCTIONS:
        module = import_module(".WRF.organization", __name__)
    elif attribute_name in _WRF_FUNCTIONS:
        module = import_module(".WRF.wrffuncs", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attribute_name!r}")

    attribute = getattr(module, attribute_name)
    globals()[attribute_name] = attribute
    return attribute


def __dir__():
    public_functions = (
        _DOWNLOAD_FUNCTIONS
        | _RADAR_FUNCTIONS
        | _WRF_ORGANIZATION_FUNCTIONS
        | _WRF_FUNCTIONS
    )
    return sorted(set(globals()) | public_functions)
