"""Lightweight helpers for organizing and locating WRF output files.

This module intentionally keeps optional scientific dependencies out of module-level
imports.  It can therefore be imported in small scripts that only need filename and
time-index bookkeeping, without importing the downloader, plotting, or radar stack.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta


__all__ = [
    "build_time_df",
    "generate_wrf_filenames",
    "get_timeidx",
    "get_timeidx_and_wrf_file",
    "parse_filename_datetime_wrf",
    "parse_wrfout_time",
    "round_to_nearest_5_minutes",
]


def round_to_nearest_5_minutes(date_time):
    """Round a datetime to the nearest five-minute boundary."""
    nearest_5 = round(date_time.minute / 5) * 5

    if nearest_5 == 60:
        date_time += timedelta(hours=1)
        nearest_5 = 0

    return date_time.replace(minute=nearest_5, second=0, microsecond=0)


def parse_filename_datetime_wrf(filepath, timeidx, timeidx_interval=5):
    """Return the valid datetime represented by a WRF filename and time index."""
    datetime_format = "wrfout_d02_%Y-%m-%d_%H_%M_%S"
    file_time = datetime.strptime(os.path.basename(filepath), datetime_format)
    return file_time + timedelta(minutes=timeidx_interval * int(timeidx))


def generate_wrf_filenames(
    start_time,
    wrf_date_time_end,
    file_interval,
    numtimeidx,
    domain=1,
    wrf_start_hour=0,
):
    """Generate WRF filenames and time indices between two datetimes.

    NumPy is imported only when this function is called, preserving the historical
    array return type while keeping module import inexpensive.
    """
    import numpy as np

    filenames = []
    time_indices = []
    current_time = start_time

    while current_time <= wrf_date_time_end:
        wrf_start_time = current_time.replace(
            hour=wrf_start_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        elapsed_minutes = (current_time - wrf_start_time).total_seconds() / 60
        wrf_offset = (elapsed_minutes // file_interval) * file_interval
        wrf_filename_time = wrf_start_time + timedelta(minutes=wrf_offset)

        time_offset = (current_time - wrf_filename_time).total_seconds() / 60
        time_step = file_interval // numtimeidx
        time_index = int(time_offset // time_step)

        filenames.append(
            f"wrfout_d0{domain}_{wrf_filename_time:%Y-%m-%d_%H:%M}:00"
        )
        time_indices.append(time_index)
        current_time += timedelta(minutes=5)

    return np.asarray(filenames), np.asarray(time_indices, dtype=int)


def get_timeidx_and_wrf_file(
    date_time,
    file_interval_sec,
    numtimeidx,
    domain=1,
    wrf_start_hour=0,
):
    """Return the WRF filename and in-file index for a datetime."""
    wrf_start_time = date_time.replace(
        hour=wrf_start_hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    elapsed_seconds = (date_time - wrf_start_time).total_seconds()
    wrf_offset_seconds = int(elapsed_seconds // file_interval_sec) * file_interval_sec
    wrf_filename_time = wrf_start_time + timedelta(seconds=wrf_offset_seconds)

    time_offset_seconds = (date_time - wrf_filename_time).total_seconds()
    time_step_seconds = file_interval_sec / numtimeidx
    time_index = int(time_offset_seconds // time_step_seconds)
    pattern = f"wrfout_d0{domain}_{wrf_filename_time:%Y-%m-%d_%H:%M:%S}"

    return time_index, pattern


def get_timeidx(wrf_date_time, file_interval, numtimeidx):
    """Return the in-file WRF time index for a datetime."""
    return int(
        (wrf_date_time.minute % file_interval)
        // (file_interval // numtimeidx)
    )


def build_time_df(path, domain):
    """Build or load a dataframe that maps WRF valid times to files and indices.

    The heavier WRF, netCDF4, and pandas packages are needed only when this helper
    actually reads model output, so they are deliberately imported here.
    """
    import pandas as pd

    wrf_files = sorted(glob.glob(os.path.join(path, f"wrfout_d0{domain}_*")))
    time_cache = os.path.join(path, f"wrfD{domain}_time_lookup.pkl")

    if os.path.exists(time_cache):
        return pd.read_pickle(time_cache)

    # These packages are only required when uncached WRF files must be inspected.
    from netCDF4 import Dataset
    from wrf import extract_times

    records = []
    for filename in wrf_files:
        with Dataset(filename) as dataset:
            times = pd.to_datetime(extract_times(dataset, timeidx=None))
            records.extend(
                (filename, time_index, valid_time)
                for time_index, valid_time in enumerate(times)
            )

    time_dataframe = pd.DataFrame(
        records,
        columns=["filename", "timeidx", "time"],
    )
    time_dataframe.to_pickle(time_cache)
    return time_dataframe


def parse_wrfout_time(filename):
    """Parse a WRF filename into filename-safe and human-readable timestamps."""
    try:
        datetime_string = filename.split("_d0")[1].split("_", 1)[1]
    except IndexError as error:
        raise ValueError(
            "Expected format: wrfout_d0X_YYYY-MM-DD_HH:MM:SS or _HH_MM_SS"
        ) from error

    for datetime_format in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H_%M_%S"):
        try:
            parsed_time = datetime.strptime(datetime_string, datetime_format)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unrecognized datetime format in filename: {datetime_string}")

    return (
        parsed_time.strftime("%Y%m%d_%H%M"),
        parsed_time.strftime("%Y-%m-%d %H:%M UTC"),
    )
