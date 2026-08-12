"""Tests for the dependency-light WRF organization API."""

import subprocess
import sys
import unittest
from datetime import datetime

from STORMY.WRF.organization import (
    get_timeidx_and_wrf_file,
    parse_wrfout_time,
    round_to_nearest_5_minutes,
)


class WRFOrganizationTests(unittest.TestCase):
    def test_finds_file_and_time_index(self):
        time_index, filename = get_timeidx_and_wrf_file(
            datetime(2022, 11, 18, 13, 50),
            file_interval_sec=3600,
            numtimeidx=12,
            domain=2,
        )

        self.assertEqual(time_index, 10)
        self.assertEqual(filename, "wrfout_d02_2022-11-18_13:00:00")

    def test_parses_colon_and_underscore_filenames(self):
        expected = ("20221118_1300", "2022-11-18 13:00 UTC")

        self.assertEqual(
            parse_wrfout_time("wrfout_d02_2022-11-18_13:00:00"),
            expected,
        )
        self.assertEqual(
            parse_wrfout_time("wrfout_d02_2022-11-18_13_00_00"),
            expected,
        )

    def test_rounds_across_hour_boundary(self):
        rounded = round_to_nearest_5_minutes(datetime(2022, 11, 18, 13, 58, 30))

        self.assertEqual(rounded, datetime(2022, 11, 18, 14, 0))

    def test_package_import_does_not_load_optional_stacks(self):
        import_check = """
import sys
import STORMY
from STORMY import get_timeidx_and_wrf_file

blocked_modules = {
    'STORMY.downloads.download_data_V2',
    'STORMY.WRF.wrffuncs',
    'cartopy',
    'netCDF4',
    'wrf',
}
loaded_blocked_modules = blocked_modules.intersection(sys.modules)
if loaded_blocked_modules:
    raise SystemExit(f'Unexpected eager imports: {sorted(loaded_blocked_modules)}')
"""
        completed_process = subprocess.run(
            [sys.executable, "-c", import_check],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed_process.returncode,
            0,
            completed_process.stderr or completed_process.stdout,
        )


if __name__ == "__main__":
    unittest.main()
