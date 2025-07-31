import os
import warnings
import numpy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyart
from pyart.testing import get_test_data
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import radarfuncs
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd

'''
Provides the average wind direction every hour given a regional subset of ERA5 netCDF dataset
'''

# --- User Input ---
dates = ["20181110", "20200227", "20220110", "20220123", "20230319","20201102","20221117","20191231","20190305"]

# Select region (lat/lon), currently over Lake Ontario
lat_range = slice(44.15, 43.15)
lon_range = slice(-77.0, -76)

#savepath = f"/data2/white/PLOTS_FIGURES/MET399/{date}/"
# --- End User Input ---

event_mean_speeds = {}

for date in dates:
    ds = xr.open_dataset(f"/data2/white/DATA/MET399/ERA5/ERA5{date}.nc")
    start_time = ds["valid_time"][0].values
    end_time = ds["valid_time"][-1].values

    # Create region subset for data
    region_subset = ds.sel(
        latitude=lat_range,
        longitude=lon_range,
    )
    speed_over_time = []
    current_time = start_time
    current_time = pd.to_datetime(start_time).to_pydatetime()
    end_time = pd.to_datetime(end_time).to_pydatetime()
    while current_time <= end_time:

        # Grab u and v at desired time
        u = region_subset['u'].sel(pressure_level=850,valid_time=current_time)
        v = region_subset['v'].sel(pressure_level=850,valid_time=current_time)

        #Calculate wind speed at each grid point
        speed = np.sqrt(u**2 + v**2)

        # Spatial average wind speed (regardless of direction)
        speed_area_avg = speed.mean(dim=('latitude', 'longitude'))
        speed_over_time.append(speed_area_avg.values)
    
        # Calculate wind direction (in degrees FROM which the wind is blowing)
        wind_dir_rad = np.arctan2(-u, -v)
        wind_dir_deg = (np.degrees(wind_dir_rad) + 360) % 360  # [0, 360] range

        # Compute *mean* wind direction using unit vectors
        u_unit = np.sin(np.radians(wind_dir_deg))
        v_unit = np.cos(np.radians(wind_dir_deg))

        # Mean over space (lat/lon) and/or time
        u_mean = u_unit.mean(dim=('latitude', 'longitude'))
        v_mean = v_unit.mean(dim=('latitude', 'longitude'))
    
        speed_mean = np.sqrt(u_mean**2 + v_mean**2)
        # Final mean direction
        mean_dir_rad = np.arctan2(u_mean, v_mean)
        mean_dir_deg = (np.degrees(mean_dir_rad) + 360) % 360

        current_time += timedelta(hours=1)

    # After loop: compute overall mean speed
    speed_mean_over_event = np.mean(speed_over_time)
    event_mean_speeds[date] = speed_mean_over_event
    print(f"Mean wind speed over event {date}: {speed_mean_over_event:.2f} m/s")

print(event_mean_speeds)
