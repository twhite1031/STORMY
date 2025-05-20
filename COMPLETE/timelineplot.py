import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from datetime import datetime
import wrffuncs # Personal library
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as crs
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

"""
Creates a time line plot of the average of Planetary Boundary Layer (PBL) Height
between start time and end time for two simulations
"""

# ---- User input for file ----
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2
file_interval = 20
numtimeidx = 4

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/PLOTS_FIGURES/SEMINAR/BOTH_ATTEMPT/"

# --- Processing to determine which WRF files to use and create filelist/timeidxlist ---
files, timeidxlist = wrffuncs.generate_wrf_filenames(start_time, end_time, file_interval,numtimeidx, domain)
# Create our full paths here
filelist_N = [os.path.join(path_N, ending) for ending in files]
filelist_F = [os.path.join(path_F, ending) for ending in files]

# Function to read and sum data from a file
def read_and_average(file_path, variable_name,timeidx,mask):
    with Dataset(file_path) as wrfin:
        try:
            data = getvar(wrfin, variable_name, timeidx=timeidx, meta=False)[mask]
            
        except IndexError:
            print("index exceeds dimension bounds")
            data = np.array([0])
        return np.mean(data)

cumulative_data1 = []
cumulative_data2 = []
time_array = []
current_time = start_time
time_diff = file_interval / numtimeidx

while current_time <= end_time:
    time_array.append(current_time)
    current_time += timedelta(minutes=time_diff)    

# Create mask to limit data to specific area
with Dataset(path_N+'wrfout_d02_1997-01-12_00:00:00') as wrfin:
        lat = getvar(wrfin, "lat", timeidx=0)
        lon = getvar(wrfin, "lon", timeidx=0)

lat_min, lat_max = 43.1386, 44.2262
lon_min, lon_max = -77.345, -74.468
lat = to_np(lat)
lon = to_np(lon)
lat_mask = (lat > lat_min) & (lat < lat_max)
lon_mask = (lon > lon_min) & (lon < lon_max)
    
# This mask can be used an any data to ensure we are in are modified region
region_mask = lat_mask & lon_mask
    
# Apply the mask to WRF data
masked_data_a1 = np.where(region_mask, lat, np.nan)

# Use this to remove nan's for statistical operations, can apply to all the data since they have matching domains
final_mask = ~np.isnan(masked_data_a1)

# Loop through all files and each time index to sum the data
for file_path1, file_path2, timeidx in zip(filelist_N, filelist_F,timeidxlist):
    cumulative_data1.append(read_and_average(file_path1, 'PBLH', timeidx,final_mask))
    cumulative_data2.append(read_and_average(file_path2, 'PBLH', timeidx,final_mask))

# Convert lists to numpy arrays for plotting
times = np.array(time_array)
cumulative_data1 = np.array(cumulative_data1)
cumulative_data2 = np.array(cumulative_data2)
#print(times.shape)
#print(cumulative_data1.shape)

# Create a line plot
fig,ax = plt.subplots(figsize=(12, 6))

ax.plot(times[1:], cumulative_data1[1:], label='Normal Simulation', color='red')
ax.plot(times[1:], cumulative_data2[1:], label='Flat Simulation', color='yellow')

ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Height (m)',fontsize=18)

ax.set_title('Planetary Boundary Layer Height',fontsize=24)
ax.legend()
ax.grid(True)
plt.show()
