import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime
import wrffuncs # Personal library
from wrf import (to_np, getvar)
from datetime import datetime

"""
Creates a time line plot of the average of Planetary Boundary Layer (PBL) Height
between start time and end time for two simulations
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,17,00,00,00), datetime(2022, 11, 22,1, 00, 00)
domain = 2
var = 'PBLH'

# Path to each WRF run (NORMAL & FLAT)
path_N = r"C:\Users\thoma\Documents\WRF_OUTPUTS"
path_F = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

# Path to save GIF or Files
savepath = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

# --- END USER INPUT ---

time_df_N = wrffuncs.build_time_df(path_N, domain)
time_df_F = wrffuncs.build_time_df(path_F, domain)

# Filter time range
mask_N = (time_df_N["time"] >= start_time) & (time_df_N["time"] <= end_time)
mask_F = (time_df_F["time"] >= start_time) & (time_df_F["time"] <= end_time)

time_df_N = time_df_N[mask_N].reset_index(drop=True)
time_df_F = time_df_F[mask_F].reset_index(drop=True)

filelist_N = time_df_N["filename"].tolist()
filelist_F = time_df_F["filename"].tolist()

# Only need one of these since they are identical between different simulations
timeidxlist = time_df_N["timeidx"].tolist()
timelist = time_df_N["time"].tolist()

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
    cumulative_data1.append(read_and_average(file_path1, var, timeidx,final_mask))
    cumulative_data2.append(read_and_average(file_path2, var, timeidx,final_mask))

# Convert lists to numpy arrays for plotting
times = np.array(timelist)
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

# Format it for a filename (no spaces/colons)
start_time, end_time = timelist[0], timelist[-1]
start_time_str, end_time_str = start_time.strftime("%Y-%m-%d_%H-%M"), end_time.strftime("%Y-%m-%d_%H-%M")

# Use in filename
filename = f"4castLES_{start_time_str}_{end_time_str}.png"

plt.savefig(savepath + filename)
plt.show()
