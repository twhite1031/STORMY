import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime
import STORMY # Personal library
from wrf import (to_np, getvar)
from datetime import datetime

"""
Creates a time line plot of the average of Planetary Boundary Layer (PBL) Height
between start time and end time for two simulations
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,17,00,00,00), datetime(2022, 11, 22,1, 00, 00)
domain = 2

var = 'PBLH' #2D VARIABLE

# Region to average the variable within
lat_min, lat_max = 43.1386, 44.2262
lon_min, lon_max = -77.345, -74.468

path_N = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_F = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_N = STORMY.build_time_df(path_N, domain)
time_df_F = STORMY.build_time_df(path_F, domain)

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

start_time, end_time = timelist[0], timelist[-1] # Adjust times based on what fit the WRF temporal resolution
# Function to read and average the data from a file
def read_and_average(file_path, variable_name, timeidx,mask):
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
with Dataset(filelist_N[0]) as wrfin:
        lat = getvar(wrfin, "lat", timeidx=0,meta=False)
        lon = getvar(wrfin, "lon", timeidx=0,meta=False)

lat_mask = (lat > lat_min) & (lat < lat_max)
lon_mask = (lon > lon_min) & (lon < lon_max)
    
# Define mask and
region_mask = lat_mask & lon_mask
masked_data = np.where(region_mask, lat, np.nan) # Apply the mask to lat data, np.nan where not in region
final_mask = ~np.isnan(masked_data) # Remove NaN values to get the final mask

# Loop through all files and each time index to average the data
for file_path1, file_path2, timeidx in zip(filelist_N, filelist_F,timeidxlist):
    cumulative_data1.append(read_and_average(file_path1, var, timeidx,final_mask))
    cumulative_data2.append(read_and_average(file_path2, var, timeidx,final_mask))

# Convert lists to numpy arrays for plotting
times = np.array(timelist)
cumulative_data1 = np.array(cumulative_data1)
cumulative_data2 = np.array(cumulative_data2)

# Create a line plot
fig,ax = plt.subplots(figsize=(12, 6))

# Plot both timelines
ax.plot(times[1:], cumulative_data1[1:], label='Normal Simulation', color='red')
ax.plot(times[1:], cumulative_data2[1:], label='Flat Simulation', color='yellow')

# Set labels
ax.set_xlabel('Time', fontsize=18)
ax.set_ylabel('Height (m)',fontsize=18)

# Add a title, legend and grid
ax.set_title('Planetary Boundary Layer Height',fontsize=24)
ax.legend()
ax.grid(True)

# Format the time for a filename (no spaces/colons), show and save figure
start_time_str, end_time_str = start_time.strftime("%Y-%m-%d_%H-%M"), end_time.strftime("%Y-%m-%d_%H-%M")
filename = f"TIMELINEAVG_{start_time_str}_{end_time_str}.png"

plt.savefig(savepath + filename)
plt.show()
