import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from datetime import datetime
import STORMY # Personal library
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times, ll_to_xy,interplevel)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as crs
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import metpy.calc
from metpy.units import units
import pandas as pd
# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,13,50,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference between model times and input time
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]
print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

cumulative_data1 = []
cumulative_data2 = []
cumulative_data3 = []
cumulative_data4 = []
cumulative_data5 = []

#Function to read and sum data from a file
def read_and_sum(file_path, variable_name,timeidx):
    with Dataset(file_path) as wrfin:
        try:
            flash_location = ll_to_xy(wrfin, 43.8693, -76.1647)
            
            data = getvar(wrfin, variable_name, timeidx=timeidx, meta=False,)[:,flash_location[1], flash_location[0]]
            print(data.shape)
        except IndexError:
            print("index exceeds dimension bounds")
            data = np.array([0])
            exit
        return data
  
cumulative_data1.append(read_and_sum(matched_file, 'QSNOW', matched_timeidx))
cumulative_data2.append(read_and_sum(matched_file, 'QICE', matched_timeidx))
cumulative_data3.append(read_and_sum(matched_file, 'QGRAUP', matched_timeidx))
cumulative_data4.append(read_and_sum(matched_file, 'height_agl', matched_timeidx))
cumulative_data5.append(read_and_sum(matched_file, 'QVAPOR', matched_timeidx))
temp_data = read_and_sum(matched_file, 'tc',matched_timeidx)
pres_data = read_and_sum(matched_file, 'pressure',matched_timeidx)
dewpoint_data = read_and_sum(matched_file, 'td',matched_timeidx)
            
lcl_pressure = metpy.calc.lcl(pres_data[0] * units.hPa,temp_data[0] * units.degC,dewpoint_data[0] * units.degC)

print('Pressure at the LCL:', lcl_pressure)
print('Temperature data: ', temp_data)
print('Dewpoint data: ',dewpoint_data)

# Read data that will be used for plotting
with Dataset(matched_file) as wrfin:
    height = getvar(wrfin, 'z', timeidx=matched_timeidx, meta=False)
    pres_data_3d = getvar(wrfin,'pressure',timeidx=matched_timeidx,meta=False)
    flash_location = ll_to_xy(wrfin, 43.8693, -76.1647)

height_interp = interplevel(height, pres_data_3d, lcl_pressure[0].magnitude)
height_at_LCL = height_interp[flash_location[1], flash_location[0]]
print(f"Height at LCL: {height_at_LCL}")

# Convert lists to numpy arrays for plotting
qsnow_data = np.array(cumulative_data1).flatten()
qice_data = np.array(cumulative_data2).flatten()
qgraup_data = np.array(cumulative_data3).flatten()
qvapor_data = np.array(cumulative_data5).flatten()
z_data = np.array(cumulative_data4).flatten()

print("Abs: ",height)
print("agl: ", z_data)
print("pres:",pres_data)

# Example list of flash times from each model run (PBL Schemes)
flash_times_YSU = [
    datetime(2022, 11, 18, 13, 40, 00),
    datetime(2022, 11, 18, 13, 45, 00),
]

flash_times_MYNN2 = [
    datetime(2022, 11, 18, 11, 5, 00),
    datetime(2022, 11, 18, 11, 35, 00),
    datetime(2022, 11, 18, 11, 40, 00),
    datetime(2022, 11, 18, 11, 50, 00),
    datetime(2022, 11, 18, 11, 55, 00),
    datetime(2022, 11, 18, 12, 5, 00),
    datetime(2022, 11, 18, 12, 35, 00),
    datetime(2022, 11, 18, 14, 5, 00),
    datetime(2022, 11, 18, 14, 15, 00),
    datetime(2022, 11, 18, 14, 20, 00),
    datetime(2022, 11, 18, 14, 25, 00),
    datetime(2022, 11, 18, 14, 30, 00),
    datetime(2022, 11, 18, 14, 35, 00),
    datetime(2022, 11, 18, 14, 40, 00),
    datetime(2022, 11, 18, 15, 10, 00),

]

fig,ax = plt.subplots(figsize=(12, 6))
ax.plot(qsnow_data * 1000, z_data, color='blue',label="Snow")
ax.plot(qice_data * 1000, z_data, color='red',label="Ice")
ax.plot(qgraup_data * 1000, z_data, color='yellow',label="Graupel")
#ax.plot(qvapor_data * 1000, z_data, color='green') # Water vapor is significantly larger

ax.minorticks_on()

# Set axes labels and limits
ax.set_xlabel('Mixing Ratio (g/kg)', fontsize=18)
ax.set_ylabel('Height AGL (m)',fontsize=18)
ax.set_ylim(0, 6000)  # Sets the y-axis from 0 to 5000 meters

# Add a title
ax.set_title('Mixing ratio with height',fontsize=24)

#If using pressure, invert axis here
#plt.gca().invert_yaxis()

ax.legend()
ax.grid(True)
plt.show()
