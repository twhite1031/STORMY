import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import concurrent.futures
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
from matplotlib.cm import (get_cmap,ScalarMappable)
import glob
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import STORMY

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
lat_lon = [(44.00, -76.75), (44.00,-75.5)] # Cross section start and end points

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.965996,-75.00000,43.000,44.273301]
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

# Read in data from the matched WRF file
with Dataset(matched_file) as ds:
    data = getvar(ds, "ELECMAG", timeidx=matched_timeidx)[0,:,:]

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(data)
cart_proj = get_cartopy(data)
WRF_ylim = cartopy_ylim(data)
WRF_xlim = cartopy_xlim(data)

# Create a figure
fig = plt.figure(figsize=(30,15),facecolor='white')
ax = plt.axes(projection=cart_proj)

# Apply cartopy features to the axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)

# Set the map bounds
ax.set_xlim(WRF_xlim)
ax.set_ylim(WRF_ylim)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
data_levels = STORMY.make_contour_levels(data, interval=10000)
data_contourf = plt.contourf(to_np(lons), to_np(lats),data,transform=crs.PlateCarree(), cmap="m_fire_r", levels=data_levels,vmin=0,vmax=180000)

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("", fontsize=22)
cbar.ax.tick_params(labelsize=14)  

# Add a title 
ax.set_title(f"Electric Field Magnitude at {matched_time}" ,fontsize=25,fontweight='bold')

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = "ELECMAG_{time_str}.png"
plt.savefig(savepath + filename)
plt.show()



    
