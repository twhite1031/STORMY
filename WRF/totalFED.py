import numpy as np
from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as crs
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from datetime import datetime
import STORMY

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,19,00,00), datetime(2022, 11, 19,20, 00)
domain = 2

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.5, -74.0,42.5, 44.5] 

# Path to each WRF run (NORMAL & FLAT)
path_1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_1 = STORMY.build_time_df(path_1, domain)
time_df_2 = STORMY.build_time_df(path_2, domain)
time_df_3 = STORMY.build_time_df(path_3, domain)

# Filter time range
mask_1 = (time_df_1["time"] >= start_time) & (time_df_1["time"] <= end_time)
mask_2 = (time_df_2["time"] >= start_time) & (time_df_2["time"] <= end_time)
mask_3 = (time_df_3["time"] >= start_time) & (time_df_3["time"] <= end_time)

time_df_1 = time_df_1[mask_1].reset_index(drop=True)
time_df_2 = time_df_2[mask_2].reset_index(drop=True)
time_df_3 = time_df_3[mask_3].reset_index(drop=True)

wrf_filelist_1 = time_df_1["filename"].tolist()
wrf_filelist_2 = time_df_2["filename"].tolist()
wrf_filelist_3 = time_df_3["filename"].tolist()

timeidxlist = time_df_1["timeidx"].tolist() # Assuming time indexes are the same
timelist = time_df_1["time"].tolist()

total_data1 = None
total_data2 = None
total_data3 = None

# Function to read and sum data from a WRF file
def read_and_sum(file_path, variable_name, timeidx):
    with Dataset(file_path) as wrfin:
        data = getvar(wrfin, variable_name, timeidx=timeidx, meta=False)     

    return data

# Loop through all files and sum 
for idx, filename in enumerate(wrf_filelist_1):
    
    data1 = read_and_sum(wrf_filelist_1[idx], 'LIGHTDENS', timeidxlist[idx])
    data2 = read_and_sum(wrf_filelist_2[idx], 'LIGHTDENS', timeidxlist[idx])
    data3 = read_and_sum(wrf_filelist_3[idx], 'LIGHTDENS', timeidxlist[idx])

    total_data1 = data1 if total_data1 is None else total_data1 + data1
    total_data2 = data2 if total_data2 is None else total_data2 + data2
    total_data3 = data3 if total_data3 is None else total_data3 + data3

# Read a file in for the metadata to create the figure
with Dataset(wrf_filelist_1[0]) as wrfin:
    data = getvar(wrfin, 'LIGHTDENS', timeidx=1)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(data)
cart_proj = get_cartopy(data)
WRF_ylim = cartopy_ylim(data)
WRF_xlim = cartopy_xlim(data)

# Create a figure
fig = plt.figure(figsize=(12,9),facecolor='white')
ax = plt.axes(projection=cart_proj)
    
# Read in detailed county lines
reader = shpreader.Reader('../COUNTY_SHAPEFILES/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

# Set the map bounds
ax.set_extent(extent)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) 

# Create a custom color map where zero values are white
colors = [(1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]  # White to Red
n_bins = 500  # Discretizes the interpolation into bins
cmap_name = 'custom_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Plot the cumulative flash extent density using pcolormesh
mesh = ax.pcolormesh(to_np(lons), to_np(lats), total_data3, cmap=custom_cmap, norm=LogNorm(1,50), transform=crs.PlateCarree())

# Add a colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1,shrink=.5)
cbar.set_label('Flash Extent Density', fontsize=18)

# Add a title and show the figure
plt.title("Flash Extent Density (Sum # of Flashes / Grid Column)", fontsize=28)
plt.show()

