from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
from wrf import (to_np,getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
from metpy.plots import USCOUNTIES
import pandas as pd
import STORMY
from datetime import datetime

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2

windbarbs = False # Set to True if you want to plot wind barbs

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

# Read in data from the matched WRF file
with Dataset(matched_file) as ds:
    #Get the CAPE values (And more, CAPE is at [0], CIN at [1], LCL at [2], and LFC at[3])
    cape = getvar(ds, "cape_2d", timeidx=matched_timeidx)

'''
# Uncomment if you have a storm report file to plot tornado locations
path = "/data1/white/Downloads/MET416/Storm_reports/"
df = pd.read_csv(path + "180515_rpts_torn.csv", index_col=False,sep=",", header=0,
                 names=["Time", "F_Scale", "Location","County","State","Lat","Lon"])

# Read in the variables of storm Report
time = df['Time'].values
torlat = df['Lat'].values
torlon = df['Lon'].values
'''
# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(cape)
cart_proj = get_cartopy(cape)
WRF_ylim = cartopy_ylim(cape)
WRF_xlim = cartopy_xlim(cape)

# Create a figure
fig = plt.figure(figsize=(30,15))

# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Apply cartopy features to the axis (States, lakes, etc.) using STORMY helper function
STORMY.add_cartopy_features(ax)

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
cape_levels = STORMY.make_contour_levels(cape[0], interval=1)

# Plot the filled contours of CAPE
cape_contour = plt.contourf(to_np(lons), to_np(lats),cape[0],levels=cape_levels,transform=crs.PlateCarree(),cmap="hot_r")

# Add a color bar
plt.colorbar()

# Set the map bounds
ax.set_xlim(WRF_xlim)
ax.set_ylim(WRF_ylim)

# Plot tornado location
#plt.scatter(to_np(torlon), to_np(torlat),transform=crs.PlateCarree())

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

# Add a title
plt.title(f"CAPE (j/kg) at {matched_time}",{"fontsize" : 14})

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"CAPE_{time_str}.png"

plt.savefig(savepath + filename)
plt.show()
