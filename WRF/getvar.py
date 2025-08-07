from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
from datetime import datetime
import STORMY
import pandas as pd

"""
A bare bones plot of a 2D WRF variable
"""

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
var = "T2" #2d variable to plot (e.g., "T2" for 2m temperature, "U10" for 10m U wind, etc.)

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
    plot_var = getvar(ds, var, timeidx=matched_timeidx)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(plot_var)
cart_proj = get_cartopy(plot_var)
WRF_ylim = cartopy_ylim(plot_var)
WRF_xlim = cartopy_xlim(plot_var)

# Create a figure
fig = plt.figure(figsize=(30,15))
ax = plt.axes(projection=cart_proj)

# Apply cartopy features to the axis(States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)

# Plot filled contours of the variable
qcs = plt.contourf(to_np(lons), to_np(lats),plot_var,transform=crs.PlateCarree(),cmap="jet")

# Add a color bar
plt.colorbar()

# Set the map bounds
ax.set_xlim(WRF_xlim)
ax.set_ylim(WRF_ylim)

# Add the title
ax.set_title(f"{var} at {matched_time}")

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"getvar{var}_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
