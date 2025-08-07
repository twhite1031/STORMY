from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,cartopy_ylim, latlon_coords)
from datetime import datetime
import STORMY
import pandas as pd

"""
Plot of the Sea Level Pressure (SLP)
"""

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]
print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

# Read data from WRF file
with Dataset(matched_file) as ds:
    slp = getvar(ds, "slp", timeidx=matched_timeidx)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(slp)
cart_proj = get_cartopy(slp)
WRF_ylim = cartopy_ylim(slp)
WRF_xlim = cartopy_xlim(slp)

# Create a figure
fig = plt.figure(figsize=(30,15))
ax = plt.axes(projection=cart_proj)

# Apply cartopy features to the axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)
ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

# Make the contour outlines and filled contours for the smoothed sea level pressure
smooth_slp = smooth2d(slp, 3, cenweight=4) # Smooth the sea level pressure,  noisy near the mountains
plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), 10, colors="black", transform=crs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), to_np(smooth_slp), 10, transform=crs.PlateCarree(),cmap=get_cmap("jet"))

# Add a color bar
cbar = plt.colorbar(ax=ax, shrink=.98)
cbar.set_label("hPa", fontsize=10)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) 

# Add a title
plt.title(f"Sea Level Pressure (hPa) at {matched_time}", fontsize="14")

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"SLP_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
