from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,cartopy_ylim, latlon_coords)
from datetime import datetime
import wrffuncs
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
time_df = wrffuncs.build_time_df(path, domain)
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

with Dataset(matched_file) as ds:
    # Get the sea level pressure
    slp = getvar(ds, "slp", timeidx=matched_timeidx)

# Smooth the sea level pressure since it tends to be noisy near the
# mountains
smooth_slp = smooth2d(slp, 3, cenweight=4)

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

# Get the cartopy mapping object
cart_proj = get_cartopy(slp)

# Create a figure
fig = plt.figure(figsize=(30,15))

# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=.1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)

# Make the contour outlines and filled contours for the smoothed sea level pressure.
plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), 10, colors="black", transform=crs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), to_np(smooth_slp), 10, transform=crs.PlateCarree(),cmap=get_cmap("jet"))

# Add a color bar
cbar = plt.colorbar(ax=ax, shrink=.98)
cbar.set_label("hPa", fontsize=10)

# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))

# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
plt.title(f"Sea Level Pressure (hPa) at {date_format}", fontsize="14")

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"SLP_{time_str}.png"

plt.savefig(savepath+filename)


plt.show()
