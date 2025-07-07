from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
from datetime import datetime
import wrffuncs
import pandas as pd

"""
A bare bones plot of a 2D WRF variable
"""

# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
var = "T2"

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = wrffuncs.build_time_df(path, domain)
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


with Dataset(matched_file) as ds:
    plot_var = getvar(ds, var, timeidx=matched_timeidx)

# Get the latitude and longitude points
lats, lons = latlon_coords(plot_var)

# Get the cartopy mapping object
cart_proj = get_cartopy(plot_var)

# Create a figure
fig = plt.figure(figsize=(30,15))

# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)

# Make the filled countours with specified levels and range
qcs = plt.contourf(to_np(lons), to_np(lats),plot_var,transform=crs.PlateCarree(),cmap="jet")

# Add a color bar
plt.colorbar()

# Set the map bounds
ax.set_xlim(cartopy_xlim(plot_var))
ax.set_ylim(cartopy_ylim(plot_var))

# Add the title
time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(matched_file, matched_timeidx)
ax.set_title(f"{var} at " + str(time_object_adjusted))

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"getvar{var}_{time_str}.png"
plt.savefig(savepath+filename)

plt.show()
