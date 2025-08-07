import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import from_levels_and_colors
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair)
import STORMY
from datetime import datetime
import pandas as pd

"""
A cross section of reflectivity given a start and end point
"""
# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
lat_lon = [(44.00, -76.75), (44.00,-75.5)] # Cross section start and end points

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

# Define the cross section start and end points
cross_start = CoordPair(lat=lat_lon[0][0], lon=lat_lon[0][1])
cross_end = CoordPair(lat=lat_lon[1][0], lon=lat_lon[1][1])

# Read in data from the matched WRF file
with Dataset(matched_file) as ds:
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)

Z = 10**(dbz/10.) # Use linear Z for interpolation

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=Dataset(matched_file), start_point=cross_start, end_point=cross_end, latlon=True, meta=True)

# Convert back to dBz after interpolation
dbz_cross = 10.0 * np.log10(z_cross)

# Add back the attributes that xarray dropped from the operations above
dbz_cross.attrs.update(dbz_cross.attrs)
dbz_cross.attrs["description"] = "radar reflectivity cross section"
dbz_cross.attrs["units"] = "dBZ"

# Make a copy of the z cross data. 
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below. Fixes the slight gap between the dbz contours and terrain 
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=cross_start,end_point=cross_end)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(ht)
cart_proj = get_cartopy(ht)

# Create the figure
fig = plt.figure(figsize=(30,15))
ax_dbz = plt.axes()

xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

# Plot the cross section plot for reflectivity
dbz_levels = np.arange(0, 75, 5)  # Define reflectivity levels for contouring
dbz_contours = ax_dbz.contourf(xs[0:41], ys[0:41] ,to_np(dbz_cross_filled)[0:41],levels=dbz_levels, cmap="NWSRef",extend="max")

# Fill in the terrain
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line),facecolor="saddlebrown")

# Add the color bar
cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)
cb_dbz.set_label("dBZ", fontsize=10)

# Grab the lat/lon points from the cross section
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
print("coord_pairs:", coord_pairs)

# Set the x-ticks to use latitude and longitude , set them evenly spaced
num_ticks = 5
x_ticks = np.arange(coord_pairs.shape[0])
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin],rotation=60,fontsize=8)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=12)
ax_dbz.set_ylabel("Height (m)", fontsize=12)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add a title
ax_dbz.set_title(f"Cross-Section of Reflectivity (dBZ) at {matched_time}", fontsize="14")

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"crosssection_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
