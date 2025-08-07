import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,cartopy_xlim, cartopy_ylim,
                 interpline, CoordPair, xy_to_ll)
from scipy.ndimage import label
import cartopy.feature as cfeature
import STORMY
from datetime import datetime
import pandas as pd

"""
A script used to automatically identify Lake-effect band locations based on connected reflectivity
values based on a threshold, still in development and not fully functional
"""

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,17,23,40,00)
domain = 2

threshold = 0 # Threshold to identify the snow band (e.g., reflectivity > 20 dBZ)


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
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx, meta=False)
    Z = 10**(dbz/10.) # Use linear Z for interpolation
    lats = getvar(ds, "lat", timeidx=matched_timeidx,meta=False)
    lons = getvar(ds, "lon",timeidx=matched_timeidx, meta=False)
    ctt = getvar(ds, "ctt",timeidx=matched_timeidx)

# Check areas where the reflectivity exceeds the threshold
snow_band = max_dbz > threshold
labeled_array, num_features = label(snow_band) # Label connected regions
region_sizes = np.bincount(labeled_array.flatten()) # Find sizes of each region
largest_region_label = np.argmax(region_sizes[1:]) + 1 # +1 to account for background label 0, find largest region
largest_region = (labeled_array == largest_region_label)
print("Largest region: ", largest_region)

# Get the coordinates of the largest region
lat_indices, lon_indices= np.where(largest_region)
lat_lon_indices = list(zip(lon_indices, lat_indices))


start_coords = xy_to_ll(Dataset(matched_file), lat_lon_indices[0][0], lat_lon_indices[0][1],timeidx=matched_timeidx)
end_coords = xy_to_ll(Dataset(matched_file), lat_lon_indices[-1][0], lat_lon_indices[-1][1],timeidx=matched_timeidx)

# Create coord pairs for the cross section start and end points
start_point = CoordPair(lat=float(start_coords[0]), lon=float(start_coords[1]))
end_point = CoordPair(lat=float(end_coords[0]), lon=float(end_coords[1]))
print(f"Start Point: {start_point}, End Point: {end_point}")

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=Dataset(matched_file), start_point=start_point, end_point=end_point, latlon=True, meta=True)
dbz_cross = 10.0 * np.log10(z_cross) # Convert back to dBz after interpolation

# Add back the attributes that xarray dropped from the operations above
dbz_cross.attrs.update(dbz_cross.attrs)
dbz_cross.attrs["description"] = "radar reflectivity cross section"
dbz_cross.attrs["units"] = "dBZ"

dbz_cross_filled = np.ma.copy(to_np(dbz_cross)) # Copy of the reflectivity cross section data

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below. Fixes the slight gap between the dbz contours and terrain 
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=start_point,end_point=end_point)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(ht)
cart_proj = get_cartopy(ht)
WRF_ylim = cartopy_ylim(ht)
WRF_xlim = cartopy_xlim(ht)

# Create the figure
fig = plt.figure(figsize=(12,9))

ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_dbz = fig.add_subplot(1,2,2)
axs = [ax_ctt, ax_dbz]

# Set the cross section x and y locations based on cross section data
xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

# Fill in the terrain heights for the cross section
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line),facecolor="saddlebrown")

# Plot the filled contours of reflectivity for the cross section
dbz_levels = np.arange(0,75,5) # Define reflectivity levels for contouring
dbz_contours = ax_dbz.contourf(xs, ys,to_np(dbz_cross_filled),levels=dbz_levels, cmap="NWSRef", extend="max")

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
ctt_levels = STORMY.make_contour_levels(to_np(ctt), interval=10)

# Plot the filled contours of cloud top temperature
ctt_contours = ax_ctt.contourf(to_np(lons), to_np(lats), to_np(ctt), ctt_levels, cmap=get_cmap("Greys"), transform=crs.PlateCarree())

# Show the cross section line on the plan view (ctt plot)
ax_ctt.plot([start_point.lon, end_point.lon],[start_point.lat, end_point.lat], color="yellow", marker="o",transform=crs.PlateCarree())

# Add the color bars
cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)
cb_dbz.set_label("dBZ", fontsize=10)

cb_ctt = fig.colorbar(ctt_contours, ax=ax_ctt)
cb_ctt.ax.tick_params(labelsize=8)
cb_ctt.set_label("degC", fontsize=10)

# Apply cartopy features to the axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax_ctt)
ax_ctt.margins(x=0, y=0, tight=True)
ax_ctt.set_xlim(WRF_xlim) # Set xlim for viewing the plots
ax_ctt.set_ylim(WRF_ylim) # Set ylim for viewing the plots
STORMY.format_gridlines(ax_ctt, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format 

# Set the x-ticks to use latitude and longitude , set them evenly spaced
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin], rotation=60, fontsize=10)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=12)
ax_dbz.set_ylabel("Height (m)", fontsize=12)

# Add titles to each plot
ax_dbz.set_title(f"Cross-Section of reflectivity (dBZ) at {matched_time}", fontsize="14")
ax_ctt.set_title(f"Plan view of cloud top temperature (degC) at {matched_time}",fontsize="14")

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"identifylesband_{time_str}.png"

plt.savefig(savepath + filename)
plt.show()
