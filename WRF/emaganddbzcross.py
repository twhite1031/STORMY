import numpy as np
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair)
import wrffuncs
from datetime import datetime, timedelta
import pandas as pd

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2

cross_sect = [(44.00, -76.75), (43.73,-75.5)] # Start and end coordinates of cross section

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

# Define the cross section start and end points
cross_start = CoordPair(lat=44.00, lon=-76.75)
cross_end = CoordPair(lat=43.73, lon=-75.5)

# Get the WRF variables
with Dataset(matched_file) as ds:
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    elecmag = getvar(ds, "ELECMAG", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    Z = 10**(dbz/10.) # Use linear Z for interpolation

    # Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
    z_cross = vertcross(Z, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
    elec_cross = vertcross(elecmag, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)


# Convert back to dBz after interpolation
dbz_cross = 10.0 * np.log10(z_cross)

# Add back the attributes that xarray dropped from the operations above
dbz_cross.attrs.update(dbz_cross.attrs)
dbz_cross.attrs["description"] = "radar reflectivity cross section"
dbz_cross.attrs["units"] = "dBZ"

# To remove the slight gap between the dbz contours and terrain due to the
# contouring of gridded data, a new vertical grid spacing, and model grid
# staggering, fill in the lower grid cells with the first non-missing value
# for each column.

# Make a copy of the z cross data. Let's use regular numpy arrays for this.
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))
elec_cross_filled = np.ma.copy(to_np(elec_cross))
# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

for i in range(elec_cross_filled.shape[-1]):
    column_vals = elec_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    elec_cross_filled[0:first_idx, i] = elec_cross_filled[first_idx, i]

with Dataset(matched_file) as ds:
    # Get the terrain heights along the cross section line
    ter_line = interpline(ter, wrfin=ds, start_point=cross_start,
                      end_point=cross_end)

# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)

# Create the figure
fig = pyplot.figure(figsize=(30,15))

ax_dbz = fig.add_subplot(1,2,1)
ax_emag = fig.add_subplot(1,2,2)
dbz_levels = np.arange(5., 75., 5.)

# Create the color table found on NWS pages.
dbz_rgb = np.array([[4,233,231],
                    [1,159,244], [3,0,244],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253],
                    [152,84,198]], np.float32) / 255.0

dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,
                                           extend="max")

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)
emag_levels = np.arange(0,180001,10000)


xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

exs = np.arange(0, elec_cross.shape[-1], 1)
eys = to_np(elec_cross.coords["vertical"])

emag_contours = ax_emag.contourf(exs[0:41], eys[0:41] ,to_np(elec_cross_filled)[0:41],levels=emag_levels, cmap="hot_r", extend="max")

dbz_contours = ax_dbz.contourf(xs[0:41], ys[0:41] ,to_np(dbz_cross_filled)[0:41],levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

# Add the color bar
cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)
#cb_dbz.set_label("dBZ", fontsize=10)

# Add the color bar
cb_emag = fig.colorbar(emag_contours, ax=ax_emag)
cb_emag.ax.tick_params(labelsize=8)
cb_emag.set_ticks(emag_levels)
#cb_dbz.set_label("dBZ", fontsize=10)

# Fill in the mountain area
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line),
                                facecolor="saddlebrown")
ht_fill_emag = ax_emag.fill_between(xs, 0, to_np(ter_line),
                                facecolor="saddlebrown")

# Set the x-ticks to use latitude and longitude labels
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
print(x_ticks)
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
print(x_labels[0])


# Set the desired number of x ticks below
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin], rotation=60, fontsize=8)
ax_emag.set_xticks(x_ticks[::thin])
ax_emag.set_xticklabels(x_labels[::thin],rotation=60,fontsize=8)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=10)
ax_dbz.set_ylabel("Height (m)", fontsize=10)
ax_emag.set_xlabel("Latitude, Longitude", fontsize=10)
ax_emag.set_ylabel("Height (m)", fontsize=10)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add a shared title at the top with the time label
fig.suptitle(date_format, fontsize=16, fontweight='bold')

# Add a title
ax_dbz.set_title(f"Cross-Section of Reflectivity (dBZ)", fontsize=16, fontweight='bold')
ax_emag.set_title(f"Cross-Section of Electric Field Magnitude (V/m)" , fontsize=16, fontweight='bold')

pyplot.show()
