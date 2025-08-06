import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import ctables
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.cm import get_cmap

from wrf import (getvar, interpline, to_np, vertcross, smooth2d, CoordPair, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
from datetime import datetime
import STORMY
import pandas as pd

"""
A three panel plot that shows a plan view of cloud top temperature with a cross section line.
Two side plots show cross sectional wind speed and reflectivity
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

# Compute absolute time difference
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]
print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")

# Read data from the matched WRF file
with Dataset(matched_file) as ds:
    slp = getvar(ds, "slp",timeidx=matched_timeidx)
    smooth_slp = smooth2d(slp, 3)
    ctt = getvar(ds, "ctt",timeidx=matched_timeidx)
    z = getvar(ds, "z",timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz",timeidx=matched_timeidx)
    wspd =  getvar(ds, "wspd_wdir",timeidx=matched_timeidx, units="kt")[0,:]
    ter = getvar(ds, "ter",timeidx=matched_timeidx, units="m")

# Define the cross section start and end points
start_point = CoordPair(lat=lat_lon[0][0], lon=lat_lon[0][1])
end_point = CoordPair(lat=lat_lon[1][0], lon=lat_lon[1][1])

Z = 10**(dbz/10.) # Use linear Z for interpolation

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, z, wrfin=Dataset(matched_file), start_point=start_point,end_point=end_point, latlon=True, meta=True)
wspd_cross = vertcross(wspd, z, wrfin=Dataset(matched_file), start_point=start_point,end_point=end_point, latlon=True, meta=True)
dbz_cross = 10.0 * np.log10(z_cross) # Back to logrithmic

# Make a copy of the z cross data. 
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below. Fixes the slight gap between the dbz contours and terrain 
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

# Same processes for wind speed
wspd_cross_filled = np.ma.copy(to_np(wspd_cross))
for i in range(wspd_cross_filled.shape[-1]):
    column_vals = wspd_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > 5).nonzero())[0])
    wspd_cross_filled[0:first_idx, i] = wspd_cross_filled[first_idx, i]

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(slp)
cart_proj = get_cartopy(slp)
WRF_ylim = cartopy_ylim(slp)
WRF_xlim = cartopy_xlim(slp)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))
ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_wspd = fig.add_subplot(2,2,2)
ax_dbz = fig.add_subplot(2,2,4)
axs = ["ax_ctt", "ax_wspd", "ax_dbz"]

# Apply cartopy features to the ctt axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)

# Set margins on all plots
for ax in axs:
    ax.margins(x=0, y=0, tight=True)

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
slp_levels = STORMY.make_contour_levels(slp, interval=5)
ctt_levels = STORMY.make_contour_levels(ctt, interval=10)

# Plot the SLP and ctt contours
c1 = ax_ctt.contour(lons, lats, to_np(smooth_slp), levels=slp_levels, colors="white", transform=crs.PlateCarree(), zorder=3, linewidths=1.0)
ctt_contours = ax_ctt.contourf(to_np(lons), to_np(lats), to_np(ctt), ctt_levels, cmap=get_cmap("Greys"), transform=crs.PlateCarree(), zorder=2)

# Plot the cross section line on the ctt plot
ax_ctt.plot([start_point.lon, end_point.lon],[start_point.lat, end_point.lat], color="yellow", marker="o",transform=crs.PlateCarree(), zorder=3)

# Set viewing limits for plots based on WRF domain
ax_ctt.set_xlim(WRF_xlim)
ax_ctt.set_ylim(WRF_ylim)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax.ctt, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=start_point,end_point=end_point)

# Set the cross section x and y locations based on cross section data
xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])[0:41]

wspdxs = np.arange(0, wspd_cross.shape[-1], 1)
wspdys = to_np(wspd_cross.coords["vertical"])[0:41]

# Make the contour plot for wind speed
wspd_contours = ax_wspd.contourf(wspdxs,wspdys, to_np(wspd_cross_filled)[0:41], cmap=get_cmap("jet"))

# Make the contour plot for dbz
dbz_levels = np.arange(0,75.,5.)
dbz_contours = ax_dbz.contourf(xs, ys, to_np(dbz_cross_filled)[0:41], levels=dbz_levels,cmap="NWSRef")

# Add the color bars
cb_ctt = fig.colorbar(ctt_contours, ax=ax_ctt, shrink=.60)
cb_ctt.ax.tick_params(labelsize=8)

cb_wspd = fig.colorbar(wspd_contours, ax=ax_wspd)
cb_wspd.ax.tick_params(labelsize=8)

cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)

# Fill in the terrain
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line), facecolor="saddlebrown")
ht_fill = ax_wspd.fill_between(wspdxs, 0, to_np(ter_line), facecolor="saddlebrown")

# Grab the lat/lon points from the cross section and use them for labeling
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]

# Set the x-ticks to use latitude and longitude , set them evenly spaced
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin], rotation=15, fontsize=7)
ax_wspd.set_xticks(x_ticks[::thin])
ax_wspd.set_xticklabels(x_labels[::thin], rotation=15, fontsize=7)

#Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=8)
ax_wspd.set_ylabel("Height (m)", fontsize=8)
ax_dbz.set_ylabel("Height (m)", fontsize=8)

# Add a shared title at the top with the time label
fig.suptitle(matched_time, fontsize=16, fontweight='bold')

# Add a title
ax_ctt.set_title("Cloud Top Temperature (degC)", fontsize=12, fontweight='bold')
ax_wspd.set_title("Cross-Section of Wind Speed (kt)", fontsize=12, fontweight='bold' )
ax_dbz.set_title("Cross-Section of Reflectivity (dBZ)", fontsize=12, fontweight='bold')

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"panelplots_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
