import numpy as np
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, COLORS
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 cartopy_xlim, cartopy_ylim, interpline, CoordPair,xy_to_ll)
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import datetime
import os
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import wrffuncs
import string
import pandas as pd
"""
A side by side plot with composite reflectivity ('mdbz') on the left with a cross section line
On the right, a cross section of reflectivity, water vapor mixing ratio, and vertical velocity vectors
"""
# --- USER INPUT ---

wrf_date_time = datetime(2022,11,18,1,00,00)
domain = 2
# Cross section start and end (lattitude, longitude)
lat_lon = [(43.65, -76.75), (43.65,-75.5)]

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

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
    # Get the WRF variables
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    mdbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    vapor = getvar(ds, "QVAPOR", timeidx=matched_timeidx)
    vapor = vapor * 1000
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    ht_agl = getvar(ds, "height_agl", timeidx=matched_timeidx)
    w = getvar(ds, 'wa',timeidx=matched_timeidx)  # vertical wind speed

# Define the cross section start and end points
cross_start = CoordPair(lat=lat_lon[0][0], lon=lat_lon[0][1])
cross_end = CoordPair(lat=lat_lon[1][0], lon=lat_lon[1][1])

Z = 10**(dbz/10.) # Use linear Z for interpolation

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=Dataset(matched_file), start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
vapor_cross = vertcross(vapor, ht, wrfin=Dataset(matched_file), start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
w_cross = vertcross(w,ht,wrfin=Dataset(matched_file), start_point=cross_start, end_point=cross_end,latlon=True)

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
vapor_cross_filled = np.ma.copy(to_np(vapor_cross))
# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

for i in range(vapor_cross_filled.shape[-1]):
    column_vals = vapor_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > 0).nonzero())[0])
    vapor_cross_filled[0:first_idx, i] = vapor_cross_filled[first_idx, i]

# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=cross_start,
                      end_point=cross_end)

# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)


# Create the figure with desired size
fig = pyplot.figure(figsize=(16, 8))

# Define GridSpec with custom width ratios
gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

# First subplot: plan view (map), uses Cartopy projection
ax_plan = fig.add_subplot(gs[0, 0], projection=cart_proj)

# Second subplot: cross-section
ax_cross = fig.add_subplot(gs[0, 1])

dbz_map, dbz_norm = wrffuncs.get_nws_cmap_norm() 

# Special stuff for counties
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/SEMINAR/countyline_files/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree())

# Add County/State borders
ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)
vapor_levels = np.arange(0,3.1,.3)

# Deal with the plan view map
#mdbz_contours = ax_plan.contour(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())
mdbz_contoursf = ax_plan.contourf(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())

ax_plan.plot([cross_start.lon, cross_end.lon],
            [cross_start.lat, cross_end.lat], color="brown", marker="o",
            transform=crs.PlateCarree())
vapor_contours = ax_plan.contourf(to_np(lons), to_np(lats), to_np(vapor)[0,:,:], levels=vapor_levels, cmap="hot_r", transform=crs.PlateCarree())

xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

exs = np.arange(0, vapor_cross.shape[-1], 1)
eys = to_np(vapor_cross.coords["vertical"])

print("XS: ", xs)
print("YS: ", ys)

# Compute the distance along the cross-section lin
# Get height and distance along cross-section
z_cross = to_np(w_cross.coords["vertical"])
distance = np.linspace(0, len(w_cross[0]), len(w_cross[0]))

# Add vectors to the plot (scale to control the arrow size)
zero_horizontal = np.zeros_like(to_np(w_cross))  

step = 5  # Adjust step to control arrow density

# Ensure the same shape after slicing
xs_sub = xs[::step]
ys_sub = ys[::step]
zero_horizontal_sub = zero_horizontal[::step]
w_cross_sub = to_np(w_cross)[::step]

print(f"xs: {xs.shape}, ys: {ys.shape}, w_cross: {w_cross_sub.shape}")

# Create reflectivity and water vapor contours
vapor_cross_contours = ax_cross.contourf(exs, eys,to_np(vapor_cross_filled),levels=vapor_levels, cmap="hot_r", extend="max")
dbz_cross_contours = ax_cross.contour(xs, ys ,to_np(dbz_cross_filled),levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

# Add wind barbs, indexed to limit amount
quiv = ax_cross.quiver(xs, ys[::10], zero_horizontal[::10], (to_np(w_cross))[::10],scale=5)

ax_cross.quiverkey(quiv, X=0.1, Y=-.2, U=.5,label='50 cm/s', labelpos='E',coordinates='axes', color='black')

# Add the color bar
cb_plan = fig.colorbar(mdbz_contoursf, ax=ax_plan, orientation="vertical",shrink=0.5)
cb_plan.ax.tick_params(labelsize=12)
cb_plan.set_label("dBZ", fontsize=16)

# Add the color bar
cb_cross = fig.colorbar(vapor_contours, ax=ax_cross, orientation='horizontal')
cb_cross.ax.tick_params(labelsize=12)
cb_cross.set_label("g/kg",fontsize=16)

# Fill in the mountain area
ht_fill_vapor = ax_cross.fill_between(xs, 0, to_np(ter_line),
                                facecolor="saddlebrown")

# Set the x-ticks to use latitude and longitude labels
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
print(x_ticks)
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
print(x_labels[0])

# Add the gridlines
gl_a1 = ax_plan.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl_a1.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
gl_a1.ylabel_style = {'size': 10}  # Change 14 to your desired font size
gl_a1.xlines = True
gl_a1.ylines = True
gl_a1.top_labels = False  # Disable top labels
gl_a1.right_labels = False  # Disable right labels
gl_a1.xpadding = 20
print("Made gridlines")


# Set the desired number of x ticks below
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_cross.set_xticks(x_ticks[::thin])
ax_cross.set_xticklabels(x_labels[::thin],rotation=60,fontsize=8)

# Set the view of the plot
ax_plan.set_extent([-77.965996,-75.00000,43.000,44.273301],crs=crs.PlateCarree())

# Set the x-axis and  y-axis labels
ax_plan.set_xlabel("Longitude", fontsize=10)
ax_plan.set_ylabel("Latitude", fontsize=10)
ax_cross.set_xlabel("Latitude, Longitude", fontsize=10)
ax_cross.set_ylabel("Height (m)", fontsize=10)

labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ...]

for i, ax in enumerate([ax_plan, ax_cross]):
    ax.text(0.01, 0.98, f"{labels[i]})", transform=ax.transAxes,fontsize=18, fontweight='bold', va='top', ha='left')

# Remove the filled contours now the we have made the colorbar
vapor_contours.remove()

# Add a title
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
ax_plan.set_title(f"Plan view of Composite Reflectivity (dBZ) at " + date_format, fontsize=14, fontweight='bold')
ax_cross.set_title(f"Cross-Section of Composite Reflectivity (dBZ), Water Vapor Mixing Ratio (g/kg), and Vertical Velocity at " + date_format, fontsize=14, fontweight='bold')



# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"plancross_{time_str}.png"
pyplot.savefig(savepath+filename)
#pyplot.show()
