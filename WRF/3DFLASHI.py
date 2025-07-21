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
import pandas as pd

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,13,50,00)
domain = 2

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

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.965996,-75.00000,43.000,44.273301]

with Dataset(matched_file) as ds:

    # Get the WRF variables
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    mdbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    elecmag = getvar(ds, "ELECMAG", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    flash = getvar(ds, "FLSHI", timeidx=matched_timeidx)
    ht_agl = getvar(ds, "height_agl", timeidx=matched_timeidx)
    ph = getvar(ds, "PH", timeidx=matched_timeidx)
    phb = getvar(ds,"PHB",timeidx=matched_timeidx)

flashindexs = np.where(to_np(flash) == 1.0) # Find where flash iniations occur

# Extract the first value from each array
print("Flash Indexes (can be multiple): ", flashindexs)

flashindex = [arr[0] for arr in flashindexs]
print("Single Flash gridbox index: ", flashindex)
with Dataset(matched_file) as ds:
    flashloc = xy_to_ll(ds, flashindex[2], flashindex[1], timeidx=matched_timeidx)

# Define the cross section start and end points based on flash location
cross_start = CoordPair(lat=flashloc[0], lon=flashloc[1]-.5)
cross_end = CoordPair(lat=flashloc[0], lon=flashloc[1]+.5)

print("Flash lat and lon: ",to_np(flashloc))
print("Height agl at 0 index: ", to_np(ht_agl[0,flashindex[1],flashindex[2]]))
print("Height (z) at 0 index: ", to_np(ht[0,flashindex[1],flashindex[2]]))

flashheight = ((ph[flashindex[0], flashindex[1],flashindex[2]] + phb[flashindex[0], flashindex[1],flashindex[2]]) /9.8) - ht[flashindex[0], flashindex[1],flashindex[2]]

print("Flashheight (m): ", to_np(flashheight))

Z = 10**(dbz/10.) # Use linear Z for interpolation

# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
with Dataset(matched_file) as ds:
    z_cross = vertcross(Z, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
    elec_cross = vertcross(elecmag, ht, wrfin=ds, start_point=cross_start, end_point=cross_end, latlon=True, meta=True)
    ter_line = interpline(ter, wrfin=ds, start_point=cross_start,end_point=cross_end)

# Convert back to dBz after interpolation
dbz_cross = 10.0 * np.log10(z_cross)

# Add back the attributes that xarray dropped from the operations above
dbz_cross.attrs.update(dbz_cross.attrs)
dbz_cross.attrs["description"] = "radar reflectivity cross section"
dbz_cross.attrs["units"] = "dBZ"

# Make a copy of the z cross data
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))
elec_cross_filled = np.ma.copy(to_np(elec_cross))

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.

for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

for i in range(elec_cross_filled.shape[-1]):
    column_vals = elec_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    elec_cross_filled[0:first_idx, i] = elec_cross_filled[first_idx, i]



# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)

# Create the figure
fig = pyplot.figure(figsize=(16,8))

ax_plan = fig.add_subplot(1,2,1, projection=cart_proj)
ax_cross = fig.add_subplot(1,2,2)
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
# Read county shapefiles into projection
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/PROJ_LEE/countyline_files/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree())

# Add County/State borders
ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=1)

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)
emag_levels = np.arange(1,180002,10000)
flash_levels = np.arange(0,2)

# Deal with the plan view map
mdbz_contours = ax_plan.contour(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())
mdbz_contoursf = ax_plan.contourf(to_np(lons), to_np(lats), to_np(mdbz), levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, transform=crs.PlateCarree())

ax_plan.plot([cross_start.lon, cross_end.lon],
            [cross_start.lat, cross_end.lat], color="brown", marker="o",
            transform=crs.PlateCarree())
emag_contours = ax_plan.contourf(to_np(lons), to_np(lats), to_np(elecmag)[0,:,:], levels=emag_levels, cmap="hot_r", transform=crs.PlateCarree())

# Put the simulated flash location on the map
ax_plan.scatter(flashloc[1],flashloc[0],s=100,marker="X",c='blue',transform=crs.PlateCarree(),zorder=5)

xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

exs = np.arange(0, elec_cross.shape[-1], 1)
eys = to_np(elec_cross.coords["vertical"])

#print("XS: ", xs)
#print("YS: ", ys)

emag_cross_contours = ax_cross.contourf(exs, eys,to_np(elec_cross_filled),levels=emag_levels, cmap="hot_r", extend="max")

dbz_cross_contours = ax_cross.contour(xs, ys ,to_np(dbz_cross_filled),levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

coord_pairs = to_np(dbz_cross.coords["xy_loc"])
print("coord_pairs:", coord_pairs)

indices = [i for i, coord in enumerate(coord_pairs) if coord.x == flashindex[2]]
print("Target x index that matches lat/lon: ", indices)

ax_cross.scatter(indices, flashheight, s=100, marker="x", c='blue',zorder=5)

# Add the color bar
cb_plan = fig.colorbar(mdbz_contoursf, ax=ax_plan, orientation="vertical")
cb_plan.ax.tick_params(labelsize=8)
cb_plan.set_label("dBZ", fontsize=10)

# Add the color bar
cb_cross = fig.colorbar(emag_contours, ax=ax_cross, orientation='horizontal')
cb_cross.ax.tick_params(labelsize=8)
tick_markers = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000]
cb_cross.set_ticks(tick_markers[::2])
selected_ticks = tick_markers[::2]
cb_cross.set_ticklabels([f'{int(tick) // 1000}' for tick in selected_ticks])
cb_cross.set_label("kV/m", fontsize=10)

# Fill in the mountain area
ht_fill_emag = ax_cross.fill_between(xs, 0, to_np(ter_line),
                                facecolor="saddlebrown")

# Set the x-ticks to use latitude and longitude labels
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
#ax_cross.set_xticks(x_ticks[::thin])
#ax_cross.set_xticklabels(x_labels[::thin],rotation=60,fontsize=8)

# Set the view of the plot
ax_plan.set_extent([-77.965996,-75.00000,43.000,44.273301],crs=crs.PlateCarree())
#ax_plan.set_extent([-82.4134,-77.8456,41.291,43.5],crs=crs.PlateCarree())

# Set the x-axis and  y-axis labels
ax_plan.set_xlabel("Longitude", fontsize=10)
ax_plan.set_ylabel("Latitude", fontsize=10)
ax_cross.set_xlabel("Latitude, Longitude", fontsize=10)
ax_cross.set_ylabel("Height (m)", fontsize=10)

# Remove the filled contours now the we have made the colorbar
mdbz_contoursf.remove()

date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
fig.tight_layout()

# Add a title
ax_plan.set_title(f"Plan view of Composite Reflectivity (dBZ) and Electric Field Magnitude (V/m) at " + date_format, fontsize=10, fontweight='bold')
ax_cross.set_title(f"Cross-Section of Composite Reflectivity (dBZ) and Electric Field Magnitude (V/m) at " + date_format, fontsize=10, fontweight='bold')

strflashheight = int(flashheight)
print("Int version of flash height: ", strflashheight)

#pyplot.savefig(savepath+f"3DFLASHI{wrf_filename.year:04d}{wrf_filename.month:02d}{wrf_filename.day:02d}{wrf_filename.hour:02d}{wrf_filename.minute:02d}D{domain}T{timeidx}A{ATTEMPT}H{strflashheight}.png")
pyplot.show()
