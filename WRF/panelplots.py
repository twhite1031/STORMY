import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import ctables
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.cm import get_cmap

from wrf import (getvar, interpline, to_np, vertcross, smooth2d, CoordPair, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
from datetime import datetime
import wrffuncs
import pandas as pd
"""
A three panel plot that shows a plan view of cloud top temperature with a cross section line.
Two side plots show cross sectional wind speed and reflectivity
"""
# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
lat_lon = [(44.00, -76.75), (44.00,-75.5)]

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
    slp = getvar(ds, "slp",timeidx=matched_timeidx)
    smooth_slp = smooth2d(slp, 3)
    ctt = getvar(ds, "ctt",timeidx=matched_timeidx)
    z = getvar(ds, "z",timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz",timeidx=matched_timeidx)
    Z = 10**(dbz/10.)
    wspd =  getvar(ds, "wspd_wdir",timeidx=matched_timeidx, units="kt")[0,:]
    ter = getvar(ds, "ter",timeidx=matched_timeidx, units="m")

# Define the cross section start and end points
start_point = CoordPair(lat=lat_lon[0][0], lon=lat_lon[0][1])
end_point = CoordPair(lat=lat_lon[1][0], lon=lat_lon[1][1])

# Compute the vertical cross-section interpolation.  Also, include the
# lat/lon points along the cross-section in the metadata by setting latlon
# to True.
z_cross = vertcross(Z, z, wrfin=Dataset(matched_file), start_point=start_point,
                    end_point=end_point, latlon=True, meta=True)
wspd_cross = vertcross(wspd, z, wrfin=Dataset(matched_file), start_point=start_point,
                       end_point=end_point, latlon=True, meta=True)
dbz_cross = 10.0 * np.log10(z_cross)

# Make a copy of the z cross data. Let's use regular numpy arrays for this.
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

# Make a copy of the wind speed cross data. Let's use regular numpy arrays for this.
wspd_cross_filled = np.ma.copy(to_np(wspd_cross))

# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(wspd_cross_filled.shape[-1]):
    column_vals = wspd_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > 5).nonzero())[0])
    wspd_cross_filled[0:first_idx, i] = wspd_cross_filled[first_idx, i]

# Get the lat/lon points
lats, lons = latlon_coords(slp)

# Get the cartopy projection object
cart_proj = get_cartopy(slp)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))
ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_wspd = fig.add_subplot(2,2,2)
ax_dbz = fig.add_subplot(2,2,4)

# Set the margins to 0
ax_ctt.margins(x=0,y=0,tight=True)
ax_wspd.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)

# Download and create the states, land, and oceans using cartopy features
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue",zorder=5)
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])
# Make the pressure contours
contour_levels = [960, 965, 970, 975, 980, 990]
c1 = ax_ctt.contour(lons, lats, to_np(smooth_slp), levels=contour_levels, colors="white", transform=crs.PlateCarree(), zorder=3, linewidths=1.0)

# Create the filled cloud top temperature contours
contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
ctt_contours = ax_ctt.contourf(to_np(lons), to_np(lats), to_np(ctt), contour_levels, cmap=get_cmap("Greys"), transform=crs.PlateCarree(), zorder=2)

ax_ctt.plot([start_point.lon, end_point.lon],
            [start_point.lat, end_point.lat], color="yellow", marker="o",
            transform=crs.PlateCarree(), zorder=3)

# Create the color bar for cloud top temperature
cb_ctt = fig.colorbar(ctt_contours, ax=ax_ctt, shrink=.60)
cb_ctt.ax.tick_params(labelsize=8)

# Draw the oceans, land, and states
ax_ctt.add_feature(land)
ax_ctt.add_feature(states, linewidth=.5, edgecolor="black")
ax_ctt.add_feature(lakes)
ax_ctt.add_feature(ocean)

# Crop the domain to the region around desired feature
ax_ctt.set_xlim(cartopy_xlim(ctt))
ax_ctt.set_ylim(cartopy_ylim(ctt))

#ax_ctt.gridlines(color="white", linestyle="dotted")

# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=start_point,
                      end_point=end_point)

# The distance both x and y for each data point to be plotted
xs = np.arange(0, dbz_cross.shape[-1], 1)
wspdxs = np.arange(0, wspd_cross.shape[-1], 1)

# This encompasses the total model height
ys = to_np(dbz_cross.coords["vertical"])[0:41]
wspdys = to_np(wspd_cross.coords["vertical"])[0:41]

print(ys)
print(xs)
# Make the contour plot for wind speed
wspd_contours = ax_wspd.contourf(wspdxs,wspdys, to_np(wspd_cross_filled)[0:41], cmap=get_cmap("jet"))

# Add the color bar
cb_wspd = fig.colorbar(wspd_contours, ax=ax_wspd)
cb_wspd.ax.tick_params(labelsize=8)

# Make the contour plot for dbz
dbz_levels = np.arange(10.,65.,5.)
#levels = [5 + 5*n for n in range(15)]
nwscmap = ctables.registry.get_colortable('NWSReflectivity')

dbz_contours = ax_dbz.contourf(xs, ys, to_np(dbz_cross_filled)[0:41], levels=dbz_levels,cmap=nwscmap)
cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)

# Try to plot flash Initiation points

#flash_cross = vertcross(flash, z, wrfin=ncfile, start_point=start_point, end_point=end_point,
#                         latlon=True, meta=True)

#ax_dbz.contourf(xs,ys,to_np(flash_cross),cmap="hot_r")


# Fill in the mountain area
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line), facecolor="saddlebrown")
ht_fill = ax_wspd.fill_between(wspdxs, 0, to_np(ter_line), facecolor="saddlebrown")

#Set the x-ticks to use latitude and longitude labels
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]

# Set the desired number of x ticks below
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

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add a shared title at the top with the time label
fig.suptitle(date_format, fontsize=16, fontweight='bold')
# Add a title
ax_ctt.set_title("Cloud Top Temperature (degC)", fontsize=12, fontweight='bold')
ax_wspd.set_title("Cross-Section of Wind Speed (kt)", fontsize=12, fontweight='bold' )
ax_dbz.set_title("Cross-Section of Reflectivity (dBZ)", fontsize=12, fontweight='bold')


# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"panelplots_{time_str}.png"

plt.savefig(savepath+filename)

plt.show()
