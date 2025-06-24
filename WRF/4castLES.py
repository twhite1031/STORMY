import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from metpy.plots import ctables
from wrf import (getvar, interpline, interplevel, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
                 get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim,extract_times)
import pyart
from matplotlib.colors import BoundaryNorm
from metpy.units import units
import wrffuncs
from datetime import datetime, timedelta
import glob
import pandas as pd
import os
"""
A Three Panel Plot for forecasting Lake-effect Snow
1. Wind Barbs
2. Simulated Reflectivity
3. Temp Difference between 850 hPa and Surface (2 meter Temp)
"""

# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
height = 850

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

# Open only the matched WRF file
with Dataset(matched_file) as ds:
    z     = getvar(ds, "z", timeidx=matched_timeidx)
    mdbz  = getvar(ds, "mdbz", timeidx=matched_timeidx)
    p     = getvar(ds, "pressure", timeidx=matched_timeidx)
    wspd  = getvar(ds, "wspd_wdir", timeidx=matched_timeidx, units="kts")[0, :]
    ua    = getvar(ds, "ua", timeidx=matched_timeidx, units="kt")
    va    = getvar(ds, "va", timeidx=matched_timeidx, units="kt")
    ter   = getvar(ds, "ter", timeidx=matched_timeidx, units="m")
    tc    = getvar(ds, "tc", timeidx=matched_timeidx)
    t2    = getvar(ds, "T2", timeidx=matched_timeidx)

# Change units to degC
t2 = to_np(t2) * units.kelvin
t2 = t2.to('degC')
t2 = t2.magnitude

temp_850 = interplevel(tc, p, 850) 
les_temp_diff = t2 - to_np(temp_850)

# Interpolate geopotential height, u, and v winds to desired height level
ht = interplevel(z, p, height)
u = interplevel(ua, p, height)
v = interplevel(va, p, height)
wspd = interplevel(wspd, p, height)

# Get the lat/lon points
lats, lons = latlon_coords(ht)

# Get the cartopy projection object
cart_proj = get_cartopy(ht)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))
ax_hgt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_dbz = fig.add_subplot(2,2,2, projection=cart_proj)
ax_temp_diff = fig.add_subplot(2,2,4, projection=cart_proj)

# Set the margins to 0
ax_hgt.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)
ax_temp_diff.margins(x=0,y=0,tight=True)


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
# Make the wind contours
contour_levels = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
c1 = ax_hgt.contourf(lons, lats, to_np(wspd), levels=contour_levels, cmap=get_cmap("rainbow"), transform=crs.PlateCarree(), zorder=2)

# Create the 500 hPa geopotential height contours
contour_levels = np.arange(1320., 1620., 30.)
hgt_contours = ax_hgt.contour(to_np(lons), to_np(lats), to_np(ht), contour_levels, colors="black", transform=crs.PlateCarree(), zorder=3)
ax_hgt.clabel(hgt_contours,inline=1, fontsize=10, fmt="%i")

# Create the color bar for wind speed temperature
cbar = fig.colorbar(c1, ax=ax_hgt, orientation="horizontal", pad=.05)
cbar.set_label('Knots',fontsize=14)


# Add the 500 hPa wind barbs, only plotting every nth data point.
ax_hgt.barbs(to_np(lons[::40,::40]), to_np(lats[::40,::40]),
        to_np(u[::40,::40]), to_np(v[::40,::40]),
          transform=crs.PlateCarree(), length=6,zorder=4)


# Draw the oceans, land, and states
ax_hgt.add_feature(land)
ax_hgt.add_feature(states, linewidth=.5, edgecolor="black")
ax_hgt.add_feature(lakes)
ax_hgt.add_feature(ocean)

ax_dbz.add_feature(land)
ax_dbz.add_feature(states, linewidth=.5, edgecolor="black")
ax_dbz.add_feature(lakes)
ax_dbz.add_feature(ocean)

ax_temp_diff.add_feature(land)
ax_temp_diff.add_feature(states, linewidth=.5, edgecolor="black")
ax_temp_diff.add_feature(lakes)
ax_temp_diff.add_feature(ocean)

ax_temp_diff.set_xlim(cartopy_xlim(mdbz))
ax_temp_diff.set_ylim(cartopy_ylim(mdbz))


# Plot mdbz
nwscmap = ctables.registry.get_colortable('NWSReflectivity')
levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
mdbz_contour = ax_dbz.contourf(to_np(lons), to_np(lats), to_np(mdbz),levels=levels,cmap=nwscmap, transform=crs.PlateCarree(),zorder=2)
mcbar = fig.colorbar(mdbz_contour, ax=ax_dbz)
mcbar.set_label("dBZ", fontsize=14)

# PLot temp difference
levels = [0,2,4,6,8,10,12,14,16,18,20,22]
isotherm_contour = ax_temp_diff.contourf(to_np(lons), to_np(lats), les_temp_diff,levels=levels,cmap="jet", transform=crs.PlateCarree(),zorder=2)
lesbar = fig.colorbar(isotherm_contour, ax=ax_temp_diff)
lesbar.set_label("degC", fontsize=14)

#Set the x-axis and  y-axis labels
ax_hgt.set_xlabel("Longitude", fontsize=8)
ax_dbz.set_ylabel("Lattitude", fontsize=8)
ax_temp_diff.set_ylabel("Lattitude", fontsize=8)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
# Add a shared title at the top with the time label
fig.suptitle(date_format, fontsize=16, fontweight='bold')

# Add a title
ax_hgt.set_title("Simulated 850hPa Wind Speed and Direction (Knots)", fontsize="14")
ax_dbz.set_title("Simulated Composite Reflectivity (dBZ)", fontsize="14")
ax_temp_diff.set_title("Surface - 850hPa Temperature (degC)", fontsize="14")

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"4castLES_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
