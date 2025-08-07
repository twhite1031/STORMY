from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,cartopy_xlim, cartopy_ylim)
import STORMY
from datetime import datetime
import pandas as pd

"""
Plot of wind barbs at a set pressure level (e.g 500 hPa) for a single WRF run
"""

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,17,23,20,00)
domain = 2
height = 850 # Pressure level for Wind Barbs

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

# Read in the data from the matched WRF file
with Dataset(matched_file) as ds:
    p = getvar(ds, "pressure",timeidx=matched_timeidx)
    z = getvar(ds, "z",timeidx=matched_timeidx, units="dm")
    ua = getvar(ds, "ua", timeidx=matched_timeidx, units="kt")
    va = getvar(ds, "va", timeidx=matched_timeidx, units="kt")
    wspd = getvar(ds, "wspd_wdir", timeidx=matched_timeidx, units="kts")[0,:]

# Interpolate geopotential height, u, and v winds to a set pressure level (hPa)
ht_500 = interplevel(z, p, height)
u_500 = interplevel(ua, p, height)
v_500 = interplevel(va, p, height)
wspd_500 = interplevel(wspd, p, height)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(p)
cart_proj = get_cartopy(p)
WRF_ylim = cartopy_ylim(p)
WRF_xlim = cartopy_xlim(p)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

# Apply cartopy features (states, lakes, etc.) using STORMY helper function
STORMY.add_cartopy_features(ax)
ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
hgt_levels = STORMY.make_contour_levels(to_np(ht_500), interval=6)
wspd_levels = STORMY.make_contour_levels(to_np(wspd_500), interval=5)

# Plot the wind speed and height contours
hgt_contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_500),levels=hgt_levels, colors="black",transform=crs.PlateCarree())
wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(wspd_500),levels=wspd_levels,cmap=get_cmap("rainbow"), transform=crs.PlateCarree())
plt.clabel(hgt_contours, inline=1, fontsize=10, fmt="%i") # Inline labels for height contours

# Add the 500 hPa wind barbs, only plotting every nth data point.
plt.barbs(to_np(lons[::50,::50]), to_np(lats[::50,::50]),to_np(u_500[::50, ::50]), to_np(v_500[::50, ::50]),transform=crs.PlateCarree(), length=6)

# Create the color bar for wind speed 
cbar = plt.colorbar(wspd_contours, ax=ax, orientation="horizontal", pad=.05)
cbar.set_label('Knots',fontsize=14)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

# Add a title
plt.title(f"{height} MB Height (dm), Wind Speed (kt), Barbs (kt) at {matched_time}",{"fontsize" : 14})

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{height}mbwspd_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
