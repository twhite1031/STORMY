from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,cartopy_xlim, cartopy_ylim)
import wrffuncs
from datetime import datetime
import pandas as pd
"""
Plot of wind barbs at a set height level
"""

# --- USER INPUT ---

wrf_date_time = datetime(2022,11,17,23,20,00)
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

# Extract the pressure, geopotential height, and wind variables
with Dataset(matched_file) as ds:
    p = getvar(ds, "pressure",timeidx=matched_timeidx)
    z = getvar(ds, "z",timeidx=matched_timeidx, units="dm")
    ua = getvar(ds, "ua", timeidx=matched_timeidx, units="kt")
    va = getvar(ds, "va", timeidx=matched_timeidx, units="kt")
    wspd = getvar(ds, "wspd_wdir", timeidx=matched_timeidx, units="kts")[0,:]

# Interpolate geopotential height, u, and v winds to 500 hPa
ht_500 = interplevel(z, p, height)
u_500 = interplevel(ua, p, height)
v_500 = interplevel(va, p, height)
wspd_500 = interplevel(wspd, p, height)

# Get the lat/lon coordinates
lats, lons = latlon_coords(ht_500)

# Get the map projection information
cart_proj = get_cartopy(ht_500)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m",facecolor="none", name="admin_1_states_provinces")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.add_feature(states, linewidth=0.5, edgecolor="black")
ax.coastlines('50m', linewidth=0.8)

# Add the 500 hPa geopotential height contours
levels = np.arange(520., 580., 6.)
contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_500),levels=levels, colors="black",transform=crs.PlateCarree())
plt.clabel(contours, inline=1, fontsize=10, fmt="%i")

# Add the wind speed contours
levels = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(wspd_500),levels=levels,cmap=get_cmap("rainbow"), transform=crs.PlateCarree())
cbar = plt.colorbar(wspd_contours, ax=ax, orientation="horizontal", pad=.05)
cbar.set_label('Knots',fontsize=14)


# Add the 500 hPa wind barbs, only plotting every nth data point.
plt.barbs(to_np(lons[::50,::50]), to_np(lats[::50,::50]),
          to_np(u_500[::50, ::50]), to_np(v_500[::50, ::50]),
          transform=crs.PlateCarree(), length=6)

# Set the map bounds
ax.set_xlim(cartopy_xlim(ht_500))
ax.set_ylim(cartopy_ylim(ht_500))

ax.gridlines()

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

plt.title(f"{height} MB Height (dm), Wind Speed (kt), Barbs (kt) at {date_format}",{"fontsize" : 14})

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"{height}mbwspd_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
