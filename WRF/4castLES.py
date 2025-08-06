import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from metpy.plots import ctables
from wrf import (getvar, interplevel, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
from metpy.units import units
import STORMY
from datetime import datetime
import pandas as pd

"""
A Three Panel Plot for forecasting Lake-effect Snow
1. Wind Barbs
2. Simulated Reflectivity
3. Temp Difference between 850 hPa and Surface (2 meter Temp)
"""

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
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
    z     = getvar(ds, "z", timeidx=matched_timeidx)
    mdbz  = getvar(ds, "mdbz", timeidx=matched_timeidx)
    p     = getvar(ds, "pressure", timeidx=matched_timeidx)
    wspd  = getvar(ds, "wspd_wdir", timeidx=matched_timeidx, units="kts")[0, :]
    ua    = getvar(ds, "ua", timeidx=matched_timeidx, units="kt")
    va    = getvar(ds, "va", timeidx=matched_timeidx, units="kt")
    ter   = getvar(ds, "ter", timeidx=matched_timeidx, units="m")
    tc    = getvar(ds, "tc", timeidx=matched_timeidx)
    t2    = getvar(ds, "T2", timeidx=matched_timeidx)

# Change temperature units to degC
t2 = to_np(t2) * units.kelvin
t2 = t2.to('degC')
t2 = t2.magnitude

# Interpolate temperature, height, u, and v winds to desired height level (850 hPa)
temp_850 = interplevel(tc, p, 850) 
ht = interplevel(z, p, height)
u = interplevel(ua, p, height)
v = interplevel(va, p, height)
wspd = interplevel(wspd, p, height)
les_temp_diff = t2 - to_np(temp_850)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(ht)
cart_proj = get_cartopy(ht)
WRF_ylim = cartopy_ylim(ht)
WRF_xlim = cartopy_xlim(ht)

# Create a figure that will have 3 subplots
fig = plt.figure(figsize=(12,9))
ax_hgt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_dbz = fig.add_subplot(2,2,2, projection=cart_proj)
ax_temp_diff = fig.add_subplot(2,2,4, projection=cart_proj)
axs = [ax_hgt, ax_dbz, ax_temp_diff]

# Apply cartopy features to each axis (States, lakes, etc.) using STORMY helper function 
for ax in axs:
    STORMY.add_cartopy_features(ax)
    ax.margins(x=0, y=0, tight=True)
    ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
    ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

# Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
# Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
wind_levels = STORMY.make_contour_levels(to_np(wspd), interval=5)
hgt_levels = STORMY.make_contour_levels(to_np(ht), interval=30)
temp_diff_levels = STORMY.make_contour_levels(les_temp_diff, interval=2)

# Plot filled contour for wind speed
c1 = ax_hgt.contourf(lons, lats, to_np(wspd), levels=wind_levels, cmap=get_cmap("rainbow"), transform=crs.PlateCarree(), zorder=2)

# Create the geopotential height contour lines
hgt_contours = ax_hgt.contour(to_np(lons), to_np(lats), to_np(ht), hgt_levels, colors="black", transform=crs.PlateCarree(), zorder=3)
ax_hgt.clabel(hgt_contours,inline=1, fontsize=10, fmt="%i") # Inline labels for height contours

# Create the color bar for wind speed temperature
cbar = fig.colorbar(c1, ax=ax_hgt, orientation="horizontal", pad=.05)
cbar.set_label('Knots',fontsize=14)

# Add the wind barbs, only plotting every nth data point.
n = 40
ax_hgt.barbs(to_np(lons[::n,::n]), to_np(lats[::n,::n]),to_np(u[::n,::n]), to_np(v[::n,::n]),transform=crs.PlateCarree(), length=6,zorder=4)

# Plot the maximum reflectivty
dbz_levels = np.arange(0, 75, 5)
mdbz_contour = ax_dbz.contourf(to_np(lons), to_np(lats), to_np(mdbz),levels=dbz_levels,cmap="NWSRef", transform=crs.PlateCarree(),zorder=2)

# PLot temperature difference
isotherm_contour = ax_temp_diff.contourf(to_np(lons), to_np(lats), les_temp_diff,levels=temp_diff_levels,cmap="jet", transform=crs.PlateCarree(),zorder=2)

# Creating and formatting colorbars
dbz_cbar = fig.colorbar(mdbz_contour, ax=ax_dbz)
dbz_cbar.set_label("dBZ", fontsize=14)

lesbar = fig.colorbar(isotherm_contour, ax=ax_temp_diff)
lesbar.set_label("degC", fontsize=14)

#Set the x-axis and  y-axis labels
ax_hgt.set_xlabel("Longitude", fontsize=8)
ax_dbz.set_ylabel("Lattitude", fontsize=8)
ax_temp_diff.set_ylabel("Lattitude", fontsize=8)

# Add a title to each plot
ax_hgt.set_title("Simulated 850hPa Wind Speed and Direction (Knots)", fontsize="14")
ax_dbz.set_title("Simulated Composite Reflectivity (dBZ)", fontsize="14")
ax_temp_diff.set_title("Surface - 850hPa Temperature (degC)", fontsize="14")
fig.suptitle(matched_time, fontsize=16, fontweight='bold') # Time of the model run

# Format date for a filename (no spaces/colons) to use in filename
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"4castLES_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
