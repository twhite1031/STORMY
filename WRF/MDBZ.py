from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from wrf import (to_np,interplevel, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import STORMY
from datetime import datetime
from metpy.plots import ctables
import pandas as pd
import numpy as np
"""
Plot of the simulated composite reflectivty ('mdbz') with/without wind barbs
"""

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,17,12,00,00)
domain = 2

windbarbs = False

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

# Read data from matched WRF file
with Dataset(matched_file) as ds:
    # Get the maxiumum reflectivity and convert units
    mdbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    ua  = getvar(ds, "ua", units="kt", timeidx=matched_timeidx)
    va = getvar(ds, "va", units="kt",timeidx=matched_timeidx)
    p = getvar(ds, "pressure",timeidx=matched_timeidx)
    u_flat = interplevel(ua, p, 900)
    v_flat= interplevel(va, p, 900)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(mdbz)
cart_proj = get_cartopy(mdbz)
WRF_ylim = cartopy_ylim(mdbz)
WRF_xlim = cartopy_xlim(mdbz)

# Create a figure
fig = plt.figure(figsize=(16,12))
ax = plt.axes(projection=cart_proj)

# Apply cartopy features to the axis (States, lakes, etc.) using STORMY helper function 
STORMY.add_cartopy_features(ax)
ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

# Plot the maximum reflectivity for the WRF run
dbz_levels = np.arange(0, 75, 5)  # Define reflectivity levels for contouring
mdbz_contourf = ax.contourf(to_np(lons), to_np(lats), mdbz,levels=dbz_levels,cmap="NWSRef", transform=crs.PlateCarree())

# Add the wind barbs at set pressure level (hPa), only plotting every nth data point.
n = 25
if windbarbs == True:
	plt.barbs(to_np(lons[::n,::n]), to_np(lats[::n,::n]),
          to_np(u_flat[::n, ::n]), to_np(v_flat[::n, ::n]),
          transform=crs.PlateCarree(), length=6)
      
# Add a color bar
cbar = plt.colorbar()
cbar.set_label("dBZ",fontsize=10)

# Set titles, get readable format from WRF time
plt.title(f"Simulated Composite Reflectivty (dBZ) at {matched_time}",fontsize="18",fontweight="bold")

# Format the time for a filename (no spaces/colons), show and save figure
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"MDBZ_{time_str}.png"

plt.savefig(savepath+filename)
plt.show()
