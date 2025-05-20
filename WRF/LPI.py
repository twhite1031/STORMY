from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import (get_cmap,ScalarMappable)
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
import matplotlib.colors as colors
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
from datetime import datetime
import wrffuncs
import pandas as pd
# --- USER INPUT ---

wrf_date_time = datetime(2018,5,15,18,00,00)
domain = 2

path = f"/data1/white/WRF_OUTPUTS/MET416/"

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
	lpi = getvar(ds, "LPI", timeidx=matched_timeidx)

# Get the latitude and longitude points
lats, lons = latlon_coords(lpi)

# Get the cartopy mapping object
cart_proj = get_cartopy(lpi)

# Create a figure
fig = plt.figure(figsize=(30,15))
# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)

# Make the filled countours with specified levels and range
lpi_levels = np.arange(0,200,10)
qcs = plt.contourf(to_np(lons), to_np(lats),lpi,levels=lpi_levels, transform=crs.PlateCarree(),cmap="YlOrRd")

# Add a color bar
plt.colorbar()

# Set the map bounds
ax.set_xlim(cartopy_xlim(lpi))
ax.set_ylim(cartopy_ylim(lpi))

# Add the gridlines
#plt.title(f"Lighting Potenital Index (J/kg) at {year}{month}{day}{hour}{minute}",{"fontsize" : 14})
#if savefig == True:
#	plt.savefig(path+f"LPI{year}{month}{day}{hour}{minute}A{ATTEMPT}.png")
plt.show()
