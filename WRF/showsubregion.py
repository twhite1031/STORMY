import numpy as np
import matplotlib.pyplot as plt
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from datetime import datetime
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon
import STORMY
import pandas as pd

"""
A plot of a box given the bounds using the same display as your WRF domain, useful if working
with subregions to see what area your focusing on
"""

# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2

lat_lon = [(44.25, -76.25), (43.25,-74.25)] # Bottom left corner , Top right corner for box

# Define important locations to plot
locations = {
    "KTYX Radar": {
        "coords": (43.755, -75.68),
        "color": "red",
        "marker": "^"
    }
}

SIMULATION = "NORMAL" # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/SEMINAR/{SIMULATION}_ATTEMPT/"

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

# Read data from WRF file
with Dataset(matched_file) as ds:
    ter = getvar(ds, "ter", timeidx=timeidx)

# Get the lat/lon points and projection object from WRF data
lats, lons = latlon_coords(mdbz)
cart_proj = get_cartopy(mdbz)
WRF_ylim = cartopy_ylim(mdbz)
WRF_xlim = cartopy_xlim(mdbz)

# Create a figure
fig = plt.figure(figsize=(30,15),facecolor='white')
ax = plt.axes(projection=cart_proj)

# Read in detailed county lines
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/SEMINAR/countyline_files/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

# Set the map bounds
ax.set_xlim(WRF_xlim)
ax.set_ylim(WRF_ylim)

# Plot the terrain filled contours
elev_contour = ax.contourf(to_np(lons), to_np(lats), ter,levels=np.arange(0, np.max(mdbz), 50), cmap="Greys_r", transform=crs.PlateCarree())

# Add a colorbar
cbar = plt.colorbar(elev_contour, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
cbar.set_label("Terrain Elevation (m)", fontsize=16)

# Add custom formatted gridlines using STORMY function
STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

# Redefine subregion points and create a shape
lat1, lon1 = lat_lon[0][0], lat_lon[0][1]  # Bottom-left corner
lat2, lon2 = lat_lon[1][0], lat_lon[1][1] # Top-right corner

coordinates = [(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]
polygon = Polygon(coordinates)

# Create and add the shape feature 
square_feature = ShapelyFeature([polygon], crs.PlateCarree(), edgecolor='red', facecolor='none',linewidth=3)
ax.add_feature(square_feature)

# Add the locations to the plot
for name, info in locations.items():
    lat, lon = info["coords"]
    ax.plot(lon, lat, marker=info["marker"], color=info["color"], markersize=10, transform=crs.PlateCarree(), zorder=10)
    ax.text(lon + 0.05, lat + 0.05, name, fontsize=16, weight='bold', transform=crs.PlateCarree(), zorder=10,
    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.show()




  
