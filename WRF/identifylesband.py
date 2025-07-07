import numpy as np
from matplotlib import pyplot
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair, xy_to_ll)
from scipy.ndimage import label
import cartopy.feature as cfeature
import wrffuncs
from datetime import datetime
import pandas as pd

"""
A script used to automatically identify Lake-effect band locations based on connected reflectivity
values based on a threshold, still in development and not fully functional
"""

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,17,23,40,00)
domain = 2

threshold = 0 # Threshold to identify the snow band (e.g., reflectivity > 20 dBZ)

SIMULATION = 1 # If comparing runs
path = r"C:\Users\thoma\Documents\WRF_OUTPUTS"
savepath = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

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

# Get the WRF variables
with Dataset(matched_file) as ds:
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    Z = 10**(dbz/10.) # Use linear Z for interpolation
    lats = getvar(ds, "lat", timeidx=matched_timeidx)
    lons = getvar(ds, "lon",timeidx=matched_timeidx)
    ctt = getvar(ds, "ctt",timeidx=matched_timeidx)

lats = to_np(lats)
lons = to_np(lons)

max_dbz = to_np(max_dbz)
snow_band = max_dbz > threshold
print(snow_band)

# Label the connected regions in the snow band
labeled_array, num_features = label(snow_band)

# Find the largest connected region
region_sizes = np.bincount(labeled_array.flatten())

largest_region_label = np.argmax(region_sizes[1:]) + 1
print(largest_region_label)

largest_region = (labeled_array == largest_region_label)
print("Largest region: ", largest_region)

# Get the coordinates of the largest region
lat_indices, lon_indices= np.where(largest_region)

#print("Lats: ", lat_indices)
#print("Lons: ", lon_indices)

lat_lon_indices = list(zip(lon_indices, lat_indices))
print(lat_lon_indices)
start_coords = xy_to_ll(Dataset(matched_file), lat_lon_indices[0][0], lat_lon_indices[0][1],timeidx=matched_timeidx)
end_coords = xy_to_ll(Dataset(matched_file), lat_lon_indices[-1][0], lat_lon_indices[-1][1],timeidx=matched_timeidx)
print("start_coords", start_coords)
print("end_coords", end_coords)

# Define start and end points as the first and last points in the snow band
#start_point_idx = coords[0]
#end_point_idx = coords[-1]
print("Calculated Starting Lat: ", start_coords[0])
print("Calculated Starting Lon: ", start_coords[1])
print("Calculated Ending Lat: ", end_coords[0])
print("Calculated Ending Lon: ", end_coords[1])

# Convert indices to coordinates
start_point = CoordPair(lat=float(start_coords[0]), lon=float(start_coords[1]))
end_point = CoordPair(lat=float(end_coords[0]), lon=float(end_coords[1]))

#print(f"Start Point: {start_point}, End Point: {end_point}")
# Compute the vertical cross-section interpolation.  Also, include the lat/lon points along the cross-section in the metadata by setting latlon to True.
z_cross = vertcross(Z, ht, wrfin=Dataset(matched_file), start_point=start_point, end_point=end_point, latlon=True, meta=True)

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
# For each cross section column, find the first index with non-missing
# values and copy these to the missing elements below.
for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    # Let's find the lowest index that isn't filled. The nonzero function
    # finds all unmasked values greater than 0. Since 0 is a valid value
    # for dBZ, let's change that threshold to be -200 dBZ instead.
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]


# Get the terrain heights along the cross section line
ter_line = interpline(ter, wrfin=Dataset(matched_file), start_point=start_point,
                      end_point=end_point)

# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)

# Create the figure
fig = pyplot.figure(figsize=(12,9))

ax_ctt = fig.add_subplot(1,2,1,projection=cart_proj)
ax_dbz = fig.add_subplot(1,2,2)
dbz_levels = np.arange(5., 75., 5.)

# Set the margins to 0
ax_ctt.margins(x=0,y=0,tight=True)
ax_dbz.margins(x=0,y=0,tight=True)

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

# Make the cross section plot for dbz
dbz_levels = np.arange(5,75,5)

xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

dbz_contours = ax_dbz.contourf(xs, ys,to_np(dbz_cross_filled),levels=dbz_levels, cmap=dbz_map, norm=dbz_norm, extend="max")

# Add the color bar
cb_dbz = fig.colorbar(dbz_contours, ax=ax_dbz)
cb_dbz.ax.tick_params(labelsize=8)
cb_dbz.set_label("dBZ", fontsize=10)

# Fill in the mountain area
ht_fill = ax_dbz.fill_between(xs, 0, to_np(ter_line),facecolor="saddlebrown")


# Download and create the states, land, and oceans using cartopy features
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                      facecolor='none',
                                      name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                    scale='50m',
                                    facecolor=cfeature.COLORS['land'])

lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue")
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])

# Create the filled cloud top temperature contours
contour_levels = [-80.0, -70.0, -60, -50, -40, -30, -20, -10, 0, 10]
ctt_contours = ax_ctt.contourf(to_np(lons), to_np(lats), to_np(ctt), contour_levels, cmap=get_cmap("Greys"), transform=crs.PlateCarree())

ax_ctt.plot([start_point.lon, end_point.lon],
            [start_point.lat, end_point.lat], color="yellow", marker="o",
            transform=crs.PlateCarree())

# Add the color bar
cb_ctt = fig.colorbar(ctt_contours, ax=ax_ctt)
cb_ctt.ax.tick_params(labelsize=8)
cb_ctt.set_label("degC", fontsize=10)



# Set the x-ticks to use latitude and longitude labels
coord_pairs = to_np(dbz_cross.coords["xy_loc"])
x_ticks = np.arange(coord_pairs.shape[0])
print(x_ticks)
x_labels = [pair.latlon_str() for pair in to_np(coord_pairs)]
print(x_labels[0])

# Set the desired number of x ticks below
num_ticks = 5
thin = int((len(x_ticks) / num_ticks) + .5)
ax_dbz.set_xticks(x_ticks[::thin])
ax_dbz.set_xticklabels(x_labels[::thin], rotation=60, fontsize=10)

# Set the x-axis and  y-axis labels
ax_dbz.set_xlabel("Latitude, Longitude", fontsize=12)
ax_dbz.set_ylabel("Height (m)", fontsize=12)

# Adjust format for date to use in figure
date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

# Add a title
ax_dbz.set_title(f"Cross-Section of reflectivity (dBZ) at {date_format}", fontsize="14")
ax_ctt.set_title(f"Plan view of cloud top temperature (degC) at {date_format}",fontsize="14")

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"4castLES_{time_str}.png"

pyplot.savefig(savepath + filename)
pyplot.show()
