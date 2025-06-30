import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair, xy_to_ll,ll_to_xy)
from scipy.ndimage import label
import cartopy.feature as cfeature
import wrffuncs
from datetime import datetime
import pandas as pd
"""
A script used to automatically identify Lake-effect band locations based on connected reflectivity
values based on a threshold, still in development.
"""

# --- USER INPUT ---

wrf_date_time = datetime(2022,11,17,23,40,00)
domain = 2
# Threshold to identify the snow band (e.g., reflectivity > 20 dBZ)
threshold = 0

lat_lon = [42.93236445072291, -78.75472132204754]  # Example coordinates

SIMULATION = 1 # If comparing runs
# Path to each WRF run (NORMAL & FLAT)
path = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

# Path to save GIF or Files
savepath = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

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


# Get the WRF variables
with Dataset(matched_file) as ds:
    # Convert desired coorindates to WRF gridbox coordinates
    x_y = ll_to_xy(ds, lat_lon[0], lat_lon[1])
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    Z = 10**(dbz/10.) # Use linear Z for interpolation
    lats = getvar(ds, "lat", timeidx=matched_timeidx)
    lons = getvar(ds, "lon",timeidx=matched_timeidx)
    ctt = getvar(ds, "ctt",timeidx=matched_timeidx)
    cloud_frac = getvar(ds, "CLDFRA",timeidx=matched_timeidx,meta=False)[30,:,:]  # Assuming CLDFRA is at level 10

lats = to_np(lats)
lons = to_np(lons)


snow_band = cloud_frac > threshold
print(snow_band)

# Label the connected regions in the snow band
labeled_array, num_features = label(snow_band)



# Find the label of the connected region that includes the starting grid box
start_label = labeled_array[x_y[1], x_y[0]]

if start_label == 0:
    print("Starting point is not part of any connected region above threshold.")
    cloud_region_mask = np.zeros_like(snow_band, dtype=bool)
else:
    # Mask only the region connected to the starting point
    cloud_region_mask = (labeled_array == start_label)

lat_inds, lon_inds = np.where(cloud_region_mask)

min_y, max_y = lat_inds.min(), lat_inds.max()
min_x, max_x = lon_inds.min(), lon_inds.max()



# Get the lat/lon points
lats, lons = latlon_coords(dbz)

# Get the cartopy projection object
cart_proj = get_cartopy(dbz)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

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
# Your main field (e.g., reflectivity)
ax.pcolormesh(lons, lats, max_dbz, cmap="viridis")  # assuming 2D lat/lon

# Draw the bounding box
rect = patches.Rectangle(
    (lons[min_y, min_x], lats[min_y, min_x]),                # bottom-left corner
    lons[max_y, max_x] - lons[min_y, min_x],                 # width
    lats[max_y, max_x] - lats[min_y, min_x],                 # height
    linewidth=2, edgecolor='red', facecolor='none'
)
ax.add_patch(rect)

plt.title("Cloud region linked to flash")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"identifycloudband_{time_str}.png"

plt.savefig(savepath + filename)
plt.show()




