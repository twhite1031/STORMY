import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair, xy_to_ll,ll_to_xy,cartopy_xlim, cartopy_ylim)
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

wrf_date_time = datetime(2022,11,18,13,50,00)
domain = 2
# Threshold to identify the snow band (e.g., cloud fraction > .1)
threshold = .95

lat_lon = [43.86935, -76.164764]  # Coordinates to start cloud check
ht_level = 15

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


# Get the WRF variables
with Dataset(matched_file) as ds:
    # Convert desired coorindates to WRF gridbox coordinates
    x_y = ll_to_xy(ds, lat_lon[0], lat_lon[1])
    ht = getvar(ds, "z", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
    Z = 10**(dbz/10.) # Use linear Z for interpolation
    lats = getvar(ds, "lat", timeidx=matched_timeidx)
    lons = getvar(ds, "lon",timeidx=matched_timeidx)
    cloud_frac = getvar(ds, "CLDFRA",timeidx=matched_timeidx)[:,:,:]  # Assuming CLDFRA is at level 10

    # Multiply by 1000 to go from kg/kg to g/kg
    wv = getvar(ds, "QVAPOR", timeidx=matched_timeidx,meta=False) * 1000
    snow = getvar(ds, "QSNOW", timeidx=matched_timeidx,meta=False) * 1000
    ice = getvar(ds, "QICE", timeidx=matched_timeidx,meta=False) * 1000
    graupel  = getvar(ds, "QGRAUP", timeidx=matched_timeidx,meta=False) * 1000

lats = to_np(lats)
lons = to_np(lons)

# Step 1: Define the 2D mask at level 15
base_level = 15
snow_band_2d = cloud_frac[base_level, :, :] >= threshold

# Step 2: Label connected horizontal regions
labeled_2d, num_features = label(snow_band_2d)

# Step 3: Get the label at the starting point (x_y = [lon_idx, lat_idx])
start_label = labeled_2d[x_y[1], x_y[0]]

if start_label == 0:
    print("Starting point not in any region.")
    cloud_region_mask = np.zeros_like(cloud_frac, dtype=bool)
else:
    # Step 4: Get the 2D footprint of the connected region
    region_mask_2d = (labeled_2d == start_label)

    # Step 5: Extend vertically, only in columns that are part of the base region
    cloud_region_mask = np.zeros_like(cloud_frac, dtype=bool)
    for z in range(cloud_frac.shape[0]):
        cloud_region_mask[z, :, :] = region_mask_2d & (cloud_frac[z, :, :] >= threshold)

cloud_heights = np.where(cloud_region_mask, ht, np.nan)
mean_height = np.nanmean(cloud_heights)
print(f"Mean cloud height at level {ht_level}: {mean_height:.1f} m")
z_inds, lat_inds, lon_inds = np.where(cloud_region_mask)

# Isolate Mixing Ratio to Cloud Bands
wv_cloud = wv[cloud_region_mask]
snow_cloud = snow[cloud_region_mask]
ice_cloud = ice[cloud_region_mask]
graupel_cloud = graupel[cloud_region_mask]

snow_avg = np.nanmean(snow_cloud)
print(f"Snow Avg: {snow_avg:.6f}")

graupel_avg = np.nanmean(graupel_cloud)
print(f"Graupel Avg: {graupel_avg:.6f}")

ice_avg = np.nanmean(ice_cloud)
print(f"Ice Avg: {ice_avg:.6f}")

wv_avg = np.nanmean(wv_cloud)
print(f"Water Vapor Avg: {wv_avg:.6f}")


if len(lat_inds) == 0 or len(lon_inds) == 0:
    print("No grid boxes in the cloud region.")
else:
    print(f"Cloud region has {len(lat_inds)} grid points")
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

# Add Cartopy features
ax.add_feature(land,zorder=0)
ax.add_feature(ocean,zorder=0)
ax.add_feature(lakes,zorder=0)
ax.add_feature(states, edgecolor='gray',zorder=2)

# Your main field (e.g., reflectivity)
#cf = ax.contour(lons, lats, max_dbz, transform=crs.PlateCarree(),cmap="viridis",vmin=10,vmax=40,zorder=1)  # assuming 2D lat/lon

# Collapse the 3D Mask into 2D for plotting. Basically saying that if any level is true, show it as a cloud (Filled) on plot
cloud_mask_2d = np.max(cloud_region_mask, axis=0)
cf = ax.contourf(
    to_np(lons), to_np(lats), to_np(cloud_mask_2d.astype(float)),
    levels=[0.5, 1.5],              # Fill values above 0.5
    colors=['red'], alpha=0.3,      # Set fill color and transparency
    transform=crs.PlateCarree(), zorder=4
)
plt.colorbar(cf, ax=ax, orientation="vertical", label="Max Ref")

# Set the map bounds
ax.set_xlim(cartopy_xlim(dbz))
ax.set_ylim(cartopy_ylim(dbz))


plt.title("Cloud region linked to flash")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"identifycloudband_{time_str}.png"

plt.savefig(savepath + filename)
plt.show()




