import cfgrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import metpy
import metpy.calc  
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
import STORMY
from wrf import getvar, latlon_coords, to_np
from netCDF4 import Dataset

# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,14,00,00)
domain = 2

SIMULATION = 1 # If comparing runs
WRF_path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

MRMS_path = f"/data2/white/DATA/MISC/MRMS/MergedReflectivityQCComposite_00.50_20221118140034.grib2"
# Area to View
lon_min, lon_max = -78.5, -74.5
lat_min, lat_max = 42.5, 44.5

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(WRF_path, domain)
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

with Dataset(matched_file) as ds:
    mdbz = getvar(ds, "mdbz", timeidx=matched_timeidx)

fig = plt.figure(figsize=(10, 5), dpi=200)
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())  

# Define base map features (thin, minimal styling)
borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m', facecolor='none')
lakes = cfeature.LAKES.with_scale('50m')
states = cfeature.STATES.with_scale('50m')

for ax in [ax1, ax2]:
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(borders, edgecolor='gray', linewidth=0.5)
    ax.add_feature(lakes, edgecolor='lightgray', facecolor='none', linewidth=0.5)
    ax.add_feature(states, edgecolor='lightgray', linewidth=0.5)
    ax.coastlines(resolution='50m', linewidth=0.5)

# === Load GRIB Data ===
ds = cfgrib.open_dataset(MRMS_path)
ds = ds.metpy.parse_cf()

valid_time = pd.to_datetime(ds['valid_time'].values)
formatted_time = valid_time.strftime("%Y-%m-%d %H:%M")  # Just to the minute
print("Valid datetime (time + step):", formatted_time)

# Open and rename for clarity
ds = ds.rename({"unknown": "reflectivity"})
data = ds['reflectivity']  

if data.ndim != 2:
    raise ValueError(f"Incompatible MRMS field: expected 2D, got {data.ndim}D")

# Make Longitude -180 to 180 (instead of 0 to 360)
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

subset = ds.sel(
    longitude=slice(lon_min, lon_max),
    latitude=slice(lat_max, lat_min)  # Note: latitude is usually decreasing in WRF/GRIB
)

# Mask low reflectivity
reflectivity = subset['reflectivity'].where(subset['reflectivity'] > 0)

# Extract arrays
z = reflectivity.values
MRMS_lats = subset.latitude.values
MRMS_lons = subset.longitude.values

# Define value boundaries and colormap
levels = np.arange(0, 75, 5)  # 0 to 70 dBZ in 5 dBZ bins
cmap = plt.get_cmap("NWSRef")  # or your defined reflectivity colormap

# Create a BoundaryNorm for shared normalization
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)
# Plot MRMS with pcolormesh 
mesh = ax1.pcolormesh(MRMS_lons, MRMS_lats, z, cmap=cmap,norm=norm,shading="auto", transform=ccrs.PlateCarree())

# Plot WRF with contourf
lats, lons = latlon_coords(mdbz)
mdbz = np.where(to_np(mdbz) < 0, np.nan, to_np(mdbz))
mdbz_plot = ax2.contourf(to_np(lons), to_np(lats), mdbz,cmap=cmap,norm=norm, transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(mesh, ticks=levels, orientation='horizontal', extend='both',
                  cax=fig.add_axes([0.1325, 0.25, 0.76, 0.02]))

cb.ax.tick_params(labelsize=5, labelcolor='black', width=0.5, length=1.5, direction='out', pad=1.0)
cb.set_label(label='{}'.format("dBZ"), size=5, color='black', weight='normal')
cb.outline.set_linewidth(0.5)


ax2.set_title(f"WRF Composite Reflectivity at {matched_time}",fontsize=10)

plt.show()

