import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, SymLogNorm, LogNorm
from matplotlib.ticker import SymmetricalLogLocator, LogFormatterSciNotation
import pyart
from netCDF4 import Dataset
from wrf import getvar, vertcross, interplevel, to_np, latlon_coords, CoordPair, get_basemap, ll_to_xy, interpline,get_cartopy, cartopy_xlim, cartopy_ylim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from datetime import datetime, timedelta
from pyproj import Geod
import wrffuncs


# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,23,00,00)
domain = 2

start_point = CoordPair(lat=43.86935, lon=-76.669)  # Example coordinates for A
end_point = CoordPair(lat=43.86935, lon=-75.659)    # Example coordinates for B

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
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
    z = getvar(ds, "z", timeidx=matched_timeidx)
    dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
    mdbz = getvar(ds, "mdbz",timeidx=matched_timeidx)
    w = getvar(ds, "wa", timeidx=matched_timeidx) # Default unit m/s
    charge_density = getvar(ds, "SCTOT", timeidx=matched_timeidx)
    flash_chan_pos = getvar(ds, "FLSHP", timeidx=matched_timeidx)
    flash_chan_neg = getvar(ds, "FLSHN",timeidx=matched_timeidx)
    flash_init = getvar(ds,"FLSHI", timeidx=matched_timeidx)
    ter = getvar(ds, "ter", timeidx=matched_timeidx)

    Z = 10**(dbz/10.) # Use linear Z for interpolation

    # Cross sections
    Z_cross = vertcross(Z, z, wrfin=ds, start_point=start_point, end_point=end_point, latlon=True)
    # Convert back to dBz after interpolation
    dbz_cross = 10.0 * np.log10(Z_cross)
    charge_cross = vertcross(charge_density, z, wrfin=ds, start_point=start_point, end_point=end_point, latlon=True)
    flash_cross = vertcross(flash_chan_pos, z, wrfin=ds, start_point=start_point, end_point=end_point, latlon=True)
    w_cross = vertcross(w, z, wrfin=ds, start_point=start_point, end_point=end_point, latlon=True)
    ter_line = interpline(ter, wrfin=ds, start_point=start_point,end_point=end_point)

# === Fixing Gap for Cross Sections ===
# Make a copy of the z cross data. 
dbz_cross_filled = np.ma.copy(to_np(dbz_cross))

for i in range(dbz_cross_filled.shape[-1]):
    column_vals = dbz_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    dbz_cross_filled[0:first_idx, i] = dbz_cross_filled[first_idx, i]

# Make a copy of the charge cross data. 
charge_cross_filled = np.ma.copy(to_np(charge_cross))

for i in range(charge_cross_filled.shape[-1]):
    column_vals = charge_cross_filled[:,i]
    first_idx = int(np.transpose((column_vals > -200).nonzero())[0])
    charge_cross_filled[0:first_idx, i] = charge_cross_filled[first_idx, i]

# === Var testing ===
min_val = float(np.nanmin(charge_density))
max_val = float(np.nanmax(charge_density))

print(f"Charge Density Range: {min_val:.2e} to {max_val:.2e} C/m³")

fig, axs = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
# Add more vertical spacing between subplots
plt.subplots_adjust(hspace=0.35)  # Increase this for more space (default is ~0.2)
axs[0].set_title(f"WRF Vertical Cross-Section (Init: 2022-11-17 00:00:00, Valid: {matched_time})", loc='left')

# Set y and x axis for cross section, all should be the same
xs = np.arange(0, dbz_cross.shape[-1], 1)
ys = to_np(dbz_cross.coords["vertical"])

# Step 1: Compute total distance (great circle) between endpoints 
geod = Geod(ellps="WGS84")
_, _, total_distance_m = geod.inv(start_point.lon, start_point.lat,
                                  end_point.lon, end_point.lat)
total_distance_km = total_distance_m / 1000

# Step 2: Map xs indices to real km distance (assuming linear spacing)
nx = len(xs)
xs_km = np.linspace(0, total_distance_km, nx)
xs = xs_km # !!! If you want the x index of each variable to mean km distance (instead of just indexes of points)
print(xs)

# === Panel A: Reflectivity ===
cs1 = axs[0].contourf(xs,ys,to_np(dbz_cross_filled), levels=np.linspace(0, 75, 500), cmap="NWSRef")
#axs[0].contour(xs, ys, to_np(dbz_cross), levels=[10, 20,30, 40, 50,60], colors='black', linewidths=0.5) # Enhance visibility of reflectivity bins

cbar = plt.colorbar(cs1, ax=axs[0], orientation="horizontal", pad=0.22, aspect=100)
cbar.set_label("Reflectivity (dBZ)")

# Set custom ticks
tick_vals = np.arange(0,75,5)
cbar.set_ticks(tick_vals)

# === Panel B: Space charge density (log scale) ===

# Use symmetric range around zero, vmin is negative vmax
vmax = 1.6e-9
vmin = -vmax
linthresh = 1e-13  # Linear threshold around 0

norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10) # Create norm for data

# Contour levels
levels_neg = -np.logspace(np.log10(linthresh), np.log10(vmax), 500)[::-1]
levels_pos = np.logspace(np.log10(linthresh), np.log10(vmax), 500)
levels = np.concatenate((levels_neg, [0], levels_pos))


cs2 = axs[1].contourf(xs, ys, to_np(charge_cross_filled), levels=levels, cmap='bwr', norm=norm)
# Colorbar
cbar = plt.colorbar(cs2, ax=axs[1], orientation='horizontal', extend='both', pad=0.22, aspect=100)
cbar.set_label("Total Charge Density (C / m^3)")
# This sets tick *locations* that match SymLogNorm behavior
cbar.locator = SymmetricalLogLocator(base=10, linthresh=1e-13)

# This formats tick labels like 1e-9, 1e-10, etc.
cbar.formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)

# Apply those changes
cbar.update_ticks()

# Set custom tick positions (must be within vmin to vmax)
tick_vals = [-1e-9, -1e-10, -1e-11, -1e-12,0,1e-12,1e-11, 1e-10, 1e-9]
cbar.set_ticks(tick_vals)
#cbar.set_label("Total Space Charge Density (C m⁻³)")

# === Panel C: Flash extent density (positive/negative) ===

# Define range
vmin = -5
vmax = 5
vcenter = 0

# Normalization centered at 0
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Contour levels: smooth, linear spacing
levels = np.linspace(vmin, vmax, 200)

cs3 = axs[2].contourf(xs,ys,to_np(flash_cross),levels=levels, cmap="bwr",norm=norm)
cbar = plt.colorbar(cs3, ax=axs[2], orientation="horizontal",pad=0.22, aspect=100)
cbar.set_label("Net Positive/Negative Channel Count (#)")
# Set custom ticks
tick_vals = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
cbar.set_ticks(tick_vals)

# === Vertical Velocity Vectors ===
# Step 1: Define strides
x_stride = 2
z_stride = 10

# Step 2: Build index grids
x_idx = np.arange(0, w_cross.shape[1], x_stride)
z_idx = np.arange(0, w_cross.shape[0], z_stride)

# Step 3: Create meshgrid of selected x and z indices
xg, zg = np.meshgrid(x_idx, z_idx)

# Step 4: Get coordinates and vertical velocity
x_plot = xs[xg]                # in km
y_plot = ys[zg]                   # in meters
w_plot = to_np(w_cross)[zg, xg]   # vertical velocity in m/s

ticks = np.arange(0, total_distance_km + 1, 5) # Set km ticks
for ax in axs:
    ax.fill_between(xs, 0, to_np(ter_line), facecolor="saddlebrown")
    #ax.quiver(x_plot, y_plot, np.zeros_like(w_plot), w_plot, scale=100, width=0.002, color='black', zorder=10)
    ax.tick_params(labelbottom=True) # Show X ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(km)}" for km in ticks])
    ax.set_ylabel("Height above MSL (m)")
    ax.set_xlabel("Distance from Point A (km)", labelpad=2)
    ax.set_ylim(0, 8000)


# === Plan View ===
lats, lons = latlon_coords(mdbz)
cart_proj = get_cartopy(mdbz)

# Extract the lat/lon of the start and end points
start_lat, start_lon = start_point.lat, start_point.lon
end_lat, end_lon = end_point.lat, end_point.lon

# Set up the map
fig_map = plt.figure(figsize=(8, 6))
ax_map = plt.axes(projection=cart_proj)

# Set the map bounds based on data
#ax_map.set_xlim(cartopy_xlim(mdbz))
#ax_map.set_ylim(cartopy_ylim(mdbz))

# Set the map bounds based on preference
lon_min, lon_max = -78.5, -74.5
lat_min, lat_max = 43.0, 44.5
ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Plot terrain (or background)
mdbz_plot = ax_map.contourf(to_np(lons), to_np(lats), to_np(mdbz),
                               levels=np.linspace(0,75,100), cmap="NWSRef", transform=ccrs.PlateCarree())

# Add cross-section line
ax_map.plot([start_lon, end_lon], [start_lat, end_lat], 'k-', marker='.',linewidth=2, transform=ccrs.PlateCarree())

# Label endpoints A and B
ax_map.text(start_lon, start_lat, 'A', fontsize=12, fontweight='bold', transform=ccrs.PlateCarree(),
            ha='right', va='bottom')
ax_map.text(end_lon, end_lat, 'B', fontsize=12, fontweight='bold', transform=ccrs.PlateCarree(),
            ha='left', va='bottom')

# Add features
ax_map.add_feature(cfeature.BORDERS, linestyle=':')
ax_map.add_feature(cfeature.STATES, linewidth=0.5)
ax_map.coastlines('10m', linewidth=0.5)

# Add the gridlines
gl = ax_map.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True

gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20

# Optional: add colorbar
cbar = plt.colorbar(mdbz_plot, ax=ax_map, orientation='vertical', shrink=0.8)
cbar.set_label("Reflectivity (dBZ)")
cbar.set_ticks(np.arange(0,75,5))
ax_map.set_title(f"Simulated Max Reflectivity at {matched_time} with Cross-Section A–B")

#plt.tight_layout()
plt.show()

