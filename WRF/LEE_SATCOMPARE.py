import GOES
import custom_color_palette as ccp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair,cartopy_ylim,cartopy_xlim)
import wrffuncs
from datetime import datetime
import pandas as pd



# --- USER INPUT ---
wrf_date_time = datetime(2022,11,18,13,50,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

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
    CTT = getvar(ds, "ctt", timeidx=matched_timeidx,units='k')

# Get the lat/lon points
lats, lons = latlon_coords(CTT)
xlim, ylim = cartopy_xlim(CTT), cartopy_ylim(CTT)
print(xlim)
print(lats)
print(lons)
# Get the cartopy projection object
cart_proj = get_cartopy(CTT)

path = '/data2/white/DATA/SATELLITE/'
file = 'OR_ABI-L2-CMIPF-M6C13_G16_s20223221350207_e20223221359527_c20223221400014.nc'

pad = 0.75  # degrees, to match WRF and Satellite
domain = [np.min(to_np(lons)) + pad,np.max(to_np(lons)) - pad,np.min(to_np(lats)) + pad,np.max(to_np(lats)) - pad]
print(domain)
ds = GOES.open_dataset(path+file)

CMI, LonCor, LatCor = ds.image('CMI', lonlat='corner', domain=domain)

sat = ds.attribute('platform_ID')
band = ds.variable('band_id').data[0]
wl = ds.variable('band_wavelength').data[0]

lower_colors = ['maroon','red','darkorange','#ffff00','forestgreen','cyan','royalblue',(148/255,0/255,211/255)]
lower_palette = [lower_colors, ccp.range(190.0,250.0,1.0)]

upper_colors = plt.cm.Greys
upper_palette = [upper_colors, ccp.range(250.0,340.0,1.0), [ccp.range(180.0,330.0,1.0),240.0,330.0]]

# pass parameters to the creates_palette module
cmap, cmticks, norm, bounds = ccp.creates_palette([lower_palette, upper_palette], extend='both')

# creating colorbar labels
ticks = ccp.range(180,330,10)

# calculates the central longitude of the plot
lon_cen = 360.0+(domain[0]+domain[1])/2.0

# creates the figure
# Create figure with two axes, each with different projections
fig = plt.figure(figsize=(10, 5), dpi=200)

ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# add the geographic boundaries
l = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none')
ax1.add_feature(l, edgecolor='gold', linewidth=0.25)
ax2.add_feature(l, edgecolor='gold', linewidth=0.25)

# plot the data
img = ax1.pcolormesh(LonCor.data, LatCor.data, CMI.data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
plt.contourf(to_np(lons), to_np(lats), to_np(CTT), cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(img, ticks=ticks, orientation='horizontal', extend='both',
                  cax=fig.add_axes([0.1325, 0.21, 0.76, 0.02]))

cb.ax.tick_params(labelsize=5, labelcolor='black', width=0.5, length=1.5, direction='out', pad=1.0)
cb.set_label(label='{} [{}]'.format(CMI.standard_name, CMI.units), size=5, color='black', weight='normal')
cb.outline.set_linewidth(0.5)

# set the titles
ax1.set_title('{} - C{:02d} [{:.1f} μm]'.format(sat,band, wl), fontsize=7, loc='left')
ax1.set_title(CMI.time_bounds.data[0].strftime('%Y/%m/%d %H:%M UTC'), fontsize=7, loc='right')

ax2.set_title(f"WRF Cloud Top Temperature at {matched_time}", fontsize=7)
# Longitude & Latitude Formatters
def format_lon(x, pos):
    return f"{x:.1f}°"

def format_lat(y, pos):
    return f"{y:.1f}°"

# Set up tick positions
dx = 1.0
dy = 0.5
xticks = np.arange(domain[0], domain[1] + dx, dx)
yticks = np.arange(domain[2], domain[3] + dy, dy)

# === Shared Axis Formatting Function ===
def format_map_axis(ax):

    
    # Ticks
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    # Format labels
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

    # Labels
    ax.set_xlabel('Longitude', color='black', fontsize=7, labelpad=3.0)
    ax.set_ylabel('Latitude', color='black', fontsize=7, labelpad=3.0)

    # Tick appearance
    ax.tick_params(left=True, right=True, bottom=True, top=True,
                   labelleft=True, labelright=False, labelbottom=True, labeltop=False,
                   length=0.0, width=0.05, labelsize=5.0, labelcolor='black')

    # Gridlines
    ax.gridlines(xlocs=xticks, ylocs=yticks, alpha=0.6, color='gray',
                 draw_labels=False, linewidth=0.25, linestyle='--')

    # Extent
    ax.set_extent([domain[0]+360.0, domain[1]+360.0, domain[2], domain[3]], crs=ccrs.PlateCarree())

# === Apply to both axes ===
format_map_axis(ax1)
format_map_axis(ax2)


plt.show()

