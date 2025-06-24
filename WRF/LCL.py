from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import (get_cmap,ScalarMappable)
import matplotlib as mpl
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
import matplotlib.colors as colors
from wrf import (to_np,interplevel, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times,cape_2d,tk)
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import from_levels_and_colors
import math
from metpy.plots import USCOUNTIES
from metpy.plots import ctables
import pandas as pd
from datetime import datetime
import wrffuncs
# --- USER INPUT ---

wrf_date_time = datetime(2018,5,15,18,00,00)
domain = 2

savefig = True
windbarbs = False
gridlines = False

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

storm_path = "/data1/white/Downloads/MET416/Storm_reports/"
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
'''
# Open storm report file
df = pd.read_csv(storm_path + "180515_rpts_torn.csv", index_col=False,sep=",", header=0,
                 names=["Time", "F_Scale", "Location","County","State","Lat","Lon"])

# Read in the variables of storm Report
time = df['Time'].values
torlat = df['Lat'].values
torlon = df['Lon'].values
'''
# Get the CAPE values (And more, CAPE is at [0], CIN at [1], LCL at [2], and LFC at[3])
with Dataset(matched_file) as ds:
	cape = getvar(ds, "cape_2d", timeidx=matched_timeidx)

# Make levels for CAPE
cape_levels = np.arange(0, 4001, 500)

# Get the latitude and longitude points
lats, lons = latlon_coords(cape)

# Get the cartopy mapping object
cart_proj = get_cartopy(cape)

# Create a figure
fig = plt.figure(figsize=(30,15))

# Set the GeoAxes to the projection used by WRF
ax = plt.axes(projection=cart_proj)

# Download and add the states, lakes and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
ax.add_feature(states, linewidth=1, edgecolor="black")
ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
ax.coastlines('50m', linewidth=1)
ax.add_feature(USCOUNTIES, alpha=0.1)


# Make the filled countours with specified levels and range
qcs = plt.contourf(to_np(lons), to_np(lats),cape[2],levels=cape_levels,transform=crs.PlateCarree(),cmap="Greys")

# Add a color bar
cbar = plt.colorbar()
#plt.colorbar(ScalarMappable(),ax=ax)
cbar.set_label("Meters",fontsize=10)

# Set the map bounds
ax.set_xlim(cartopy_xlim(cape))
ax.set_ylim(cartopy_ylim(cape))

# Plot tornado location
#plt.scatter(to_np(torlon), to_np(torlat),transform=crs.PlateCarree())

# Add the gridlines

gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlines = False
gl.ylines = False
gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels

#Add the 500 hPa wind barbs, only plotting every 125th data point.
if windbarbs == True:
	plt.barbs(to_np(lons[::25,::25]), to_np(lats[::25,::25]),
          to_np(u_500[::25, ::25]), to_np(v_500[::25, ::25]),
          transform=crs.PlateCarree(), length=6)

#Show and/or Save
plt.title(f"LCL Height (m) at " + str(matched_time),{"fontsize" : 14})

time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"LCL_{time_str}.png"

plt.savefig(savepath+filename)

plt.show()
