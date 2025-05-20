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
# ---- User input for file ----

# Format in YEAR, MONTH, DAY, HOUR, MINUTE, DOMAIN
fileinfo = [2018, "05", 15, "18", "20", 2]

# Time index of file
timeidx = 9

# Boolean to save figure and display wind barbs, gridlines
savefig = True
windbarbs = False
gridlines = False

# ---- End User input ----

# Open the NetCDF file
path = f"/data1/white/WRF_Outputs/MET416/"
pattern = f"wrfout_d0{fileinfo[5]}_{fileinfo[0]}-{fileinfo[1]}-{fileinfo[2]}_{fileinfo[3]}:{fileinfo[4]}:00"
ncfile = Dataset(path+pattern)

# Open storm report file
path = "/data1/white/Downloads/MET416/Storm_reports/"
df = pd.read_csv(path + "180515_rpts_torn.csv", index_col=False,sep=",", header=0,
                 names=["Time", "F_Scale", "Location","County","State","Lat","Lon"])

# Read in the variables of storm Report
time = df['Time'].values
torlat = df['Lat'].values
torlon = df['Lon'].values

# Get the CAPE values (And more, CAPE is at [0], CIN at [1], LCL at [2], and LFC at[3])
cape = getvar(ncfile, "cape_2d", timeidx=timeidx)

# Make levels for CAPE
cape_levels = np.arange(0, 3000, 500)

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
qcs = plt.contourf(to_np(lons), to_np(lats),cape[0],levels=cape_levels,transform=crs.PlateCarree(),cmap="hot_r")

# Add a color bar
plt.colorbar()
#plt.colorbar(ScalarMappable(),ax=ax)
#plt.colorbar.set_label("dBZ",fontsize=10)

# Set the map bounds
ax.set_xlim(cartopy_xlim(cape))
ax.set_ylim(cartopy_ylim(cape))

# Plot tornado location
plt.scatter(to_np(torlon), to_np(torlat),transform=crs.PlateCarree())

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
date = np.datetime_as_string(extract_times(ncfile,timeidx=timeidx,do_xtime=False))[0:19]
plt.title(f"CAPE (j/kg) at " + date,{"fontsize" : 14})

if savefig == True:
	plt.savefig("/data1/white/PLOTS_FIGURES/MET416/CAPE" + date + ".png")
plt.show()
