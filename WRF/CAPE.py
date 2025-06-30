from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
from wrf import (to_np,getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
from metpy.plots import USCOUNTIES
import pandas as pd
import wrffuncs
from datetime import datetime

# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2
height = 850

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# Boolean to save figure and display wind barbs, gridlines
savefig = True
windbarbs = False
gridlines = False

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

    #Get the CAPE values (And more, CAPE is at [0], CIN at [1], LCL at [2], and LFC at[3])
    cape = getvar(ds, "cape_2d", timeidx=matched_timeidx)
'''
# Open storm report file
path = "/data1/white/Downloads/MET416/Storm_reports/"
df = pd.read_csv(path + "180515_rpts_torn.csv", index_col=False,sep=",", header=0,
                 names=["Time", "F_Scale", "Location","County","State","Lat","Lon"])

# Read in the variables of storm Report
time = df['Time'].values
torlat = df['Lat'].values
torlon = df['Lon'].values
'''

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

plt.title(f"CAPE (j/kg) at ",{"fontsize" : 14})

# Format it for a filename (no spaces/colons)
time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
# Use in filename
filename = f"CAPE_{time_str}.png"

plt.savefig(savepath + filename)
plt.show()
