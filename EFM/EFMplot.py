from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.cm import (get_cmap,ScalarMappable)
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import numpy as np
import matplotlib.colors as colors
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times,ll_to_xy)
	
'''
Matching the electric field obs for IOP_2 and WRF
'''

# This script is to match the electric field obs for IOP_2
lat_lon2300 = [43.983548, -76.133277]
lat_lon140 = [43.8883, -76.1567]
lat_lon250 = [43.8883, -76.1567]

attempt = 1

# Open the NetCDF file

path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{attempt}/"

#Timeidx 0
pattern1 = f"wrfout_d02_2022-11-18_23:00:00"

#Timeidx 0
pattern2 = f"wrfout_d02_2022-11-19_01:40:00"

#Timeidx 2
pattern3 = f"wrfout_d02_2022-11-19_02:40:00"
x_y2300 = ll_to_xy(Dataset(path+pattern1), lat_lon2300[0], lat_lon2300[1])
x_y140 = ll_to_xy(Dataset(path+pattern1), lat_lon140[0], lat_lon140[1])
x_y250 = ll_to_xy(Dataset(path+pattern1), lat_lon250[0], lat_lon250[1])

# Get the electric field z component and temperature in celsius
emag1 = getvar(Dataset(path+pattern1), "ELECZ", timeidx=0)[:,x_y2300[1],x_y2300[0]]
tc1 = getvar(Dataset(path+pattern1), "tc", timeidx=0)[:,x_y2300[1],x_y2300[0]]
z = getvar(Dataset(path+pattern1), "z",timeidx=0)[:,x_y2300[1],x_y2300[0]]


emag2 = getvar(Dataset(path+pattern2), "ELECZ", timeidx=0)[:,x_y140[1],x_y140[0]]
tc2 = getvar(Dataset(path+pattern2), "tc", timeidx=0)[:,x_y140[1],x_y140[0]]


tc3 = getvar(Dataset(path+pattern3), "tc", timeidx=2)[:,x_y250[1],x_y250[0]]
emag3 = getvar(Dataset(path+pattern3), "ELECZ", timeidx=2)[:,x_y250[1],x_y250[0]]


fig, ax = plt.subplots(figsize=(6,8))

# Set the font properties to match the example
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'  

for idx,value in enumerate(emag1):
    emag1[idx] = value/1000
    emag2[idx] = emag2[idx]/1000
    emag3[idx] = emag3[idx]/1000

# Function to plot segments with color based on slope
def plot_segments(x, y, ax):
    for i in range(len(x) - 1):
        slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        color = 'red' if slope > 0 else 'blue'
        ax.plot(x[i:i + 2], y[i:i + 2], color=color)

# Determine the color based on the slope
plot_segments(emag1, z, ax)
plot_segments(emag2, z, ax)
plot_segments(emag3, z, ax)

ax.plot(tc1,z,c="black")
ax.plot(tc2,z,c="black")
ax.plot(tc3,z,c="black")

ax.annotate('2300 UTC -', xy=(-3.6, 915), xytext=(-44.5, 915),fontsize='14')

ax.annotate('- 0140 UTC', xy=(4, 776), xytext=(4, 776),fontsize='14')
ax.annotate('- 0250 UTC', xy=(39.3, 1015), xytext=(40.0, 1015),fontsize='14')
plt.xticks([-100,-50,0,50,100], fontsize='20')
plt.xlabel('T ($^\circ$C), E$_z$ (kV m$^{-1}$)', fontsize='20')
plt.setp(ax.spines.values(), linewidth=2)
plt.ylim(0,6000)
plt.yticks([0,1000,2000,3000,4000,5000,6000], labels=["0","1","2","3","4","5","6"],fontsize='20')
plt.ylabel('Altitude (km)',fontsize='20')
fig.tight_layout()

# Add a thin vertical line at x=0
ax.axvline(x=0, color='black', linewidth=0.5)
# Directory to store saved files
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
plt.savefig(savepath+f"EFMplotIOP2A{attempt}.png")

#plt.legend()
plt.show()
