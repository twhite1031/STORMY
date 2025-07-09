import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import concurrent.futures
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords,extract_times)
from matplotlib.cm import (get_cmap,ScalarMappable)
import glob
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import colorcet as cc
import wrffuncs
from datetime import datetime, timedelta



# ---- User input for file ----

# User input for file
wrf_date_time = datetime(2022,11,18,1,55,00)
domain = 2
numtimeidx = 4
IOP = 2
ATTEMPT = 1

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.965996,-75.00000,43.000,44.273301]

wrf_date_time = wrffuncs.round_to_nearest_5_minutes(wrf_date_time)
print("Closest WRF time to match input time: ", wrf_date_time)
timeidx = int((wrf_date_time.minute % 20) // (20 // numtimeidx))
wrf_filename = wrf_date_time
wrf_min = (round((wrf_filename.minute)// 20) * 20)
wrf_filename = wrf_filename.replace(minute=wrf_min)
print("Using data from this file: ", wrf_filename)


# ---- End User input for file ----

# Open the NetCDF file
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_{IOP}/ATTEMPT_{ATTEMPT}/"
pattern = f"wrfout_d0{domain}_{wrf_filename.year:04d}-{wrf_filename.month:02d}-{wrf_filename.day:02d}_{wrf_filename.hour:02d}:{wrf_filename.minute:02d}:00"
wrffile = path + pattern

def generate_frame(wrffile, timeidx):

    # Read data from file
    with Dataset(wrffile) as wrfin:
        data = getvar(wrfin, "ELECMAG", timeidx=timeidx)[0,:,:]
           
    # Create a figure
    fig = plt.figure(figsize=(30,15),facecolor='white')

    # Get the latitude and longitude points
    lats, lons = latlon_coords(data)

    # Get the cartopy mapping object
    cart_proj = get_cartopy(data)
    
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)

    # Download and add the states, lakes  and coastlines
    states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
    ax.add_feature(states, linewidth=.1, edgecolor="black")
    ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
    ax.coastlines('50m', linewidth=1)
    ax.add_feature(USCOUNTIES, alpha=0.1)
    
    # Set the map bounds
    ax.set_xlim(cartopy_xlim(data))
    ax.set_ylim(cartopy_ylim(data))
    
    ax.set_facecolor('white')

    # Add the gridlines
    gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabel_style = {'rotation': 'horizontal','size': 28,'ha':'center'} # Change 14 to your desired font size
    gl.ylabel_style = {'size': 28}  # Change 14 to your desired font size
    gl.xlines = True
    gl.ylines = True
    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xpadding = 20
    
    
    #norm = Normalize(vmin=100000,vmax=180000)
    #data = data[0,:,:]

    # Define contour levels
    levels = np.arange(0,180000,1000)    
    qcs = plt.contourf(to_np(lons), to_np(lats),data,transform=crs.PlateCarree(), cmap=cc.m_fire_r, levels=levels,vmin=0,vmax=180000)

    # Add a fixed colorbar
    #cbar = plt.colorbar(ScalarMappable(cmap='Spectral_r',norm=qcs.norm), ax=ax)
    cbar = plt.colorbar()
    cbar.set_label("", fontsize=22)
    cbar.ax.tick_params(labelsize=14)  
    
    # Adjust format for date to use in figure
    date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")

    
    ax.set_title(f"Electric Field Magnitude at {date_format}" ,fontsize=25,fontweight='bold')
    
    savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{ATTEMPT}/"

    # Save the figure to a file
    filename = "ELECMAG{wrf_date_time.year:04d}{wrf_date_time.month:02d}{wrf_date_time.day:02d}{wrf_date_time.hour:02d}{wrf_date_time.minute:02d}D{domain}T{timeidx}A{ATTEMPT}.png"
    plt.savefig(savepath + filename)
    plt.show()

if __name__ == "__main__":
    generate_frame(wrffile, timeidx)

    
