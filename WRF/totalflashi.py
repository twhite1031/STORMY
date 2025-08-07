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
from metpy.plots import USCOUNTIES, ctables
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import STORMY
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,18,13,50), datetime(2022, 11, 18,14, 00)
domain = 2

# Area you would like the plan view to look at (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-77.5, -74.0,42.5, 44.5] 
hend_harb = [43.88356, -76.155543] # Henderson Harbor coordinates

# Path to each WRF run (NORMAL & FLAT)
path_1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_1 = STORMY.build_time_df(path_1, domain)
time_df_2 = STORMY.build_time_df(path_2, domain)
time_df_3 = STORMY.build_time_df(path_3, domain)

# Filter time range
mask_1 = (time_df_1["time"] >= start_time) & (time_df_1["time"] <= end_time)
mask_2 = (time_df_2["time"] >= start_time) & (time_df_2["time"] <= end_time)
mask_3 = (time_df_3["time"] >= start_time) & (time_df_3["time"] <= end_time)

time_df_1 = time_df_1[mask_1].reset_index(drop=True)
time_df_2 = time_df_2[mask_2].reset_index(drop=True)
time_df_3 = time_df_3[mask_3].reset_index(drop=True)

wrf_filelist_1 = time_df_1["filename"].tolist()
wrf_filelist_2 = time_df_2["filename"].tolist()
wrf_filelist_3 = time_df_3["filename"].tolist()

timeidxlist = time_df_1["timeidx"].tolist() # Assuming time indexes are the same
timelist = time_df_1["time"].tolist()

def generate_frame(file_path1,file_path2,file_path3, timeidx,time):
    print("Starting generate frame")
        
    try:
    # Read data from WRF file
        with Dataset(file_path1) as wrfin:
            data1 = getvar(wrfin, "FLSHI", timeidx=timeidx, meta=False)
        with Dataset(file_path2) as wrfin:
            data2 = getvar(wrfin, "FLSHI", timeidx=timeidx, meta=False)
            
        with Dataset(file_path3) as wrfin:
            data3 = getvar(wrfin, "FLSHI", timeidx=timeidx,meta=False)

    # Convert arrays intop numpy arrays, np.sum is used so we only get the x,y coordinates of the flash          
        flash_data1 = np.sum(data1, axis=0)
        flash_data2 = np.sum(data2, axis=0)
        flash_data3 = np.sum(data3, axis=0)

    # Plot each lon and lat points, based on if there is data (1.0) in the respective flash_data array
        ax.scatter(lons[flash_data1 == 1.0],lats[flash_data1 == 1.0],s=75,marker="*",c='yellow',transform=crs.PlateCarree())
        ax.scatter(lons[flash_data2 == 1.0],lats[flash_data2 == 1.0],s=75,marker="*",c='red',transform=crs.PlateCarree())
        ax.scatter(lons[flash_data3 == 1.0],lats[flash_data3 == 1.0],s=75,marker="*",c='blue',transform=crs.PlateCarree())


        global count
        global coords
        # If we wanted to track the coords and counts of certain flashes, we can use this
        if any(lons[flash_data1 == 1.0]):
            count += 1
            times.append(time)
            coords.append((lats[flash_data1 == 1.0],lons[flash_data1 == 1.0]))            

    except Exception as e:
        print(f"Error processing files, see {e}")
            
if __name__ == "__main__":

    count = 0
    coords = []
    times = []

    # Read data from any matching WRF file to set up the plot, not for the data
    with Dataset(wrf_filelist_1[0]) as wrfin:
        data = getvar(wrfin, "FLSHI", timeidx=0)
     
    # Get the lat/lon points and projection object from WRF data
    lats, lons = latlon_coords(data)
    cart_proj = get_cartopy(data)
    WRF_ylim = cartopy_ylim(data)
    WRF_xlim = cartopy_xlim(data)

    # Create a figure
    fig = plt.figure(figsize=(12,9),facecolor='white')
    ax = plt.axes(projection=cart_proj)
    
    # Read in detailed county lines
    reader = shpreader.Reader('../COUNTY_SHAPEFILES/countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=5)
    ax.add_feature(COUNTIES,facecolor='none', edgecolor='black',linewidth=1)

    # Set the map bounds
    ax.set_extent(extent)

    # Add custom formatted gridlines using STORMY function
    STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) 
   
    lats = to_np(lats)
    lons = to_np(lons)
    # Loop through each file, plotting and storing flash initation information
    for idx, file_path in enumerate(wrf_filelist_1):
        generate_frame(wrf_filelist_1[idx], wrf_filelist_2[idx], wrf_filelist_3[idx], timeidxlist[idx],timelist[idx])
        
    # Add a title and show the Figure
    plt.title("Total Flash Initiation Locations",fontsize=28)
    plt.show()         

       
