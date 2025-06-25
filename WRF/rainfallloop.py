import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import concurrent.futures
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,cartopy_xlim, cartopy_ylim,extract_times)
from matplotlib.cm import (get_cmap,ScalarMappable)
import glob
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES, ctables
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime,timedelta
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import wrffuncs
import string
"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF

# --- USER INPUT ---
start_time, end_time  = datetime(2023,1,9,7,40,00), datetime(2023, 1, 9,7, 55, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path = f"/data2/white/wrf/WRFV4.5.2/run/"
# Path to save GIF or Files
savepath = f""

# --- END USER INPUT ---
time_df = wrffuncs.build_time_df(path, domain)

mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()

# Get Starting point data to substract from (Start at 0 mm)
with Dataset(filelist[0]) as ds:
            start_rain_nc = getvar(ds, "RAINNC",timeidx=timeidxlist[0])
            start_rain_c = getvar(ds, "RAINC",timeidx=timeidxlist[0])

start_total_rain = start_rain_nc + start_rain_c
start_time = wrffuncs.parse_filename_datetime_wrf(filelist[0],timeidxlist[0])



# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, timeidx = args
    try:
   
    # Read data from file
        with Dataset(file_path_N) as ncfile:
            rain_nc = getvar(ncfile, "RAINNC",timeidx=timeidx)
            rain_c = getvar(ncfile, "RAINC",timeidx=timeidx)

            ua = getvar(ncfile, "ua",timeidx=timeidx, units="m s-1")
            va = getvar(ncfile, "va",timeidx=timeidx, units="m s-1")
            wspd = getvar(ncfile, "uvmet_wspd_wdir", timeidx=timeidx,units="m s-1")[0,:]
        #print(f"q.shape: {q.shape}")   # Should be (levels, latitudes, longitudes)
        #print(f"ua.shape: {ua.shape}") # Should be (levels, latitudes, longitudes)
        #print(f"va.shape: {va.shape}") # Should be (levels, latitudes, longitudes)
        #print(f"p.shape: {p.shape}")   # Should be (levels, latitudes, longitudes)

        #print("Read in WRF data")
        cart_proj = get_cartopy(ua)
  
    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_N = plt.axes(projection=cart_proj)

        #print("Created Figures")

    # Get the latitude and longitude points
        lats, lons = latlon_coords(ua)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax_N.add_feature(states, linewidth=.1, edgecolor="black")
        ax_N.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_N.coastlines('50m', linewidth=1)
        ax_N.add_feature(USCOUNTIES, alpha=0.1)
        #print("Made land features")
    # Set the map bounds
        ax_N.set_xlim(cartopy_xlim(ua))
        ax_N.set_ylim(cartopy_ylim(ua))
        #print("Set map bounds")

    # Add the gridlines
        gl_N = ax_N.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_N.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_N.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_N.xlines = True
        gl_N.ylines = True
        gl_N.top_labels = False  # Disable top labels
        gl_N.right_labels = True # Disable right labels
        gl_N.xpadding = 20
        #gl_N.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        #gl_N.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees
        # Format the gridline labels
        gl_N.xformatter = LONGITUDE_FORMATTER
        gl_N.yformatter = LATITUDE_FORMATTER


        wspd_interval = .2
        #vmin = (np.min(ivt) // wspd_interval) * wspd_interval
        #vmax = (np.max(ivt) // wspd_interval + 1) * wspd_interval
        vmin = 0
        vmax = 12
        #print(f"vmin: {vmin} , vmax: {vmax}")
        levels = np.arange(vmin, vmax + wspd_interval, wspd_interval)
        
        # Get total rain from start time, then convert to inches
        total_rain = ((rain_nc + rain_c) - start_total_rain) / 25.4
        wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(total_rain),levels=levels,cmap=get_cmap("turbo"), transform=crs.PlateCarree())
        
        cbar = plt.colorbar(wspd_contours, ax=ax_N, orientation="horizontal", pad=.075,shrink=0.6)
        cbar.set_label('Rainfall (mm)',fontsize=14)

        
    # Set titles, get readable format from WRF time
        time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(file_path_N, timeidx)
        ax_N.set_title(f"Rainfall starting at " + str(start_time),fontsize=18,fontweight='bold')
                     

    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        #print(f"Max IVT for this frame: {max_ivt}")
        #print("Saving Frame")
        plt.savefig(filename)
        #plt.show()
        #print(f"{os.path.basename(file_path_N)} Processed!")
        plt.close()

        return filename
    except ValueError:
        print("Error processing files")
        
def create_gif(frame_filenames, output_filename):

    frames = []
    for filename in frame_filenames:
            new_frame = Image.open(filename)
            frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(savepath + output_filename, format='GIF', append_images=frames[1:],save_all=True,duration=75, loop=0)
    
if __name__ == "__main__":
    # Generate tasks
    tasks = zip(filelist, timeidxlist)
    start_str = start_time.strftime("%Y%m%d_%H%M")
    end_str   = end_time.strftime("%Y%m%d_%H%M")
    output_gif = f'rainfall_LOOP_D{domain}{start_str}_to_{end_str}.gif'

    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        results = list(executor.map(generate_frame, tasks))
        frame_filenames = results
        frame_filenames = list(frame_filenames)
            
    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]  

    create_gif(sorted(filtered_list), output_gif)
    # Clean up the frame files
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
