import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import STORMY
import string
import pandas as pd

"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""
# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2

SLE_ratio = 10

path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_N = STORMY.build_time_df(path_N, domain)
time_df_F = STORMY.build_time_df(path_F, domain)

# Filter time range
mask_N = (time_df_N["time"] >= start_time) & (time_df_N["time"] <= end_time)
mask_F = (time_df_F["time"] >= start_time) & (time_df_F["time"] <= end_time)

time_df_N = time_df_N[mask_N].reset_index(drop=True)
time_df_F = time_df_F[mask_F].reset_index(drop=True)

filelist_N = time_df_N["filename"].tolist()
filelist_F = time_df_F["filename"].tolist()
timelist = time_df_N["time"].tolist()
timeidxlist = time_df_N["timeidx"].tolist()

# Get Starting point data to substract from since SNOWNC 
with Dataset(filelist_N[0]) as ds:
    start_snow_N = getvar(ds, "SNOWNC", timeidx=timeidxlist[0])

with Dataset(filelist_F[0]) as ds:
    start_snow_F = getvar(ds, "SNOWNC", timeidx=timeidxlist[0])


start_time, end_time = timelist[0], timelist[-1]

# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, file_path_F, timeidx, time = args
    try:
   
    # Read data from WRF file
        with Dataset(file_path_N) as wrfin:
            SWE_N = getvar(wrfin, "SNOWNC", timeidx=timeidx) 
        with Dataset(file_path_F) as wrfin2:
            SWE_F = getvar(wrfin2, "SNOWNC", timeidx=timeidx) 
            
        print("Read in WRF data")

    # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(SWE_N)
        cart_proj = get_cartopy(SWE_N)
        WRF_ylim = cartopy_ylim(SWE_N)
        WRF_xlim = cartopy_xlim(SWE_N)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_N = fig.add_subplot(1,2,1, projection=cart_proj)
        ax_F = fig.add_subplot(1,2,2, projection=cart_proj)
        axs = [ax_N, ax_F]
        print("Created Figures")

    # Apply cartopy features to the axes (States, lakes, etc.) using STORMY helper function 
        for ax in axs:
            STORMY.add_cartopy_features(ax)
            STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines
            ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
            ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

        print("Made land features and set map bounds")

    # Convert mm to inches and then apply snow ratio
        SWE_N = (SWE_N / 25.4) * SLE_ratio
        SWE_F = (SWE_F / 25.4) * SLE_ratio

    # Plot the snow contours
        levels = np.arange(1,90,8) 
        SWE_N_contour = ax_N.contourf(to_np(lons), to_np(lats), SWE_N,levels=levels,cmap="YlGnBu", transform=crs.PlateCarree())
        SWE_F_contour = ax_F.contourf(to_np(lons), to_np(lats), SWE_F,levels=levels,cmap="YlGnBu", transform=crs.PlateCarree())
        print("Made all contours")
    
    # Create a shared colorbar to the right of both subplots
        cbar = fig.colorbar(SWE_N_contour, ax=[ax_N, ax_F], orientation='vertical', shrink=0.5, pad=0.02)
        cbar.set_label("Snow Water Equivalent (mm)", fontsize=16, fontweight='bold')
        
    # Add identifiers to each subplot (a) and (b)
        labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ...]
        for i, ax in enumerate([ax_N, ax_F]):
            ax.text(0.01, 0.98, f"{labels[i]})", transform=ax.transAxes,fontsize=18, fontweight='bold', va='top', ha='left')

    # Add titles
        ax_N.set_title(f"Normal Simulation Total Snow Water Equivalent (mm) at {time}",fontsize=22,fontweight='bold')
        ax_F.set_title(f"Flat Simulation Total Snow Water Equivalent (mm) at {time}",fontsize=22,fontweight='bold')
              
    # Format the time for a filename (no spaces/colons), show and save figure
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        plt.close()

        return filename
    except Exception as e:
        print(f"Error processing files, see {e}")

if __name__ == "__main__":

    # Generate tasks
    tasks = zip(filelist_N, filelist_F, timeidxlist, timelist)
    print("Finished gathering tasks")
      
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        results = list(executor.map(generate_frame, tasks))
        frame_filenames = results
        frame_filenames = list(frame_filenames)

    # Filter any failed filenames
    filtered_list = [filename for filename in frame_filenames if filename is not None] 

    # Create readable GIF name
    start_str = start_time.strftime("%Y%m%d%H%M")
    end_str   = end_time.strftime("%Y%m%d%H%M")
    output_gif = f"SNOWCOMPARE_{start_time}_{end_time}"   
    
    # Create the GIF
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
       print("Removing: ", filename)
       os.remove(filename)
