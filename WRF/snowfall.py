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
from datetime import datetime,timedelta
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import wrffuncs
import string
import pandas as pd
"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""
# --- USER INPUT ---

wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2

SIMULATION = "NORMAL" # If comparing runs

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# --- END USER INPUT ---

time_df_N = wrffuncs.build_time_df(path_N, domain)
time_df_F = wrffuncs.build_time_df(path_F, domain)

# Filter time range
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference
closest_idx = (time_df_N["time"] - obs_time).abs().argmin()

# Extract the matched row
match_N = time_df_N.iloc[closest_idx]
match_F = time_df_F.iloc[closest_idx]

# Unpack matched file info
matched_file_N = match_N["filename"]
matched_file_F = match_F["filename"]
matched_timeidx = match_N["timeidx"]
matched_time = match_N["time"]

print(f"Closest match for Normal: {matched_time} in file {matched_file_N} at time index {matched_timeidx}")


# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, file_path_F, timeidx = args
    try:
   
    # Read data from file
        with Dataset(file_path_N) as wrfin:
            SWE_N = getvar(wrfin, "SNOWNC", timeidx=timeidx) # Conversion factor to inches
        with Dataset(file_path_F) as wrfin2:
            SWE_F = getvar(wrfin2, "SNOWNC", timeidx=timeidx) 
        print("Read in WRF data")
        cart_proj = get_cartopy(SWE_N)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_N = fig.add_subplot(1,2,1, projection=cart_proj)
        ax_F = fig.add_subplot(1,2,2, projection=cart_proj)
        print("Created Figures")

    # Get the latitude and longitude points
        lats, lons = latlon_coords(SWE_N)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax_N.add_feature(states, linewidth=.1, edgecolor="black")
        ax_N.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_N.coastlines('50m', linewidth=1)
        ax_N.add_feature(USCOUNTIES, alpha=0.1)
        print("Made land features")
    # Set the map bounds
        ax_N.set_xlim(cartopy_xlim(SWE_N))
        ax_N.set_ylim(cartopy_ylim(SWE_N))
        print("Set map bounds")

    # Add the gridlines
        gl_N = ax_N.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_N.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_N.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_N.xlines = True
        gl_N.ylines = True
        gl_N.top_labels = False  # Disable top labels
        gl_N.right_labels = True # Disable right labels
        gl_N.xpadding = 20
        gl_N.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        gl_N.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

        print("Made gridlines")

    # Download and add the states, lakes  and coastlines
        ax_F.add_feature(states, linewidth=.1, edgecolor="black")
        ax_F.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_F.coastlines('50m', linewidth=1)
        ax_F.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_F.set_xlim(cartopy_xlim(SWE_F))
        ax_F.set_ylim(cartopy_ylim(SWE_F))

    # Add the gridlines
        gl_F = ax_F.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_F.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_F.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_F.xlines = True
        gl_F.ylines = True
        gl_F.top_labels = False  # Disable top labels
        gl_F.right_labels = True  # Disable right labels
        gl_F.xpadding = 20

        # Increase the number of gridlines by setting locators
        #gl_F.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        #gl_F.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees

        # Format the gridline labels
        gl_F.xformatter = LONGITUDE_FORMATTER
        gl_F.yformatter = LATITUDE_FORMATTER
        

      # Read in cmap and map contours
        levels = np.arange(1,90,8) 
        
        # Convert mm to inches and then apply snow ratio
        #SWE_N = (SWE_N / 25.4) * 10
        #SWE_F = (SWE_F / 25.4) * 10

        print("Control max", np.max(SWE_N))
        print("Flat max", np.max(SWE_F))

        SWE_N_contour = ax_N.contourf(to_np(lons), to_np(lats), SWE_N,levels=levels,cmap="YlGnBu", transform=crs.PlateCarree())
        SWE_F_contour = ax_F.contourf(to_np(lons), to_np(lats), SWE_F,levels=levels,cmap="YlGnBu", transform=crs.PlateCarree())
        
        # Create a shared colorbar to the right of both subplots
        cbar = fig.colorbar(SWE_N_contour, ax=[ax_N, ax_F], orientation='vertical', shrink=0.5, pad=0.02)
        cbar.set_label("Snow Water Equivalent (mm)", fontsize=16, fontweight='bold')

        print("Made all contours")
        
        labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ...]

        for i, ax in enumerate([ax_N, ax_F]):
            ax.text(0.01, 0.98, f"{labels[i]})", transform=ax.transAxes,fontsize=18, fontweight='bold', va='top', ha='left')

    # Set titles, get readable format from WRF time
        time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(file_path_N, timeidx)
        ax_N.set_title(f"Normal Simulation Total Snow Water Equivalent (mm)",fontsize=22,fontweight='bold')
        ax_F.set_title(f"Flat Simulation Total Snow Water Equivalent (mm)",fontsize=22,fontweight='bold')
              

    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")
        plt.show()
        plt.close()

        return filename
    except IndexError:
        print("Error processing files")
 
if __name__ == "__main__":

    # Generate tasks
    tasks = [(matched_file_N, matched_file_F, matched_timeidx)]
    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        generate_frame(tasks[0])    
