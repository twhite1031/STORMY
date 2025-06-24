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
"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

time_df_N = wrffuncs.build_time_df(path_N, domain)
time_df_F = wrffuncs.build_time_df(path_F, domain)

# Filter time range
mask_N = (time_df_N["time"] >= start_time) & (time_df_N["time"] <= end_time)
mask_F = (time_df_F["time"] >= start_time) & (time_df_F["time"] <= end_time)

time_df_N = time_df_N[mask_N].reset_index(drop=True)
time_df_F = time_df_F[mask_F].reset_index(drop=True)

filelist_N = time_df_N["filename"].tolist()
filelist_F = time_df_F["filename"].tolist()
timeidxlist = time_df_N["timeidx"].tolist()


# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, file_path_F, timeidx = args
    try:
   
    # Read data from file
        with Dataset(file_path_N) as wrfin:
            ter_N = getvar(wrfin, "ter", timeidx=timeidx)
        with Dataset(file_path_F) as wrfin2:
            ter_F = getvar(wrfin2, "ter", timeidx=timeidx)
        #print("Read in WRF data")
        cart_proj = get_cartopy(ter_N)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_N = fig.add_subplot(1,2,1, projection=cart_proj)
        ax_F = fig.add_subplot(1,2,2, projection=cart_proj)
        #print("Created Figures")

    # Get the latitude and longitude points
        lats, lons = latlon_coords(ter_N)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax_N.add_feature(states, linewidth=.1, edgecolor="black")
        ax_N.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_N.coastlines('50m', linewidth=1)
        ax_N.add_feature(USCOUNTIES, alpha=0.1)
        #print("Made land features")
    # Set the map bounds
        ax_N.set_xlim(cartopy_xlim(ter_N))
        ax_N.set_ylim(cartopy_ylim(ter_N))
        #print("Set map bounds")

    # Add the gridlines
        gl_N = ax_N.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
        gl_N.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_N.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_N.xlines = True
        gl_N.ylines = True
        gl_N.top_labels = False  # Disable top labels
        gl_N.right_labels = True # Disable right labels
        gl_N.left_labels = True
        gl_N.bottom_labels = True
        gl_N.xpadding = 20
        #gl_N.xlocator = mticker.FixedLocator([-77.5,-76.5,-75.5])  # Set longitude gridlines every 10 degrees
        #gl_N.ylocator = mticker.FixedLocator(np.arange(41, 47, .5))    # Set latitude gridlines every 10 degrees
        # Format the gridline labels
        gl_N.xformatter = LONGITUDE_FORMATTER
        gl_N.yformatter = LATITUDE_FORMATTER

        #print("Made gridlines")

    # Download and add the states, lakes  and coastlines
        ax_F.add_feature(states, linewidth=.1, edgecolor="black")
        ax_F.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_F.coastlines('50m', linewidth=1)
        ax_F.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_F.set_xlim(cartopy_xlim(ter_F))
        ax_F.set_ylim(cartopy_ylim(ter_F))

    # Add the gridlines
        gl_F = ax_F.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=True)
     
        gl_F.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_F.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_F.xlines = True
        gl_F.ylines = True
        gl_F.left_labels = False
        gl_F.bottom_labels = True
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
        elev_contour_N = ax_N.contourf(to_np(lons), to_np(lats), ter_N,levels=np.arange(0, np.max(ter_N), 50), cmap="Greys_r", transform=crs.PlateCarree())
        elev_contour_F = ax_F.contourf(to_np(lons), to_np(lats), ter_F,levels=np.arange(0, np.max(ter_F), 50), cmap="Greys_r", transform=crs.PlateCarree())

        # Create a shared colorbar to the right of both subplots
        cbar = fig.colorbar(elev_contour_N, ax=[ax_N, ax_F], orientation='vertical', shrink=0.5, pad=0.02)
        cbar.set_label("Terrain Height (m)", fontsize=16, fontweight='bold')

        #cbar = fig.colorbar(mdbz_N_contour, ax=ax_N, orientation='vertical', fraction=0.05, pad=0.02)
        #print("Made all contours")
        ax_N.set_xlabel("Longitude",fontsize=16, fontweight='bold')
        ax_N.set_ylabel("Lattitude", fontsize=16, fontweight='bold')
    
        ax_F.set_xlabel("Longitude", fontsize=16, fontweight='bold')
        ax_F.set_ylabel("Lattitude", fontsize=16, fontweight='bold')

        labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ...]

        for i, ax in enumerate([ax_N, ax_F]):
            ax.text(0.01, 0.98, f"{labels[i]})", transform=ax.transAxes,fontsize=18, fontweight='bold', va='top', ha='left')
    # Set titles, get readable format from WRF time
        time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(file_path_N, timeidx)
        ax_N.set_title(f"Normal Simulation Terrain Height (m)",fontsize=18,fontweight='bold')
        ax_F.set_title(f"Flat Simulation Terrain Height (m)",fontsize=18,fontweight='bold')
              

    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")
        #plt.show()
        plt.close()

        return filename
    except IndexError:
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
    tasks = zip(filelist_N, filelist_F, timeidxlist)
    output_gif = f'comparebandloc{domain}.gif'
    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
       
    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]    
    create_gif(sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
       print("Removing: ", filename)
       os.remove(filename)
    
