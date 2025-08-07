import matplotlib.pyplot as plt
import os
import concurrent.futures
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES, ctables
from PIL import Image
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import STORMY
import string
import numpy as np

"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# Path to save GIF or Files
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

# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, file_path_F, timeidx,WRF_time = args
    try:
   
    # Read data from WRF file
        with Dataset(file_path_N) as wrfin:
            mdbz_N = getvar(wrfin, "mdbz", timeidx=timeidx)
        with Dataset(file_path_F) as wrfin2:
            mdbz_F = getvar(wrfin2, "mdbz", timeidx=timeidx)
        #print("Read in WRF data")
        
    # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(mdbz_N)
        cart_proj = get_cartopy(mdbz_N)
        WRF_ylim = cartopy_ylim(mdbz_N)
        WRF_xlim = cartopy_xlim(mdbz_N)

    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax_N = fig.add_subplot(1,2,1, projection=cart_proj)
        ax_F = fig.add_subplot(1,2,2, projection=cart_proj)
        axs = [ax_N, ax_F]

    # Apply cartopy features to each axis (States, lakes, etc.) using STORMY helper function 
        for ax in axs:
            STORMY.add_cartopy_features(ax)
            ax.margins(x=0, y=0, tight=True)
            ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
            ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots
            STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

        # Plot the maximum reflectivity for both WRF runs
        dbz_levels = np.arange(0, 75, 5)  # Define reflectivity levels for contouring
        mdbz_N_contour = ax_N.contourf(to_np(lons), to_np(lats), mdbz_N,levels=dbz_levels,cmap="NWSRef", transform=crs.PlateCarree())
        mdbz_F_contour = ax_F.contourf(to_np(lons), to_np(lats), mdbz_F,levels=dbz_levels,cmap="NWSRef", transform=crs.PlateCarree())
        
        # Create a shared colorbar to the right of both subplots
        cbar = fig.colorbar(mdbz_N_contour, ax=axs, orientation='vertical', shrink=0.5, pad=0.02)
        cbar.set_label("Reflectivity (dBZ)", fontsize=16, fontweight='bold')

    # Set titles, get readable format from WRF time
        ax_N.set_title(f"Normal Simulation Composite Reflectivity (dBZ) at {WRF_time}",fontsize=18,fontweight='bold')
        ax_F.set_title(f"Flat Simulation Compostite Reflectivity (dBZ) at  {WRF_time}",fontsize=18,fontweight='bold')
        
    # Add identifiers to each subplot (a) and (b)
        labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ...]
        for i, ax in enumerate([ax_N, ax_F]):
            ax.text(0.01, 0.98, f"{labels[i]})", transform=ax.transAxes,fontsize=18, fontweight='bold', va='top', ha='left')

    # Save the figure to a set filename to be used in the GIF
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        plt.close()

        return filename
    except Exception as e:
        print(f"Error processing files, see {e}")
        
def create_gif(frame_filenames, output_filename):

    frames = []
    for filename in frame_filenames:
            new_frame = Image.open(filename)
            frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(savepath + output_filename, format='GIF', append_images=frames[1:],save_all=True,duration=75, loop=0)
    
if __name__ == "__main__":
    # Generate tasks
    tasks = zip(filelist_N, filelist_F, timeidxlist, timelist)
    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
       
    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]  
    output_gif = f'MDBZCOMPARE{domain}.gif'  
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up (Remove) the frame files that were created for the GIF
    for filename in filtered_list:
       print("Removing: ", filename)
       os.remove(filename)
    
