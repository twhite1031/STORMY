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
import wrffuncs
import cartopy.io.shapereader as shpreader
import pyart
import multiprocessing as mp

"""
A foundation for using a multi processor loop to speed up processing data
Be weary of how many you use based on the task and how many may be available (type 'top' in terminal, 'q' to quit)
"""

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2

SIMULATION = "NORMAL"
path = f"/data2/white/WRF_OUTPUTS/SEMINAR/{SIMULATION}_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/{SIMULATION}_ATTEMPT/"

# --- END USER INPUT ---

time_df = wrffuncs.build_time_df(path, domain)

# Filter time range
mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)


filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()


# ---- Function to Loop (Each Frame) ----

def generate_frame(args):
    print("Starting generate frame")
    file_path, timeidx = args    

    
    try:
   
    # Read data from file
        wrfin = Dataset(file_path) 
        data = getvar(wrfin, "mdbz", timeidx=timeidx)
        #print("Read in WRF data")
        cart_proj = get_cartopy(data)

    # Create a figure
        fig = plt.figure(figsize=(30,15))
        ax = plt.axes(projection=cart_proj)

    # Get the latitude and longitude points
        lats, lons = latlon_coords(data)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax.add_feature(states, linewidth=.1, edgecolor="black")
        ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax.coastlines('50m', linewidth=1)
        ax.add_feature(USCOUNTIES, alpha=0.1)
        #print("Made land features")

    # Set the map bounds
        ax.set_xlim(cartopy_xlim(data))
        ax.set_ylim(cartopy_ylim(data))
        #print("Set map bounds")
    
    # Add the gridlines
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xpadding = 20
        #print("Made gridlines")

        
    # Get composite reflectivity from observed LVL2 data
        nwscmap = ctables.registry.get_colortable('NWSReflectivity')
        levels = np.arange(0, 55, 5)

        data_contour = ax.contourf(to_np(lons), to_np(lats), data,cmap=nwscmap,levels=levels, transform=crs.PlateCarree())

    # Set up Colorbar
        cbar = fig.colorbar(data_contour,ax=ax)
        cbar.set_label("dBZ", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
    # Set up Titles
        wrf_time = wrffuncs.parse_filename_datetime_wrf(file_path, timeidx)
        ax.set_title(f"Maximum Reflectivity at " + str(wrf_time),fontsize=12,fontweight='bold')        
        #plt.suptitle(datetime_obs)
        
    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
    
        print(f"{os.path.basename(file_path)} Processed!")
        #plt.show()
        plt.close()

        return filename
    except IndexError:
        print("Error processing files due to Index Error")
        
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
    output_gif = f'{SIMULATION}loopD{domain}{start_time.month:02d}{start_time.day:02d}{start_time.hour:02d}{start_time.minute:02d}to{end_time.month:02d}{end_time.day:02d}{end_time.hour:02d}{end_time.minute:02d}.gif'
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
    
