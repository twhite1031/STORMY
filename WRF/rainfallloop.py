import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
from wrf import (getvar, to_np, latlon_coords, get_cartopy,cartopy_xlim, cartopy_ylim)
from matplotlib.cm import (get_cmap)
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset
from metpy.plots import USCOUNTIES
from PIL import Image
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import STORMY

"""
A side by side comparison of two WRF runs using plots of simulated reflectivity ('mdbz)
A GIF will be made using the plots between the time periods
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2023,1,9,7,40,00), datetime(2023, 1, 9,7, 55, 00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/wrf/WRFV4.5.2/run/"
savepath = f""

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)

mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()
timelist = time_df["time"].tolist()

# Get Starting point data to substract from since RAINNC and RAINC is cumulative
with Dataset(filelist[0]) as ds:
            start_rain_nc = getvar(ds, "RAINNC",timeidx=timeidxlist[0])
            start_rain_c = getvar(ds, "RAINC",timeidx=timeidxlist[0])

start_total_rain = start_rain_nc + start_rain_c
start_time = timelist[0]

# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, timeidx, time= args
    try:
   
    # Read data from WRF file
        with Dataset(file_path_N) as ncfile:
            rain_nc = getvar(ncfile, "RAINNC",timeidx=timeidx)
            rain_c = getvar(ncfile, "RAINC",timeidx=timeidx)

            ua = getvar(ncfile, "ua",timeidx=timeidx, units="m s-1")
            va = getvar(ncfile, "va",timeidx=timeidx, units="m s-1")
            wspd = getvar(ncfile, "uvmet_wspd_wdir", timeidx=timeidx,units="m s-1")[0,:]

        print("Read in WRF data")

        # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(ua)
        cart_proj = get_cartopy(ua)
        WRF_ylim = cartopy_ylim(ua)
        WRF_xlim = cartopy_xlim(ua)


    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax = plt.axes(projection=cart_proj)

    # Apply cartopy features to the ctt axis (States, lakes, etc.) using STORMY helper function 
        STORMY.add_cartopy_features(ax)

    # Set the map bounds
        ax.set_xlim(WRF_xlim)
        ax.set_ylim(WRF_ylim)

    # Add custom formatted gridlines using STORMY function
        STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

        rain_interval = .2
        start = 0
        stop = 12
        levels = np.arange(start, stop + rain_interval, rain_interval)
        
    # Get total rain from start time, then convert to inches
        total_rain = ((rain_nc + rain_c) - start_total_rain) / 25.4

    # Plot the total_rain filled contours
        rain_contours = plt.contourf(to_np(lons), to_np(lats), to_np(total_rain),levels=levels,cmap=get_cmap("turbo"), transform=crs.PlateCarree())
        
    # Create a colorbar
        cbar = plt.colorbar(rain_contours, ax=ax, orientation="horizontal", pad=.075,shrink=0.6)
        cbar.set_label('Rainfall (in)',fontsize=14)

    # Set titles, get readable format from WRF time
        ax.set_title(f"Rainfall starting at {time}" ,fontsize=18,fontweight='bold')
                     
    # Save the figure to a set filename to be used in the GIF
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'

        plt.savefig(filename)
        plt.close()

        return filename
    except ValueError:
        print("Error processing files")
        
if __name__ == "__main__":
    print("")
    # Generate tasks
    tasks = zip(filelist, timeidxlist)


    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        results = list(executor.map(generate_frame, tasks))
        frame_filenames = results
        frame_filenames = list(frame_filenames)
            
    # Filter failed files from list
    filtered_list = [filename for filename in frame_filenames if filename is not None]  

    # Create readable GIF name
    start_str = start_time.strftime("%Y%m%d%H%M")
    end_str   = end_time.strftime("%Y%m%d%H%M")
    output_gif = f'rainfall_LOOP_D{domain}{start_str}_to_{end_str}.gif'

    # Create GIF
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
