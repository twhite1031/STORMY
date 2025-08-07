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

# Path to each WRF run 
path = f"/data2/white/wrf/WRFV4.5.2/run/"
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)

mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()
timelist = time_df["time"].tolist()

def generate_frame(args):
    print("Starting generate frame")
    file_path_N, timeidx, time = args
    try:
   
    # Read data from file
        with Dataset(file_path_N) as ncfile:
            p = getvar(ncfile, "p")
            q = getvar(ncfile, 'QVAPOR') 
            q = q / (1 + q)
            ua = getvar(ncfile, "ua", units="m s-1")
            va = getvar(ncfile, "va", units="m s-1")

    # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(p)
        cart_proj = get_cartopy(p)
        WRF_ylim = cartopy_ylim(p)
        WRF_xlim = cartopy_xlim(p)
  
    # Create a figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax = plt.axes(projection=cart_proj)

    # Apply cartopy features to thie axis (States, lakes, etc.) using STORMY helper function 
        STORMY.add_cartopy_features(ax)

    # Add custom formatted gridlines using STORMY function
        STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines

    # Calculate wind speed up to 29 levels, roughly the IVT definition of 300hPa
        V = np.sqrt(ua[:29, :, :]**2 + va[:29, :, :]**2)  

    # Compute pressure differences along vertical (level) axis
        dp = np.diff(p[:29, :, :], axis=0)  

    # Multiply q, V, and dp — exclude last level for q and V to match dp shape
        ivt_full = np.sum(q[:28, :, :] * V[:28, :, :] * np.abs(dp), axis=0)  
        ivt = ivt_full / 9.81  # Convert to kg/ms

    # Set fixed levels for IVT contouring
        ivt_interval = 20
        start = 0
        stop = 1600
        ivt_levels = np.arange(start, stop + ivt_interval, ivt_interval)

    # Plot filled contours of IVT
        ivt_contours = plt.contourf(to_np(lons), to_np(lats), to_np(ivt),levels=ivt_levels,cmap=get_cmap("rainbow"), transform=crs.PlateCarree())
        
    # Add a color bar, formatted to fit 
        cbar = plt.colorbar(ivt_contours, ax=ax, orientation="horizontal", pad=.075,shrink=0.6)
        cbar.set_label('IVT (kg/ms)',fontsize=14)

    # Set titles, get readable format from WRF time
        time_object_adjusted = STORMY.parse_filename_datetime_wrf(file_path_N, timeidx)
        ax.set_title(f"Integrated Vapor Transport (IVT) at " + time,fontsize=18,fontweight='bold')
                     
    # Save the figure to a set filename to be used in the GIF
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        plt.close()

        return filename
    except ValueError:
        print("Error processing files")
        
if __name__ == "__main__":

    # Generate tasks
    tasks = zip(filelist, timeidxlist,timelist)
    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        results = list(executor.map(generate_frame, tasks))
        frame_filenames, max_ivts = zip(*results)
        frame_filenames = list(frame_filenames)


    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]  # Remove any failed filenames that became None
    start_str, end_str = start_time.strftime("%Y%m%d_%H%M"), end_time.strftime("%Y%m%d_%H%M")
    output_gif = f'IVT_LOOP_D{domain}{start_str}_to_{end_str}.gif'
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up (Remove) the frame files that were created for the GIF
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
