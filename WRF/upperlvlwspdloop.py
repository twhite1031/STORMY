import numpy as np
import matplotlib.pyplot as plt
import os
from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,cartopy_xlim, cartopy_ylim)
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
start_time, end_time  = datetime(2022,11,17,23,20,00), datetime(2022, 11, 17,23, 40, 00)
domain = 2

height = 850 # Pressure level for Wind Barbs

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)

mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()
timelist = time_df["time"].tolist()


# ---- Function to Loop (Each Frame) ----
def generate_frame(args):
    print("Starting generate frame")
    file_path_N, timeidx, time = args
    try:
   
    # Read data from WRF file
        with Dataset(file_path_N) as ncfile:
            p = getvar(ncfile, "pressure")
            z = getvar(ncfile, "z", units="dm")
            ua = getvar(ncfile, "ua", units="kt")
            va = getvar(ncfile, "va", units="kt")
            wspd = getvar(ncfile, "uvmet_wspd_wdir", units="kts")
            wpsd = wspd[0,:]   # Extract wind speed from the first element of the tuple

    # Interpolate geopotential height, u, and v winds to 500 hPa
        ht_500 = interplevel(z, p, height)
        u_500 = interplevel(ua, p, height)
        v_500 = interplevel(va, p, height)
        wspd_500 = interplevel(wspd, p, height)[0,:]

    # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(p)
        cart_proj = get_cartopy(p)
        WRF_ylim = cartopy_ylim(p)
        WRF_xlim = cartopy_xlim(p)

    # Create the figure
        fig = plt.figure(figsize=(30,15),facecolor='white')
        ax = plt.axes(projection=cart_proj)
        
    # Apply cartopy features (states, lakes, etc.) using STORMY helper function
        STORMY.add_cartopy_features(ax)
        ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
        ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots

    # Add custom formatted gridlines using STORMY function
        STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20)

    # Create contour levels in a predetermined interval based on the data range for the color maps using STORMY helper function
    # Note: This interval is dynamic based on the data range, not ideal for comparing multiple runs
        hgt_levels = STORMY.make_contour_levels(to_np(ht_500), interval=6)
        wspd_levels = STORMY.make_contour_levels(to_np(wspd_500), interval=5)

    # Plot the wind speed and height contours
        hgt_contours = plt.contour(to_np(lons), to_np(lats), to_np(ht_500),levels=hgt_levels, colors="black",transform=crs.PlateCarree())
        wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(wspd_500),levels=wspd_levels,cmap=get_cmap("rainbow"), transform=crs.PlateCarree())
        plt.clabel(hgt_contours, inline=1, fontsize=10, fmt="%i") # Inline labels for height contours

    # Add the 500 hPa wind barbs, only plotting every nth data point.
        n = 50
        plt.barbs(to_np(lons[::n,::n]), to_np(lats[::n,::n]),to_np(u_500[::n, ::n]), to_np(v_500[::n, ::n]),transform=crs.PlateCarree(), length=6)
        
    # Create the color bar for wind speed
        cbar = plt.colorbar(wspd_contours, ax=ax, orientation="horizontal", pad=.05,shrink=0.6)
        cbar.set_label('Knots',fontsize=14)
    
    # Add a title
        ax.set_title(f"{height}hPa Wind Speed (m/s) and Heights (dm) at {time}" ,fontsize=18,fontweight='bold')
                     
    # Save the figure to a set filename to be used in the GIF
        frame_number = os.path.splitext(os.path.basename(file_path_N))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        
        plt.savefig(filename)
        plt.show()
        plt.close()

        print(f"{os.path.basename(file_path_N)} Processed!")

        return filename
    except Exception as e:
        print(f"Error processing files, see {e}")
        
if __name__ == "__main__":

    # Generate tasks
    tasks = zip(filelist, timeidxlist,timelist)
    print("Finished gathering tasks")
    
    # Use multiprocessing to generate frames in parallel
    #with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
    #    print("Starting multiprocessing")
    #   frame_filenames_gen = executor.map(generate_frame, tasks)
    #    frame_filenames = list(frame_filenames_gen)  # Convert generator to list
    
    frame_filenames = []
    for task in tasks:
        result = generate_frame(task)
        frame_filenames.append(result)

    # Filter any failed filenames
    filtered_list = [filename for filename in frame_filenames if filename is not None]    
    
    # Create an accurate GIF name
    start_str = start_time.strftime("%Y%m%d_%H%M")
    end_str   = end_time.strftime("%Y%m%d_%H%M")
    output_gif = f'WSPD{height}hPa_LOOP_D{domain}{start_str}_to_{end_str}.gif'

    # Create the GIF
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up (Remove) the frame files that were created for the GIF
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
