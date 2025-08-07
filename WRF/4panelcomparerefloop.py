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
start_time, end_time  = datetime(2022,11,19,00,00), datetime(2022, 11, 19,20, 00)
domain = 2

# Path to each WRF run 
path_1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Path to radar files
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

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
timelist = time_df_1["time"].tolist() # Assuming times are the same
# ---- End User input for file ----

def generate_frame(args):
    print("Starting generate frame")
    file_path_a1, file_path_a2, file_path_a3, timeidx, WRF_time = args    

    try:
   
    # Read in the data from the matched WRF file for each attempt
        with Dataset(file_path_a1) as wrfin:
            mdbz_a1 = getvar(wrfin, "mdbz", timeidx=timeidx)
        with Dataset(file_path_a2) as wrfin2:
            mdbz_a2 = getvar(wrfin2, "mdbz", timeidx=timeidx)
        with Dataset(file_path_a3) as wrfin3:
            mdbz_a3 = getvar(wrfin3, "mdbz", timeidx=timeidx)
        print("Read in WRF data")

    # Define the format of the datetime string in your filename to parse
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"
        time_object = datetime.strptime(os.path.basename(file_path_a2), datetime_format)
        print("Made time_object")

    # Find the closest radar file
        # Locate radar data directory
        KTYX_closest_file = STORMY.find_closest_radar_file(WRF_time, radar_data_dir, "KTYX")
        KBUF_closest_file = STORMY.find_closest_radar_file(WRF_time, radar_data_dir, "KBUF")
        KBGM_closest_file = STORMY.find_closest_radar_file(WRF_time, radar_data_dir, "KBGM")
        print("Found closest radar files")

    # Get the observed variables
        KBUF_obs_dbz = pyart.io.read_nexrad_archive(KBUF_closest_file)
        KBUF_display = pyart.graph.RadarMapDisplay(KBUF_obs_dbz)
        KTYX_obs_dbz = pyart.io.read_nexrad_archive(KTYX_closest_file)
        KTYX_display = pyart.graph.RadarMapDisplay(KTYX_obs_dbz)
        KBGM_obs_dbz = pyart.io.read_nexrad_archive(KBGM_closest_file)
        KBGM_display = pyart.graph.RadarMapDisplay(KBGM_obs_dbz)
        print("Got observed variables")

    # Get the lat/lon points and projection object from WRF data
        lats, lons = latlon_coords(mdbz_a1)
        cart_proj = get_cartopy(mdbz_a1)
        WRF_ylim = cartopy_ylim(mdbz_a1)
        WRF_xlim = cartopy_xlim(mdbz_a1)

    # Create a figure
        fig = plt.figure(figsize=(12,9),facecolor='white')
        ax_a1 = fig.add_subplot(2,2,1, projection=cart_proj)
        ax_a2 = fig.add_subplot(2,2,2, projection=cart_proj)
        ax_a3 = fig.add_subplot(2,2,3, projection=cart_proj)
        ax_obs = fig.add_subplot(2,2,4, projection=cart_proj)
        axs = [ax_a1, ax_a2, ax_a3, ax_obs]
        print("Created Figure with axes")

  
    # Apply cartopy features to each axis (States, lakes, etc.) using STORMY helper function
        for ax in axs:
            STORMY.add_cartopy_features(ax)
            ax.set_xlim(WRF_xlim) # Set xlim for viewing the plots
            ax.set_ylim(WRF_ylim) # Set ylim for viewing the plots
            STORMY.format_gridlines(ax, x_inline=False, y_inline=False, xpadding=20, ypadding=20) # Format gridlines
        print("Added cartopy features and formatted gridlines")
    
    # Retrieve composite reflectivity from radar files
        KBUF_comp_ref = pyart.retrieve.composite_reflectivity(KBUF_obs_dbz, field="reflectivity")
        KBUF_display = pyart.graph.RadarMapDisplay(KBUF_comp_ref)
        KTYX_comp_ref = pyart.retrieve.composite_reflectivity(KTYX_obs_dbz, field="reflectivity")
        KTYX_display = pyart.graph.RadarMapDisplay(KTYX_comp_ref)
        KBGM_comp_ref = pyart.retrieve.composite_reflectivity(KBGM_obs_dbz, field="reflectivity")
        KBGM_display = pyart.graph.RadarMapDisplay(KBGM_comp_ref)
        print("Calculated composite reflectivity from radar files")
       
    # Plot the composite reflectivity for each radar
        KBGM_obs_contour = KBGM_display.plot_ppi_map("composite_reflectivity",vmin=0,vmax=75,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)
        KBUF_obs_contour = KBUF_display.plot_ppi_map("composite_reflectivity",vmin=0,vmax=75,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)
        KTYX_obs_contour = KTYX_display.plot_ppi_map("composite_reflectivity",vmin=0,vmax=75,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)
        print("Plotted composite reflectivity for each observed radar")

    # Read in cmap and map contours
        dbz_levels = np.arange(0, 75, 5)

    # Mask the modeled reflectivity data to a specific range to match observations
        mdbz_a1 = np.ma.masked_outside(to_np(mdbz_a1),0,75)
        mdbz_a2 = np.ma.masked_outside(to_np(mdbz_a2),0,75)
        mdbz_a3 = np.ma.masked_outside(to_np(mdbz_a3),0,75)

    # Plot the composite reflectivity for each WRF attempt
        mdbz_a1_contour = ax_a1.contourf(to_np(lons), to_np(lats), mdbz_a1,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())
        mdbz_a2_contour = ax_a2.contourf(to_np(lons), to_np(lats), mdbz_a2,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())
        mdbz_a3_contour = ax_a3.contourf(to_np(lons), to_np(lats), mdbz_a3,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())
        print("Made all contours")
        
    # Mark Henderson Harbor on each plot
        hend_harb = [43.88356, -76.155543]
        ax_a1.plot(hend_harb[1], hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(),markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_a2.plot(hend_harb[1], hend_harb[0],marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_a3.plot(hend_harb[1], hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_obs.plot(hend_harb[1],hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')

    # Add the colorbar manually to fit the figure
        cbar_a2 = fig.add_axes([ax_a2.get_position().x1 + 0.01,ax_a2.get_position().y0,0.02,ax_a2.get_position().height])

        cbar1 = fig.colorbar(mdbz_a2, cax=cbar_a2)
        cbar1.set_label("dBZ", fontsize=12)
        cbar1.ax.tick_params(labelsize=10)

    # Format the datetime into a more readable format
        datetime_obs = STORMY.parse_filename_datetime_obs(KTYX_closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        print("Made formatted datetime for obs")

    # Set the titles for each subplot
        ax_a1.set_title(f"Attempt 1 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_a2.set_title(f"Attempt 2 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_a3.set_title(f"Attempt 3 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        ax_obs.set_title(f"Observation at" + formatted_datetime_obs, fontsize=12,fontweight='bold')
        plt.suptitle(formatted_datetime_obs)

    # Save the figure to a set filename to be used in the GIF
        frame_number = os.path.splitext(os.path.basename(file_path_a1))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")

    #print(f"{os.path.basename(file_path)} Processed!")
        if time_object_adjusted.day == 19 and time_object_adjusted.hour == 0 and time_object_adjusted.minute == 0:
            plt.show()
        plt.close()

        return filename
    except Exception as e:
        print(f"Error processing files, see {e}")
        
if __name__ == "__main__":

    # Generate tasks
    tasks = zip(wrf_filelist_1, wrf_filelist_2, wrf_filelist_3, timeidxlist, timelist)
    print("Finished gathering tasks")

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
    '''
    # For Normal Processing
    frame_filenames = []
    for file_path, timeidx in tasks:
        filename = generate_frame(file_path, timeidx)
        if filename:
            frame_filenames.append(filename)    
    '''
    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]    
    output_gif = f'4panelcomparerefloopD{domain}{start_time.month:02d}{start_time.day:02d}{start_time.hour:02d}{start_time.minute:02d}to{end_time.month:02d}{end_time.day:02d}{end_time.hour:02d}{end_time.minute:02d}.gif'
    STORMY.create_gif(savepath, sorted(filtered_list), output_gif)

    # Clean up (Remove) the frame files that were created for the GIF
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
