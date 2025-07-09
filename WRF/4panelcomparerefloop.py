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

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,19,00,00), datetime(2022, 11, 19,20, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path_1 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/"
path_2 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_2/"
path_3 = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_3/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df_1 = wrffuncs.build_time_df(path_1, domain)
time_df_2 = wrffuncs.build_time_df(path_2, domain)
time_df_3 = wrffuncs.build_time_df(path_3, domain)

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

# ---- End User input for file ----

def generate_frame(args):
    print("Starting generate frame")
    file_path_a1, file_path_a2, file_path_a3, timeidx = args    

    def parse_filename_datetime_obs(filepath):
    
    # Get the filename 
        filename = filepath.split('/')[-1]
    # Extract the date part (8 characters starting from the 5th character)
        date_str = filename[4:12]
    # Extract the time part (6 characters starting from the 13th character)
        time_str = filename[13:19]
    # Combine the date and time strings
        datetime_str = date_str + time_str
    # Convert the combined string to a datetime object
    #datetime_obj = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
    #formatted_datetime_obs = parse_filename_datetime_obs(radar_data)datetime_obj.strftime('%B %d, %Y %H:%M:%S') 
        return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

    try:
   
    # Read data from file
        with Dataset(file_path_a1) as wrfin:
            mdbz_a1 = getvar(wrfin, "mdbz", timeidx=timeidx)
        with Dataset(file_path_a2) as wrfin2:
            mdbz_a2 = getvar(wrfin2, "mdbz", timeidx=timeidx)
        with Dataset(file_path_a3) as wrfin3:
            mdbz_a3 = getvar(wrfin3, "mdbz", timeidx=timeidx)

        print("Read in WRF data")
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H:%M:%S"
        print("made datetime format")

    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(file_path_a2), datetime_format)
        print("Made time_object")

    # Add timeidx value
        add_time = 5 * float(timeidx)
        time_object_adjusted = time_object + timedelta(minutes=add_time)
        print("Adjusted WRF time")

    # Find the closest radar file
        # Locate radar data directory
        radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"
        KTYX_closest_file = wrffuncs.find_closest_radar_file(time_object_adjusted, radar_data_dir, "KTYX")
        KBUF_closest_file = wrffuncs.find_closest_radar_file(time_object_adjusted, radar_data_dir, "KBUF")
        KBGM_closest_file = wrffuncs.find_closest_radar_file(time_object_adjusted, radar_data_dir, "KBGM")
       
        print("Found closest radar file")
    # Get the observed variables
        KBUF_obs_dbz = pyart.io.read_nexrad_archive(KBUF_closest_file)
        KBUF_display = pyart.graph.RadarMapDisplay(KBUF_obs_dbz)


        KTYX_obs_dbz = pyart.io.read_nexrad_archive(KTYX_closest_file)
        KTYX_display = pyart.graph.RadarMapDisplay(KTYX_obs_dbz)

        KBGM_obs_dbz = pyart.io.read_nexrad_archive(KBGM_closest_file)
        KBGM_display = pyart.graph.RadarMapDisplay(KBGM_obs_dbz)

        print("Got observed variables")

    # Get the cartopy projection object
        cart_proj = get_cartopy(mdbz_a1)

    # Create a figure
        fig = plt.figure(figsize=(12,9),facecolor='white')
        ax_a1 = fig.add_subplot(2,2,1, projection=cart_proj)
        ax_a2 = fig.add_subplot(2,2,2, projection=cart_proj)
        ax_a3 = fig.add_subplot(2,2,3, projection=cart_proj)
        ax_obs = fig.add_subplot(2,2,4, projection=cart_proj)
        print("Created Figures")

    # Get the latitude and longitude points
        lats, lons = latlon_coords(mdbz_a1)

    # Download and add the states, lakes  and coastlines
        states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
        ax_a1.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a1.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a1.coastlines('50m', linewidth=1)
        ax_a1.add_feature(USCOUNTIES, alpha=0.1)
        print("Made land features")

    # Set the map bounds
        ax_a1.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a1.set_ylim(cartopy_ylim(mdbz_a2))
        print("Set map bounds")

    # Add the gridlines
        gl_a1 = ax_a1.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl_a1.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a1.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a1.xlines = True
        gl_a1.ylines = True
        gl_a1.top_labels = False  # Disable top labels
        gl_a1.right_labels = False  # Disable right labels
        gl_a1.xpadding = 20
        print("Made gridlines")

    # Download and add the states, lakes  and coastlines
        ax_a2.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a2.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a2.coastlines('50m', linewidth=1)
        ax_a2.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_a2.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a2.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_a2 = ax_a2.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl_a2.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a2.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a2.xlines = True
        gl_a2.ylines = True
        gl_a2.top_labels = False  # Disable top labels
        gl_a2.right_labels = False  # Disable right labels
        gl_a2.xpadding = 20

    # Download and add the states, lakes  and coastlines
        ax_a3.add_feature(states, linewidth=.1, edgecolor="black")
        ax_a3.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_a3.coastlines('50m', linewidth=1)
        ax_a3.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_a3.set_xlim(cartopy_xlim(mdbz_a1))
        ax_a3.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_a3 = ax_a3.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl_a3.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_a3.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_a3.xlines = True
        gl_a3.ylines = True
        gl_a3.top_labels = False  # Disable top labels
        gl_a3.right_labels = False  # Disable right labels
        gl_a3.xpadding = 20
    
    # Download and add the states, lakes  and coastlines
        ax_obs.add_feature(states, linewidth=.1, edgecolor="black")
        ax_obs.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
        ax_obs.coastlines('50m', linewidth=1)
        ax_obs.add_feature(USCOUNTIES, alpha=0.1)

    # Set the map bounds
        ax_obs.set_xlim(cartopy_xlim(mdbz_a1))
        ax_obs.set_ylim(cartopy_ylim(mdbz_a2))

    # Add the gridlines
        gl_obs = ax_obs.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
        gl_obs.xlabel_style = {'rotation': 'horizontal','size': 10,'ha':'center'} # Change 14 to your desired font size
        gl_obs.ylabel_style = {'size': 10}  # Change 14 to your desired font size
        gl_obs.xlines = True
        gl_obs.ylines = True
        gl_obs.top_labels = False  # Disable top labels
        gl_obs.right_labels = False  # Disable right labels
        gl_obs.xpadding = 20
    
    # Get composite reflectivity from observed LVL2 data
        nwscmap = ctables.registry.get_colortable('NWSReflectivity')
    

        KBUF_comp_ref = pyart.retrieve.composite_reflectivity(KBUF_obs_dbz, field="reflectivity")
        KBUF_display = pyart.graph.RadarMapDisplay(KBUF_comp_ref)

        KTYX_comp_ref = pyart.retrieve.composite_reflectivity(KTYX_obs_dbz, field="reflectivity")
        KTYX_display = pyart.graph.RadarMapDisplay(KTYX_comp_ref)

        KBGM_comp_ref = pyart.retrieve.composite_reflectivity(KBGM_obs_dbz, field="reflectivity")
        KBGM_display = pyart.graph.RadarMapDisplay(KBGM_comp_ref)

        print("Calculated composite reflectivity")
       
        KBGM_obs_contour = KBGM_display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)
        KBUF_obs_contour = KBUF_display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)

        KTYX_obs_contour = KTYX_display.plot_ppi_map("composite_reflectivity",vmin=10,vmax=60,mask_outside=True,ax=ax_obs, colorbar_flag=False, title_flag=False, add_grid_lines=False, cmap=nwscmap)

        print("Made observation contours")

    # Read in cmap and map contours
        levels = [10, 15, 20, 25, 30, 35, 40, 45,50,55,60]


        mdbz_a1 = np.ma.masked_outside(to_np(mdbz_a1),10,65)
        mdbz_a2 = np.ma.masked_outside(to_np(mdbz_a2),10,65)
        mdbz_a3 = np.ma.masked_outside(to_np(mdbz_a3),10,65)

        mdbz_a1_contour = ax_a1.contourf(to_np(lons), to_np(lats), mdbz_a1,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())

        mdbz_a2_contour = ax_a2.contourf(to_np(lons), to_np(lats), mdbz_a2,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())

        mdbz_a3_contour = ax_a3.contourf(to_np(lons), to_np(lats), mdbz_a3,levels=levels,vmin=10,vmax=60,cmap=nwscmap, transform=crs.PlateCarree())
        print("Made all contours")
        
    # Henderson Harbor coordinates
        hend_harb = [43.88356, -76.155543]

        ax_a1.plot(hend_harb[1], hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(),markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_a2.plot(hend_harb[1], hend_harb[0],marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_a3.plot(hend_harb[1], hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')
        ax_obs.plot(hend_harb[1],hend_harb[0], marker='^', color='brown', transform=crs.PlateCarree(), markersize=3)  # 'ro' means red color ('r') and circle marker ('o')




    # Add the colorbar for the first plot with respect to the position of the plots
        #cbar_a2 = fig.add_axes([ax_a2.get_position().x1 + 0.01,
        #                ax_a2.get_position().y0,
        #                 0.02,
        #                 ax_a2.get_position().height])

        #cbar1 = fig.colorbar(mdbz_a2, cax=cbar_a2)
        #cbar1.set_label("dBZ", fontsize=12)
        #cbar1.ax.tick_params(labelsize=10)

    # Format the datetime into a more readable format
        datetime_obs = parse_filename_datetime_obs(KTYX_closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        print("Made formatted datetime for obs")
        #ax_a1.set_title(f"Attempt 1 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_a2.set_title(f"Attempt 2 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_a3.set_title(f"Attempt 3 at " + str(time_object_adjusted),fontsize=12,fontweight='bold')
        #ax_obs.set_title(f"Observation at" + formatted_datetime_obs, fontsize=12,fontweight='bold')
        plt.suptitle(datetime_obs)
    # Save the figure to a file
        frame_number = os.path.splitext(os.path.basename(file_path_a1))[0]
        filename = f'frame_{frame_number}{timeidx}.png'
        plt.savefig(filename)
        print("Saving Frame")

    #print(f"{os.path.basename(file_path)} Processed!")
        if time_object_adjusted.day == 19 and time_object_adjusted.hour == 0 and time_object_adjusted.minute == 0:
            plt.show()
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
    tasks = zip(wrf_filelist_1, wrf_filelist_2, wrf_filelist_3, timeidxlist)

    #output_gif = f'4panelcomparerefloopD{domain}{wrf_date_time_start.month:02d}{wrf_date_time_start.day:02d}{wrf_date_time_start.hour:02d}{wrf_date_time_start.minute:02d}to{wrf_date_time_end.month:02d}{wrf_date_time_end.day:02d}{wrf_date_time_end.hour:02d}{wrf_date_time_end.minute:02d}.gif'
    
    print("Finished gathering tasks")
    #print(tasks[:5])

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
        frame_filenames = list(frame_filenames_gen)  # Convert generator to list
    
    
    # For Normal Processing
    #frame_filenames = []
    #for file_path, timeidx in tasks:
    #    filename = generate_frame(file_path, timeidx)
    #    if filename:
    #        frame_filenames.append(filename)    
    

    # Create the GIF
    filtered_list = [filename for filename in frame_filenames if filename is not None]    
    #create_gif(sorted(filtered_list), output_gif)

    # Clean up the frame files
    for filename in filtered_list:
        print("Removing: ", filename)
        os.remove(filename)
    
