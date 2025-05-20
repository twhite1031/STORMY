import os
import warnings
import numpy
import cartopy.crs as ccrs
from metpy.plots import ctables
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyart
from pyart.testing import get_test_data
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import radarfuncs
from datetime import datetime, timedelta
import nexradaws
conn = nexradaws.NexradAwsInterface()
'''
Loops through a start and end time creating plan view maps
of 0.5 deg Reflectivity, will download non-existing files
'''

# === User Input ===
start_time = datetime(2019,12,31,0,0)
end_time = datetime(2019,12,31,0,0)
step = timedelta(minutes=5)
radar = 'KTYX'
var = 'reflectivity' # Level II Data
radar_data_dir = f"/data2/white/DATA/MET399/NEXRADLVL2/{start_time.strftime('%Y%m%d')}/"
savepath = f"/data2/white/PLOTS_FIGURES/MET399/{start_time.strftime('%Y%m%d')}/"

# --- End Input ---

# === LOOP THROUGH TIME ===
current_time = start_time
while current_time <= end_time:
    if current_time.minute == 0:  # only do files for each hour
        radar_filepath = radarfuncs.find_closest_radar_file(current_time, radar_data_dir, radar)

        if radar_filepath:
            
            radar_object = pyart.io.read(radar_filepath)
            display = pyart.graph.RadarMapDisplay(radar_object,grid_projection=None)
            fig = plt.figure(figsize=(30,15),facecolor='white')
            ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())


            # Reflectiviy cmap
            cmap = ctables.registry.get_colortable('NWSReflectivity')

            plot_data = display.plot_ppi_map(var,sweep=0,mask_outside=True,vmin=10,vmax=60,ax=ax, colorbar_flag=True, title_flag=True, add_grid_lines=False, cmap=cmap,zorder= 5)

            
            # Remove any state lines (ShapelyFeature instances from cartopy)
            # Loop through and remove unwanted artists (like state lines to fix them)
            for artist in list(ax.artists):  # list() to avoid modifying during iteration
                artist.remove()

            # Download and add the states, lakes  and coastlines
            states = NaturalEarthFeature(category="cultural", scale="50m", facecolor="none", name="admin_1_states_provinces")
            ax.add_feature(states, linewidth=.1, edgecolor="black")
            ax.add_feature(cfeature.LAKES.with_scale('50m'),linewidth=1, facecolor="none",  edgecolor="black")
            ax.coastlines('50m', linewidth=1)
            ax.add_feature(USCOUNTIES, alpha=0.1)
            #print("Made land features")
            
            gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
            gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
            gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
            gl.xlines = True
            gl.ylines = True

            gl.top_labels = False  # Disable top labels
            gl.right_labels = False  # Disable right labels
            gl.xpadding = 20


            ax.set_extent([-79.5, -74.0, 42.5, 44.5], crs=ccrs.PlateCarree())
            filename = os.path.basename(radar_filepath)
            # Slice out date and time
            date_part = filename[4:12]   # '20200226'
            time_part = filename[13:19]  # '235717'

            # Combine into datetime object
            dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
                        
            filename_str = dt.strftime("%Y%m%d%H%M")
            plt.title("0.5 deg Reflectivity at " +  filename_str)
            plt.savefig(savepath + f"REFmap{filename_str}.png")
            #plt.show()

        else:
            print(f"No file found for {current_time}, downloading the data")

            # Define the time range for available scans
            start = current_time - timedelta(minutes=10)
            end = current_time + timedelta(minutes=10)
            
            scans = conn.get_avail_scans_in_range(start, end, radar)
            print(f"There are {len(scans)} scans available for {radar} between {start} and {end}\n")

            # Download the data
            downloaded_files = conn.download(scans, radar_data_dir)



            current_time -= step
    current_time += step


