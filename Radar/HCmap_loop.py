import os
import warnings
import numpy
import cartopy.crs as ccrs
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

'''
Loops through a start and end time creating plan view maps
PROVIDED that files exist in the specified directory
'''

# === User Input ===
start_time = datetime(2023,3,19,3,0)
end_time = datetime(2023,3,19,3,0)
step = timedelta(minutes=5) # Timestep to loop through
radar = 'KTYX'
var = 'N0H' # Level III
radar_data_dir = "/data2/white/DATA/MET399/NEXRADLVL3/20230319/"
savepath = "/data2/white/PLOTS_FIGURES/MET399/20230319/"

# === LOOP THROUGH TIME ===
current_time = start_time
while current_time <= end_time:
    if current_time.minute == 0:  # only do files for each hour
        radar_file = radarfuncs.find_closest_radar_lvl3_file(radar_data_dir, var, radar, current_time)

        if radar_file:

            src = os.path.join(radar_data_dir, radar_file)
            radar_object = pyart.io.read_nexrad_level3(src)
            hhc = radar_object.fields['radar_echo_classification']['data']

            display = pyart.graph.RadarMapDisplay(radar_object,grid_projection=None)
            fig = plt.figure(figsize=(30,15),facecolor='white')
            ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())


            labels = ["ND","BI","GC","IC","DS","WS","RA","HR","BD","GR","HA","LH","GH","--","UK","RH"]
            ticks = np.arange(5,156,10)

            boundaries = np.arange(0, 161, 10)
            norm = mpl.colors.BoundaryNorm(boundaries, 256)
            
            display.plot_ppi_map('radar_echo_classification',0,ax=ax, norm=norm, ticks=ticks,ticklabs=labels,projection=ccrs.PlateCarree(),resolution='50m')

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
            
            gl = ax.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
            gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
            gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
            gl.xlines = True
            gl.ylines = True

            gl.top_labels = False  # Disable top labels
            gl.right_labels = False  # Disable right labels
            gl.xpadding = 20

            # Area to plot
            ax.set_extent([-79.5, -74.0, 42.5, 44.5], crs=ccrs.PlateCarree())
            
            # Mask out class 0
            hhc_filtered = hhc[hhc >= 10]
            counts, bin_edges = numpy.histogram(hhc_filtered,bins=boundaries)

            # Get the index of the bin with the most counts
            max_index = np.argmax(counts)

            # Get the bin value (e.g., center or lower edge of the bin)
            most_frequent_class_num = boundaries[max_index]  
            most_frequent_class = labels[int(most_frequent_class_num/10)] # Grabs the name rather than number (i.e. IC instead of 30)
            
            # Make a String of the file time
            parts = radar_file.split('_')
            filename_str = parts[-1]

            print(f"Most frequent hydrometeor class for {filename_str}: {most_frequent_class}")
            plt.savefig(savepath + f"HCmap{filename_str}.png")

            #plt.show()
        else:
            print(f"No file found for {current_time}")
    current_time += step


