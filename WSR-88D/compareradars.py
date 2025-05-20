# @twhite
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from metpy.plots import ctables
from wrf import (getvar, interpline, interplevel, to_np, vertcross, smooth2d, CoordPair, GeoBounds,get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)
import pyart
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader
import os
import glob
import radarfuncs
from datetime import datetime, timedelta
import nexradaws

"""
This script plots NEXRAD radar data for a specified time period. It supports multiple radar sites and will automatically download the required data if not already present in the local directory.

Additionally, the script compares NEXRAD reflectivity data with output from a Climavision gap-filling radar located in Pleasantville, PA.
"""

# --- USER INPUT ----
var = "cross_correlation_ratio" # "reflectivity", "cross_correlation_ratio", "spectrum_width","differential_phase","velocity"
time_of_interest = datetime(2025,2,16,6,11,0)
radar_data_dir = '/data2/white/DATA/LESPaRC/NEXRADLVL2/'
savepath = f"/data2/white/PLOTS_FIGURES/LESPaRC/"

# Area for the plan view (Left Lon, Right Lon, Bottom Lat, Top Lat)
extent = [-82,-77.00000,40.5,42.5]

# Radars to get data from, use None when not needed
radars = ["KCLE", "KBUF", "KCCX"]
radar_coords = {
    'KCLE': (41.4122, -81.8594),
    'KCCX': (41.1286, -77.9956),
    'V109': (41.5901, -79.572),
}
# --- END USER INPUT ---

radar_files = {}

for radar in radars:
    if radar is not None:
        file_path = radarfuncs.find_closest_radar_file(time_of_interest, radar_data_dir, radar)

        # Check if the file exists before storing it
        if file_path and os.path.exists(file_path):
            radar_files[radar] = file_path
        else:
            radar_files[radar] = None  # Mark as missing

# Check for missing files and handle downloading if needed
missing_radars = [radar for radar, file in radar_files.items() if file is None]
conn = nexradaws.NexradAwsInterface()

if missing_radars:
    print(f"Missing radar files for: {missing_radars}")

    # Define the time range for available scans
    start = time_of_interest - timedelta(minutes=3)
    end = time_of_interest + timedelta(minutes=3)

    # Check available scans and download data for missing radars
    for radar in missing_radars:
        scans = conn.get_avail_scans_in_range(start, end, radar)
        print(f"There are {len(scans)} scans available for {radar} between {start} and {end}\n")

        # Download the data
        downloaded_files = conn.download(scans, radar_data_dir)

    # **Step 3: Re-check for Files After Download**
for radar in missing_radars:
    file_path = uncs.find_closest_radar_file(time_of_interest, radar_data_dir, radar)
    if file_path and os.path.exists(file_path):
        radar_files[radar] = file_path
    else:
        print(f"Warning: Still missing file for {radar} after download.")

# **Step 4: Read Radar Data**
for radar, file_path in radar_files.items():
    if file_path:
        print(f"Reading radar data from {file_path} for {radar}...")
        radar_data = pyart.io.read(file_path)
        # Process radar_data as needed
   
print(radar_files)

# Read radar files into Py-ART objects
radar_objects = {}  # Store Py-ART radar objects
display_objects = {}  # Store Py-ART display objects

for radar, file_path in radar_files.items():
    if file_path:
        print(f"Reading radar data from {file_path} for {radar}...")
        radar_obj = pyart.io.read(file_path)
        radar_objects[radar] = radar_obj  # Store radar object
        display_objects[radar] = pyart.graph.RadarMapDisplay(radar_obj)  # Create display object

# Example of accessing start and end times for data search
'''
for radar, radar_obj in radar_objects.items():
    start_time = radar_obj.time["data"][0]  # First time step
    end_time = radar_obj.time["data"][-1]  # Last time step
    print(radar_obj.time["units"])
    print(f"{radar}: Start Time = {start_time}, End Time = {end_time}") 
'''

# Locate radar data directory for Climavision radar
radar_data_dir = "/data2/white/DATA/LESPaRC/NYDOT/20250216/"
radar_file = radarfuncs.find_closest_radar_file(time_of_interest, radar_data_dir, "v109")

# Get the observed variables
obs_dbz = pyart.io.read(radar_file)

# Retrieve lat/lon of Climavision radar
#print(f"Latitude: {obs_dbz.latitude['data'][0]}, Longitude: {obs_dbz.longitude['data'][0]}")
display = pyart.graph.RadarMapDisplay(obs_dbz)

# Use these lines to see the fields in the Climavision file
radar_fields = obs_dbz.fields
#print("Observed radar fields: ", radar_fields)

# Create a figure that will have 2 subplots
fig = plt.figure(figsize=(30,15))
ax_ctrl = fig.add_subplot(1,2,1, projection=crs.PlateCarree())
ax_gap = fig.add_subplot(1,2,2, projection=crs.PlateCarree())

# Set the margins to 0
ax_ctrl.margins(x=0,y=0,tight=True)
ax_gap.margins(x=0,y=0,tight=True)

# Download and create the states, land, and oceans using cartopy features. (Can add if Needed)
'''
states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',facecolor='none',name='admin_1_states_provinces')
land = cfeature.NaturalEarthFeature(category='physical', name='land',scale='50m',facecolor=cfeature.COLORS['land'])
lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue",zorder=2)
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',scale='50m',facecolor=cfeature.COLORS['water'])
'''

# Special stuff for counties
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/LESPaRC/county_shapefiles/countyl010g.shp')
counties = list(reader.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, crs.PlateCarree(),zorder=7)

# Special stuff for roads
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/LESPaRC/road_shapefiles/tl_2024_us_primaryroads/tl_2024_us_primaryroads.shp')
roads = list(reader.geometries())
ROADS = cfeature.ShapelyFeature(roads, crs.PlateCarree(),zorder=8)

# Add County/State borders
ax_ctrl.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=.8,linestyle='--')
ax_gap.add_feature(COUNTIES, facecolor='none', edgecolor='black',linewidth=.8,linestyle='--')

# Add Primary (Interstates) roads
ax_ctrl.add_feature(ROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_gap.add_feature(ROADS, facecolor='none', edgecolor='black',linewidth=.75)


# Reads in super deatiled roads
'''
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/LESPaRC/tl_2024_36_prisecroads/tl_2024_36_prisecroads.shp')
NYroads = list(reader.geometries())
NYROADS = cfeature.ShapelyFeature(NYroads, crs.PlateCarree(),zorder=7)
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/LESPaRC/tl_2024_42_prisecroads/tl_2024_42_prisecroads.shp')
PAroads = list(reader.geometries())
PAROADS = cfeature.ShapelyFeature(PAroads, crs.PlateCarree(),zorder=7)
reader = shpreader.Reader('/data2/white/PYTHON_SCRIPTS/LESPaRC/tl_2024_39_prisecroads/tl_2024_39_prisecroads.shp')
OHroads = list(reader.geometries())
OHROADS = cfeature.ShapelyFeature(OHroads, crs.PlateCarree(),zorder=7)
'''

# Plots detailed roads if needed
'''
ax_ctrl.add_feature(NYROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_gap.add_feature(NYROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_ctrl.add_feature(PAROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_gap.add_feature(PAROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_ctrl.add_feature(OHROADS, facecolor='none', edgecolor='black',linewidth=.75)
ax_gap.add_feature(OHROADS, facecolor='none', edgecolor='black',linewidth=.75)
'''

gl = ax_ctrl.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl.xlines = True
gl.ylines = True

gl.top_labels = False  # Disable top labels
gl.right_labels = False  # Disable right labels
gl.xpadding = 20

gl2 = ax_gap.gridlines(color="black", linestyle="dotted",draw_labels=True, x_inline=False, y_inline=False)
gl2.xlabel_style = {'rotation': 'horizontal','size': 14,'ha':'center'} # Change 14 to your desired font size
gl2.ylabel_style = {'size': 14}  # Change 14 to your desired font size
gl2.xlines = True
gl2.ylines = True
gl2.top_labels = False  # Disable top labels
gl2.right_labels = False  # Disable right labels
gl2.xpadding = 20

# ZDR cmap
colors1 = plt.cm.binary_r(np.linspace(0.,0.8,33))
colors2= plt.cm.gist_ncar(np.linspace(0.,1,100))
colors = np.vstack((colors1, colors2[10:121]))
ZDR_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# Dictionary for each variable setting
field_settings = {
    'reflectivity': {
        'vmin': 5,
        'vmax': 70,
        'cmap': 'pyart_NWSRef',
        'label': 'Reflectivity',
        'units': 'dBZ',
        },
    'velocity': {
        'vmin': -40,
        'vmax': 40,
        'cmap': 'pyart_NWSVel',
        'label': 'Radial Velocity',
        'units': 'm/s',
        },
    'cross_correlation_ratio': {
        'vmin': 0.7,
        'vmax': 1.01,
        'cmap': 'pyart_SCook18',
        'label': 'Correlation Coefficient',
        'units': '',
    },
    'differential_reflectivity': {
        'vmin': -2,
        'vmax': 6,
        'cmap': ZDR_cmap,
        'label': 'Differential Reflectivity',
        'units': 'dB',
    },
    # Add more fields as needed
}

settings = field_settings[var]
radar_zorder_start = 2

# Plotting WSR-88D Radars 
for i, radar in enumerate(radars):
    CTRL_contour = display_objects[radar].plot_ppi_map(var, mask_outside=True, vmin=settings['vmin'], vmax=settings['vmax'], ax=ax_ctrl,colorbar_flag=False,title_flag=False,add_grid_lines=False,cmap=settings['cmap'],colorbar_label=settings['label'],zorder=radar_zorder_start + i)
    OBS_contour = display_objects[radar].plot_ppi_map(var, mask_outside=True, vmin=settings['vmin'], vmax=settings['vmax'], ax=ax_gap,colorbar_flag=False,title_flag=False,add_grid_lines=False,cmap=settings['cmap'],colorbar_label=settings['label'],zorder=radar_zorder_start + i)

# Adding gap filling radar
obs_contour = display.plot_ppi_map(var,mask_outside=True, vmin=settings['vmin'], vmax=settings['vmax'], ax=ax_gap,colorbar_flag=False,title_flag=False,add_grid_lines=False,cmap=settings['cmap'],colorbar_label=settings['label'],zorder=len(radars) + 1)

# Manually create a mappable with identical settings to make this work correctly and not mess up the sizes of the plots
norm = mcolors.Normalize(vmin=settings['vmin'], vmax=settings['vmax'])
mappable = ScalarMappable(norm=norm, cmap=settings['cmap'])
mappable.set_array([])

# Add the colorbar for the plots with respect to the position of the plots
cbar_ax1 = fig.add_axes([ax_gap.get_position().x1 + 0.01,ax_gap.get_position().y0+.1,0.015,ax_gap.get_position().height-.21])
cbar1 = fig.colorbar(mappable, cax=cbar_ax1,shrink=0.5)
cbar1.set_label(settings['units'], fontsize=12,labelpad=6)
cbar1.ax.tick_params(labelsize=10)

# Plot radar locations on map
for radar_id, (lat, lon) in radar_coords.items():
    if radar_id != "V109":
        ax_ctrl.plot(lon, lat, marker='^', color='black', markersize=8, transform=crs.PlateCarree(), zorder=10)
        #ax_ctrl.text(lon + 0.3, lat + 0.3, radar_id, transform=crs.PlateCarree(), fontsize=10)
        ax_gap.plot(lon, lat, marker='^', color='black', markersize=8, transform=crs.PlateCarree(), zorder=10)
        #ax_gap.text(lon + 0.03, lat + 0.3, radar_id, transform=crs.PlateCarree(), fontsize=10)
    else:
        ax_gap.plot(lon, lat, marker='*', color='yellow', markersize=8, transform=crs.PlateCarree(), zorder=10)
        #ax_ctrl.text(lon + 0.3, lat + 0.3, radar_id, transform=crs.PlateCarree(), fontsize=10)


# Set the view of the plot
ax_gap.set_extent(extent, crs=crs.PlateCarree())
ax_ctrl.set_extent(extent, crs=crs.PlateCarree())

#Set the x-axis and  y-axis labels
ax_gap.set_xlabel("Longitude", fontsize=8)
ax_ctrl.set_ylabel("Lattitude", fontsize=8)
ax_gap.set_ylabel("Lattitude", fontsize=8)

# Format the datetime into a more readable format
datetime_obs = radarfuncs.parse_filename_datetime_obs(radar_data_dir + radar_file)
formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
plt.suptitle("Climavision Scan at " + formatted_datetime_obs)

# Add a title
ax_ctrl.set_title(settings['label'] + ' Without Gap-Filling Radar',  fontsize=14, fontweight='bold')
ax_gap.set_title(settings['label'] + ' With Gap-Filling Radar',fontsize=14, fontweight='bold')

# Save Figure
plt.savefig(savepath + f"Compare_{var[:4]}_{datetime_obs.year:04d}{datetime_obs.month:02d}{datetime_obs.day:02d}{datetime_obs.hour:02d}{datetime_obs.minute:02d}.png")

# Show Figure
plt.show()
