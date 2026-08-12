import glob
import os
from datetime import timedelta
import numpy as np
from matplotlib.colors import from_levels_and_colors
import requests
from PIL import Image
import cartopy.feature as cfeature
import STORMY 

# Keep the historical ``STORMY.WRF.wrffuncs`` imports working while the focused
# implementation lives in the lightweight organization module.
from .organization import (
    build_time_df,
    generate_wrf_filenames,
    get_timeidx,
    get_timeidx_and_wrf_file,
    parse_filename_datetime_wrf,
    parse_wrfout_time,
    round_to_nearest_5_minutes,
)

def find_closest_radar_file(target_datetime, directory, radar_prefix=None):
    """Finds the file in the directory with the datetime closest to the target datetime."""
    closest_file = None
    closest_diff = None
    
    # Iterate over all files in the directory
    if radar_prefix:
        search_pattern = os.path.join(directory, f'{radar_prefix}*.ar2v')
    else:
        search_pattern = os.path.join(directory, '*.ar2v')

    for filepath in glob.glob(search_pattern):
        # Extract the filename
        filename = os.path.basename(filepath)
        try:
            # Parse the datetime from the filename
            file_datetime = STORMY.parse_filename_datetime_obs(filename)
            # Calculate the difference between the file's datetime and the target datetime
            diff = abs((file_datetime - target_datetime).total_seconds())
            # Update the closest file if this file is closer
            if closest_diff is None or diff < closest_diff:
                closest_file = filepath
                closest_diff = diff
        except ValueError:
            # If the filename does not match the expected format, skip it
            continue
    
    return closest_file

def get_nws_cmap_norm():
    dbz_levels = np.arange(5., 75., 5.)

    # Create the color table found on NWS pages.
    dbz_rgb = np.array([[4,233,231],
                    [1,159,244], [3,0,244],
                    [2,253,2], [1,197,1],
                    [0,142,0], [253,248,2],
                    [229,188,0], [253,149,0],
                    [253,0,0], [212,0,0],
                    [188,0,0],[248,0,253],
                    [152,84,198]], np.float32) / 255.0
    dbz_map, dbz_norm = from_levels_and_colors(dbz_levels, dbz_rgb,
                                           extend="max")
    return dbz_map, dbz_norm

# tbuffer in seconds
def get_LMA_flash_data(start,tbuffer):
    filenames = []
    filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format(start.strftime('%y%m%d_%H%M')[:-1])
    filenames.append(filename)
    if (glob.glob(filename) == []): # Check if file exists 
        url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format(start.strftime('%Y/%m/%d'),os.path.basename(filename))
        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f'{filename} downloaded successfully.')
    if (tbuffer > 600):
        for i in range(int(tbuffer/600)):
            filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format((start+timedelta(seconds=(i*600))).strftime('%y%m%d_%H%M')[:-1])
            filenames.append(filename)
            if (glob.glob(filename) == []): # Check if file exists
                url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format((start+timedelta(seconds=(i*600))).strftime('%Y/%m/%d'),os.path.basename(filename))
                print(url)
                response = requests.get(url)
                with open(filename, "wb") as file:
                    file.write(response.content)
                print(f'{filename} downloaded successfully.')
    filename = '/data2/white/DATA/PROJ_LEE/LMADATA/LYLOUT_{}000_0600.dat.flash.h5'.format((start+timedelta(seconds=tbuffer)).strftime('%y%m%d_%H%M')[:-1])
    if filename not in filenames:
        filenames.append(filename)
        if (glob.glob(filename) == []):
                url = 'https://data.nssl.noaa.gov/thredds/fileServer/WRDD/OKLMA/deployments/flashsort_6/h5_files/{}/{}'.format((start+timedelta(seconds=tbuffer)).strftime('%Y/%m/%d'),os.path.basename(filename))
                response = requests.get(url)
                with open(filename, "wb") as file:
                    file.write(response.content)
                print(f'{filename} downloaded successfully.')

def format_gridlines(ax, **kwargs):
    """
    Apply consistent gridline formatting to a Cartopy Axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The axis to format.

    kwargs : dict
        Additional keyword arguments passed to ax.gridlines().
        For example: color="black", linestyle="dotted", x_inline=False, y_inline=False
    """
    gl = ax.gridlines(draw_labels=True, **kwargs)
    gl.xlabel_style = {'rotation': 'horizontal', 'size': 14, 'ha': 'center'}
    gl.ylabel_style = {'size': 14}
    gl.xlines = True
    gl.ylines = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xpadding = 20
    return gl

def add_cartopy_features(ax, 
                         add_borders=True, 
                         add_states=True, 
                         add_lakes=True, 
                         add_ocean=True, 
                         add_land=True):
    """
    Add common cartopy map features to an axis.

    Parameters
    ----------
    ax : matplotlib axis with cartopy projection
        The axis to add features to.
    add_borders, add_states, add_lakes, add_ocean, add_land : bool
        Control which features are added.
    """
   
    
    if add_borders:
        
        borders = cfeature.NaturalEarthFeature(
            'cultural', 'admin_0_countries', '50m', facecolor='none'
        )
        ax.add_feature(borders, edgecolor='black', linewidth=1)

    if add_states:
        states = cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces', '50m', facecolor='none'
        )
        ax.add_feature(states, edgecolor='gray', linewidth=1)

    if add_lakes:
        lakes = cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m', facecolor='none', edgecolor='blue'
        )
        ax.add_feature(lakes, linewidth=1)

    if add_ocean:
        ocean = cfeature.NaturalEarthFeature(
            'physical', 'ocean', '50m', facecolor=cfeature.COLORS['water']
        )
        # ax.add_feature(ocean)

    if add_land:
        land = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m', facecolor=cfeature.COLORS['land']
        )
       #ax.add_feature(land)

def make_contour_levels(data, interval):
    """
    Create contour levels for a given dataset and interval.
    
    Parameters
    ----------
    data : array-like
        Input data array.
    interval : float
        Interval between contour levels.
        
    Returns
    -------
    np.ndarray
        Array of contour levels covering the full data range.
    """
    data = np.asarray(data)
    start = np.floor(np.nanmin(data) / interval) * interval
    end   = np.ceil(np.nanmax(data) / interval) * interval
    return np.arange(start, end + interval, interval)

# Function to create a GIF from the generated frames
def create_gif(path, frame_filenames, output_filename):

    frames = []
    for filename in frame_filenames:
            new_frame = Image.open(filename)
            frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(path + output_filename, format='GIF', append_images=frames[1:],save_all=True,duration=75, loop=0)
