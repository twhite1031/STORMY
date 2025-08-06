import glob
import os
from datetime import datetime, timedelta
import numpy as np
from matplotlib.colors import from_levels_and_colors
import pandas as pd
from netCDF4 import Dataset
from wrf import extract_times
import requests
from PIL import Image

# Adjust datetime to match filenames
def round_to_nearest_5_minutes(dt):
    # Extract the minute value
    minute = dt.minute

    # Calculate the nearest 5-minute mark
    nearest_5 = round(minute / 5) * 5

    # Handle the case where rounding up to 60 minutes
    if nearest_5 == 60:
        dt = dt + timedelta(hours=1)
        nearest_5 = 0

    # Replace the minute value with the nearest 5-minute mark
    rounded_dt = dt.replace(minute=nearest_5, second=0, microsecond=0)

    return rounded_dt

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
            file_datetime = parse_filename_datetime_obs(filename)
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

def parse_filename_datetime_wrf(filepath, timeidx, timeidx_interval=5):
    
    # Define the format of the datetime string in your filename
        datetime_format = "wrfout_d02_%Y-%m-%d_%H_%M_%S"

    # Parse the datetime string into a datetime object
        time_object = datetime.strptime(os.path.basename(filepath), datetime_format)

    # Add timeidx value
        add_time = timeidx_interval * int(timeidx)
        time_object_adjusted = time_object + timedelta(minutes=add_time)

        return time_object_adjusted

def generate_wrf_filenames(start_time, wrf_date_time_end,file_interval, numtimeidx,domain=1,wrf_start_hour=0):

    """
    Generates WRF output filenames and time indices for each time step between start and end time.

    Parameters:
    - current_time (datetime): The starting datetime
    - wrf_date_time_end (datetime): The ending datetime
    - domain (int): The WRF domain (e.g., 1 or 2)
    - numtimeidx (int): Number of time indices per WRF file

    Returns:
    - filelist (numpy array): List of generated WRF filenames
    - timeidxlist (numpy array): Corresponding time indices
    """

    filelist = np.array([])
    timeidxlist = np.array([], dtype=int)

    while start_time <= wrf_date_time_end:
               
        # Define the first WRF file time of the current day
        wrf_start_time = start_time.replace(hour=wrf_start_hour, minute=0, second=0, microsecond=0)

        # Compute the nearest WRF file start time before or at the given date_time
        elapsed_minutes = (start_time - wrf_start_time).total_seconds() / 60
        wrf_offset = (elapsed_minutes // file_interval) * file_interval
        wrf_filename_time = wrf_start_time + timedelta(minutes=wrf_offset)

        # Compute time index within the selected WRF file
        time_offset = (start_time - wrf_filename_time).total_seconds() / 60
        time_step = file_interval // numtimeidx
        timeidx = int(time_offset // time_step)
        
        # Construct WRF file name pattern
        pattern = f"wrfout_d0{domain}_{wrf_filename_time.year:04d}-{wrf_filename_time.month:02d}-{wrf_filename_time.day:02d}_{wrf_filename_time.hour:02d}:{wrf_filename_time.minute:02d}:00"

        filelist = np.append(filelist, pattern)
        timeidxlist = np.append(timeidxlist, timeidx)

        # Increment time by 5 minutes
        start_time += timedelta(minutes=5)

    return filelist, timeidxlist
from datetime import datetime, timedelta

def get_timeidx_and_wrf_file(date_time, file_interval_sec, numtimeidx, domain=1, wrf_start_hour=0):
    """
    Determines the correct WRF file and time index based on the given datetime.

    Parameters:
    - date_time (datetime): The datetime to plot.
    - file_interval_sec (int): The time interval (in **seconds**) between each WRF file.
    - numtimeidx (int): Number of time indices per WRF file.
    - domain (int): The WRF domain (e.g., 1 or 2).
    - wrf_start_hour (int): Hour of the first WRF file in a day (default is 0).

    Returns:
    - timeidx (int): The time index within the selected WRF file.
    - pattern (str): The corresponding WRF file name string.
    """
    # Set the base start time (start of the current day at wrf_start_hour)
    wrf_start_time = date_time.replace(hour=wrf_start_hour, minute=0, second=0, microsecond=0)

    # Compute how many seconds have passed since start
    elapsed_sec = (date_time - wrf_start_time).total_seconds()
    
    # Find the start time of the WRF file that covers this datetime
    wrf_offset_sec = int(elapsed_sec // file_interval_sec) * file_interval_sec
    wrf_filename_time = wrf_start_time + timedelta(seconds=wrf_offset_sec)

    # Determine time index within the WRF file
    time_offset_sec = (date_time - wrf_filename_time).total_seconds()
    time_step_sec = file_interval_sec / numtimeidx
    timeidx = int(time_offset_sec // time_step_sec)

    # Construct filename pattern
    pattern = f"wrfout_d0{domain}_{wrf_filename_time.strftime('%Y-%m-%d_%H:%M:%S')}"

    return timeidx, pattern

def get_timeidx(wrf_date_time, file_interval, numtimeidx):
    timeidx = int((wrf_date_time.minute % file_interval) // (file_interval // numtimeidx))
    return timeidx

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

def build_time_df(path, domain):
    wrf_files = sorted(glob.glob(os.path.join(path, f"wrfout_d0{domain}_*")))
    time_cache = os.path.join(path, f"wrfD{domain}_time_lookup.pkl")

    if os.path.exists(time_cache):
        return pd.read_pickle(time_cache)

    records = []
    for f in wrf_files:
        with Dataset(f) as ds:
            times = pd.to_datetime(extract_times(ds,timeidx=None))
            for i, t in enumerate(times):
                records.append((f, i, t))
    df = pd.DataFrame(records, columns=["filename", "timeidx", "time"])
    df.to_pickle(time_cache)
    return df

def parse_wrfout_time(filename):
    """
    Parses WRF output filenames with either:
    - colons:    wrfout_d01_2022-11-17_13:00:00
    - underscores: wrfout_d01_2022-11-17_13_00_00

    Returns:
    - file_time: safe for filenames (e.g., 20221117_1300)
    - title_time: human-readable string (e.g., 2022-11-17 13:00 UTC)
    """
    try:
        # Extract timestamp string after domain
        datetime_str = filename.split('_d0')[1].split('_', 1)[1]
    except IndexError:
        raise ValueError("Expected format: wrfout_d0X_YYYY-MM-DD_HH:MM:SS or _HH_MM_SS")

    # Try all supported time formats
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H_%M_%S"):
        try:
            dt = datetime.strptime(datetime_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Unrecognized datetime format in filename: {datetime_str}")

    # Windows-safe and readable
    file_time = dt.strftime("%Y%m%d_%H%M")
    title_time = dt.strftime("%Y-%m-%d %H:%M UTC")
    
    return file_time, title_time

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
        ax.add_feature(borders, edgecolor='black', linewidth=0.8, zorder=2)

    if add_states:
        states = cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces', '50m', facecolor='none'
        )
        ax.add_feature(states, edgecolor='gray', linewidth=0.5, zorder=2)

    if add_lakes:
        lakes = cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m', facecolor='none', edgecolor='blue'
        )
        ax.add_feature(lakes, linewidth=0.5, zorder=1)

    if add_ocean:
        ocean = cfeature.NaturalEarthFeature(
            'physical', 'ocean', '50m', facecolor=cfeature.COLORS['water']
        )
        ax.add_feature(ocean, zorder=0)

    if add_land:
        land = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m', facecolor=cfeature.COLORS['land']
        )
        ax.add_feature(land, zorder=0)

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