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
from metpy.plots import USCOUNTIES
from matplotlib.colors import Normalize
from PIL import Image
from datetime import datetime, timedelta
import cartopy.io.shapereader as shpreader
import wrffuncs

"""
A simple script to loop through a range of your files to detect
a given WRF variable.
"""

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2
var = "T2"

# Path to each WRF run (NORMAL & FLAT)
path = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"

# --- END USER INPUT ---

time_df = wrffuncs.build_time_df(path, domain)

# Filter time range
mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timeidxlist = time_df["timeidx"].tolist()

# Open the NetCDF file
def generate_frame(args):
    try:
    # Seperate these for normal processing and place in args 
        file_path, timeidx = args
        # Read data from file
        with Dataset(file_path) as wrfin:
            data = getvar(wrfin, var, timeidx=timeidx)
            
            if np.any(to_np(data) > 0): 
                print("Data detected at: " + os.path.basename(file_path) + " Timeindex: " + str(timeidx))
               
    except TypeError:
        print("Error processing files")

if __name__ == "__main__":

    tasks = zip(filelist, timeidxlist)

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:  # Use ProcessPoolExecutor for CPU-bound tasksI
        frame_filenames_gen = executor.map(generate_frame, tasks)
        
        
