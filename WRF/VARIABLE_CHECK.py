import numpy as np
import os
import concurrent.futures
from wrf import (to_np, getvar)
from netCDF4 import Dataset
from datetime import datetime
import wrffuncs

"""
A simple script to loop through a range of your files to detect
a given WRF variable.
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,17,00,2,00), datetime(2022, 11, 20,00, 18, 00)
domain = 2
var = "T2"

# Path to each WRF run (NORMAL & FLAT)
path = r"C:\Users\thoma\Documents\WRF_OUTPUTS"

# Path to save GIF or Files
savepath = r"C:\Users\thoma\Documents\WRF_OUTPUTS"
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
            
            # Handle case where data is a dict or None
            if isinstance(data, dict):
                for key, val in data.items():
                    arr = to_np(val)
                    if arr is not None and np.any(arr > 0):
                        print(f"Data detected for key '{key}' at: " + os.path.basename(file_path) + " Timeindex: " + str(timeidx))
            elif data is not None:
                arr = to_np(data)
                if arr is not None and np.any(arr > 0):
                    print("Data detected at: " + os.path.basename(file_path) + " Timeindex: " + str(timeidx))
               
    except TypeError:
        print("Error processing files")

if __name__ == "__main__":

    tasks = zip(filelist, timeidxlist)

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:  # Use ProcessPoolExecutor for CPU-bound tasksI
        frame_filenames_gen = executor.map(generate_frame, tasks)
        
        
