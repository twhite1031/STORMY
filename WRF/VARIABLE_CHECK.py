import numpy as np
import os
import concurrent.futures
from wrf import (to_np, getvar,latlon_coords)
from netCDF4 import Dataset
from datetime import datetime
import STORMY

"""
A simple script to loop through a range of your files to detect
a given WRF variable.
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,18,00,00,00), datetime(2022, 11, 19,12, 00, 00)
domain = 2
var = "T2"

path = "/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_1/" # Path to WRF output files

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = STORMY.build_time_df(path, domain)

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
            print(f"Starting: {os.path.basename(file_path)} | Time Index: {timeidx}", flush=True)

            data = getvar(wrfin, var, timeidx=timeidx)
            p = getvar(wrfin, "pressure", timeidx=timeidx,meta=False)
            tc = getvar(wrfin, "tc", timeidx=timeidx,meta=False)
            td = getvar(wrfin, "td", timeidx=timeidx,meta=False)

            # Get lat/lon
            lats, lons = latlon_coords(data)

            lat_np, lon_np = to_np(lats), to_np(lons)

            # Mask where dewpoint > temperature
            bad_mask = td > tc
            if np.any(bad_mask):
                print(f"\n--- Dewpoint > Temperature Detected ---")
            
                indices = np.argwhere(bad_mask)

                for idx in indices:
                    if p.ndim == 3:
                        k, j, i = idx

                        t_val = tc[k, j, i]
                        td_val = td[k, j, i]
                        pres = p[k, j, i]
                        lat = lat_np[j, i]
                        lon = lon_np[j, i]
                        # RH calculation
                        e_t = 6.112 * np.exp((17.67 * t_val) / (t_val + 243.5))
                        e_td = 6.112 * np.exp((17.67 * td_val) / (td_val + 243.5))
                        rh = 100 * (e_td / e_t)
                        if rh > 104.0:
                            print(f"File: {os.path.basename(file_path)} | Time Index: {timeidx}")
                            print(f"RH = {rh} | [k={k}, j={j}, i={i}] | T={t_val:.2f} °C | Td={td_val:.2f} °C | "
                              f"P={pres:.1f} hPa | Lat={lat:.2f} | Lon={lon:.2f}")

            else:  # if 2D
                j, i = idx

                t_val = tc[j, i]
                td_val = td[j, i]
                lat = lat_np[j, i]
                lon = lon_np[j, i]

                print(f"[j={j}, i={i}] | T={t_val:.2f} °C | Td={td_val:.2f} °C | Surface Level | Lat={lat:.2f} | Lon={lon:.2f}")
            '''
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
            '''
    except Exception as e:
        print(f"Error processing file {file_path} at time index {timeidx}: {e}")

if __name__ == "__main__":

    tasks = zip(filelist, timeidxlist)

    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:  # Use ProcessPoolExecutor for CPU-bound tasksI
        frame_filenames_gen = executor.map(generate_frame, tasks)
        
        
