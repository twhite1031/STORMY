import numpy as np
import concurrent.futures
from wrf import (to_np, getvar)
from netCDF4 import Dataset
from datetime import datetime
import pyart
import numpy.ma as ma
import wrffuncs

# MAY NEED TO USE IN SHELL
#export PROJ_NETWORK=OFF

# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,2,00), datetime(1997, 1, 10,00, 18, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"
# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

# --- END USER INPUT ---

time_df_N = wrffuncs.build_time_df(path_N, domain)
time_df_F = wrffuncs.build_time_df(path_F, domain)

# Filter time range
mask_N = (time_df_N["time"] >= start_time) & (time_df_N["time"] <= end_time)
mask_F = (time_df_F["time"] >= start_time) & (time_df_F["time"] <= end_time)

time_df_N = time_df_N[mask_N].reset_index(drop=True)
time_df_F = time_df_F[mask_F].reset_index(drop=True)

filelist_N = time_df_N["filename"].tolist()
filelist_F = time_df_F["filename"].tolist()
timeidxlist = time_df_N["timeidx"].tolist()


# ---- End User input for file ----

def generate_frame(args):
    print("Starting generate frame")
    file_path_N, file_path_F, timeidx = args
    try:
        
    # Read data from file
        with Dataset(file_path_N) as wrfin:
            mdbz_N = getvar(wrfin, "mdbz", timeidx=timeidx)
            mdbz_N = to_np(mdbz_N)
        with Dataset(file_path_F) as wrfin2:
            mdbz_F = getvar(wrfin2, "mdbz", timeidx=timeidx)
            mdbz_F = to_np(mdbz_F)
        print('Got WRF data')

    # Find the closest radar file
        time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(file_path_N, timeidx)
        print(time_object_adjusted)
    # Locate radar data directory
        radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"
        closest_file = wrffuncs.find_closest_radar_file(time_object_adjusted, radar_data_dir,radar_prefix="KTYX")
        print(closest_file)
        with Dataset(path_N+'wrfout_d02_1997-01-14_00:00:00') as wrfin:
            lon = getvar(wrfin, "lon", timeidx=0)
            lat = getvar(wrfin, "lat", timeidx=0)
        lat_min, lat_max = 43.1386, 44.2262
        lon_min, lon_max = -77.345, -74.468
        lat = to_np(lat)
        lon = to_np(lon)
        lat_mask = (lat > lat_min) & (lat < lat_max)
        lon_mask = (lon > lon_min) & (lon < lon_max)
    # This mask can be used an any data to ensure we are in are modified region
        region_mask = lat_mask & lon_mask
    
    # Apply the mask to WRF data
        masked_data = np.where(region_mask, lat, np.nan)

    # Use this to remove nan's for statistical operations, can apply to all the data since they have matching domains
        final_mask = ~np.isnan(masked_data)
       
  
    # Get the observed variables
        
        obs_dbz = pyart.io.read_nexrad_archive(closest_file)
        comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
        comp_ref_data = comp_ref.fields['composite_reflectivity']['data']

    # Filters and flattens
        filtered_data_N = mdbz_N[final_mask]
        filtered_data_F = mdbz_F[final_mask]
        print(filtered_data_N)

        # Additional mask to include values only over 10 dBZ
        dbz_mask_N = filtered_data_N > 10
        dbz_mask_F = filtered_data_F > 10
        dbz_mask_obs = (comp_ref_data > 10) & (comp_ref_data < 60)

        filtered_data_N = filtered_data_N[dbz_mask_N]
        filtered_data_F = filtered_data_F[dbz_mask_F]
        comp_ref_data = comp_ref_data[dbz_mask_obs]
        print('All masks completed')

        # Convert to Z for stats evalutation
        for idx,value in enumerate(filtered_data_N):
            filtered_data_N[idx] = 10**(value/10.) # Use linear Z for interpolation
        for idx,value in enumerate(filtered_data_F):
            filtered_data_F[idx] = 10**(value/10.) # Use linear Z for interpolation
        for idx,value in enumerate(comp_ref_data):
            comp_ref_data[idx] = 10**(value/10.) # Use linear Z for interpolation
        print('Data Converted')
        print(filtered_data_N)
        max_N = np.max(filtered_data_N)
        max_F = np.max(filtered_data_F)
        max_obs = np.max(comp_ref_data)

        #Back to logrithamic
        max_N = 10.0 * np.log10(max_N)
        max_F = 10.0 * np.log10(max_F)
        max_obs = 10.0 * np.log10(max_obs)
        print('Max calculated')

        min_N = np.min(filtered_data_N)
        min_F = np.min(filtered_data_F)
        min_obs = np.min(comp_ref_data)
        
        #Back to logrithamic
        min_N = 10.0 * np.log10(min_N)
        min_F = 10.0 * np.log10(min_F)
        min_obs = 10.0 * np.log10(min_obs)
        print("Min Calculated")

        mean_N = np.mean(filtered_data_N)
        mean_F = np.mean(filtered_data_F)
        mean_obs = np.mean(comp_ref_data)
        
        #Back to logrithamic
        mean_N = 10.0 * np.log10(mean_N)
        mean_F = 10.0 * np.log10(mean_F)
        mean_obs = 10.0 * np.log10(mean_obs)
        print("Mean calculated")

        median_N = ma.median(filtered_data_N)
        median_F = ma.median(filtered_data_F)
        median_obs = ma.median(comp_ref_data)
        print("Median calculated one")

        #Back to logrithamic
        median_N = 10.0 * np.log10(median_N)
        median_F = 10.0 * np.log10(median_F)
        median_obs = 10.0 * np.log10(median_obs)
        print("Median calculated")

        std_N = np.std(filtered_data_N)
        std_F = np.std(filtered_data_F)
        std_obs = np.std(comp_ref_data)
    
        #Back to logrithamic
        std_N = 10.0 * np.log10(std_N)
        std_F = 10.0 * np.log10(std_F)
        std_obs = 10.0 * np.log10(std_obs)

        range_N = max_N - min_N
        range_F = max_F - min_F
        range_obs = max_obs - min_obs

        datetime_obs = wrffuncs.parse_filename_datetime_obs(closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        
        #print("All stats completed")
        if time_object_adjusted.minute == 0 and time_object_adjusted.second == 0:
            print("Attempt 1 at" + str(time_object_adjusted) + " - Max: " + str(round(max_N,2)) + " Mean: " + str(round(mean_N,2)) + " Median: " + str(round(median_N,2)) + " Standard Dev: " + str(round(std_N, 2)))
            print("Attempt 2 at" + str(time_object_adjusted) + " - Max: " + str(round(max_F,2)) + " Mean: " + str(round(mean_F,2)) + " Median: " + str(round(median_F,2)) + " Standard Dev: " + str(round(std_F, 2)))
            print("Observation at" + formatted_datetime_obs + " - Max: " + str(round(max_obs,2)) + " Mean: " + str(round(mean_obs,2)) + " Median: " + str(round(median_obs,2)) + " Standard Dev: " + str(round(std_obs, 2)))
            print("------------------------------------------------------------")

    except Exception as e:
        return f"Error: {e}"  # Return the error message instead of crashing:
                
if __name__ == "__main__":

  
    tasks = zip(filelist_N, filelist_F, timeidxlist)    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
    
      
