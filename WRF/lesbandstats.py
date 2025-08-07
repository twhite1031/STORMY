import numpy as np
import concurrent.futures
from wrf import (to_np, getvar)
from netCDF4 import Dataset
from datetime import datetime
import pyart
import numpy.ma as ma
import STORMY


# --- USER INPUT ---
start_time, end_time  = datetime(1997,1,10,00,55,00), datetime(1997, 1, 10,1, 00, 00)
domain = 2

# Path to each WRF run (NORMAL & FLAT)
path_N = f"/data2/white/WRF_OUTPUTS/SEMINAR/NORMAL_ATTEMPT/"
path_F = f"/data2/white/WRF_OUTPUTS/SEMINAR/FLAT_ATTEMPT/"

# Path to save GIF or Files
savepath = f"/data2/white/WRF_OUTPUTS/SEMINAR/BOTH_ATTEMPT/"
# Locate radar data directory
radar_data_dir = "/data2/white/DATA/PROJ_LEE/IOP_2/NEXRADLVL2/HAS012534416/"

# --- END USER INPUT ---

time_df_N = STORMY.build_time_df(path_N, domain)
time_df_F = STORMY.build_time_df(path_F, domain)

# Filter time range
mask_N = (time_df_N["time"] >= start_time) & (time_df_N["time"] <= end_time)
mask_F = (time_df_F["time"] >= start_time) & (time_df_F["time"] <= end_time)

time_df_N = time_df_N[mask_N].reset_index(drop=True)
time_df_F = time_df_F[mask_F].reset_index(drop=True)

filelist_N = time_df_N["filename"].tolist()
filelist_F = time_df_F["filename"].tolist()
timeidxlist = time_df_N["timeidx"].tolist()
timelist = time_df_N["time"].tolist()

def generate_frame(args):
    file_path_N, file_path_F, timeidx,time = args
    print("Starting generate frame")
    try:
        
    # Read in data from the matched WRF file for each model run
        with Dataset(file_path_N) as wrfin:
            mdbz_N = getvar(wrfin, "mdbz", timeidx=timeidx, meta=False)
        with Dataset(file_path_F) as wrfin2:
            mdbz_F = getvar(wrfin2, "mdbz", timeidx=timeidx,meta=False)

    # Locate radar data directory
        closest_file = STORMY.find_closest_radar_file(time, radar_data_dir,radar_prefix="KTYX")
    
    # Read in data from any WRF file to grab lat/lon 
        with Dataset(path_N+'wrfout_d02_1997-01-14_00:00:00') as wrfin:
            lon = getvar(wrfin, "lon", timeidx=0,meta=False)
            lat = getvar(wrfin, "lat", timeidx=0,meta=False)

    # Define the region of interest as similar to KTYX
        lat_min, lat_max = 43.1386, 44.2262
        lon_min, lon_max = -77.345, -74.468
        lat_mask = (lat > lat_min) & (lat < lat_max)
        lon_mask = (lon > lon_min) & (lon < lon_max)

    # This mask can be used an any data to ensure we are in are modified region
        region_mask = lat_mask & lon_mask
        masked_data = np.where(region_mask, lat, np.nan) # Apply the mask to lat data, np.nan where not in region
        final_mask = ~np.isnan(masked_data) # Remove NaN values to get the final mask
       
    # Get the observed variables
        obs_dbz = pyart.io.read_nexrad_archive(closest_file)
        comp_ref = pyart.retrieve.composite_reflectivity(obs_dbz, field="reflectivity")
        comp_ref_data = comp_ref.fields['composite_reflectivity']['data']

    # Apply the region mask to the modeled maximum reflectivity data
        filtered_data_N = mdbz_N[final_mask]
        filtered_data_F = mdbz_F[final_mask]

    # Additional mask to include values only over 10 dBZ and less than 60 dBZ to clean noise
        dbz_mask_N = filtered_data_N > 10
        dbz_mask_F = filtered_data_F > 10
        dbz_mask_obs = (comp_ref_data > 10) & (comp_ref_data < 60)

    # Apply new mask on top of the region masked data
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
        
    # !!! Each Statisitc is calculated on Z, then converted back to dBZ for display !!!
        
    # Maximum
        max_N = np.max(filtered_data_N)
        max_F = np.max(filtered_data_F)
        max_obs = np.max(comp_ref_data)
        max_N = 10.0 * np.log10(max_N)
        max_F = 10.0 * np.log10(max_F)
        max_obs = 10.0 * np.log10(max_obs)
        print('Max calculated')

    # Minimum
        min_N = np.min(filtered_data_N)
        min_F = np.min(filtered_data_F)
        min_obs = np.min(comp_ref_data)
        min_N = 10.0 * np.log10(min_N)
        min_F = 10.0 * np.log10(min_F)
        min_obs = 10.0 * np.log10(min_obs)
        print("Min Calculated")

    # Mean
        mean_N = np.mean(filtered_data_N)
        mean_F = np.mean(filtered_data_F)
        mean_obs = np.mean(comp_ref_data)
        mean_N = 10.0 * np.log10(mean_N)
        mean_F = 10.0 * np.log10(mean_F)
        mean_obs = 10.0 * np.log10(mean_obs)
        print("Mean calculated")

    # Median
        median_N = ma.median(filtered_data_N)
        median_F = ma.median(filtered_data_F)
        median_obs = ma.median(comp_ref_data)
        median_N = 10.0 * np.log10(median_N)
        median_F = 10.0 * np.log10(median_F)
        median_obs = 10.0 * np.log10(median_obs)
        print("Median calculated")

    # Standard Deviation
        std_N = np.std(filtered_data_N)
        std_F = np.std(filtered_data_F)
        std_obs = np.std(comp_ref_data)
        std_N = 10.0 * np.log10(std_N)
        std_F = 10.0 * np.log10(std_F)
        std_obs = 10.0 * np.log10(std_obs)
        print("Standard deviation calculated")

    # Format the radar observation time for display
        datetime_obs = STORMY.parse_filename_datetime_obs(closest_file)
        formatted_datetime_obs = datetime_obs.strftime('%Y-%m-%d %H:%M:%S')
        
        if time.minute == 0 and time.second == 0:
            print(f"Attempt 1 at {time} - Max: " + str(round(max_N,2)) + " Mean: " + str(round(mean_N,2)) + " Median: " + str(round(median_N,2)) + " Standard Dev: " + str(round(std_N, 2)))
            print(f"Attempt 2 at {time} - Max: " + str(round(max_F,2)) + " Mean: " + str(round(mean_F,2)) + " Median: " + str(round(median_F,2)) + " Standard Dev: " + str(round(std_F, 2)))
            print("Observation at" + formatted_datetime_obs + " - Max: " + str(round(max_obs,2)) + " Mean: " + str(round(mean_obs,2)) + " Median: " + str(round(median_obs,2)) + " Standard Dev: " + str(round(std_obs, 2)))
            print("------------------------------------------------------------")

    except Exception as e:
        return f"Error: {e}"  # Return the error message instead of crashing:
                
if __name__ == "__main__":

  
    tasks = zip(filelist_N, filelist_F, timeidxlist,timelist)    
    # Use multiprocessing to generate frames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor: # Use ProcessPoolExecutor for CPU-bound tasks
        print("Starting multiprocessing")
        frame_filenames_gen = executor.map(generate_frame, tasks)
    
      
