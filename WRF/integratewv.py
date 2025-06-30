import numpy as np
import matplotlib.pyplot as plt
from wrf import (to_np, getvar)
from netCDF4 import Dataset
from datetime import datetime
import wrffuncs

"""
A line plot of integrated water vapor using the mixing ratio given from
WRF output with the ideal gas law and area of each gridbox.
"""

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
def generate_frame(file_path_N, file_path_F, timeidx):
    try:
        with Dataset(path_N+'wrfout_d02_1997-01-10_00:00:00') as wrfin:
            lat = getvar(wrfin, "lat", timeidx=0)
            lon = getvar(wrfin, "lon", timeidx=0)

        lat_min, lat_max = 43.25, 44.25
        lon_min, lon_max = -76.25, -74.25
        lat = to_np(lat)
        lon = to_np(lon)
        lat_mask = (lat > lat_min) & (lat < lat_max)
        lon_mask = (lon > lon_min) & (lon < lon_max)
    
        # This mask can be used an any data to ensure we are in are modified region
        final_mask = lat_mask & lon_mask
        print(f"Total grid points in region: {np.count_nonzero(final_mask)}")
    

          
        # Apply the mask to WRF data
        #masked_data = np.where(region_mask, lat, np.nan)
    
        # Use this to remove nan's for statistical operations, can apply to all the data since they have matching domains
        #final_mask = ~np.isnan(region_)

        # Read data from files
        with Dataset(file_path_N) as wrfin:
            WV_N = getvar(wrfin, "QVAPOR", timeidx=timeidx,meta=False) * 1000
            T_N = getvar(wrfin, "tk", timeidx=timeidx,meta=False)
            P_N = getvar(wrfin, "pressure", timeidx=timeidx,meta=False)
            Z_N = getvar(wrfin, "z", timeidx=timeidx,meta=False)
            S_N = getvar(wrfin, "QSNOW", timeidx=timeidx,meta=False) * 1000
            ICE_N = getvar(wrfin, "QICE", timeidx=timeidx,meta=False) * 1000
            GRA_N = getvar(wrfin, "QGRAUP", timeidx=timeidx,meta=False) * 1000

        with Dataset(file_path_F) as wrfin2:
            WV_F = getvar(wrfin2, "QVAPOR", timeidx=timeidx,meta=False) * 1000
            T_F = getvar(wrfin2, "tk", timeidx=timeidx,meta=False)
            P_F = getvar(wrfin2, "pressure", timeidx=timeidx,meta=False)
            Z_F = getvar(wrfin2, "z", timeidx=timeidx,meta=False)
            S_F = getvar(wrfin2, "QSNOW", timeidx=timeidx,meta=False) * 1000
            ICE_F = getvar(wrfin2, "QICE", timeidx=timeidx,meta=False) * 1000
            GRA_F = getvar(wrfin2, "QGRAUP", timeidx=timeidx,meta=False) * 1000

        
        print("Read in WRF data")
               
        # Apply horizontal mask
        print(file_path_N)
        print("Shape of final_mask:", final_mask.shape)
        print(WV_N.shape)
        WV_region_N = WV_N[:, final_mask]
        P_region_N = P_N[:, final_mask]
        T_region_N = T_N[:, final_mask]
        z_region_N = Z_N[:, final_mask]
        S_region_N = S_N[:, final_mask]
        ICE_region_N = ICE_N[:, final_mask]
        GRA_region_N = GRA_N[:, final_mask]


        # Compute air density (ρ) using the ideal gas law
        Rd = 287.05  # Gas constant for dry air (J/kg·K)
        rho_N = (P_region_N * 100) / (Rd * T_region_N)  # Convert hPa to Pa for pressure
        print(f"rho_N {rho_N}")
      

        # Apply horizontal mask
        print(file_path_F)
        WV_region_F = WV_F[:, final_mask]
        P_region_F = P_F[:, final_mask]
        T_region_F = T_F[:, final_mask]
        z_region_F = Z_F[:, final_mask]
        S_region_F = S_F[:, final_mask]
        ICE_region_F = ICE_F[:, final_mask]
        GRA_region_F = GRA_F[:, final_mask]
        
        print(f"T_region_F {T_region_F}")
        print(f"T_region_N {T_region_N}")
        

        print(f"P_region_F {P_region_F}")
        print(f"P_region_N {P_region_N}")

        # Compute air density (ρ) using the ideal gas law
        rho_F = (P_region_F * 100) / (Rd * T_region_F)  # Convert hPa to Pa for pressure
        print(f"rho_F {rho_F}")
        # Compute vertical grid spacing (dz) at each level for both simulations
        dz_N = np.gradient(to_np(z_region_N), axis=0)  # Computes dz between unevenly spaced levels
        dz_F = np.gradient(to_np(z_region_F), axis=0)  # Computes dz between unevenly spaced levels
        
        # Grid spacing in horizontal direction (modify if needed)
        dx = 3000  # Grid spacing in meters (example, adjust based on WRF grid)
        dy = 3000  # Grid spacing in meters

        # Compute volume of each grid cell
        dV_N = dx * dy * dz_N # Grid cell volume (m³)
        dV_F = dx * dy * dz_F  # Grid cell volume (m³)
        

        # Compute total water vapor mass in region (sum over all grid cells)
        print(f"dZ_N: {dz_N}")
        print(f"dZ_F: {dz_F}")

        WV_mass_N = np.sum(WV_region_N * rho_N * dV_N)  # In grams if QVAPOR is in g/kg
        WV_mass_F = np.sum(WV_region_F * rho_F * dV_F)

        S_mass_N = np.sum(S_region_N * rho_N * dV_N)  
        S_mass_F = np.sum(S_region_F * rho_F * dV_F)

        ICE_mass_N = np.sum(ICE_region_N * rho_N * dV_N)  
        ICE_mass_F = np.sum(ICE_region_F * rho_F * dV_F)

        GRA_mass_N = np.sum(GRA_region_N * rho_N * dV_N)  
        GRA_mass_F = np.sum(GRA_region_F * rho_F * dV_F)
        
        WV_profile_N = np.mean(WV_region_N * rho_N * dV_N, axis=1)  # Mean WV at each height
        WV_profile_F = np.mean(WV_region_F * rho_F * dV_F, axis=1)
        print(WV_profile_N)
        # Parse time
        time_object_adjusted = wrffuncs.parse_filename_datetime_wrf(file_path_N, timeidx)

        # Append results to lists
        datetime_array.append(time_object_adjusted.strftime("%d %H:%M"))
        print(f"WV_region_N: {WV_region_N.shape}, rho_N: {rho_N.shape}, dV_N: {dV_N.shape}")
        print(f"WV_mass_N: {WV_mass_N}, WV_mass_F: {WV_mass_F}")
        print(f"Difference: {WV_mass_F - WV_mass_N}")
        print("Water Vapor Diff: ", np.sum(WV_mass_F) - np.sum(WV_mass_N))
        WV_array_N.append(WV_mass_N / 1000)
        WV_array_F.append(WV_mass_F / 1000)

        S_array_N.append(S_mass_N)
        S_array_F.append(S_mass_F)

        ICE_array_N.append(ICE_mass_N)
        ICE_array_F.append(ICE_mass_F)

        GRA_array_N.append(GRA_mass_N)
        GRA_array_F.append(GRA_mass_F)

        print("Data appended")
    except ValueError as e:
        print(f"Error processing files: {e}")



if __name__ == "__main__":

    datetime_array = []
    WV_array_N = []
    WV_array_F = []
    S_array_N = []
    S_array_F = []
    ICE_array_N = []
    ICE_array_F = []
    GRA_array_N = []
    GRA_array_F = []


    #plt.figure(figsize=(8, 6))  # Create a figure with a specific size
    # Set the GeoAxes to the projection used by WRF
   
    #plt.pcolormesh(region_mask, cmap="gray")  # Display the mask
    #plt.colorbar(label="Mask (1=True, 0=False)")  # Add a colorbar for reference
    #plt.title("Masked Region")  # Set the title
    #plt.show()  # Display the figure
        # Apply the mask to WRF data
    #masked_data = np.where(region_mask, lat, np.nan)
    
    # Use this to remove nan's for statistical operations, can apply to all the data since they have matching domains
    #final_mask = ~np.isnan(region_)
  
 

    tasks = zip(filelist_N, filelist_F, timeidxlist)
   
    for task in tasks:
        generate_frame(*task)

    # Create the Line Plot
    plt.figure(figsize=(12, 5))
    #WV_array_N = np.array(WV_array_N)
    #WV_array_F = np.array(WV_array_F)
    #relative_difference = ((WV_array_N - WV_array_F) / WV_array_F) * 100

    plt.plot(datetime_array, WV_array_N, label="Water Vapor (Normal)",color='red') 
    plt.plot(datetime_array, WV_array_F, label="Water Vapor (Flat)",color='red',linestyle = "--") 
    
    #plt.plot(datetime_array, S_array_N, label="Snow (Normal)",color='blue') 
    #plt.plot(datetime_array, S_array_F, label="Snow (Flat)",color='blue',linestyle = "--") 
    
    #plt.plot(datetime_array, ICE_array_N, label="Ice (Normal)",color='yellow') 
    #plt.plot(datetime_array, ICE_array_F, label="Ice (Flat)",color='yellow',linestyle = "--") 

    #plt.plot(datetime_array, GRA_array_N, label="Graupel (Normal)",color='black') 
    #plt.plot(datetime_array, GRA_array_F, label="Graupel (Flat)",color='black',linestyle = "--") 


    print(len(WV_array_N))
    #plt.plot(WV_array_N[0], range(len(WV_array_N[0])), label="Normal")
    #plt.plot(WV_array_F[0], range(len(WV_array_F[0])), label="Flat", linestyle="dashed")
    #plt.plot(WV_array_F[0] - WV_array_N[0], range(len(WV_array_F[0])), label="Flat", linestyle="dashed")


    #plt.xlabel("Mean Water Vapor (g/kg)")
    #plt.ylabel("Model Level")
    #plt.legend()
    #plt.show()

    # Formatting
    plt.xlabel("Time",fontsize=16,fontweight='bold')
    plt.ylabel("Water Vapor Mass in Subregion (kg)",fontsize=16,fontweight='bold')
    plt.title("Change in Mass of Water Vapor (kg)", fontsize=22,fontweight='bold')
    plt.xticks(datetime_array[::3],rotation=45)
    plt.legend()
    plt.grid()

    # Format it for a filename (no spaces/colons)
    time_str = matched_time.strftime("%Y-%m-%d_%H-%M-%S")
    # Use in filename
    filename = f"integratewv.png"

    plt.savefig(savepath+filename)
    plt.show()
    print(datetime_array)
    
  
      
