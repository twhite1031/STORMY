import numpy as np
from netCDF4 import Dataset
import wrffuncs
from datetime import datetime
import pandas as pd
from wrf import getvar
# --- USER INPUT ---
wrf_date_time = datetime(1997,1,12,1,52,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = wrffuncs.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference between model times and input time
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]

print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")


# Constants
g = 9.81  # gravity (m/s^2)

# Open your WRF output file
wrf_file = Dataset(matched_file)

# Load relevant variables
PH = getvar(wrf_file, "PH", timeidx=matched_timeidx)     # [bottom_top_stag, south_north, west_east]
PHB = getvar(wrf_file, "PHB", timeidx=matched_timeidx) 
HGT = getvar(wrf_file, "ter", timeidx=matched_timeidx)    # [south_north, west_east]
hgt_agl =  getvar(wrf_file, "height_agl", timeidx=matched_timeidx) 
Z = getvar(wrf_file, "z", timeidx=matched_timeidx)
# Calculate full geopotential height (Z) at w-levels
z_full = (PH + PHB) / g  # in meters, shape: (Time, nz+1, ny, nx)

z_mass = 0.5 * (z_full[:-1, :, :] + z_full[1:, :, :])    # (nz, ny, nx)
height_agl = z_mass - HGT 

k = 0      # bottom mass level
j, i = 100, 100  # grid point


print(f"Z @ level {k} (MSL): {z_mass[k, j, i]:.2f} m")
print(f"Terrain height (HGT): {HGT[j, i]:.2f} m")
print(f"Height AGL (computed): {height_agl[k, j, i]:.2f} m")
print(f"Consistency check: Z - HGT = {z_mass[k, j, i] - HGT[j, i]:.2f} m")
print(f"WRF native height_agl: {hgt_agl[k, j, i]:.2f} m")
print(f"WRF native height: {Z[k,j,i]:.2f} m")
