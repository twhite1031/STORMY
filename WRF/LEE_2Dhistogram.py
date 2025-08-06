from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np,interplevel, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import STORMY
from datetime import datetime
from metpy.plots import ctables
import pandas as pd
from collections import defaultdict
import numpy as np

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,18,00,00), datetime(2022,11,19,12,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
datapath = "/data2/white/DATA/PROJ_LEE/IOP_2/CLOUDDATA/"

# --- END USER INPUT ---


# Read the raw data CSV
df = pd.read_csv(datapath + "cloud_var_rawdata_202211180000_202211191200.csv")

# Build a nested dictionary: data_dict[group][variable] = list of (value, height)
data_dict = defaultdict(lambda: defaultdict(list))

# Read through each row to grab the value and associated height
for _, row in df.iterrows():
    group = row["Group"]
    var = row["Variable"]
    time = row["Time"]

    # Skip rows where height is missing
    if pd.isna(row["Height_AGL"]):
        continue

    # No longer need to skip or separately fetch height_agl — it's already here
    value = row["Value"]
    height = row["Height_AGL"]

    data_dict[group][var].append((value, height)) # Store groups seperately based on variables, keeping the value and height for plotting


# Bin settings
height_bins = np.arange(0, 6000, 250)  # AGL in meters, 250m bins
x_bins = np.linspace(0, df["Value"].max(), 100)  # Adjust max if needed
'''
for group, variables in data_dict.items():
    for var, val_height_pairs in variables.items():
        if not val_height_pairs:
            continue

        values, heights = zip(*val_height_pairs)
        values = np.array(values)
        heights = np.array(heights)

        x_bins = np.linspace(0, np.nanmax(values), 100)
        hist_2d, xedges, yedges = np.histogram2d(values, heights, bins=[x_bins, height_bins])
        mean_heights = 0.5 * (yedges[1:] + yedges[:-1])

        # Mask 0-frequency bins for white background
        hist_masked = np.ma.masked_where(hist_2d == 0, hist_2d)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')

        # === Compute mean and median for each height bin ===
        mean_profile = []
        median_profile = []

        mean_profile_nozero = []
        median_profile_nozero = []
        for j in range(len(yedges) - 1):
            mask = (heights >= yedges[j]) & (heights < yedges[j+1]) # Ensure the value sits in the correct bin
            bin_values = values[mask]
            bin_values_nozero = bin_values[bin_values > 0]  # filter zero values

            # Check if the value exists (e.g. > 0)
            if len(bin_values) > 0:
                mean_profile.append(np.mean(bin_values))
                median_profile.append(np.median(bin_values))

                mean_profile_nozero.append(np.mean(bin_values_nozero))
                median_profile_nozero.append(np.median(bin_values_nozero))

            else:
                mean_profile.append(np.nan)
                median_profile.append(np.nan)

                mean_profile_nozero.append(np.nan)
                median_profile_nozero.append(np.nan)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_bins[:-1], mean_heights, hist_masked.T, cmap=cmap, shading='auto')
        plt.plot(mean_profile, mean_heights, color='black', label='Mean', linewidth=2)
        plt.plot(median_profile, mean_heights, color='red', linestyle='--', label='Median', linewidth=2)

        plt.plot(mean_profile_nozero, mean_heights, color='blue', label='Mean Nozero', linewidth=2)
        plt.plot(median_profile_nozero, mean_heights, color='yellow', linestyle='--', label='Median Nozero', linewidth=2)



        plt.xlabel(f"{var} (g/kg)")
        plt.ylabel("Height AGL (m)")
        plt.title(f"2D Histogram: {var} in {group}")
        plt.colorbar(label="Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath + f"{group}_{var}_2Dhist_CFA.png")
        plt.show()
        plt.close()
'''

# Setting up fixed per mixing ratio bins
var_max_dict = {}

for group, variables in data_dict.items():
    for var, val_height_pairs in variables.items():
        if not val_height_pairs:
            continue
        values, _ = zip(*val_height_pairs)
        current_max = np.nanmax(values)

        if var not in var_max_dict or current_max > var_max_dict[var]:
            var_max_dict[var] = current_max

for group, variables in data_dict.items():
    for var, val_height_pairs in variables.items():
        if not val_height_pairs:
            continue
        
        # Fixed bins for this variable
        max_val = var_max_dict[var]
        x_bins = np.linspace(0, max_val, 100)
        print(max_val)
        # === Extract values and heights ===
        values, heights = zip(*val_height_pairs)
        values = np.array(values)
        heights = np.array(heights)

        # === 2D Histogram ===
        hist_2d, xedges, yedges = np.histogram2d(values, heights, bins=[x_bins, height_bins])
        mean_heights = 0.5 * (yedges[1:] + yedges[:-1])
        
        # Sum over columns (each column is a height bin)
        col_sums = hist_2d.sum(axis=0)

        # Avoid division by zero using np.errstate and where=
        with np.errstate(divide='ignore', invalid='ignore'):
            hist_norm = np.divide(hist_2d, col_sums, where=col_sums != 0)

        # Create a 2D mask from the 1D col_sums == 0
        mask = np.tile(col_sums == 0, (hist_2d.shape[0], 1))  # shape: (99, 23)

        # Apply mask
        hist_normalized = np.ma.masked_array(hist_norm, mask=mask)
       
        # === Mask 0s for clean background ===
        hist_masked = np.ma.masked_where(hist_normalized == 0, hist_normalized)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')

        # === Compute mean and median profiles ===
        mean_profile, median_profile = [], []
        mean_profile_nozero, median_profile_nozero = [], []

        for j in range(len(yedges) - 1):
            mask = (heights >= yedges[j]) & (heights < yedges[j + 1])
            bin_values = values[mask]
            bin_values_nozero = bin_values[bin_values > 0]

            if len(bin_values) > 0:
                mean_profile.append(np.mean(bin_values))
                median_profile.append(np.median(bin_values))
                mean_profile_nozero.append(np.mean(bin_values_nozero))
                median_profile_nozero.append(np.median(bin_values_nozero))
            else:
                mean_profile.append(np.nan)
                median_profile.append(np.nan)
                mean_profile_nozero.append(np.nan)
                median_profile_nozero.append(np.nan)

        # === Plotting ===
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x_bins[:-1], mean_heights, hist_masked.T, cmap=cmap, shading='auto')
        plt.plot(mean_profile, mean_heights, color='black', label='Mean', linewidth=2)
        plt.plot(median_profile, mean_heights, color='red', linestyle='--', label='Median', linewidth=2)
        
        #plt.plot(mean_profile_nozero, mean_heights, color='blue', label='Mean Nozero', linewidth=2)
        #plt.plot(median_profile_nozero, mean_heights, color='yellow', linestyle='--', label='Median Nozero', linewidth=2)

        plt.xlabel(f"{var} (g/kg)")
        plt.ylabel("Height AGL (m)")
        plt.title(f"Normalized 2D Histogram: {var} in {group}")
        cbar = plt.colorbar(label="Relative Frequency (Normalized per Height Bin)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(savepath + f"{group}_{var}_2Dhist_NORMALIZED.png")
        plt.show()
        plt.close()
