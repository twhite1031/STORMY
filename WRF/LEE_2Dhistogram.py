from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from wrf import (to_np,interplevel, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import wrffuncs
from datetime import datetime
from metpy.plots import ctables
import pandas as pd
from collections import defaultdict
import numpy as np

# --- USER INPUT ---
wrf_date_time  = datetime(2022,11,18,00,00)
domain = 2

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# !!! Not Used Right Now !!!
# Build/Find the time data for the model runs
time_df = wrffuncs.build_time_df(path, domain)
obs_time = pd.to_datetime(wrf_date_time)

# Compute absolute time difference
closest_idx = (time_df["time"] - obs_time).abs().argmin()

# Extract the matched row
match = time_df.iloc[closest_idx]

# Unpack matched file info
matched_file = match["filename"]
matched_timeidx = match["timeidx"]
matched_time = match["time"]

print(f"Closest match: {matched_time} in file {matched_file} at time index {matched_timeidx}")


# Read the raw data CSV
df = pd.read_csv(savepath + "cloud_var_rawdata_202211181330_202211181400.csv")

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
        plt.savefig(f"{group}_{var}_2Dhist_CFA.png")
        plt.show()
        plt.close()
