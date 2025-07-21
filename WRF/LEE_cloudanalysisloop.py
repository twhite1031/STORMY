import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches
from cartopy import crs
from netCDF4 import Dataset
from wrf import (getvar, to_np, get_cartopy, latlon_coords, vertcross,
                 interpline, CoordPair, xy_to_ll,ll_to_xy,cartopy_xlim, cartopy_ylim)
from scipy.ndimage import label,  generate_binary_structure
import cartopy.feature as cfeature
import wrffuncs
from datetime import datetime
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import sys
import os
"""
A script used to automatically identify Lake-effect band locations based on connected cloud fractions at a certain level, 
then classifying the region as a cloud band for analysis
"""

# --- USER INPUT ---
start_time, end_time  = datetime(2022,11,17,12,00,00), datetime(2022, 11, 19,12, 00, 00)
domain = 2

INTERACTIVE = False  # Set to False for non-interactive (auto-run) mode
SHOW_FIGS = False # Do not display completed figures
USE_MAX_DBZ = True  # Toggle this to switch between lat/lon and max dBZ

# Threshold to identify the snow band (e.g., cloud fraction > .1)
threshold = 1
lat_lon = [43.86935, -76.164764]  # Coordinates to start cloud check
ht_level = 15

SIMULATION = 1 # If comparing runs
path = f"/data2/white/WRF_OUTPUTS/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"
savepath = f"/data2/white/PLOTS_FIGURES/PROJ_LEE/ELEC_IOP_2/ATTEMPT_{SIMULATION}/"

# --- END USER INPUT ---

# Build/Find the time data for the model runs
time_df = wrffuncs.build_time_df(path, domain)

# Filter time range based on start_time and end_time
time_mask = (time_df["time"] >= start_time) & (time_df["time"] <= end_time)

time_df = time_df[time_mask].reset_index(drop=True)

filelist = time_df["filename"].tolist()
timelist = time_df["time"].tolist()
timeidxlist = time_df["timeidx"].tolist()


# Define dict of variables we will store over time
cloud_with_flash = {
    "height_agl": [],
    "reflectivity": [],
    "Snow": [],
    "Graupel": [],
    "Ice": [],
    "Water Vapor": [],
    "Cloud": [],
    "Hail": []
}
cloud_without_flash = {
    "height_agl": [],
    "reflectivity": [],
    "Snow": [],
    "Graupel": [],
    "Ice": [],
    "Water Vapor": [],
    "Cloud": [],
    "Hail": []
}

cloud_all = {
    "height_agl": [],
    "reflectivity": [],
    "Snow": [],
    "Graupel": [],
    "Ice": [],
    "Water Vapor": [],
    "Cloud": [],
    "Hail": []
}

cloud_all_with_lightning = {
    "height_agl": [],
    "reflectivity": [],
    "Snow": [],
    "Graupel": [],
    "Ice": [],
    "Water Vapor": [],
    "Cloud": [],
    "Hail": []
}

cloud_all_without_lightning = {
    "height_agl": [],
    "reflectivity": [],
    "Snow": [],
    "Graupel": [],
    "Ice": [],
    "Water Vapor": [],
    "Cloud": [],
    "Hail": []
}


for idx, filename in enumerate(filelist):

    matched_time = timelist[idx]
    matched_timeidx = timeidxlist[idx]

    timestamp_str = timelist[idx].strftime("%Y%m%d_%H%M%S")

    print(f"Closest match: {matched_time} in file {filename} at time index {matched_timeidx}")
    # Get the WRF variables
    with Dataset(filename) as ds:

        # Convert desired coorindates to WRF gridbox coordinates
        x_y = ll_to_xy(ds, lat_lon[0], lat_lon[1])
        ht = getvar(ds, "z", timeidx=matched_timeidx)
        ht_agl = getvar(ds, "height_agl",timeidx=matched_timeidx)
        dbz = getvar(ds, "dbz", timeidx=matched_timeidx)
        max_dbz = getvar(ds, "mdbz", timeidx=matched_timeidx)
        lats = getvar(ds, "lat", timeidx=matched_timeidx)
        lons = getvar(ds, "lon",timeidx=matched_timeidx)
        cloud_frac = getvar(ds, "CLDFRA",timeidx=matched_timeidx) # Assuming CLDFRA is at level 10
        fed = getvar(ds,"LIGHTDENS", timeidx=matched_timeidx,meta=False)
        vert_velocity = getvar(ds, "wa",timeidx=matched_timeidx,meta=False)

        # Multiply by 1000 to go from kg/kg to g/kg while reading data in
        wv = getvar(ds, "QVAPOR", timeidx=matched_timeidx,meta=False) * 1000
        snow = getvar(ds, "QSNOW", timeidx=matched_timeidx,meta=False) * 1000
        ice = getvar(ds, "QICE", timeidx=matched_timeidx,meta=False) * 1000
        graupel  = getvar(ds, "QGRAUP", timeidx=matched_timeidx,meta=False) * 1000
        cloud = getvar(ds, "QCLOUD", timeidx=matched_timeidx, meta=False) * 1000
        hail = getvar(ds,"QHAIL", timeidx=matched_timeidx, meta=False) * 1000


    # Define dict where masks will be stored
    mask_cases = {}
    
    # Define dict of mixing ratios to look at
    mixing_ratios = {
        "Snow": snow,
        "Graupel": graupel,
        "Ice": ice,
        "Water Vapor": wv,
        "Cloud":cloud,
        "Hail":hail
    }

    def identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y):

        base_level = ht_level
        snow_band_2d = cloud_frac[base_level, :, :] >= threshold
        labeled_2d, num_features = label(snow_band_2d)
        
        if USE_MAX_DBZ:
            # Isolate Lake Ontario (LEE) Region
            lat_min, lat_max = 43, 44
            lon_min, lon_max = -78.5, -75.5

            dbz_base = dbz.isel(bottom_top=base_level)
            lats, lons = latlon_coords(dbz_base)
            region_mask = (lats >= lat_min) & (lats <= lat_max) & \
                          (lons >= lon_min) & (lons <= lon_max)

            masked_dbz = dbz_base.where(region_mask)
            print("masked_dbz shape:", masked_dbz.shape)

            # Flatten and sort dBZ values with their 2D indices
            flat_dbz = masked_dbz.values.flatten()
            sorted_indices = np.argsort(flat_dbz)[::-1]  # descending order

            found_valid_start = False

            for idx in sorted_indices:
                if np.isnan(flat_dbz[idx]):
                    continue  # skip NaNs

                y_idx, x_idx = np.unravel_index(idx, masked_dbz.shape)
                start_label = labeled_2d[y_idx, x_idx]

                if start_label != 0:
                    with Dataset(filename) as ds:
                        dbz_x_y = xy_to_ll(ds, x_idx, y_idx)

                    print(f"Starting gridbox (from ranked dBZ): x={x_idx}, y={y_idx}, Coordinates {dbz_x_y.values}, dBZ={flat_dbz[idx]}")
                    found_valid_start = True
                    break  # valid start found

            if not found_valid_start:
                print("No valid dBZ point found in a connected region.")
                return "NO_CLOUD"

        else:
            # Manual input
            start_label = labeled_2d[x_y[1], x_y[0]]

        cloud_region_mask = np.zeros_like(cloud_frac, dtype=bool)

        if start_label == 0:
            print("!!! Starting point not in any region. Skipping to the next time !!!")
            return "NO_CLOUD"
        else:
            region_mask_2d = (labeled_2d == start_label)
            cloud_region_mask[base_level, :, :] = region_mask_2d
            # Grow upward
            prev_layer = region_mask_2d.copy()
            for z in range(base_level + 1, cloud_frac.shape[0]):
                current_layer = (cloud_frac[z, :, :] >= threshold) & prev_layer
                if not np.any(current_layer):
                    break
                cloud_region_mask[z, :, :] = current_layer
                prev_layer = current_layer

            # Grow downward
            prev_layer = region_mask_2d.copy()
            for z in range(base_level - 1, -1, -1):
                current_layer = (cloud_frac[z, :, :] >= threshold) & prev_layer
                if not np.any(current_layer):
                    break
                cloud_region_mask[z, :, :] = current_layer
                prev_layer = current_layer

        # Store compled cloud mask
        mask_cases["cloud"] = cloud_region_mask

        # Index of cloud region
        z_inds, lat_inds, lon_inds = np.where(cloud_region_mask)

        # Convert FED if needed
        fed_np = fed if isinstance(fed, np.ndarray) else fed.values

        if np.any((fed_np > 0) & cloud_region_mask):
            had_lightning = True # Identify cloud as lightning cloud
            flash_mask = cloud_region_mask & (fed_np > 0)
            no_flash_mask = cloud_region_mask & (fed_np == 0)

        # Add to dictionary or process them
            mask_cases["cloudlight"] = flash_mask
            mask_cases["cloudnolight"] = no_flash_mask
        else:
            print("No flash pixels in cloud region — skipping flash mask creation.")
            had_lightning = False

        # Mean height calculations
        if np.any(cloud_region_mask):
            mean_start_height = ht.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
            mean_cloud_height = ht.where(cloud_region_mask).mean().item()
            print(f"\nBase level mean height (level {base_level}): {mean_start_height:.1f} m")
            print(f"Total cloud region mean height: {mean_cloud_height:.1f} m")

            mean_start_height_agl = ht_agl.isel(bottom_top=base_level).where(region_mask_2d).mean().item()
            mean_cloud_height_agl = ht_agl.where(cloud_region_mask).mean().item()
            print(f"\nBase level mean height agl (level {base_level}): {mean_start_height_agl:.1f} m")
            print(f"Total cloud region mean height agl: {mean_cloud_height_agl:.1f} m")

        else:
            mean_start_height = float('nan')
            mean_cloud_height = float('nan')

        return (
            mean_start_height, mean_cloud_height,
            z_inds, lat_inds, lon_inds, had_lightning
        )
    
    # Get cloud and flash masks once 
    result = identify_connected_cloud(cloud_frac, ht, ht_agl, max_dbz, fed, threshold, ht_level, x_y)

    if result == "NO_CLOUD":
        continue  # Skip to next file in loop

    (mean_start_height, mean_cloud_height, z_inds, lat_inds, lon_inds, had_lightning) = result
    
    ht_agl = to_np(ht_agl)

    cloud_mask = mask_cases["cloud"]
    target_dict = cloud_all_with_lightning if had_lightning else cloud_all_without_lightning

    for name, var in mixing_ratios.items():
        target_dict[name].append(var[cloud_mask])

    target_dict["height_agl"].append(ht_agl[cloud_mask])
    #target_dict["reflectivity"].append(max_dbz[cloud_mask])
    print("\nValues stored for clouds with or without lightning")

    # Unified loop over existing groups
    for group_name, storage_dict in [
        ("cloudlight", cloud_with_flash),
        ("cloudnolight", cloud_without_flash),
        ("cloud", cloud_all)]:

        if group_name not in mask_cases:
            continue

        mask = mask_cases[group_name]

        for name, var in mixing_ratios.items():
            storage_dict[name].append(var[mask])

        storage_dict["height_agl"].append(ht_agl[mask])
        #storage_dict["reflectivity"].append(max_dbz[mask])
    print("Values stored for clouds, cloud regions with lightning and cloud regions without lightning")
   
    def prompt_plot(prompt_text, plot_func):
    
        valid_responses = {'y', 'n', 'exit'}
    
        while True:

            if not INTERACTIVE:
                print(f"[AUTO] {prompt_text} ... running plot.")
                plot_func()

                break

            response = input(f"{prompt_text} (y/n/exit): ").strip().lower()
        
            if response not in valid_responses:
                print("Invalid input. Please type 'y' for yes, 'n' for no, or 'exit' to quit.")
                continue
        
            if response == 'y':
                plot_func()
                break
            elif response == 'n':
                print("Skipped.\n")
                break
            else:  
                print("Exiting script.")
                sys.exit()

    # === Stats for Cloud ===
    for case_name, mask in mask_cases.items():
        print(f"\n===== STATS FOR {case_name.upper()} =====")
        print(f"\n{case_name.upper()} has {len(lat_inds)} grid points")

        for name, var in mixing_ratios.items():
            flat_data = var[mask]
            flat_data = flat_data[~np.isnan(flat_data)]
            flat_data_compressed = flat_data.compressed()
            
            # Optional: print stats
            print(f"\n{name} Mean: {np.mean(flat_data_compressed):.3f}, Median: {np.median(flat_data_compressed):.3f}, Std: {np.std(flat_data_compressed):.3f}")

    # === Histograms for Cloud ===
    
    for case_name, mask in mask_cases.items():
        print(f"\n===== HISTOGRAM FOR {case_name.upper()} =====")
            
        for name, var in mixing_ratios.items():
            flat_data = var[mask]
            flat_data = flat_data[~np.isnan(flat_data)]

            def plot_q_histogram():
                plt.figure(figsize=(8, 10))
                plt.hist(flat_data, bins=50, edgecolor='black')
                plt.title(f"{name} Mixing Ratio Histogram (g/kg)")
                plt.xlabel("Mixing Ratio (g/kg)")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.suptitle(
                    f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Gridboxes: {len(lat_inds)} | T: {threshold} | "
                    f"A{SIMULATION} D{domain}"
                )
                filename = f"{case_name}_hist{name[0:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
                plt.savefig(savepath + filename)
                plt.show() if SHOW_FIGS else plt.close()

            def plot_q_2d_histogram():
                z_levels = var.shape[0]
                mix_bins = np.linspace(0, np.nanmax(flat_data), 100)
                masked_var = np.where(mask, var, np.nan)
              
                mean_heights = []
                hist_rows = []

                for z in range(z_levels):
                    if not np.any(mask[z]):
                        continue
                    level_data = masked_var[z].flatten()
                    level_data = level_data[~np.isnan(level_data)]
                    if len(level_data) == 0:
                        continue
                    hist, _ = np.histogram(level_data, bins=mix_bins)
                    hist_rows.append(hist)
                    level_heights = ht_agl[z][mask[z]]
                    mean_heights.append(np.nanmean(level_heights))

                hist_2d = np.array(hist_rows)
                mean_heights = np.array(mean_heights)

                plt.figure(figsize=(10, 8))
                plt.pcolormesh(mix_bins[:-1], mean_heights, hist_2d, shading='auto')
                plt.xlabel("Mixing Ratio (g/kg)")
                plt.ylabel("Mean Cloud Height (m)")
                plt.title(f"{name} 2D Histogram ({case_name})")
                plt.colorbar(label="Frequency")
                plt.tight_layout()
                filename2d = f"{case_name}_2Dhist{name[0:3]}_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
                plt.savefig(savepath + filename2d)
                plt.show() if SHOW_FIGS else plt.close()

            # Prompt user for each type
            prompt_plot(f"Plot {name} histogram for {case_name}?", plot_q_histogram)
            prompt_plot(f"Plot {name} 2D histogram by height for {case_name}?", plot_q_2d_histogram)
            
        flat_data = vert_velocity[mask]
        flat_data = flat_data[~np.isnan(flat_data)]

        def plot_w_histogram():
            plt.figure(figsize=(8, 10))
            plt.hist(flat_data, bins=50, edgecolor='black')
            plt.title(f"Vertical Velocity Histogram (m/s)")
            plt.xlabel("Vertical Velocity (m/s)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.suptitle(
                f"Mean height {mean_start_height:.1f}m | Level {ht_level} | Gridboxes: {len(lat_inds)} | T: {threshold} | "
                f"A{SIMULATION} D{domain}"
            )

            filename = f"{case_name}_histw_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
            plt.savefig(savepath + filename)
            plt.show() if SHOW_FIGS else plt.close()

        prompt_plot(f"Plot Vertical Velocity histogram for {case_name}?", plot_w_histogram)

         
    # === 3D Scatter Plot (with real lat/lon/height) ===

    def plot_3d_scatter():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert WRF lat/lon to NumPy arrays
        lats, lons = latlon_coords(max_dbz)
        lats, lons = to_np(lats), to_np(lons)

        # Get indices of cloud voxels
        z_inds, y_inds, x_inds = np.where(mask_cases["cloud"])

        # Extract geographic + height values
        lat_vals = lats[y_inds, x_inds]
        lon_vals = lons[y_inds, x_inds]
        height_vals = ht_agl[z_inds, y_inds, x_inds]  # meters AGL

        # Sanity checks
        print("Min height:", np.min(height_vals))
        print("Max height:", np.max(height_vals))

        # Plot point cloud
        ax.scatter(lon_vals, lat_vals, height_vals, s=2, c='cyan', alpha=.8)

        # Tidy appearance
        ax.view_init(elev=10, azim=-90)
        ax.tick_params(axis='y', pad=10)
        ax.tick_params(axis='z', pad=10)
        ax.set_yticks([])
        ax.set_xlabel("Longitude")
        ax.set_zlabel("Height AGL (m)", labelpad=15)
        ax.set_title("3D Cloud Structure with Real Coordinates")

        plt.tight_layout()
        plt.savefig(savepath + f"3Dcloudscat_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
        plt.show() if SHOW_FIGS else plt.close()

    prompt_plot(f"Plot 3D scatter of the cloud?", plot_3d_scatter)

    # === 3D Voxel Plot (grid-based, no lat/lon) ===
    '''
    def plot_3d_voxel():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Print original shape (z, y, x)
        z_dim, y_dim, x_dim = mask_cases["cloud"].shape
        print("Original cloud mask shape (z, y, x):", z_dim, y_dim, x_dim)

        # Transpose to match (x, y, z) voxel format
        cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0)) # Downsample here if needed (e.g., [::2, ::2, ::2])

        print("Voxel shape (x, y, z):", cloud_mask_vox.shape)

        # *** New Updraft Check ***
        cloud_mask_vox = np.transpose(mask_cases["cloud"], (2, 1, 0))
        strong_updraft_mask = np.transpose((mask_cases["cloud"]) & (vert_velocity > 2.0), (2, 1, 0))

        # Plot voxels
        ax.voxels(strong_updraft_mask, facecolors='magenta',alpha=0.6)
        ax.voxels(cloud_mask_vox, facecolors='lightblue', edgecolor=None, alpha=0.4)
        
        # Axes and appearance
        ax.view_init(elev=10, azim=-90)

        
        x_min, x_max = lon_inds.min(), lon_inds.max()
        y_min, y_max = lat_inds.min(), lat_inds.max()
        z_min, z_max = z_inds.min(), z_inds.max()
        
        ax.set_xlim(x_min - 5, (x_max + 5))
        ax.set_ylim(y_min - 5, (y_max + 5))
        ax.set_zlim(0, (z_max + 3))
        
        ax.set_xlabel("X (grid)")
        ax.set_ylabel("Y (grid)")
        ax.set_zlabel("Z (grid)")
        ax.set_title("3D Cloud Structure (Grid Space)")

        plt.tight_layout()
        plt.savefig(savepath + f"3Dcloudvox_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png")
        plt.show() if SHOW_FIGS else plt.close()

    prompt_plot(f"Plot 3D voxels of the cloud?", plot_3d_voxel)
    '''
    # === Plan View Map of Cloud and flash regions

    def plot_plan_cloud():
        
        # Collapse cloud mask (this one should always exist)
        cloud_mask_2d = np.max(mask_cases["cloud"], axis=0)

        # Optional masks: flash_mask and no_flash_mask
        flash_mask_2d = np.max(mask_cases["cloudlight"], axis=0) if "cloudlight" in mask_cases else None
            
        # Get the lat/lon points 
        lats, lons = latlon_coords(max_dbz)

        # Get the cartopy projection object
        cart_proj = get_cartopy(max_dbz)

        # Create the figure
        fig = plt.figure(figsize=(12,9))
        ax = plt.axes(projection=cart_proj)

        # Download and create the states, land, and oceans using cartopy features
        states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
                                          facecolor='none',
                                          name='admin_1_states_provinces')
        land = cfeature.NaturalEarthFeature(category='physical', name='land',
                                        scale='50m',
                                        facecolor=cfeature.COLORS['land'])

        lakes = cfeature.NaturalEarthFeature(category='physical',name='lakes',scale='50m',facecolor="none",edgecolor="blue")
        ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                         scale='50m',
                                        facecolor=cfeature.COLORS['water'])

        # Add Cartopy features
        ax.add_feature(land,zorder=0)
        ax.add_feature(ocean,zorder=0)
        ax.add_feature(lakes,zorder=0)
        ax.add_feature(states, edgecolor='gray',zorder=2)

        # Plot Cloud Region (red)
        cf = ax.contourf(
            to_np(lons), to_np(lats), to_np(cloud_mask_2d.astype(float)),
            levels=[0.5, 1.5], colors=['red'], alpha=0.3,
            transform=crs.PlateCarree(), zorder=4
        )

        # Plot Flash Region (yellow)
        if flash_mask_2d is not None:
            cf_flash = ax.contourf(
                to_np(lons), to_np(lats), to_np(flash_mask_2d.astype(float)),
                levels=[0.5, 1.5], colors=['yellow'], alpha=0.6,
                transform=crs.PlateCarree(), zorder=5
            )

         
        # Add patch legend
        legend_elements = [
            Patch(facecolor='red', edgecolor='red', label='Cloud Region'),
            Patch(facecolor='yellow', edgecolor='yellow', label='Flash Region')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Final touches
        ax.set_xlim(cartopy_xlim(max_dbz))
        ax.set_ylim(cartopy_ylim(max_dbz))
        plt.title("Cloud region linked to flash")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.suptitle(
            f"Mean height {mean_start_height} index {ht_level} | {len(lat_inds)} gridboxes | Threshold {threshold} | Attempt {SIMULATION}\n"
            f"Starting Check at {lat_lon[0]} {lat_lon[1]}"
        )

        # Save the figure
        os.makedirs(savepath, exist_ok=True)
        filename = f"plancloud_ht{ht_level}T{threshold}_A{SIMULATION}D{domain}_{timestamp_str}.png"
        plt.savefig(os.path.join(savepath, filename), dpi=150)
        plt.show() if SHOW_FIGS else plt.close()

    prompt_plot("Plot plan view of the cloud highlighting flash regions?", plot_plan_cloud)

def flatten_group(group_dict):
    for key in group_dict:
        if len(group_dict[key]) == 0:
            print(f"Warning: No data to concatenate for {key}. Skipping.")
            group_dict[key] = np.array([])  # Or continue if you prefer to leave it untouched
        else:
            group_dict[key] = np.concatenate(group_dict[key])
    


# === Summary statistics collector ===
summary_stats = []

all_groups = {
    "cloud_with_flash": cloud_with_flash,
    "cloud_without_flash": cloud_without_flash,
    "cloud_all_with_lightning": cloud_all_with_lightning,
    "cloud_all_without_lightning": cloud_all_without_lightning,
    "cloud_all": cloud_all,
}

for group_name, group_dict in all_groups.items():
    flatten_group(group_dict)

    for var_name, values in group_dict.items():
        if values.size == 0:
            continue

        stats = {
            "Group": group_name,
            "Variable": var_name,
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Std Dev": np.std(values),
            "Min": np.min(values),
            "Max": np.max(values),
        }
        summary_stats.append(stats)

# === Save statistics as CSV ===
df_stats = pd.DataFrame(summary_stats)

# Set print options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.3f}".format)

# Print to terminal
print("\n=== Summary Statistics for Cloud Variables ===\n")
print(df_stats.to_string(index=False))
df_stats.to_csv(os.path.join(savepath, "cloud_variable_stats.csv"))
print(f"Saved histograms and summary statistics in: {savepath}")

